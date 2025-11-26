# This file is part of ap_association.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Generate per-visit solar system ephemerides using Sorcha.

Sorcha is an open-source solar system survey simulation tool, which
efficiently generates ephemerides for solar system objects from input orbits
and observation pointings. It is currently the fastest robust and maintained
ephemeris generator, and can fit our use case well.

Sorcha is a command line tool, and is not designed for direct use in python.
This task creates a temporary directory in which to run Sorcha as designed,
providing it with all required input files and reading the Sorcha-generated
ephemerides from a csv. While awkward and un-pipetask-like, it works and takes
advantage of Sorcha's as-designed speed.

Eventually, this should be replaced with adam_core's ephemeris generation
tools which propagate forward orbital uncertainty into on-sky ellipses.
Doing so will require re-implementing the healpix binning method described
in https://arxiv.org/abs/2506.02140. Doing so will not only improve this code
but also allow on-sky uncertainties to be used during association. When this is
done, mpsky should be modified to do the same.
"""

__all__ = ["GenerateEphemeridesConfig", "GenerateEphemeridesTask"]


import numpy as np
import os
import pandas as pd
from importlib import resources
from sqlite3 import connect as sqlite_connect
from subprocess import run, PIPE
from tempfile import TemporaryDirectory
from textwrap import dedent


from lsst.pex.config import Field
from lsst.pipe.base import connectionTypes, NoWorkFound, PipelineTask, \
    PipelineTaskConfig, PipelineTaskConnections, Struct
import lsst.pipe.tasks
from lsst.utils.timer import timeMethod


class GenerateEphemeridesConnections(PipelineTaskConnections,
                                     dimensions=("instrument",)):

    visitSummaries = connectionTypes.Input(
        doc="Summary of visit information including ra, dec, and time",
        name="preliminary_visit_summary",
        storageClass="ExposureCatalog",
        dimensions={"instrument", "visit"},
        multiple=True
    )
    mpcorb = connectionTypes.Input(
        doc="Minor Planet Center orbit table used for association",
        name="mpcorb",
        storageClass="DataFrame",
        dimensions={},
    )

    # The following 9 prequisite inputs are Sorcha's required auxiliary files.
    de440s = connectionTypes.PrerequisiteInput(
        doc="NAIF DE440 ephemeris file (de440s.bsp)",
        name="de440s",
        storageClass="SSPAuxiliaryFile",
        dimensions={},
    )
    sb441_n16 = connectionTypes.PrerequisiteInput(
        doc="NAIF DE440 ephemeris file (sb441_n16.bsp)",
        name="sb441_n16",
        storageClass="SSPAuxiliaryFile",
        dimensions={},
    )
    obsCodes = connectionTypes.PrerequisiteInput(
        doc="MPC observatory code file (ObsCodes.json)",
        name="obscodes",
        storageClass="SSPAuxiliaryFile",
        dimensions={},
    )
    linux_p1550p2650 = connectionTypes.PrerequisiteInput(
        doc="TODO (linux_p1550p2650.440)",
        name="linux_p1550p2650",
        storageClass="SSPAuxiliaryFile",
        dimensions={},
    )
    pck00010 = connectionTypes.PrerequisiteInput(
        doc="orientation of planets, moons, the Sun, and selected asteroids. (pck00010.pck)",
        name="pck00010",
        storageClass="SSPAuxiliaryFile",
        dimensions={},
    )
    earth_latest_high_prec = connectionTypes.PrerequisiteInput(
        doc="High-precision Earth orientation parameters (EOP) kernel",
        name="earth_latest_high_prec",
        storageClass="SSPAuxiliaryFile",
        dimensions={},
    )
    earth_620120_250826 = connectionTypes.PrerequisiteInput(
        doc="Historical EOP",
        name="earth_620120_250826",
        storageClass="SSPAuxiliaryFile",
        dimensions={},
    )
    earth_2025_250826_2125_predict = connectionTypes.PrerequisiteInput(
        doc="Longterm EOP predictions",
        name="earth_2025_250826_2125_predict",
        storageClass="SSPAuxiliaryFile",
        dimensions={},
    )
    naif0012 = connectionTypes.PrerequisiteInput(
        doc="Leapsecond tls file",
        name="naif0012",
        storageClass="SSPAuxiliaryFile",
        dimensions={},
    )

    ssObjects = connectionTypes.Output(
        doc="Sorcha-provided Solar System objects observable in this detector-visit",
        name="preloaded_ss_object_visit",
        storageClass="DataFrame",
        dimensions=("instrument", "visit"),
        multiple=True,
    )


class GenerateEphemeridesConfig(
        PipelineTaskConfig,
        pipelineConnections=GenerateEphemeridesConnections):
    observatoryCode = Field(
        dtype=str,
        doc="IAU Minor Planet Center observer code for queries "
            "(default is X05 for Rubin Obs./LSST)",
        default='X05'
    )
    observatoryFOVRadius = Field(
        dtype=float,
        doc="The field of view of the observatory (degrees)",
        default=2.06,
    )


class GenerateEphemeridesTask(PipelineTask):
    """Task to query the Sorcha service and retrieve the solar system objects
    that are observable within the input visit.
    """
    ConfigClass = GenerateEphemeridesConfig
    _DefaultName = "GenerateEphemerides"

    def runQuantum(
        self,
        butlerQC,
        inputRefs,
        outputRefs,
    ):
        """Do butler IO and transform to provide in memory
        objects for tasks `~Task.run` method.

        Parameters
        ----------
        butlerQC : `QuantumContext`
            A butler which is specialized to operate in the context of a
            `lsst.daf.butler.Quantum`.
        inputRefs : `InputQuantizedConnection`
            Datastructure whose attribute names are the names that identify
            connections defined in corresponding `PipelineTaskConnections`
            class. The values of these attributes are the
            `lsst.daf.butler.DatasetRef` objects associated with the defined
            input/prerequisite connections.
        outputRefs : `OutputQuantizedConnection`
            Datastructure whose attribute names are the names that identify
            connections defined in corresponding `PipelineTaskConnections`
            class. The values of these attributes are the
            `lsst.daf.butler.DatasetRef` objects associated with the defined
            output connections.
        """
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        for ref in outputRefs.ssObjects:
            dataId = ref.dataId
            ephemeris_visit = outputs.ssObjects[dataId['visit']]
            butlerQC.put(ephemeris_visit, ref)

    @timeMethod
    def run(self, visitSummaries, mpcorb, de440s, sb441_n16, obsCodes, linux_p1550p2650, pck00010,
            earth_latest_high_prec, earth_620120_250826, earth_2025_250826_2125_predict, naif0012):
        """Parse the information on the current visit and retrieve the
        observable solar system objects from Sorcha.

        Parameters
        ----------
        visitSummary : `lsst.afw.table.ExposureCatalog`
            Has rows with .getVisitInfo, which give the center and time of exposure

        mpcorb, de440s, sb441_n16, obsCodes, linux_p1550p2650, pck00010,
            earth_latest_high_prec, earth_620120_250826, earth_2025_250826_2125_predict,
            naif0012 : `lsst.pipe.tasks.sspAuxiliaryFile.SSPAuxiliaryFile`s
            Minor Planet Center orbit table used for association

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results struct with components:

            - ``ssObjects`` : `pandas.DataFrame`
                DataFrame containing Solar System Objects near the detector
                footprint as retrieved by Sorcha. The columns are as follows:

                ``Name``
                    object name (`str`)
                ``ra``
                    RA in decimal degrees (`float`)
                ``dec``
                    DEC in decimal degrees (`float`)

        """
        if len(visitSummaries) == 0:
            raise NoWorkFound("No visits input!")
        visitInfos = [vs[0].getVisitInfo() for vs in visitSummaries]
        fieldRA = np.array([vi.getBoresightRaDec().getRa().asDegrees() for vi in visitInfos])
        fieldDec = np.array([vi.getBoresightRaDec().getDec().asDegrees() for vi in visitInfos])
        epochMJD = np.array([vi.date.toAstropy().tai.mjd for vi in visitInfos])
        visits = [vi.id for vi in visitInfos]

        # Confused about seconds/days units here
        n = len(epochMJD)
        visitTime = np.ones(n) * 30.0  # seconds
        inputVisits = pd.DataFrame({
            "observationMidpointMJD": epochMJD,
            "fieldRA": fieldRA,
            "fieldDec": fieldDec,
            "observationId": visits,
            "visitTime": np.ones(n) * visitTime,
            "observationStartMJD": epochMJD - (visitTime / 2) / 86400.0,
            "visitExposureTime": visitTime,

            # The following columns are required by Socha but only used after ephemeris generation
            "filter": ["r"] * n,
            "seeingFwhmGeom": [-1] * n,
            "seeingFwhmEff": [-1] * n,
            "fiveSigmaDepth": [-1] * n,
            "rotSkyPos": [-1] * n,
        })

        inputOrbits = mpcorb[
            ['packed_primary_provisional_designation', 'q', 'e', 'i',
             'node', 'argperi', 'peri_time', 'epoch_mjd']
        ].rename(columns={'packed_primary_provisional_designation': 'ObjID', 'epoch_mjd': 'epochMJD_TDB',
                          'i': 'inc', 'argperi': 'argPeri', 'peri_time': 't_p_MJD_TDB'})
        inputOrbits['FORMAT'] = 'COM'
        # Colors exactly like jake's prep_input_colors
        inputColors = inputOrbits[["ObjID"]].copy()
        inputColors["H_r"] = mpcorb['h']
        inputColors["GS"] = 0.15  # Traditional

        eph_str = resources.files(lsst.pipe.tasks).parents[3].joinpath("data/eph.ini").read_text()
        eph_str = eph_str.replace("{OBSCODE}", self.config.observatoryCode)
        eph_str = eph_str.replace("{FOV}", str(self.config.observatoryFOVRadius))

        with TemporaryDirectory() as tmpdirname:
            self.log.info(f'temp dir: {tmpdirname}')

            # Orbits
            inputOrbits.to_csv(f'{tmpdirname}/orbits.csv', index=False)
            # Observations SQLite
            conn = sqlite_connect(f'{tmpdirname}/pointings.db')
            inputVisits.to_sql('observations', conn, if_exists='replace', index=False)
            conn.close()

            with open(f'{tmpdirname}/eph.ini', 'w') as ephFile:
                ephFile.write(eph_str)

            inputColors.to_csv(f'{tmpdirname}/colors.csv', index=False)

            cache = f'{tmpdirname}/sorcha_cache/'
            self.log.info('making cache')
            os.mkdir(cache)
            # DONE
            for filename, fileref in [
                ('de440s.bsp', de440s),
                ('sb441-n16.bsp', sb441_n16),
                ('ObsCodes.json', obsCodes),
                ('linux_p1550p2650.440', linux_p1550p2650),
                ('pck00010.pck', pck00010),
                ('earth_latest_high_prec.bpc', earth_latest_high_prec),
                ('earth_620120_250826.bpc', earth_620120_250826),
                ('earth_2025_250826_2125_predict.bpc', earth_2025_250826_2125_predict),
                ('naif0012.tls', naif0012),
            ]:
                self.log.info(f'writing {filename}')
                with open(cache + filename, 'wb') as file:
                    file.write(fileref.fileContents.read())

            abspath = f'{tmpdirname}/sorcha_cache/'
            split = 79
            n_iter = int(len(str(abspath)) / split)  # Number of splits required
            for n in range(n_iter, 0, -1):
                abspath = abspath[:split * n] + "+' '" + abspath[split * n:]

            meta_kernel_text = dedent(f"""\
                                      \\begindata

                                      PATH_VALUES = ('{abspath}')

                                      PATH_SYMBOLS = ('A')

                                      KERNELS_TO_LOAD=(
                                          '$A/naif0012.tls',
                                          '$A/earth_620120_250826.bpc',
                                          '$A/earth_2025_250826_2125_predict.bpc',
                                          '$A/pck00010.pck',
                                          '$A/de440s.bsp',
                                          '$A/earth_latest_high_prec.bpc',
                                      )

                                      \\begintext
                                      """)
            with open(f'{tmpdirname}/sorcha_cache/meta_kernel.txt', 'w') as meta_kernel_file:
                meta_kernel_file.write(meta_kernel_text)
            self.log.info('Sorcha process begun')

            result = run(
                sorcha_run
                + [
                    "-c", f"{tmpdirname}/eph.ini",
                    "-o", f"{tmpdirname}/",
                    "--ob", f"{tmpdirname}/orbits.csv",
                    "-p", f"{tmpdirname}/colors.csv",
                    "--pd", f"{tmpdirname}/pointings.db",
                    "--ew", f"{tmpdirname}/ephemeris",
                    "--ar", f"{tmpdirname}/sorcha_cache/"
                ],
                stdout=PIPE,
                stderr=PIPE,
                text=True
            )

            self.log.info(f"Sorcha STDOUT:\n {result.stdout}")
            self.log.info(f"Sorcha STDERR:\n {result.stderr}")

            eph_path = f'{tmpdirname}/ephemeris.csv'
            if not os.path.exists(eph_path):
                raise FileNotFoundError(
                    " Sorcha did not create ephemeris. Check STDOUT/STDERR above. "
                    f"Directory contents:\n{os.listdir(tmpdirname)}"
                )

            # Return Sorcha output directly
            ephemeris = pd.read_csv(eph_path)
        perVisitSsObjects = {FieldID: group for FieldID, group in ephemeris.groupby("FieldID")}
        for v in visits:
            if v not in perVisitSsObjects:
                perVisitSsObjects[v] = ephemeris.iloc[:0]
        return Struct(ssObjects=perVisitSsObjects)
