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

"""Solar System Object Query to Sorcha in place of a internal Rubin solar
system object caching/retrieval code.

Will compute the location for of known SSObjects within a visit-detector combination.
"""

__all__ = ["SorchaEphemerisQueryDRPConfig", "SorchaEphemerisQueryDRPTask"]


import os
import pandas as pd
import numpy as np
import pyarrow as pa
from importlib import resources
from time import sleep

#new packages to add from jake's code
import sqlite3
import lsst.daf.butler as dafButler
import astropy.table as tb
import matplotlib.pyplot as plt
from jpl_small_bodies_de441_n16 import _de441_n16_md5, de441_n16   
from mpc_obscodes import _mpc_obscodes_md5, mpc_obscodes
from naif_de440 import _de440_md5, de440
from naif_eop_high_prec import _eop_high_prec_md5, eop_high_prec
from naif_eop_historical import _eop_historical_md5, eop_historical
from naif_eop_predict import _eop_predict_md5, eop_predict
from naif_leapseconds import _leapseconds_md5, leapseconds
import tempfile
import time
import glob
from time import sleep
import os
#end of jake's code imports
from lsst.geom import SpherePoint, degrees
import lsst.pex.config as pexConfig
import lsst.pipe.tasks
from lsst.pipe.tasks.associationUtils import obj_id_to_ss_object_id
from lsst.sphgeom import ConvexPolygon
from lsst.utils.timer import timeMethod

from lsst.pipe.base import connectionTypes, NoWorkFound, PipelineTask, \
    PipelineTaskConfig, PipelineTaskConnections, Struct


class SorchaEphemerisQueryDRPConnections(PipelineTaskConnections,
                                        dimensions=("instrument",
                                                )):
    visitInfo = connectionTypes.Input(
        doc="Summary of visit information including ra, dec, and time",
        name="preliminary_visit_image.visitInfo",
        storageClass="VisitInfo",
        dimensions={"instrument", "visit"},
        multiple = True
    )

    inputOrbits = connectionTypes.Input(
        doc="Minor Planet Center orbit table used for association",
        name="mpcorb",
        storageClass="DataFrame",
        dimensions={},
    )

    ssObjects = connectionTypes.Output(
        doc="Sorcha-provided Solar System objects observable in this detector-visit",
        name="preloaded_ss_object_visit",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
    )


class SorchaEphemerisQueryDRPConfig(
        PipelineTaskConfig,
        pipelineConnections=SorchaEphemerisQueryDRPConnections):
    observerCode = pexConfig.Field(
        dtype=str,
        doc="IAU Minor Planet Center observer code for queries "
            "(default is X05 for Rubin Obs./LSST)",
        default='X05'
    )
    queryBufferRadiusDegrees = pexConfig.Field(
        dtype=float,
        doc="Buffer radius in degrees added to detector bounding circle for ephemeris "
            "cone search. Defaults to 10 deg/day * 30 minutes",
        default=0.208
    )



class SorchaEphemerisQueryDRPTask(PipelineTask):
    """Task to query the Sorcha service and retrieve the solar system objects
    that are observable within the input visit.
    """
    ConfigClass = SorchaEphemerisQueryDRPConfig
    _DefaultName = "SorchaEphemerisQueryDRP"

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
        n = len(outputs.ssObjects)

        for i in range(n):
            dataId = outputRefs.ssObjects[i]
            butlerQC.put(outputs.ssObjects[i], dataId)
            

    @timeMethod
    def run(self, visitInfo, inputOrbits):
        """Parse the information on the current visit and retrieve the
        observable solar system objects from Sorcha.

        Parameters
        ----------
        finalVisitSummary : `lsst.afw.table.ExposureCatalog`
            visitInfo including center and time of exposure

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
        fieldRA = np.array([vi.getBoresightRaDec().getRa().asDegrees() for vi in visitInfo])
        fieldDec = np.array([vi.getBoresightRaDec().getDec().asDegrees() for vi in visitInfo])
        epochMJD = np.array([vi.date.toAstropy().tai.mjd for vi in visitInfo])

        queryRadius = self.config.queryBufferRadiusDegrees

        # Confused about seconds/days units here
        n = len(epochMJD)
        visitTime = np.ones(n) * 30.0  # seconds
        inputVisits = pd.DataFrame({
            "observationMidpointMJD": epochMJD,
            "fieldRA": fieldRA,
            "fieldDec": fieldDec,
            "observationId": np.arange(n),
            "visitTime" : np.ones(n) * visitTime,
            "observationStartMJD": epochMJD - (visitTime / 2) / 86400.0,
            "visitExposureTime": visitTime,
            "filter": ["r"] * n,
            "seeingFwhmGeom": [-1] * n,
            "seeingFwhmEff": [-1] * n,
            "fiveSigmaDepth": [-1] * n,
            "rotSkyPos": [-1] * n,
        })

        # Colors exactly like jake's prep_input_colors
        inputColors = inputOrbits[["ObjID"]].copy()
        inputColors["H_r"] = 0
        inputColors["GS"] = 0.15
        eph_str = resources.files(lsst.pipe.tasks).parents[3].joinpath("data/eph.ini").read_text()
        pck_str = resources.files(lsst.pipe.tasks).parents[3].joinpath('data/pck00010.pck').read_text()
        # same as code
        with tempfile.TemporaryDirectory() as tmpdirname:
            print('temp dir:', tmpdirname)
            inputOrbits = inputOrbits.iloc[:10]
            inputColors = inputColors.iloc[:10]
            # Orbits
            print(inputOrbits)
            print(inputColors)

            inputOrbits.to_csv(f'{tmpdirname}/orbits.csv', index=False)
            # Observations SQLite
            conn = sqlite3.connect(f'{tmpdirname}/pointings.db')
            inputVisits.to_sql('observations', conn, if_exists='replace', index=False)
            conn.close()
            # eph.ini
            open(f'{tmpdirname}/eph.ini', 'w').write(eph_str)
            # Colors
            inputColors.to_csv(f'{tmpdirname}/colors.csv', index=False)

            # Kernels/cache exactly like the code
            print('making', f'{tmpdirname}/sorcha_cache')
            os.mkdir(f'{tmpdirname}/sorcha_cache')
            os.system(f'cp {de441_n16} {tmpdirname}/sorcha_cache')
            os.system(f'cp {mpc_obscodes} {tmpdirname}/sorcha_cache')
            os.system(f'cp {de440} {tmpdirname}/sorcha_cache')
            os.system(f'cp {eop_high_prec} {tmpdirname}/sorcha_cache')
            os.system(f'cp {eop_historical} {tmpdirname}/sorcha_cache')
            os.system(f'cp {eop_predict} {tmpdirname}/sorcha_cache')
            os.system(f'cp {leapseconds} {tmpdirname}/sorcha_cache')
            open(f'{tmpdirname}/pck00010.pck', 'w').write(pck_str)
            print('copied everything to', tmpdirname, 'sleeping.')
            sleep(60)

            import subprocess
            print('sorcha running')

            result = subprocess.run(
                [
                    "sorcha",
                    "run",
                    "-c", f"{tmpdirname}/eph.ini",
                    "-o", f"{tmpdirname}/",
                    "--ob", f"{tmpdirname}/orbits.csv",
                    "-p", f"{tmpdirname}/colors.csv",
                    "--pd", f"{tmpdirname}/pointings.db",
                    "--ew", f"{tmpdirname}/eph.csv",
                    "--ar", f"{tmpdirname}/sorcha_cache"
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            print(" Sorcha STDOUT:\n", result.stdout)
            print(" Sorcha STDERR:\n", result.stderr)
            
            eph_path = f'{tmpdirname}/eph.csv.csv'
            
            if not os.path.exists(eph_path):
                raise FileNotFoundError(
                    f" Sorcha did not create eph.csv.csv. Check STDOUT/STDERR above. Directory contents:\n{os.listdir(tmpdirname)}"
                )

            # Return Sorcha output directly
            SorchaSsObjects = pd.read_csv(eph_path)

        return Struct(ssObjects=SorchaSsObjects)


    

