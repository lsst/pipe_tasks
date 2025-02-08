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

"""Solar System Object Query to MPSky in place of a internal Rubin solar
system object caching/retrieval code.

Will compute the location for of known SSObjects within a visit-detector combination.
"""

__all__ = ["MPSkyEphemerisQueryConfig", "MPSkyEphemerisQueryTask"]


import os
import pandas as pd
import pyarrow as pa
import requests

from lsst.ap.association.utils import getMidpointFromTimespan, objID_to_ssObjectID
from lsst.geom import SpherePoint
import lsst.pex.config as pexConfig
from lsst.utils.timer import timeMethod

from lsst.pipe.base import connectionTypes, NoWorkFound, PipelineTask, \
    PipelineTaskConfig, PipelineTaskConnections, Struct


class MPSkyEphemerisQueryConnections(PipelineTaskConnections,
                                     dimensions=("instrument",
                                                 "visit", "detector")):
    finalVisitSummary = connectionTypes.Input(
        doc="Summary of visit information including ra, dec, and time",
        name="finalVisitSummary",
        storageClass="ExposureCatalog",
        dimensions={"instrument", "visit", "detector"},
    )

    ssObjects = connectionTypes.Output(
        doc="MPSky-provided Solar System objects observable in this detector-visit",
        name="preloaded_SsObjects",
        storageClass="DataFrame",
        dimensions=("instrument", "visit", "detector"),
    )


class MPSkyEphemerisQueryConfig(
        PipelineTaskConfig,
        pipelineConnections=MPSkyEphemerisQueryConnections):
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
    mpSkyRequestTimeoutSeconds = pexConfig.Field(
        dtype=float,
        doc="Time in seconds to wait for mpSky request before failing ",
        default=1.0
    )
    mpSkyFallbackURL = pexConfig.Field(
        dtype=str,
        doc="mpSky default URL if MP_SKY_URL environment variable unset",
        default="http://sdfiana014.sdf.slac.stanford.edu:3666/ephemerides/"
    )


class MPSkyEphemerisQueryTask(PipelineTask):
    """Task to query the MPSky service and retrieve the solar system objects
    that are observable within the input visit.
    """
    ConfigClass = MPSkyEphemerisQueryConfig
    _DefaultName = "mpSkyEphemerisQuery"

    def runQuantum(
        self,
        butlerQC: QuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
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
        detector = butlerQC.quantum.dataId["detector"]
        outputs = self.run(**inputs, detector)
        butlerQC.put(outputs, outputRefs)


    @timeMethod
    def run(self, finalVisitSummary, detector):
        """Parse the information on the current visit and retrieve the
        observable solar system objects from MPSky.

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
                footprint as retrieved by MPSky. The columns are as follows:

                ``Name``
                    object name (`str`)
                ``ra``
                    RA in decimal degrees (`float`)
                ``dec``
                    DEC in decimal degrees (`float`)
                ``obj_X_poly``, ``obj_Y_poly``, ``obj_Z_poly``
                    Chebyshev coefficients for object path
                ``obs_X_poly``, ``obs_Y_poly``, ``obs_Z_poly``
                    Chebyshev coefficients for observer path
                ``t_min``
                    Lower time bound for polynomials
                ``t_max``
                    Upper time bound for polynomials
        """
        row = finalVisitSummary.find(detector)
        visitInfo = row.visitInfo
        ra, dec = row['ra'], row['dec']
        corners = np.vstack([row['raCorners'], row['decCorners']]).T
        corner_coords = []
        for corner in corners:
            corner_coords.append(SpherePoint(corner[0], corner[1], units=degrees).getVector())
        detectorPolygon = ConvexPolygon(corner_coords)
        radius = detectorPolygon.getBoundingCircle().getOpeningAngle().asDegrees()
        expMidPointEPOCH = visitInfo.date.toAstropy().mjd

        # MPSky service query
        mpSkyURL = os.environ.get('MP_SKY_URL', self.config.mpSkyFallbackURL)
        mpSkySsObjects = self._mpSkyConeSearch(expCenter, expMidPointEPOCH,
                                               expRadius + self.config.queryBufferRadiusDegrees, mpSkyURL)
        return Struct(
            ssObjects=mpSkySsObjects,
        )

    def read_mp_sky_response(self, response):
        """Extract ephemerides from an MPSky web request

        Parameters
        ----------
        response : `requests.Response`
            MPSky message

        Returns
        -------
        objID : `np.ndarray`
            Designations of nearby objects
        ra : `np.ndarray`
            Array of object right ascensions
        dec : `np.ndarray`
            Array of object declinations
        object_polynomial : `np.ndarray`, (N,M)
            Array of object cartesian position polynomials
        observer_polynomial : `np.ndarray`, (N,M)
            Array of observer cartesian position polynomials
        t_min : `np.ndarray`
            Lower time bound for polynomials
        t_max : `np.ndarray`
            Upper time bound for polynomials

        """
        with pa.input_stream(memoryview(response.content)) as stream:
            stream.seek(0)
            object_polynomial = pa.ipc.read_tensor(stream).to_numpy()
            observer_polynomial = pa.ipc.read_tensor(stream).to_numpy()
            with pa.ipc.open_stream(stream) as reader:
                columns = next(reader)
        objID = columns["name"].to_numpy(zero_copy_only=False)
        ra = columns["ra"].to_numpy()
        dec = columns["dec"].to_numpy()
        t_min = columns["tmin"].to_numpy()
        t_max = columns["tmax"].to_numpy()
        return objID, ra, dec, object_polynomial, observer_polynomial, t_min, t_max

    def _mpSkyConeSearch(self, expCenter, epochMJD, queryRadius, mpSkyURL):
        """Query MPSky ephemeris service for objects near the expected detector position

        Parameters
        ----------
        expCenter : `lsst.geom.SpherePoint`
            Center of search cone
        epochMJD : `float`
            Epoch of cone search, (MJD in UTC).
        queryRadius : `float`
            Radius of the cone search in degrees.
        mpSkyURL : `str`
            URL to query for MPSky.

        Returns
        -------
        mpSkySsObjects : `pandas.DataFrame`
            DataFrame with Solar System Object information and RA/DEC position
            within the visit.
        """
        fieldRA = expCenter.getRa().asDegrees()
        fieldDec = expCenter.getDec().asDegrees()

        params = {
            "t": epochMJD,
            "ra": fieldRA,
            "dec": fieldDec,
            "radius": queryRadius
        }

        try:
            response = requests.get(mpSkyURL, params=params, timeout=self.config.mpSkyRequestTimeoutSeconds)
            response.raise_for_status()
            response = self.read_mp_sky_response(response)
            objID, ra, dec, object_polynomial, observer_polynomial, tmin, tmax = response

            mpSkySsObjects = pd.DataFrame()
            mpSkySsObjects['ObjID'] = objID
            mpSkySsObjects['ra'] = ra
            mpSkySsObjects['obj_x_poly'] = [poly[0] for poly in object_polynomial.T]
            mpSkySsObjects['obj_y_poly'] = [poly[1] for poly in object_polynomial.T]
            mpSkySsObjects['obj_z_poly'] = [poly[2] for poly in object_polynomial.T]
            mpSkySsObjects['obs_x_poly'] = [observer_polynomial.T[0] for
                                            i in range(len(mpSkySsObjects))]
            mpSkySsObjects['obs_y_poly'] = [observer_polynomial.T[1] for
                                            i in range(len(mpSkySsObjects))]
            mpSkySsObjects['obs_z_poly'] = [observer_polynomial.T[2] for
                                            i in range(len(mpSkySsObjects))]
            mpSkySsObjects['tmin'] = tmin
            mpSkySsObjects['tmax'] = tmax
            mpSkySsObjects['dec'] = dec
            mpSkySsObjects['Err(arcsec)'] = 2
            mpSkySsObjects['ssObjectId'] = [objID_to_ssObjectID(v) for v in mpSkySsObjects['ObjID'].values]
            nFound = len(mpSkySsObjects)

            if nFound == 0:
                self.log.info("No Solar System objects found for visit.")
            else:
                self.log.info("%d Solar System Objects in visit", nFound)
        except requests.RequestException as e:
            raise NoWorkFound(f"Failed to connect to the remote ephemerides service: {e}") from e

        return mpSkySsObjects
