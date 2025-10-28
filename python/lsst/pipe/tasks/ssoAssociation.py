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

"""Spatial association for Solar System Objects."""

__all__ = ["SolarSystemAssociationConfig", "SolarSystemAssociationTask"]

from astropy import units as u
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import healpy as hp
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev, chebval
from scipy.spatial import cKDTree

from lsst.afw.image.exposure.exposureUtils import bbox_contains_sky_coords
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.utils.timer import timeMethod
from lsst.pipe.tasks.associationUtils import obj_id_to_ss_object_id


class SolarSystemAssociationConfig(pexConfig.Config):
    """Config class for SolarSystemAssociationTask.
    """
    maxDistArcSeconds = pexConfig.Field(
        dtype=float,
        doc='Maximum distance in arcseconds to test for a DIASource to be a '
            'match to a SSObject.',
        default=1.0,
    )
    maxPixelMargin = pexConfig.RangeField(
        doc="Maximum padding to add to the ccd bounding box before masking "
            "SolarSystem objects to the ccd footprint. The bounding box will "
            "be padded by the minimum of this number or the max uncertainty "
            "of the SolarSystemObjects in pixels.",
        dtype=int,
        default=100,
        min=0,
    )


class SolarSystemAssociationTask(pipeBase.Task):
    """Associate DIASources into existing SolarSystem Objects.

    This task performs the association of detected DIASources in a visit
    with known solar system objects.
    """
    ConfigClass = SolarSystemAssociationConfig
    _DefaultName = "ssoAssociation"

    @timeMethod
    def run(self, diaSourceCatalog, ssObjects, visitInfo, bbox, wcs):
        """Create a searchable tree of unassociated DiaSources and match
        to the nearest ssoObject.

        Parameters
        ----------
        diaSourceCatalog : `astropy.table.Table`
            Catalog of DiaSources. Modified in place to add ssObjectId to
            successfully associated DiaSources.
        ssObjects : `astropy.table.Table`
            Set of solar system objects that should be within the footprint
            of the current visit.
        visitInfo : `lsst.afw.image.VisitInfo`
            visitInfo of exposure used for exposure time
        bbox : `lsst.geom.Box2I`
            bbox of exposure used for masking
        wcs : `lsst.afw.geom.SkyWcs`
            wcs of exposure used for masking

        Returns
        -------
        resultsStruct : `lsst.pipe.base.Struct`

            - ``ssoAssocDiaSources`` : DiaSources that were associated with
              solar system objects in this visit. (`astropy.table.Table`)
            - ``unAssocDiaSources`` : Set of DiaSources that were not
              associated with any solar system object. (`astropy.table.Table`)
            - ``nTotalSsObjects`` : Total number of SolarSystemObjects
              contained in the CCD footprint. (`int`)
            - ``nAssociatedSsObjects`` : Number of SolarSystemObjects
              that were associated with DiaSources. (`int`)
            - ``ssSourceData`` : ssSource table data. (`Astropy.table.Table`)
        """

        nSolarSystemObjects = len(ssObjects)
        if nSolarSystemObjects <= 0:
            return self._return_empty(diaSourceCatalog, ssObjects)

        mjd_midpoint = visitInfo.date.toAstropy().tai.mjd
        if 'obs_position' in ssObjects.columns:
            ref_time = mjd_midpoint - ssObjects["tmin"].value[0]  # all tmin should be identical
            ssObjects['obs_position'] = [
                np.array([chebval(ref_time, row['obs_x_poly']),
                          chebval(ref_time, row['obs_y_poly']),
                          chebval(ref_time, row['obs_z_poly'])])
                for row in ssObjects]
            ssObjects['obs_velocity'] = [
                np.array([chebval(ref_time, Chebyshev(row['obs_x_poly']).deriv().coef),
                          chebval(ref_time, Chebyshev(row['obs_y_poly']).deriv().coef),
                          chebval(ref_time, Chebyshev(row['obs_z_poly']).deriv().coef)])
                for row in ssObjects]
            ssObjects['obj_position'] = [
                np.array([chebval(ref_time, row['obj_x_poly']),
                          chebval(ref_time, row['obj_y_poly']),
                          chebval(ref_time, row['obj_z_poly'])])
                for row in ssObjects]
            ssObjects['obj_velocity'] = [
                np.array([chebval(ref_time, Chebyshev(row['obj_x_poly']).deriv().coef),
                          chebval(ref_time, Chebyshev(row['obj_y_poly']).deriv().coef),
                          chebval(ref_time, Chebyshev(row['obj_z_poly']).deriv().coef)])
                for row in ssObjects]
            vector = np.vstack(ssObjects['obj_position'].value - ssObjects['obs_position'].value)
            ras, decs = np.vstack(hp.vec2ang(vector, lonlat=True))
            ssObjects['ra'] = ras
            ssObjects['dec'] = decs
            ssObjects['obs_position_x'], ssObjects['obs_position_y'], \
                ssObjects['obs_position_z'] = ssObjects['obs_position'].value.T
            ssObjects['heliocentricX'], ssObjects['heliocentricY'], \
                ssObjects['heliocentricZ'] = ssObjects['obj_position'].value.T
            ssObjects['obs_velocity_x'], ssObjects['obs_velocity_y'], \
                ssObjects['obs_velocity_z'] = ssObjects['obs_velocity'].value.T
            ssObjects['heliocentricVX'], ssObjects['heliocentricVY'], \
                ssObjects['heliocentricVZ'] = ssObjects['obj_velocity'].value.T
            ssObjects['topocentric_position'], ssObjects['topocentric_velocity'] = (
                ssObjects['obj_position'] - ssObjects['obs_position'],
                ssObjects['obj_velocity'] - ssObjects['obs_velocity'],
            )
            ssObjects['topocentricX'], ssObjects['topocentricY'], ssObjects['topocentricZ'] = (
                np.array(list(ssObjects['topocentric_position'].value)).T
            )
            ssObjects['topocentricVX'], ssObjects['topocentricVY'], ssObjects['topocentricVZ'] = (
                np.array(list(ssObjects['topocentric_velocity'].value)).T
            )
            ssObjects['heliocentricVX'], ssObjects['heliocentricVY'], \
                ssObjects['heliocentricVZ'] = np.array(list(ssObjects['obj_velocity'].value)).T
            ssObjects['heliocentricDist'] = np.linalg.norm(ssObjects['obj_position'], axis=1)
            ssObjects['topocentricDist'] = np.linalg.norm(ssObjects['topocentric_position'], axis=1)
            ssObjects['phaseAngle'] = np.degrees(np.arccos(np.sum(
                ssObjects['obj_position'].T * ssObjects['topocentric_position'].T
                / ssObjects['heliocentricDist'] / ssObjects['topocentricDist'], axis=0
            )))
            marginArcsec = ssObjects["Err(arcsec)"].max()

            columns_to_drop = [
                "obs_position", "obs_velocity", "obj_position", "obj_velocity", "topocentric_position",
                "topocentric_velocity", "obs_x_poly", "obs_y_poly", "obs_z_poly", "obj_x_poly", "obj_y_poly",
                "obj_z_poly", "associated"
            ]

        else:
            ssObjects.rename_columns(
                ['RA_deg', 'Dec_deg', 'phase_deg', 'Range_LTC_km', 'Obj_Sun_x_LTC_km', 'Obj_Sun_y_LTC_km',
                 'Obj_Sun_z_LTC_km', 'Obj_Sun_vx_LTC_km_s', 'Obj_Sun_vy_LTC_km_s', 'Obj_Sun_vz_LTC_km_s'],
                ['ra', 'dec', 'phaseAngle', 'topocentricDist', 'heliocentricX', 'heliocentricY',
                 'heliocentricZ', 'heliocentricVX', 'heliocentricVY', 'heliocentricVZ'])
            ssObjects['ssObjectId'] = [obj_id_to_ss_object_id(v) for v in ssObjects['ObjID']]
            ssObjects['heliocentricDist'] = (
                np.sqrt(ssObjects['heliocentricX']**2 + ssObjects['heliocentricY']**2
                        + ssObjects['heliocentricZ']**2)
            )
            for substring1, substring2 in [('X', 'x_km'), ('Y', 'y_km'), ('Z', 'z_km'),
                                           ('VX', 'vx_km_s'), ('VY', 'vy_km_s'), ('VZ', 'vz_km_s')]:
                topoName = 'topocentric' + substring1
                helioName = 'heliocentric' + substring1
                obsName = 'Obs_Sun_' + substring2
                ssObjects[topoName] = ssObjects[helioName] - ssObjects[obsName]

            marginArcsec = 1.0  # TODO: justify

            columns_to_drop = ['FieldID', 'fieldMJD_TAI', 'fieldJD_TDB',
                               'RangeRate_LTC_km_s', 'RARateCosDec_deg_day',
                               'DecRate_deg_day', 'Obs_Sun_x_km', 'Obs_Sun_y_km', 'Obs_Sun_z_km',
                               'Obs_Sun_vx_km_s', 'Obs_Sun_vy_km_s', 'Obs_Sun_vz_km_s',
                               '__index_level_0__']

            # Format DRP ephemerides. Needs:
            # ra
            # dec
            # ssObjectId

        stateVectorColumns = ['heliocentricX', 'heliocentricY', 'heliocentricZ', 'heliocentricVX',
                              'heliocentricVY', 'heliocentricVZ', 'topocentricX', 'topocentricY',
                              'topocentricZ', 'topocentricVX', 'topocentricVY', 'topocentricVZ']

        mpcorbColumns = [col for col in ssObjects.columns if col[:7] == 'MPCORB_']

        maskedObjects = self._maskToCcdRegion(
            ssObjects,
            bbox,
            wcs,
            marginArcsec).copy()
        nSolarSystemObjects = len(maskedObjects)
        if nSolarSystemObjects <= 0:
            return self._return_empty(diaSourceCatalog, maskedObjects)

        maxRadius = np.deg2rad(self.config.maxDistArcSeconds / 3600)

        # Transform DIA RADEC coordinates to unit sphere xyz for tree building.
        vectors = self._radec_to_xyz(diaSourceCatalog["ra"],
                                     diaSourceCatalog["dec"])

        # Create KDTree of DIA sources
        tree = cKDTree(vectors)

        nFound = 0
        # Query the KDtree for DIA nearest neighbors to SSOs. Currently only
        # picks the DiaSource with the shortest distance. We can do something
        # fancier later.
        ssSourceData, ssObjectIds = [], []
        ras, decs, residual_ras, residual_decs, dia_ids = [], [], [], [], []
        diaSourceCatalog["ssObjectId"] = 0
        source_column = 'id'
        maskedObjects['associated'] = False
        if 'diaSourceId' in diaSourceCatalog.columns:
            source_column = 'diaSourceId'
        for ssObject in maskedObjects:
            index = ssObject.index
            ssoVect = self._radec_to_xyz(ssObject["ra"], ssObject["dec"])
            # Which DIA Sources fall within r?
            dist, idx = tree.query(ssoVect, distance_upper_bound=maxRadius)
            if len(idx) == 1 and np.isfinite(dist[0]):
                nFound += 1
                diaSourceCatalog[idx[0]]["ssObjectId"] = ssObject["ssObjectId"]
                ssObjectIds.append(ssObject["ssObjectId"])
                all_cols = ["phaseAngle", "heliocentricDist",
                            "topocentricDist"] + stateVectorColumns + mpcorbColumns
                ssSourceData.append(list(ssObject[all_cols].values()))
                dia_ra = diaSourceCatalog[idx[0]]["ra"]
                dia_dec = diaSourceCatalog[idx[0]]["dec"]
                dia_id = diaSourceCatalog[idx[0]][source_column]
                ras.append(dia_ra)
                decs.append(dia_dec)
                dia_ids.append(dia_id)
                residual_ras.append(dia_ra - ssObject["ra"])
                residual_decs.append(dia_dec - ssObject["dec"])
                maskedObjects['associated'][index] = True
            else:
                maskedObjects['associated'][index] = False

        self.log.info("Successfully associated %d / %d SolarSystemObjects.", nFound, nSolarSystemObjects)
        self.metadata['nAssociatedSsObjects'] = nFound
        self.metadata['nExpectedSsObjects'] = nSolarSystemObjects
        assocSourceMask = diaSourceCatalog["ssObjectId"] != 0
        unAssocObjectMask = np.logical_not(maskedObjects['associated'].value)
        ssSourceData = np.array(ssSourceData)
        ssSourceData = Table(ssSourceData,
                             names=[
                                 "phaseAngle", "heliocentricDist", "topocentricDist"
                             ] + stateVectorColumns + mpcorbColumns)
        ssSourceData['ssObjectId'] = Column(data=ssObjectIds, dtype=int)
        ssSourceData["ra"] = ras
        ssSourceData["dec"] = decs
        ssSourceData["residualRa"] = residual_ras
        ssSourceData["residualDec"] = residual_decs
        ssSourceData[source_column] = dia_ids
        coords = SkyCoord(ra=ssSourceData['ra'].value * u.deg, dec=ssSourceData['dec'].value * u.deg)
        ssSourceData['galacticL'] = coords.galactic.l.deg
        ssSourceData['galacticB'] = coords.galactic.b.deg
        ssSourceData['eclipticLambda'] = coords.barycentrictrueecliptic.lon.deg
        ssSourceData['eclipticBeta'] = coords.barycentrictrueecliptic.lat.deg
        unassociatedObjects = maskedObjects[unAssocObjectMask]
        unassociatedObjects.remove_columns(columns_to_drop)
        return pipeBase.Struct(
            ssoAssocDiaSources=diaSourceCatalog[assocSourceMask],
            unAssocDiaSources=diaSourceCatalog[~assocSourceMask],
            nTotalSsObjects=nSolarSystemObjects,
            nAssociatedSsObjects=nFound,
            associatedSsSources=ssSourceData,
            unassociatedSsObjects=unassociatedObjects)

    def _maskToCcdRegion(self, ssObjects, bbox, wcs, marginArcsec):
        """Mask the input SolarSystemObjects to only those in the exposure
        bounding box.

        Parameters
        ----------
        ssObjects : `astropy.table.Table`
            SolarSystemObjects to mask to ``exposure``.
        bbox :
            Exposure bbox used for masking
        wcs :
            Exposure wcs used for masking
        marginArcsec : `float`
            Maximum possible matching radius to pad onto the exposure bounding
            box. If greater than ``maxPixelMargin``, ``maxPixelMargin`` will
            be used.

        Returns
        -------
        maskedSolarSystemObjects : `astropy.table.Table`
            Set of SolarSystemObjects contained within the exposure bounds.
        """
        if len(ssObjects) == 0:
            return ssObjects
        padding = min(
            int(np.ceil(marginArcsec / wcs.getPixelScale(bbox.getCenter()).asArcseconds())),
            self.config.maxPixelMargin)

        return ssObjects[bbox_contains_sky_coords(
            bbox,
            wcs,
            ssObjects['ra'].value * u.degree,
            ssObjects['dec'].value * u.degree,
            padding)]

    def _radec_to_xyz(self, ras, decs):
        """Convert input ra/dec coordinates to spherical unit-vectors.

        Parameters
        ----------
        ras : `array-like`
            RA coordinates of objects in degrees.
        decs : `array-like`
            DEC coordinates of objects in degrees.

        Returns
        -------
        vectors : `numpy.ndarray`, (N, 3)
            Output unit-vectors
        """
        ras = np.radians(ras)
        decs = np.radians(decs)
        try:
            vectors = np.empty((len(ras), 3))
        except TypeError:
            vectors = np.empty((1, 3))

        sin_dec = np.sin(np.pi / 2 - decs)
        vectors[:, 0] = sin_dec * np.cos(ras)
        vectors[:, 1] = sin_dec * np.sin(ras)
        vectors[:, 2] = np.cos(np.pi / 2 - decs)

        return vectors

    def _return_empty(self, diaSourceCatalog, emptySolarSystemObjects):
        """Return a struct with all appropriate empty values for no SSO associations.

        Parameters
        ----------
        diaSourceCatalog : `astropy.table.Table`
            Used for column names
        emptySolarSystemObjects : `astropy.table.Table`
            Used for column names.
        Returns
        -------
        results : `lsst.pipe.base.Struct`
            Results struct with components.
            - ``ssoAssocDiaSources`` : Empty. (`astropy.table.Table`)
            - ``unAssocDiaSources`` : Input DiaSources. (`astropy.table.Table`)
            - ``nTotalSsObjects`` : Zero. (`int`)
            - ``nAssociatedSsObjects`` : Zero.
            - ``associatedSsSources`` : Empty. (`Astropy.table.Table`)
            - ``unassociatedSsObjects`` : Empty. (`Astropy.table.Table`)


        Raises
        ------
        RuntimeError
            Raised if duplicate DiaObjects or duplicate DiaSources are found.
        """
        self.log.info("No SolarSystemObjects found in detector bounding box.")
        return pipeBase.Struct(
            ssoAssocDiaSources=Table(names=diaSourceCatalog.columns),
            unAssocDiaSources=diaSourceCatalog,
            nTotalSsObjects=0,
            nAssociatedSsObjects=0,
            associatedSsSources=Table(names=['phaseAngle', 'heliocentricDist', 'topocentricDist',
                                             'heliocentricX', 'heliocentricY', 'heliocentricZ',
                                             'heliocentricVX', 'heliocentricVY', 'heliocentricVZ',
                                             'topocentricX', 'topocentricY', 'topocentricZ',
                                             'topocentricVX', 'topocentricVY', 'topocentricVZ',
                                             'residualRa', 'residualDec', 'eclipticLambda', 'eclipticBeta',
                                             'galacticL', 'galacticB', 'ssObjectId', 'diaSourceId'],
                                      dtype=[float] * 21 + [int] * 2),
            unassociatedSsObjects=Table(names=emptySolarSystemObjects.columns)
        )
