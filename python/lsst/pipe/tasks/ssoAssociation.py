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
from astropy.coordinates import SkyCoord
from astropy.table import join, Column, Table

import healpy as hp
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev, chebval
from scipy.spatial import cKDTree

from lsst.afw.image.exposure.exposureUtils import bbox_contains_sky_coords
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.utils.timer import timeMethod
from lsst.pipe.tasks.associationUtils import obj_id_to_ss_object_id

from .ssp.ssobject import DIA_COLUMNS, DIA_DTYPES


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

        exposure_midpoint = visitInfo.date.toAstropy()
        if 'obs_x_poly' in ssObjects.columns:  # mpSky ephemeris
            ref_time = exposure_midpoint.tai.mjd - ssObjects["tmin"].value[0]  # all tmin should be identical
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
            ssObjects['ephRa'] = ras
            ssObjects['ephDec'] = decs
            ssObjects['obs_position_x'], ssObjects['obs_position_y'], \
                ssObjects['obs_position_z'] = ssObjects['obs_position'].value.T
            ssObjects['helio_x'], ssObjects['helio_y'], \
                ssObjects['helio_z'] = ssObjects['obj_position'].value.T
            ssObjects['obs_velocity_x'], ssObjects['obs_velocity_y'], \
                ssObjects['obs_velocity_z'] = ssObjects['obs_velocity'].value.T
            ssObjects['helio_vx'], ssObjects['helio_vy'], \
                ssObjects['helio_vz'] = ssObjects['obj_velocity'].value.T
            ssObjects['topocentric_position'], ssObjects['topocentric_velocity'] = (
                ssObjects['obj_position'] - ssObjects['obs_position'],
                ssObjects['obj_velocity'] - ssObjects['obs_velocity'],
            )
            ssObjects['topo_x'], ssObjects['topo_y'], ssObjects['topo_z'] = (
                np.array(list(ssObjects['topocentric_position'].value)).T
            )
            ssObjects['topo_vx'], ssObjects['topo_vy'], ssObjects['topo_vz'] = (
                np.array(list(ssObjects['topocentric_velocity'].value)).T
            )
            ssObjects['helio_vx'], ssObjects['helio_vy'], \
                ssObjects['helio_vz'] = np.array(list(ssObjects['obj_velocity'].value)).T
            ssObjects['helioRange'] = np.linalg.norm(ssObjects['obj_position'], axis=1)
            ssObjects['topoRange'] = np.linalg.norm(ssObjects['topocentric_position'], axis=1)
            ssObjects['phaseAngle'] = np.degrees(np.arccos(np.sum(
                ssObjects['obj_position'].T * ssObjects['topocentric_position'].T
                / ssObjects['helioRange'] / ssObjects['topoRange'], axis=0
            )))
            # Add other required columns with dummy values until we compute them properly.
            # Fix in DM-53463
            ssObjects['RARateCosDec_deg_day'] = 0
            ssObjects['DecRate_deg_day'] = 0
            ssObjects['PSFMagTrue'] = 0
            ssObjects['RangeRate_LTC_km_s'] = 0

            marginArcsec = ssObjects["Err(arcsec)"].max()

            columns_to_drop = [
                "obs_position", "obs_velocity", "obj_position", "obj_velocity", "topocentric_position",
                "topocentric_velocity", "obs_x_poly", "obs_y_poly", "obs_z_poly", "obj_x_poly", "obj_y_poly",
                "obj_z_poly", "associated"
            ]

        else:  # Sorcha ephemerides
            if 'PSFMagTrue' not in ssObjects.columns:  # Only possible for historical CI ephemerides
                ssObjects['PSFMagTrue'] = 0  # Fix in DM-53462
            ssObjects.rename_columns(
                ['RATrue_deg', 'DecTrue_deg', 'phase_deg', 'Range_LTC_km', 'Obj_Sun_x_LTC_km',
                 'Obj_Sun_y_LTC_km', 'Obj_Sun_z_LTC_km', 'Obj_Sun_vx_LTC_km_s', 'Obj_Sun_vy_LTC_km_s',
                 'Obj_Sun_vz_LTC_km_s'],
                ['ephRa', 'ephDec', 'phaseAngle', 'topoRange', 'helio_x', 'helio_y',
                 'helio_z', 'helio_vx', 'helio_vy', 'helio_vz'])
            ssObjects['ssObjectId'] = [obj_id_to_ss_object_id(v) for v in ssObjects['ObjID']]
            ssObjects['helioRange'] = (
                np.sqrt(ssObjects['helio_x']**2 + ssObjects['helio_y']**2
                        + ssObjects['helio_z']**2)
            )
            for substring1, substring2 in [('x', 'x_km'), ('y', 'y_km'), ('z', 'z_km'),
                                           ('vx', 'vx_km_s'), ('vy', 'vy_km_s'), ('vz', 'vz_km_s')]:
                topoName = 'topo_' + substring1
                helioName = 'helio_' + substring1
                obsName = 'Obs_Sun_' + substring2
                ssObjects[topoName] = ssObjects[helioName] - ssObjects[obsName]

            marginArcsec = 1.0  # TODO: justify

            columns_to_drop = ['FieldID', 'fieldMJD_TAI', 'fieldJD_TDB',
                               'Obs_Sun_x_km', 'Obs_Sun_y_km', 'Obs_Sun_z_km',
                               'Obs_Sun_vx_km_s', 'Obs_Sun_vy_km_s', 'Obs_Sun_vz_km_s',
                               '__index_level_0__']

        stateVectorColumns = ['helio_x', 'helio_y', 'helio_z', 'helio_vx',
                              'helio_vy', 'helio_vz', 'topo_x', 'topo_y',
                              'topo_z', 'topo_vx', 'topo_vy', 'topo_vz']

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
        ssSourceData, ssObjectIds, prov_ids = [], [], []
        ras, decs, residual_ras, residual_decs, dia_ids = [], [], [], [], []
        diaSourceCatalog["ssObjectId"] = 0
        source_column = 'id'
        maskedObjects['associated'] = False
        if 'diaSourceId' in diaSourceCatalog.columns:
            source_column = 'diaSourceId'

        # Find all pairs of a source and an object within maxRadius
        nearby_obj_source_pairs = []
        for obj_idx, (ra, dec) in enumerate(zip(maskedObjects["ephRa"].data, maskedObjects["ephDec"].data)):
            ssoVect = self._radec_to_xyz(ra, dec)
            dist, idx = tree.query(ssoVect, distance_upper_bound=maxRadius)
            if not np.isfinite(dist):
                continue
            for i in range(len(dist)):
                nearby_obj_source_pairs.append((dist, obj_idx, idx[i]))
        nearby_obj_source_pairs = sorted(nearby_obj_source_pairs)

        # From closest to farthest, associate diaSources to SSOs.
        # Skipping already-associated sources and objects.
        used_src_indices, used_obj_indices = set(), set()
        maskedObjects['associated'] = False
        for dist, obj_idx, src_idx in nearby_obj_source_pairs:
            if src_idx in used_src_indices or obj_idx in used_obj_indices:
                continue
            maskedObject = maskedObjects[obj_idx]
            used_src_indices.add(src_idx)
            used_obj_indices.add(obj_idx)
            diaSourceCatalog[src_idx]["ssObjectId"] = maskedObject["ssObjectId"]
            ssObjectIds.append(maskedObject["ssObjectId"])
            all_cols = (
                ["ObjID", "phaseAngle", "helioRange", "topoRange"] + stateVectorColumns + mpcorbColumns
                + ["ephRa", "ephDec", "RARateCosDec_deg_day",
                   "DecRate_deg_day", "PSFMagTrue", "RangeRate_LTC_km_s"]
            )
            ssSourceData.append(list(maskedObject[all_cols].values()))
            dia_ra = diaSourceCatalog[src_idx]["ra"]
            dia_dec = diaSourceCatalog[src_idx]["dec"]
            dia_id = diaSourceCatalog[src_idx][source_column]
            ras.append(dia_ra)
            decs.append(dia_dec)
            dia_ids.append(dia_id)
            residual_ras.append(dia_ra - maskedObject["ephRa"])
            residual_decs.append(dia_dec - maskedObject["ephDec"])
            prov_ids.append(maskedObject['ObjID'])
            maskedObjects['associated'][obj_idx] = True
        nFound = len(ras)

        self.log.info("Successfully associated %d / %d SolarSystemObjects.", nFound, nSolarSystemObjects)
        self.metadata['nAssociatedSsObjects'] = nFound
        self.metadata['nExpectedSsObjects'] = nSolarSystemObjects
        assocSourceMask = diaSourceCatalog["ssObjectId"] != 0
        unAssocObjectMask = np.logical_not(maskedObjects['associated'].value)
        ssSourceData = np.array(ssSourceData)
        colnames = ["designation", "phaseAngle", "helioRange", "topoRange"]
        colnames += stateVectorColumns + mpcorbColumns
        colnames += ["ephRa", "ephDec", "ephRateRa", "ephRateDec", "ephVmag", "topoRangeRate"]
        ssSourceData = Table(ssSourceData, names=colnames, dtype=[str] + [np.float64] * (len(colnames) - 1))
        ssSourceData['ssObjectId'] = Column(data=ssObjectIds, dtype=int)
        ssSourceData["ra"] = ras
        ssSourceData["dec"] = decs
        ephOffsetRa = np.array(residual_ras * np.cos(np.radians(ssSourceData["dec"]))) * 3600  # in arcsec
        ephOffsetDec = np.array(residual_decs) * 3600  # in arcsec
        ssSourceData["ephOffsetRa"] = ephOffsetRa
        ssSourceData["ephOffsetDec"] = ephOffsetDec
        ephOffsetVec = np.array([ephOffsetRa, ephOffsetDec])
        ssSourceData[source_column] = dia_ids
        coords = SkyCoord(ra=ssSourceData['ra'].value * u.deg, dec=ssSourceData['dec'].value * u.deg)
        ssSourceData['galLon'] = coords.galactic.l.deg
        ssSourceData['galLat '] = coords.galactic.b.deg
        ssSourceData['eclLambda'] = coords.barycentrictrueecliptic.lon.deg
        ssSourceData['eclBeta'] = coords.barycentrictrueecliptic.lat.deg
        ssSourceData['designation'] = prov_ids
        ssSourceData['ephRate'] = np.sqrt((ssSourceData['ephRateRa']) ** 2
                                          + (ssSourceData['ephRateDec']) ** 2)
        ssSourceData['ephOffset'] = np.sqrt((ssSourceData['ephOffsetRa']) ** 2
                                            + (ssSourceData['ephOffsetDec']) ** 2)
        ssSourceData['topo_vtot'] = np.sqrt(ssSourceData['topo_vx'] ** 2
                                            + ssSourceData['topo_vy'] ** 2
                                            + ssSourceData['topo_vy'] ** 2)
        ssSourceData['helio_vtot'] = np.sqrt(ssSourceData['helio_vx'] ** 2
                                             + ssSourceData['helio_vy'] ** 2
                                             + ssSourceData['helio_vy'] ** 2)
        skyMotionNormal0 = (ssSourceData['ephRateRa']/ssSourceData['ephRate']).data
        skyMotionNormal1 = (ssSourceData['ephRateDec']/ssSourceData['ephRate']).data
        skyMotionNormal = np.array([skyMotionNormal0, skyMotionNormal1])
        skyMotionOrthogonal = np.array([[0, -1], [1, 0]]) @ skyMotionNormal
        ssSourceData['ephOffsetAlongTrack'] = (ephOffsetVec * skyMotionNormal).sum(axis=0)
        ssSourceData['ephOffsetCrossTrack'] = (ephOffsetVec * skyMotionOrthogonal).sum(axis=0)
        ssSourceData['diaDistanceRank'] = 1
        sun_obs_x = ssSourceData['topo_x'] - ssSourceData['helio_x']
        sun_obs_y = ssSourceData['topo_y'] - ssSourceData['helio_y']
        sun_obs_z = ssSourceData['topo_z'] - ssSourceData['helio_z']
        sun_obs_range = np.sqrt(sun_obs_x**2 + sun_obs_y**2 + sun_obs_z**2)
        sun_obj_dot = (sun_obs_x * ssSourceData['topo_x'] + sun_obs_y * ssSourceData['topo_y']
                       + sun_obs_z * ssSourceData['topo_z'])
        ssSourceData["elongation"] = np.degrees(np.arccos(sun_obj_dot
                                                          / (sun_obs_range * ssSourceData['topoRange'])))
        ssSourceData["helioRangeRate"] = ((ssSourceData["helio_vx"] * ssSourceData["helio_x"]
                                           + ssSourceData["helio_vy"] * ssSourceData["helio_y"]
                                           + ssSourceData["helio_vz"] * ssSourceData["helio_z"])
                                          / ssSourceData["helioRange"])

        for distanceName in ['helio_x', 'helio_y', 'helio_z', 'topo_x', 'topo_y', 'topo_z',
                             'helioRange', 'topoRange']:
            ssSourceData[distanceName] = ssSourceData[distanceName].astype(np.float64)
            ssSourceData[distanceName] *= (u.km).to(u.AU)
        unassociatedObjects = maskedObjects[unAssocObjectMask]
        unassociatedObjects.remove_columns(columns_to_drop)
        unassociatedObjects['ra'] = unassociatedObjects['ephRa']
        unassociatedObjects['dec'] = unassociatedObjects['ephDec']

        # Add diaSource columns we care about when producing the SSObject table
        if len(ssSourceData):

            # Extract only DIA_COLUMNS
            dia = diaSourceCatalog[DIA_COLUMNS].copy()

            # Prefix all except diaSourceId
            for c in DIA_COLUMNS:
                if c != "diaSourceId":
                    dia.rename_column(c, f"DIA_{c}")

            # Join on diaSourceId, keeping all rows of ssSourceData
            ssSourceData = join(ssSourceData, dia, keys="diaSourceId", join_type="left", uniq_col_name="")

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
            ssObjects['ephRa'].value * u.degree,
            ssObjects['ephDec'].value * u.degree,
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
        dia_columns = [col for col in DIA_COLUMNS if not col == 'diaSourceId']
        dia_dtypes = [d[0] for d in zip(DIA_DTYPES, DIA_COLUMNS) if not d[1] == 'diaSourceId']
        names = ['designation', 'eclBeta', 'eclLambda', 'ephDec', 'ephOffsetDec', 'ephOffsetRa', 'ephRa',
                 'galLat', 'galLon', 'elongation', 'ephOffset', 'ephOffsetAlongTrack', 'ephOffsetCrossTrack',
                 'ephRate', 'ephRateDec', 'ephRateRa', 'ephVmag', 'helio_vtot', 'helio_vx', 'helio_vy',
                 'helio_vz', 'helio_x', 'helio_y', 'helio_z', 'helioRange', 'helioRangeRate', 'phaseAngle',
                 'topo_vtot', 'topo_vx', 'topo_vy', 'topo_vz', 'topo_x', 'topo_y', 'topo_z', 'topoRange',
                 'topoRangeRate', 'diaSourceId', 'ssObjectId', 'diaDistanceRank'] + dia_columns
        dtypes = [str] + [float] * 35 + [int] * 3 + dia_dtypes
        return pipeBase.Struct(
            ssoAssocDiaSources=Table(names=diaSourceCatalog.columns),
            unAssocDiaSources=diaSourceCatalog,
            nTotalSsObjects=0,
            nAssociatedSsObjects=0,
            associatedSsSources=Table(names=names,
                                      dtype=dtypes),
            unassociatedSsObjects=Table(names=emptySolarSystemObjects.columns)
        )
