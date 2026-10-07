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
from lsst.ip.isr.shutterTiming import ShutterTimingStatus
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
    def run(self, diaSourceCatalog, ssObjects, visitInfo, bbox, wcs, shutterTiming=None):
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
            wcs of exposure used for masking, and for mapping predicted
            positions to pixels when ``shutterTiming`` is given.
        shutterTiming : `lsst.ip.isr.shutterTiming.ShutterTiming`, optional
            Per-pixel shutter-corrected mid-exposure times of this detector.
            If given, each object's prediction is evaluated at the time of
            the pixel where it is first predicted at ``visitInfo.date``
            (Chebyshev ephemerides are re-evaluated; precomputed Sorcha
            positions are moved along their sky rates).  Objects whose
            per-source status is UNAVAILABLE, or whose time is not finite,
            keep ``visitInfo.date``.  If `None` (default), every prediction
            is at ``visitInfo.date``.

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

        # TODO DM-53699: the source_column should be made consistent, and
        # should match the definition in sdm_schemas.
        source_columns = DIA_COLUMNS.copy()
        if 'diaSourceId' in diaSourceCatalog.columns:
            source_column = 'diaSourceId'
        else:
            source_column = 'id'
            source_columns[source_columns.index('diaSourceId')] = source_column
        nSolarSystemObjects = len(ssObjects)
        self.metadata['nSsoShutterEpochs'] = 0
        self.metadata['ssoShutterEpochMaxShift'] = 0.0

        if nSolarSystemObjects <= 0:
            return self._return_empty(diaSourceCatalog, ssObjects, source_column=source_column)

        for c in ssObjects.columns:
            if ssObjects[c][0] is np.ma.core.MaskedConstant:
                ssObjects[c] = None

        exposure_midpoint = visitInfo.date.toAstropy()
        if 'obs_x_poly' in ssObjects.columns:  # mpSky ephemeris
            # tmin, tmax are mjd. All tmin should be identical, so just take the first.
            tmin = ssObjects["tmin"].quantity.to_value(u.d)[0]
            ref_time = exposure_midpoint.tai.mjd - tmin
            # *_poly are Chebyshev polynomials encoding the observer and object positions.
            # Positions are received in equatorial, cartesian au. Velocities are in au/d.
            states = self._evaluateChebyshevStates(ssObjects, np.full(len(ssObjects), ref_time))
            if shutterTiming is not None:
                # Second pass: re-evaluate each object (and the observer) at
                # the shutter-corrected mid-exposure time of the pixel where
                # the first pass predicts it.  Everything derived below
                # (ephRa/ephDec, hence matching and ephOffset*; the helio,
                # topo and observer state vectors; ranges, phase angle,
                # elongation, helioRangeRate) is then at that epoch.
                ras, decs = self._topocentricRaDec(states[2] - states[0])
                epochs, reevaluate = self._shutterEpochs(ras, decs, wcs, exposure_midpoint.tai.mjd,
                                                         shutterTiming)
                if np.any(reevaluate):
                    newStates = self._evaluateChebyshevStates(ssObjects[reevaluate],
                                                              epochs[reevaluate] - tmin)
                    for state, newState in zip(states, newStates):
                        state[reevaluate] = newState
            obs_position, obs_velocity, obj_position, obj_velocity = states
            ssObjects['obs_position'] = obs_position * u.au
            ssObjects['obs_velocity'] = obs_velocity * u.au/u.d
            ssObjects['obj_position'] = obj_position * u.au
            ssObjects['obj_velocity'] = obj_velocity * u.au/u.d

            ras, decs = self._topocentricRaDec(ssObjects['obj_position'].quantity.to_value(u.au)
                                               - ssObjects['obs_position'].quantity.to_value(u.au))
            # Angles in degrees
            ssObjects['ephRa'] = ras * u.deg
            ssObjects['ephDec'] = decs * u.deg
            # Positions in au, velocities in km/s.
            ssObjects['obs_position_x'], ssObjects['obs_position_y'], \
                ssObjects['obs_position_z'] = ssObjects['obs_position'].quantity.to_value(u.au).T
            ssObjects['helio_x'], ssObjects['helio_y'], \
                ssObjects['helio_z'] = ssObjects['obj_position'].quantity.to_value(u.au).T
            ssObjects['obs_velocity_x'], ssObjects['obs_velocity_y'], \
                ssObjects['obs_velocity_z'] = ssObjects['obs_velocity'].quantity.to_value(u.km/u.s).T
            ssObjects['helio_vx'], ssObjects['helio_vy'], \
                ssObjects['helio_vz'] = ssObjects['obj_velocity'].quantity.to_value(u.km/u.s).T
            ssObjects['topocentric_position'], ssObjects['topocentric_velocity'] = (
                ssObjects['obj_position'] - ssObjects['obs_position'],
                ssObjects['obj_velocity'] - ssObjects['obs_velocity'],
            )
            ssObjects['topo_x'], ssObjects['topo_y'], ssObjects['topo_z'] = (
                np.array(list(ssObjects['topocentric_position'].quantity.to_value(u.au))).T
            )
            ssObjects['topo_vx'], ssObjects['topo_vy'], ssObjects['topo_vz'] = (
                np.array(list(ssObjects['topocentric_velocity'].quantity.to_value(u.km/u.s))).T
            )
            ssObjects['helioRange'] = np.linalg.norm(ssObjects['obj_position'], axis=1)
            ssObjects['topoRange'] = np.linalg.norm(ssObjects['topocentric_position'], axis=1)

            # Phase angle in degrees
            ssObjects['phaseAngle'] = np.degrees(np.arccos(np.sum(
                ssObjects['obj_position'].T * ssObjects['topocentric_position'].T
                / ssObjects['helioRange'] / ssObjects['topoRange'], axis=0
            )))

            #  ssObjects['trailedSourceMagTrue'] should already be filled
            # Add other required columns with dummy values until we compute them properly.
            # Fix in DM-53463
            ssObjects['RARateCosDec_deg_day'] = 0
            ssObjects['DecRate_deg_day'] = 0
            ssObjects['RangeRate_LTC_km_s'] = 0

            marginArcsec = ssObjects["Err(arcsec)"].max()
            ssObjects['designation'] = ssObjects['MPCORB_unpacked_primary_provisional_designation']

            columns_to_drop = [
                "obs_position", "obs_velocity", "obj_position", "obj_velocity", "topocentric_position",
                "topocentric_velocity", "obs_x_poly", "obs_y_poly", "obs_z_poly", "obj_x_poly", "obj_y_poly",
                "obj_z_poly", "associated"
            ]

        else:  # Sorcha ephemerides
            if 'trailedSourceMagTrue' not in ssObjects.columns:  # Only possible for historical CI ephemerides
                ssObjects['trailedSourceMagTrue'] = 0  # Fix in DM-53462
            ssObjects.rename_columns(
                ['RATrue_deg', 'DecTrue_deg', 'phase_deg', 'Range_LTC_km', 'Obj_Sun_x_LTC_km',
                 'Obj_Sun_y_LTC_km', 'Obj_Sun_z_LTC_km', 'Obj_Sun_vx_LTC_km_s', 'Obj_Sun_vy_LTC_km_s',
                 'Obj_Sun_vz_LTC_km_s'],
                ['ephRa', 'ephDec', 'phaseAngle', 'topoRange', 'helio_x', 'helio_y',
                 'helio_z', 'helio_vx', 'helio_vy', 'helio_vz'])
            for col in ['topoRange', 'helio_x', 'helio_y', 'helio_z']:
                ssObjects[col] = ssObjects[col].quantity.to_value(u.au)
            for col in ['Obs_Sun_x_km', 'Obs_Sun_y_km', 'Obs_Sun_z_km']:
                ssObjects[col.replace('km', 'au')] = ssObjects[col].quantity.to_value(u.au)
            for col in ['helio_vx', 'helio_vy', 'helio_vz', 'Obs_Sun_vx_km_s', 'Obs_Sun_vy_km_s',
                        'Obs_Sun_vz_km_s']:
                ssObjects[col] = ssObjects[col].quantity.to_value(u.km/u.s)

            ssObjects['ssObjectId'] = [obj_id_to_ss_object_id(v) for v in ssObjects['packed_desig']]
            for substring1, substring2 in [('x', 'x_au'), ('y', 'y_au'), ('z', 'z_au'),
                                           ('vx', 'vx_km_s'), ('vy', 'vy_km_s'), ('vz', 'vz_km_s')]:
                topoName = 'topo_' + substring1
                helioName = 'helio_' + substring1
                obsName = 'Obs_Sun_' + substring2
                ssObjects[topoName] = ssObjects[helioName] - ssObjects[obsName]

            ssObjects['helioRange'] = (
                np.sqrt(ssObjects['helio_x']**2 + ssObjects['helio_y']**2
                        + ssObjects['helio_z']**2)
            )

            ssObjects['designation'] = ssObjects['ObjID']

            if shutterTiming is not None:
                # Sorcha supplies RA/Dec and their rates (RARateCosDec_deg_day,
                # DecRate_deg_day; both also copied to SSSource below) at the
                # visit time, so move each prediction linearly to its pixel's
                # mid-exposure time.  The state vectors, ranges and phase angle
                # stay at the visit time: over the <~0.5 s shift they change
                # by a few km.
                visitMjdTai = exposure_midpoint.tai.mjd
                ephRa = np.array(ssObjects['ephRa'], dtype=float)
                ephDec = np.array(ssObjects['ephDec'], dtype=float)
                epochs, reevaluate = self._shutterEpochs(ephRa, ephDec, wcs, visitMjdTai, shutterTiming)
                if np.any(reevaluate):
                    dt = epochs[reevaluate] - visitMjdTai  # days
                    raRate = np.array(ssObjects['RARateCosDec_deg_day'], dtype=float)[reevaluate]
                    decRate = np.array(ssObjects['DecRate_deg_day'], dtype=float)[reevaluate]
                    cosDec = np.cos(np.radians(ephDec[reevaluate]))
                    ephRa[reevaluate] = (ephRa[reevaluate] + raRate*dt/cosDec) % 360.0
                    ephDec[reevaluate] += decRate*dt
                    ssObjects['ephRa'][:] = ephRa
                    ssObjects['ephDec'][:] = ephDec

            marginArcsec = 1.0  # TODO: justify

            columns_to_drop = ['FieldID', 'fieldMJD_TAI', 'fieldJD_TDB',
                               'Obs_Sun_x_km', 'Obs_Sun_y_km', 'Obs_Sun_z_km',
                               'Obs_Sun_vx_km_s', 'Obs_Sun_vy_km_s', 'Obs_Sun_vz_km_s',
                               'Obs_Sun_x_au', 'Obs_Sun_y_au', 'Obs_Sun_z_au']

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
            return self._return_empty(diaSourceCatalog, maskedObjects, source_column=source_column)

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
        maskedObjects['associated'] = False

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
        all_cols = (
            ["designation", "phaseAngle", "helioRange", "topoRange"] + stateVectorColumns + mpcorbColumns
            + ["ephRa", "ephDec", "RARateCosDec_deg_day",
               "DecRate_deg_day", "trailedSourceMagTrue", "RangeRate_LTC_km_s"]
        )

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
        dtypes = [float if d is np.ma.core.MaskedConstant else type(d)
                  for d in maskedObjects[0][all_cols].values()]
        ssSourceData = np.ma.filled(np.array(ssSourceData), None)
        colnames = ["designation", "phaseAngle", "helioRange", "topoRange"]
        colnames += stateVectorColumns + mpcorbColumns
        colnames += ["ephRa", "ephDec", "ephRateRa", "ephRateDec", "ephVmag", "topoRangeRate"]
        ssSourceData = Table(ssSourceData, names=colnames, dtype=dtypes)
        if 'MPCORB_created_at' in ssSourceData.columns:
            ssSourceData['MPCORB_created_at'] = 0
            ssSourceData['MPCORB_updated_at'] = 0
            ssSourceData['MPCORB_fitting_datetime'] = 0
            for c in ['MPCORB_created_at', 'MPCORB_updated_at', 'MPCORB_fitting_datetime',
                      'MPCORB_u_param']:
                ssSourceData[c] = ssSourceData[c].astype(np.float64).astype(np.int64)
            ssSourceData['MPCORB_orbit_type_int'] = -1  # TODO: DM-54214. Mixed string and nan types.
            ssSourceData['MPCORB_orbit_type_int'] = ssSourceData['MPCORB_orbit_type_int'].astype(np.int64)
        ssSourceData['ssObjectId'] = Column(data=ssObjectIds, dtype=int)
        ssSourceData["ra"] = ras
        ssSourceData["dec"] = decs
        ephOffsetRa = np.array(residual_ras * np.cos(np.radians(ssSourceData["dec"]))) * 3600  # in arcsec
        ephOffsetDec = np.array(residual_decs) * 3600  # in arcsec
        ssSourceData["ephOffsetRa"] = ephOffsetRa
        ssSourceData["ephOffsetDec"] = ephOffsetDec
        ephOffsetVec = np.array([ephOffsetRa, ephOffsetDec])
        ssSourceData[source_column] = dia_ids
        coords = SkyCoord(ra=ssSourceData['ra'] * u.deg, dec=ssSourceData['dec'] * u.deg)
        ssSourceData['galLon'] = coords.galactic.l.deg
        ssSourceData['galLat'] = coords.galactic.b.deg
        ssSourceData['eclLambda'] = coords.barycentrictrueecliptic.lon.deg
        ssSourceData['eclBeta'] = coords.barycentrictrueecliptic.lat.deg
        ssSourceData['ephRate'] = np.sqrt((ssSourceData['ephRateRa']) ** 2
                                          + (ssSourceData['ephRateDec']) ** 2)
        ssSourceData['ephOffset'] = np.sqrt((ssSourceData['ephOffsetRa']) ** 2
                                            + (ssSourceData['ephOffsetDec']) ** 2)
        ssSourceData['topo_vtot'] = np.sqrt(ssSourceData['topo_vx'] ** 2
                                            + ssSourceData['topo_vy'] ** 2
                                            + ssSourceData['topo_vz'] ** 2)
        ssSourceData['helio_vtot'] = np.sqrt(ssSourceData['helio_vx'] ** 2
                                             + ssSourceData['helio_vy'] ** 2
                                             + ssSourceData['helio_vz'] ** 2)
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

        unassociatedObjects = maskedObjects[unAssocObjectMask]
        unassociatedObjects.remove_columns(columns_to_drop)
        unassociatedObjects['ra'] = unassociatedObjects['ephRa']
        unassociatedObjects['dec'] = unassociatedObjects['ephDec']

        # Add diaSource columns we care about when producing the SSObject table
        if len(ssSourceData):

            # Extract only DIA_COLUMNS
            dia = diaSourceCatalog[source_columns].copy()

            # Prefix all except diaSourceId
            for c in source_columns:
                if c != source_column:
                    dia.rename_column(c, f"DIA_{c}")

            # Join on diaSourceId, keeping all rows of ssSourceData
            ssSourceData = join(ssSourceData, dia, keys=source_column, join_type="left", uniq_col_name="")

        return pipeBase.Struct(
            ssoAssocDiaSources=diaSourceCatalog[assocSourceMask],
            unAssocDiaSources=diaSourceCatalog[~assocSourceMask],
            nTotalSsObjects=nSolarSystemObjects,
            nAssociatedSsObjects=nFound,
            associatedSsSources=ssSourceData,
            unassociatedSsObjects=unassociatedObjects)

    @staticmethod
    def _evaluateChebyshevStates(ssObjects, refTimes):
        """Evaluate the mpSky Chebyshev ephemerides of each object.

        Parameters
        ----------
        ssObjects : `astropy.table.Table`
            Objects with ``obs_[xyz]_poly`` and ``obj_[xyz]_poly`` columns.
        refTimes : `numpy.ndarray`
            Per-object evaluation time, days since ``tmin``.

        Returns
        -------
        obsPosition, obsVelocity, objPosition, objVelocity : `numpy.ndarray`
            (N, 3) observer and object positions (au) and velocities (au/d).
        """
        def evaluate(prefix, deriv):
            values = []
            for refTime, row in zip(refTimes, ssObjects):
                coefs = [row[f'{prefix}_{c}_poly'] for c in 'xyz']
                if deriv:
                    coefs = [Chebyshev(coef).deriv().coef for coef in coefs]
                values.append(np.array([chebval(refTime, coef) for coef in coefs]))
            return np.array(values, dtype=float).reshape(len(values), 3)

        return (evaluate('obs', False), evaluate('obs', True),
                evaluate('obj', False), evaluate('obj', True))

    @staticmethod
    def _topocentricRaDec(vector):
        """RA, Dec (degrees) of (N, 3) topocentric vectors."""
        return np.vstack(hp.vec2ang(np.vstack(vector), lonlat=True))

    def _shutterEpochs(self, ras, decs, wcs, visitMjdTai, shutterTiming):
        """Per-object shutter-corrected mid-exposure times.

        Parameters
        ----------
        ras, decs : `numpy.ndarray`
            Predicted positions at the visit time (degrees).
        wcs : `lsst.afw.geom.SkyWcs`
            Maps the predictions to pixels.
        visitMjdTai : `float`
            ``visitInfo.date`` (MJD TAI).
        shutterTiming : `lsst.ip.isr.shutterTiming.ShutterTiming`
            Per-pixel times of this detector.

        Returns
        -------
        epochs : `numpy.ndarray`
            Per-object time (MJD TAI); ``visitMjdTai`` where not
            re-evaluated.
        reevaluate : `numpy.ndarray` [`bool`]
            Objects to re-evaluate at ``epochs``: per-source status not
            UNAVAILABLE and a finite time.

        Notes
        -----
        Sets the task metadata ``nSsoShutterEpochs`` (number of objects
        re-evaluated) and ``ssoShutterEpochMaxShift`` (largest
        ``|epoch - visitMjdTai|``, seconds).
        """
        ras = np.asarray(ras, dtype=float)
        decs = np.asarray(decs, dtype=float)
        x, y = wcs.skyToPixelArray(ras, decs, degrees=True)
        times = np.asarray(shutterTiming.tMidMjdTai(x, y), dtype=float)
        status = np.asarray(shutterTiming.sourceStatus(x, y))
        reevaluate = (status != ShutterTimingStatus.UNAVAILABLE) & np.isfinite(times)
        epochs = np.where(reevaluate, times, visitMjdTai)
        nShifted = int(np.count_nonzero(reevaluate))
        maxShift = float(np.max(np.abs(epochs - visitMjdTai))) * 86400.0 if nShifted else 0.0
        self.metadata['nSsoShutterEpochs'] = nShifted
        self.metadata['ssoShutterEpochMaxShift'] = maxShift
        self.log.debug("Shutter timing: %d / %d solar system predictions moved to their pixel's "
                       "mid-exposure time (max |shift| %.3f s).", nShifted, len(ras), maxShift)
        return epochs, reevaluate

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
            ssObjects['ephRa'],
            ssObjects['ephDec'],
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

    def _return_empty(self, diaSourceCatalog, emptySolarSystemObjects, source_column):
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
        # TODO DM-53699: these column names and dtypes should match the
        # definitions in sdm_schemas.
        dia_columns = [col for col in DIA_COLUMNS if not col == source_column]
        dia_dtypes = [d[0] for d in zip(DIA_DTYPES, DIA_COLUMNS) if not d[1] == source_column]
        names = ['designation', 'eclBeta', 'eclLambda', 'ephDec', 'ephOffsetDec', 'ephOffsetRa', 'ephRa',
                 'galLat', 'galLon', 'elongation', 'ephOffset', 'ephOffsetAlongTrack', 'ephOffsetCrossTrack',
                 'ephRate', 'ephRateDec', 'ephRateRa', 'ephVmag', 'helio_vtot', 'helio_vx', 'helio_vy',
                 'helio_vz', 'helio_x', 'helio_y', 'helio_z', 'helioRange', 'helioRangeRate', 'phaseAngle',
                 'topo_vtot', 'topo_vx', 'topo_vy', 'topo_vz', 'topo_x', 'topo_y', 'topo_z', 'topoRange',
                 'topoRangeRate', source_column, 'ssObjectId', 'diaDistanceRank'] + dia_columns
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
