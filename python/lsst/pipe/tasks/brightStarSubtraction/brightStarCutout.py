# This file is part of pipe_tasks.
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

"""Extract bright star cutouts; normalize and warp to the same pixel grid."""

__all__ = ["BrightStarCutoutConnections", "BrightStarCutoutConfig", "BrightStarCutoutTask"]

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS
from lsst.afw.detection import FootprintSet, Threshold
from lsst.afw.geom import makeModifiedWcs
from lsst.afw.geom.transformFactory import makeTransform
from lsst.afw.image import ExposureF, PhotoCalib
from lsst.afw.math import WarpingControl, warpImage
from lsst.geom import (AffineTransform, Box2I, Extent2D, Extent2I, Point2D, Point2I, SpherePoint, arcseconds,
                       floor, radians)
from lsst.meas.algorithms import LoadReferenceObjectsConfig, ReferenceObjectLoader, Stamp, Stamps, WarpedPsf
from lsst.pex.config import ChoiceField, ConfigField, Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output, PrerequisiteInput
from lsst.utils.timer import timeMethod


class BrightStarCutoutConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
):
    """Connections for BrightStarCutoutTask."""

    refCat = PrerequisiteInput(
        doc="Reference catalog that contains bright star positions.",
        name="gaia_dr3_20230707",
        storageClass="SimpleCatalog",
        dimensions=("skypix",),
        multiple=True,
        deferLoad=True,
    )
    inputExposure = Input(
        doc="Input exposure from which to extract bright star stamp cutouts.",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("visit", "detector"),
    )
    inputBackground = Input(
        doc="Background model for the input exposure.",
        name="calexpBackground",
        storageClass="Background",
        dimensions=("visit", "detector"),
    )
    brightStarStamps = Output(
        doc="Set of preprocessed postage stamp cutouts, each centered on a single bright star.",
        name="brightStarStamps2",
        storageClass="Stamps",
        dimensions=("visit", "detector"),
    )


class BrightStarCutoutConfig(
    PipelineTaskConfig,
    pipelineConnections=BrightStarCutoutConnections,
):
    """Configuration parameters for BrightStarCutoutTask."""

    # Star selection
    magLimit = Field[float](
        doc="Magnitude limit, in Gaia G. Cutouts will be made for all stars brighter than this magnitude.",
        default=18,
    )
    excludeArcsecRadius = Field[float](
        doc="Stars with a star brighter than ``excludeMagLimit`` in ``excludeArcsecRadius`` are not be used.",
        default=5,
    )
    excludeMagLimit = Field[float](
        doc="Stars with a star brighter than ``excludeMagLimit`` in ``excludeArcsecRadius`` are not be used.",
        default=20,
    )
    minAreaFraction = Field[float](
        doc="Minimum fraction of the stamp area, post-masking, that must remain for a cutout to be retained.",
        default=0.1,
    )
    badMaskPlanes = ListField[str](
        doc="Mask planes that identify excluded pixels for the calculation of ``minAreaFraction``.",
        default=["BAD", "CR", "CROSSTALK", "EDGE", "NO_DATA", "SAT", "SUSPECT", "UNMASKEDNAN"],
    )

    # Cutout geometry
    stampSize = ListField[int](
        doc="Size of the stamps to be extracted, in pixels.",
        default=(1001, 1001),
    )
    stampSizePadding = Field[float](
        doc="Multiplicative factor applied to the cutout stamp size, to guard against post-warp flux loss.",
        default=1.1,
    )
    warpingKernelName = ChoiceField[str](
        doc="Warping kernel.",
        default="lanczos5",
        allowed={
            "bilinear": "bilinear interpolation",
            "lanczos3": "Lanczos kernel of order 3",
            "lanczos4": "Lanczos kernel of order 4",
            "lanczos5": "Lanczos kernel of order 5",
        },
    )

    # Misc
    neighborMaskPlane = Field[str](
        doc="Mask plane to identify pixels that are bright star neighbors.",
        default="NEIGHBOR",
    )
    loadReferenceObjectsConfig = ConfigField[LoadReferenceObjectsConfig](
        doc="Reference object loader for astrometric calibration.",
    )


class BrightStarCutoutTask(PipelineTask):
    """Extract bright star cutouts; normalize and warp to the same pixel grid.

    This task is used to extract, process, and store small image cutouts
    (or "postage stamps") around bright stars. This task essentially consists
    of three principal steps. First, it identifies bright stars within an
    exposure using a reference catalog and extracts a stamp around each.
    Second, it shifts and warps each stamp to remove optical distortions and
    sample all stars on the same pixel grid. Finally, it computes the flux of
    an object in an annulus and normalizes it. This is required to normalize
    each bright star stamp as their central pixels are likely saturated and/or
    contain ghosts, and cannot be used.
    """

    ConfigClass = BrightStarCutoutConfig
    _DefaultName = "brightStarCutout"

    def __init__(self, initInputs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        stampSize = Extent2D(*self.config.stampSize.list())
        stampRadius = floor(stampSize / 2)
        self.stampBBox = Box2I(corner=Point2I(0, 0), dimensions=Extent2I(1, 1)).dilatedBy(stampRadius)
        paddedStampSize = stampSize * self.config.stampSizePadding
        self.paddedStampRadius = floor(paddedStampSize / 2)
        self.paddedStampBBox = Box2I(corner=Point2I(0, 0), dimensions=Extent2I(1, 1)).dilatedBy(
            self.paddedStampRadius
        )

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["dataId"] = str(butlerQC.quantum.dataId)
        refObjLoader = ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefs.refCat],
            refCats=inputs.pop("refCat"),
            name=self.config.connections.refCat,
            config=self.config.loadReferenceObjectsConfig,
        )
        output = self.run(**inputs, refObjLoader=refObjLoader)
        # Only ingest Stamp if it exists; prevent ingesting an empty FITS file
        if output:
            butlerQC.put(output, outputRefs)

    @timeMethod
    def run(self, inputExposure, inputBackground, refObjLoader=None, dataId=None, skyCorr=None):
        """Identify bright stars within an exposure using a reference catalog,
        extract stamps around each, then preprocess them.

        Bright star preprocessing steps are: shifting, warping and potentially
        rotating them to the same pixel grid; computing their annular flux,
        and; normalizing them.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The image from which bright star stamps should be extracted.
        refObjLoader : `~lsst.meas.algorithms.ReferenceObjectLoader`, optional
            Loader to find objects within a reference catalog.
        dataId : `dict` or `~lsst.daf.butler.DataCoordinate`
            The dataId of the exposure (including detector) that bright stars
            should be extracted from.
        skyCorr : `~lsst.afw.math.backgroundList.BackgroundList`, optional
            Full focal plane sky correction obtained by `SkyCorrectionTask`.

        Returns
        -------
        brightStarResults : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``brightStarStamps``
                (`~lsst.meas.algorithms.brightStarStamps.BrightStarStamps`)
        """
        wcs = inputExposure.getWcs()
        bbox = inputExposure.getBBox()
        warpCont = WarpingControl(self.config.warpingKernelName)
        neighborMP = self.config.neighborMaskPlane

        refCatBright = self._getRefCatBright(refObjLoader, wcs, bbox)
        zipRaDec = zip(refCatBright["coord_ra"] * radians, refCatBright["coord_dec"] * radians)
        spherePoints = [SpherePoint(ra, dec) for ra, dec in zipRaDec]
        pixCoords = wcs.skyToPixel(spherePoints)

        # Restore original subtracted background
        inputMI = inputExposure.getMaskedImage()
        inputMI += inputBackground.getImage()

        # Set up NEIGHBOR mask plane; associate footprints with stars
        inputExposure.mask.addMaskPlane(neighborMP)
        allFootprints, associations = self._associateFootprints(inputExposure, pixCoords, plane="DETECTED")

        # TODO: If we eventually have better PhotoCalibs (eg FGCM), apply here
        inputMI = inputExposure.getPhotoCalib().calibrateImage(inputMI, False)

        # Set up transform
        detector = inputExposure.detector
        pixelScale = wcs.getPixelScale().asArcseconds() * arcseconds
        pixToFocalPlaneTan = detector.getTransform(PIXELS, FIELD_ANGLE).then(
            makeTransform(AffineTransform.makeScaling(1 / pixelScale.asRadians()))
        )

        # Loop over each bright star
        stamps = []
        for starIndex, (obj, sp, pc) in enumerate(zip(refCatBright, spherePoints, pixCoords)):  # type: ignore
            footprintIndex = associations.get(starIndex, None)
            stamp_im = ExposureF(self.paddedStampBBox)

            # Set NEIGHBOR footprints in the mask plane
            if footprintIndex:
                neighborFootprints = [fp for i, fp in enumerate(allFootprints) if i != footprintIndex]
                self._setFootprints(inputExposure, neighborFootprints, neighborMP)
            else:
                self._setFootprints(inputExposure, allFootprints, neighborMP)

            # Define linear shifting to recenter stamps
            coordFocalPlaneTan = pixToFocalPlaneTan.applyForward(pc)  # center of warped star
            shift = makeTransform(AffineTransform(Point2D(0, 0) - coordFocalPlaneTan))
            angle = np.arctan2(coordFocalPlaneTan.getY(), coordFocalPlaneTan.getX()) * radians
            rotation = makeTransform(AffineTransform.makeRotation(-angle))
            pixToPolar = pixToFocalPlaneTan.then(shift).then(rotation)

            # Apply the warp to the star stamp (in-place)
            warpImage(stamp_im.maskedImage, inputExposure.maskedImage, pixToPolar, warpCont)
            stamp_im.setWcs(makeModifiedWcs(pixToPolar, wcs, False))
            stamp_im.setPhotoCalib(PhotoCalib(1.0))
            stamp_im.setPsf(WarpedPsf(inputExposure.getPsf(), pixToPolar, warpCont))

            # Trim to the base stamp size, check mask coverage, update metadata
            stamp_im = stamp_im[self.stampBBox]
            badMaskBitMask = stamp_im.mask.getPlaneBitMask(self.config.badMaskPlanes)
            goodFrac = np.sum(stamp_im.mask.array & badMaskBitMask == 0) / stamp_im.mask.array.size
            if goodFrac < self.config.minAreaFraction:
                continue
            md = stamp_im.getMetadata()
            md.update(dict(obj))

            stamps.append(Stamp(stamp_im, archive_element=pixToPolar, position=sp))

        grammar = "stamp" if len(stamps) == 1 else "stamps"
        self.log.info(
            "Extracted %i bright star %s; %i excluded due to insufficient usable area (less than %s%%).",
            len(stamps),
            grammar,
            len(pixCoords) - len(stamps),
            self.config.minAreaFraction * 100,
        )
        brightStarStamps = Stamps(stamps)
        return Struct(brightStarStamps=brightStarStamps)

    def _getRefCatBright(self, refObjLoader, wcs, bbox):
        """Get a bright star subset of the reference catalog.

        Trim the reference catalog to only those objects within the exposure
        bounding box dilated by half the bright star stamp size.
        This ensures all stars that overlap the exposure are included.

        Parameters
        ----------
        refObjLoader : `~lsst.meas.algorithms.ReferenceObjectLoader`
            Loader to find objects within a reference catalog.
        wcs : `~lsst.afw.geom.SkyWcs`
            World coordinate system.
        bbox : `~lsst.geom.Box2I`
            Bounding box of the exposure.

        Returns
        -------
        refCatBright : `~astropy.table.Table`
            Bright star subset of the reference catalog.
        """
        dilatedBBox = bbox.dilatedBy(self.paddedStampRadius)
        withinExposure = refObjLoader.loadPixelBox(dilatedBBox, wcs, filterName="phot_g_mean")
        refCatFull = withinExposure.refCat
        fluxField: str = withinExposure.fluxField

        proxFluxLimit = ((self.config.excludeMagLimit * u.ABmag).to(u.nJy)).to_value()
        brightFluxLimit = ((self.config.magLimit * u.ABmag).to(u.nJy)).to_value()

        subsetStars = refCatFull[fluxField] > np.min((proxFluxLimit, brightFluxLimit))
        refCatSubset = Table(refCatFull.extract("id", "coord_ra", "coord_dec", fluxField, where=subsetStars))
        proxStars = refCatSubset[fluxField] >= proxFluxLimit
        brightStars = refCatSubset[fluxField] >= brightFluxLimit

        coords = SkyCoord(refCatSubset["coord_ra"], refCatSubset["coord_dec"], unit="rad")
        refCatBrightIsolated = []
        for coord in coords[brightStars]:
            neighbors = coords[proxStars]
            seps = coord.separation(neighbors).to(u.arcsec)
            tooClose = (seps > 0) & (seps <= self.config.excludeArcsecRadius * u.arcsec)  # not self matched
            refCatBrightIsolated.append(not tooClose.any())

        refCatBright = refCatSubset[brightStars][refCatBrightIsolated]

        # refCatBright["mag"] = (refCatBright[fluxField][:] * u.nJy).to(u.ABmag).to_value()  # AB magnitudes

        grammar = "star" if len(refCatFull) == 1 else "stars"
        self.log.info(
            "Identified %i of %i %s overlapping the frame brighter than %s mag and with no nearby neighbors.",
            len(refCatBright),
            len(refCatFull),
            grammar,
            self.config.magLimit,
        )

        return refCatBright

    def _setFootprints(self, inputExposure, footprints, maskPlane):
        """Set footprints in a given mask plane.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The input exposure to modify.
        footprints : `list` [`~lsst.afw.detection.Footprint`]
            The footprints to set in the mask plane.
        maskPlane : `str`
            The mask plane to set the footprints in.

        Notes
        -----
        This method modifies the ``inputExposure`` object in-place.
        """
        detThreshold = Threshold(inputExposure.mask.getPlaneBitMask(maskPlane), Threshold.BITMASK)
        detThresholdValue = int(detThreshold.getValue())
        footprintSet = FootprintSet(inputExposure.mask, detThreshold)

        # Wipe any existing footprints in the mask plane
        inputExposure.mask.clearMaskPlane(int(np.log2(detThresholdValue)))

        # Set the footprints in the mask plane
        footprintSet.setFootprints(footprints)
        footprintSet.setMask(inputExposure.mask, maskPlane)

    def _associateFootprints(self, inputExposure, pixCoords, plane):
        """Associate footprints from a given mask plane with specific objects.

        Footprints from the given mask plane are associated with objects at the
        coordinates provided, where possible.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The input exposure with a mask plane.
        pixCoords : `list` [`~lsst.geom.Point2D`]
            The pixel coordinates of the objects.
        plane : `str`
            The mask plane used to identify masked pixels.

        Returns
        -------
        footprints : `list` [`~lsst.afw.detection.Footprint`]
            The footprints from the input exposure.
        associations : `dict`
            Association indices between objects (key) and footprints (value).
        """
        detThreshold = Threshold(inputExposure.mask.getPlaneBitMask(plane), Threshold.BITMASK)
        footprintSet = FootprintSet(inputExposure.mask, detThreshold)
        footprints = footprintSet.getFootprints()
        associations = {}
        for starIndex, pixCoord in enumerate(pixCoords):
            for footprintIndex, footprint in enumerate(footprints):
                if footprint.contains(Point2I(pixCoord)):
                    associations[starIndex] = footprintIndex
                    break
        grammar = "footprint" if len(footprints) == 1 else "footprints"
        self.log.info(
            "Associated %i of %i %s %s with stars in the reference catalog.",
            len(associations),
            len(footprints),
            plane,
            grammar,
        )
        return footprints, associations
