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

"""Extract bright star cutouts; normalize and warp, optionally fit the PSF."""

__all__ = ["BrightStarCutoutConnections", "BrightStarCutoutConfig", "BrightStarCutoutTask"]

from typing import Any, Iterable, cast

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from lsst.afw.cameraGeom import FIELD_ANGLE, PIXELS, FOCAL_PLANE
from lsst.afw.detection import Footprint, FootprintSet, Threshold
from lsst.afw.geom import SkyWcs, SpanSet, makeModifiedWcs
from lsst.afw.geom.transformFactory import makeTransform
from lsst.afw.image import ExposureF, ImageD, ImageF, MaskedImageF
from lsst.afw.math import BackgroundList, FixedKernel, WarpingControl, warpImage
from lsst.daf.butler import DataCoordinate
from lsst.geom import (
    AffineTransform,
    Box2I,
    Extent2D,
    Extent2I,
    Point2D,
    Point2I,
    SpherePoint,
    arcseconds,
    floor,
    radians,
    Angle,
)
from lsst.meas.algorithms import (
    BrightStarStamp,
    BrightStarStamps,
    KernelPsf,
    LoadReferenceObjectsConfig,
    ReferenceObjectLoader,
    WarpedPsf,
)
from lsst.pex.config import ChoiceField, ConfigField, Field, ListField
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections, Struct
from lsst.pipe.base.connectionTypes import Input, Output, PrerequisiteInput
from lsst.utils.timer import timeMethod
from copy import deepcopy
import math


NEIGHBOR_MASK_PLANE = "NEIGHBOR"


class BrightStarCutoutConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit", "detector"),
):
    """Connections for BrightStarCutoutTask."""

    refCat = PrerequisiteInput(
        name="gaia_dr3_20230707",
        storageClass="SimpleCatalog",
        doc="Reference catalog that contains bright star positions.",
        dimensions=("skypix",),
        multiple=True,
        deferLoad=True,
    )
    inputExposure = Input(
        name="calexp",
        storageClass="ExposureF",
        doc="Background-subtracted input exposure from which to extract bright star stamp cutouts.",
        dimensions=("visit", "detector"),
    )
    inputBackground = Input(
        name="calexpBackground",
        storageClass="Background",
        doc="Background model for the input exposure, to be added back on during processing.",
        dimensions=("visit", "detector"),
    )
    extendedPsf = Input(
        name="extendedPsf2",
        storageClass="ImageF",
        doc="Extended PSF model, built from stacking bright star cutouts.",
        dimensions=("band",),
    )
    brightStarStamps = Output(
        name="brightStarStamps",
        storageClass="BrightStarStamps",
        doc="Set of preprocessed postage stamp cutouts, each centered on a single bright star.",
        dimensions=("visit", "detector"),
    )

    def __init__(self, *, config: "BrightStarCutoutConfig | None" = None):
        super().__init__(config=config)
        assert config is not None
        if not config.useExtendedPsf:
            self.inputs.remove("extendedPsf")


class BrightStarCutoutConfig(
    PipelineTaskConfig,
    pipelineConnections=BrightStarCutoutConnections,
):
    """Configuration parameters for BrightStarCutoutTask."""

    # Star selection
    magRange = ListField[float](
        doc="Magnitude range in Gaia G. Cutouts will be made for all stars in this range.",
        default=[0, 18],
    )
    excludeArcsecRadius = Field[float](
        doc="Stars with a star in the range ``excludeMagRange`` mag in ``excludeArcsecRadius`` are not used.",
        default=5,
    )
    excludeMagRange = ListField[float](
        doc="Stars with a star in the range ``excludeMagRange`` mag in ``excludeArcsecRadius`` are not used.",
        default=[0, 20],
    )
    minAreaFraction = Field[float](
        doc="Minimum fraction of the stamp area, post-masking, that must remain for a cutout to be retained.",
        default=0.1,
    )
    badMaskPlanes = ListField[str](
        doc="Mask planes that identify excluded pixels for the calculation of ``minAreaFraction`` and, "
        "optionally, fitting of the PSF.",
        default=[
            "BAD",
            "CR",
            "CROSSTALK",
            "EDGE",
            "NO_DATA",
            "SAT",
            "SUSPECT",
            "UNMASKEDNAN",
            NEIGHBOR_MASK_PLANE,
        ],
    )
    stampSize = ListField[int](
        doc="Size of the stamps to be extracted, in pixels.",
        default=(251, 251),
    )
    stampSizePadding = Field[float](
        doc="Multiplicative factor applied to the cutout stamp size, to guard against post-warp data loss.",
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
    maskWarpingKernelName = ChoiceField[str](
        doc="Warping kernel for mask.",
        default="bilinear",
        allowed={
            "bilinear": "bilinear interpolation",
            "lanczos3": "Lanczos kernel of order 3",
            "lanczos4": "Lanczos kernel of order 4",
            "lanczos5": "Lanczos kernel of order 5",
        },
    )
    scalePsfModel = Field[bool](
        doc="If True, uses a scale factor to bring the PSF model data to the same level as the star data.",
        default=True,
    )

    # PSF Fitting
    useExtendedPsf = Field[bool](
        doc="Use the extended PSF model to normalize bright star cutouts.",
        default=False,
    )
    doFitPsf = Field[bool](
        doc="Fit a scaled PSF and a pedestal to each bright star cutout.",
        default=True,
    )
    useMedianVariance = Field[bool](
        doc="Use the median of the variance plane for PSF fitting.",
        default=False,
    )
    psfMaskedFluxFracThreshold = Field[float](
        doc="Maximum allowed fraction of masked PSF flux for PSF fitting to occur.",
        default=0.97,
    )
    fitIterations = Field[int](
        doc="Number of iterations over pedestal-gradient and scaling fit.",
        default=5,
    )
    offFrameMagLim = Field[float](
        doc="Stars fainter than this limit are only included if they appear within the frame boundaries.",
        default=15.0,
    )

    # Misc
    loadReferenceObjectsConfig = ConfigField[LoadReferenceObjectsConfig](
        doc="Reference object loader for astrometric calibration.",
    )


class BrightStarCutoutTask(PipelineTask):
    """Extract bright star cutouts; normalize and warp to the same pixel grid.

    The BrightStarCutoutTask is used to extract, process, and store small image
    cutouts (or "postage stamps") around bright stars.
    This task essentially consists of three principal steps.
    First, it identifies bright stars within an exposure using a reference
    catalog and extracts a stamp around each.
    Second, it shifts and warps each stamp to remove optical distortions and
    sample all stars on the same pixel grid.
    Finally, it optionally fits a PSF plus plane flux model to the cutout.
    This final fitting procedure may be used to normalize each bright star
    stamp prior to stacking when producing extended PSF models.
    """

    ConfigClass = BrightStarCutoutConfig
    _DefaultName = "brightStarCutout"
    config: BrightStarCutoutConfig

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
        self.modelScale = 1

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["dataId"] = butlerQC.quantum.dataId
        refObjLoader = ReferenceObjectLoader(
            dataIds=[ref.datasetRef.dataId for ref in inputRefs.refCat],
            refCats=inputs.pop("refCat"),
            name=self.config.connections.refCat,
            config=self.config.loadReferenceObjectsConfig,
        )
        extendedPsf = inputs.pop("extendedPsf", None)
        output = self.run(**inputs, extendedPsf=extendedPsf, refObjLoader=refObjLoader)
        # Only ingest Stamp if it exists; prevents ingesting an empty FITS file
        if output:
            butlerQC.put(output, outputRefs)

    @timeMethod
    def run(
        self,
        inputExposure: ExposureF,
        inputBackground: BackgroundList,
        extendedPsf: ImageF | None,
        refObjLoader: ReferenceObjectLoader,
        dataId: dict[str, Any] | DataCoordinate,
    ):
        """Identify bright stars within an exposure using a reference catalog,
        extract stamps around each, warp/shift stamps onto a common frame and
        then optionally fit a PSF plus plane model.

        Parameters
        ----------
        inputExposure : `~lsst.afw.image.ExposureF`
            The background-subtracted image to extract bright star stamps.
        inputBackground : `~lsst.afw.math.BackgroundList`
            The background model associated with the input exposure.
        refObjLoader : `~lsst.meas.algorithms.ReferenceObjectLoader`, optional
            Loader to find objects within a reference catalog.
        dataId : `dict` or `~lsst.daf.butler.DataCoordinate`
            The dataId of the exposure that bright stars are extracted from.
            Both 'visit' and 'detector' will be persisted in the output data.

        Returns
        -------
        brightStarResults : `~lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``brightStarStamps``
                (`~lsst.meas.algorithms.brightStarStamps.BrightStarStamps`)
        """
        wcs = inputExposure.getWcs()
        bbox = inputExposure.getBBox()
        warpingControl = WarpingControl(self.config.warpingKernelName, self.config.maskWarpingKernelName)

        refCatBright = self._getRefCatBright(refObjLoader, wcs, bbox)
        zipRaDec = zip(refCatBright["coord_ra"] * radians, refCatBright["coord_dec"] * radians)
        spherePoints = [SpherePoint(ra, dec) for ra, dec in zipRaDec]
        pixCoords = wcs.skyToPixel(spherePoints)

        # Restore original subtracted background
        inputMI = inputExposure.getMaskedImage()
        inputMI += inputBackground.getImage()

        # Set up NEIGHBOR mask plane; associate footprints with stars
        inputExposure.mask.addMaskPlane(NEIGHBOR_MASK_PLANE)
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
        stamps, goodFracs, stamps_fitPsfResults = [], [], []
        for starIndex, (obj, pixCoord) in enumerate(zip(refCatBright, pixCoords)):  # type: ignore
            # Excluding faint stars that are not within the frame.
            if obj["mag"] > self.config.offFrameMagLim and not self.star_in_frame(pixCoord, bbox):
                continue
            footprintIndex = associations.get(starIndex, None)
            stampMI = MaskedImageF(self.paddedStampBBox)

            # Set NEIGHBOR footprints in the mask plane
            if footprintIndex:
                neighborFootprints = [fp for i, fp in enumerate(allFootprints) if i != footprintIndex]
                self._setFootprints(inputMI, neighborFootprints, NEIGHBOR_MASK_PLANE)
            else:
                self._setFootprints(inputMI, allFootprints, NEIGHBOR_MASK_PLANE)

            # Define linear shifting to recenter stamps
            coordFocalPlaneTan = pixToFocalPlaneTan.applyForward(pixCoord)  # center of warped star
            shift = makeTransform(AffineTransform(Point2D(0, 0) - coordFocalPlaneTan))
            angle = np.arctan2(coordFocalPlaneTan.getY(), coordFocalPlaneTan.getX()) * radians
            rotation = makeTransform(AffineTransform.makeRotation(-angle))
            pixToPolar = pixToFocalPlaneTan.then(shift).then(rotation)

            # Apply the warp to the star stamp (in-place)
            warpImage(stampMI, inputMI, pixToPolar, warpingControl)

            # Trim to the base stamp size, check mask coverage, update metadata
            stampMI = stampMI[self.stampBBox]
            badMaskBitMask = stampMI.mask.getPlaneBitMask(self.config.badMaskPlanes)
            goodFrac = np.sum(stampMI.mask.array & badMaskBitMask == 0) / stampMI.mask.array.size
            goodFracs.append(goodFrac)
            if goodFrac < self.config.minAreaFraction:
                continue

            # Fit a scaled PSF and a pedestal to each bright star cutout
            psf = WarpedPsf(inputExposure.getPsf(), pixToPolar, warpingControl)
            constantPsf = KernelPsf(FixedKernel(psf.computeKernelImage(Point2D(0, 0))))
            if self.config.useExtendedPsf:
                psfImage = deepcopy(extendedPsf)  # Assumed to be warped, center at [0,0]
            else:
                psfImage = constantPsf.computeKernelImage(constantPsf.getAveragePosition())
                # TODO: maybe we want to generate a smaller psf in case the following happens?
                # The following could happen for when the user chooses small stampSize ~(50, 50)
                if (
                    psfImage.array.shape[0] > stampMI.image.array.shape[0]
                    or psfImage.array.shape[1] > stampMI.image.array.shape[1]
                ):
                    continue
            # Computing an scale factor that brings the model to the similar level of the star.
            self.computeModelScale(stampMI, psfImage)
            psfImage.array *= self.modelScale  # ####### model scale correction ########

            fitPsfResults = {}

            if self.config.doFitPsf:
                fitPsfResults = self._fitPsf(stampMI, psfImage)
            stamps_fitPsfResults.append(fitPsfResults)

            # Save the stamp if the PSF fit was successful or no fit requested
            if fitPsfResults or not self.config.doFitPsf:
                distance_mm, theta_angle = self.star_location_on_focal(pixCoord, detector)

                stamp = BrightStarStamp(
                    stamp_im=stampMI,
                    psf=constantPsf,
                    wcs=makeModifiedWcs(pixToPolar, wcs, False),
                    visit=cast(int, dataId["visit"]),
                    detector=cast(int, dataId["detector"]),
                    ref_id=obj["id"],
                    ref_mag=obj["mag"],
                    position=pixCoord,
                    focal_plane_radius=distance_mm,
                    focal_plane_angle=theta_angle,  # TODO: add the lsst.geom.Angle here
                    scale=fitPsfResults.get("scale", None),
                    scale_err=fitPsfResults.get("scaleErr", None),
                    pedestal=fitPsfResults.get("pedestal", None),
                    pedestal_err=fitPsfResults.get("pedestalErr", None),
                    pedestal_scale_cov=fitPsfResults.get("pedestalScaleCov", None),
                    gradient_x=fitPsfResults.get("xGradient", None),
                    gradient_y=fitPsfResults.get("yGradient", None),
                    global_reduced_chi_squared=fitPsfResults.get("globalReducedChiSquared", None),
                    global_degrees_of_freedom=fitPsfResults.get("globalDegreesOfFreedom", None),
                    psf_reduced_chi_squared=fitPsfResults.get("psfReducedChiSquared", None),
                    psf_degrees_of_freedom=fitPsfResults.get("psfDegreesOfFreedom", None),
                    psf_masked_flux_fraction=fitPsfResults.get("psfMaskedFluxFrac", None),
                )
                print(
                    obj["mag"],
                    fitPsfResults.get("globalReducedChiSquared", None),
                    fitPsfResults.get("globalDegreesOfFreedom", None),
                    fitPsfResults.get("psfReducedChiSquared", None),
                    fitPsfResults.get("psfDegreesOfFreedom", None),
                    fitPsfResults.get("psfMaskedFluxFrac", None),
                )
                stamps.append(stamp)

        self.log.info(
            "Extracted %i bright star stamp%s. "
            "Excluded %i star%s: insufficient area (%i), PSF fit failure (%i).",
            len(stamps),
            "" if len(stamps) == 1 else "s",
            len(refCatBright) - len(stamps),
            "" if len(refCatBright) - len(stamps) == 1 else "s",
            np.sum(np.array(goodFracs) < self.config.minAreaFraction),
            (
                np.sum(np.isnan([x.get("pedestal", np.nan) for x in stamps_fitPsfResults]))
                if self.config.doFitPsf
                else 0
            ),
        )
        brightStarStamps = BrightStarStamps(stamps)
        return Struct(brightStarStamps=brightStarStamps)

    def star_location_on_focal(self, pixCoord, detector):
        star_focal_plane_coords = detector.transform(pixCoord, PIXELS, FOCAL_PLANE)
        star_x_fp = star_focal_plane_coords.getX()
        star_y_fp = star_focal_plane_coords.getY()
        distance_mm = np.sqrt(star_x_fp**2 + star_y_fp**2)
        theta_rad = math.atan2(star_y_fp, star_x_fp)
        theta_angle = Angle(theta_rad, radians)
        return distance_mm, theta_angle

    def star_in_frame(self, pixCoord, inputExposureBBox):
        if (
            pixCoord[0] < 0
            or pixCoord[1] < 0
            or pixCoord[0] > inputExposureBBox.getDimensions()[0]
            or pixCoord[1] > inputExposureBBox.getDimensions()[1]
        ):
            return False
        return True

    def _getRefCatBright(self, refObjLoader: ReferenceObjectLoader, wcs: SkyWcs, bbox: Box2I) -> Table:
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

        proxFluxRange = sorted(((self.config.excludeMagRange * u.ABmag).to(u.nJy)).to_value())
        brightFluxRange = sorted(((self.config.magRange * u.ABmag).to(u.nJy)).to_value())

        subsetStars = (refCatFull[fluxField] > np.min((proxFluxRange[0], brightFluxRange[0]))) & (
            refCatFull[fluxField] < np.max((proxFluxRange[1], brightFluxRange[1]))
        )
        refCatSubset = Table(refCatFull.extract("id", "coord_ra", "coord_dec", fluxField, where=subsetStars))

        proxStars = (refCatSubset[fluxField] >= proxFluxRange[0]) & (
            refCatSubset[fluxField] <= proxFluxRange[1]
        )
        brightStars = (refCatSubset[fluxField] >= brightFluxRange[0]) & (
            refCatSubset[fluxField] <= brightFluxRange[1]
        )

        coords = SkyCoord(refCatSubset["coord_ra"], refCatSubset["coord_dec"], unit="rad")
        excludeArcsecRadius = self.config.excludeArcsecRadius * u.arcsec  # type: ignore
        refCatBrightIsolated = []
        for coord in cast(Iterable[SkyCoord], coords[brightStars]):
            neighbors = coords[proxStars]
            seps = coord.separation(neighbors).to(u.arcsec)
            tooClose = (seps > 0) & (seps <= excludeArcsecRadius)  # not self matched
            refCatBrightIsolated.append(not tooClose.any())

        refCatBright = cast(Table, refCatSubset[brightStars][refCatBrightIsolated])

        fluxNanojansky = refCatBright[fluxField][:] * u.nJy  # type: ignore
        refCatBright["mag"] = fluxNanojansky.to(u.ABmag).to_value()  # AB magnitudes

        self.log.info(
            "Identified %i of %i star%s which satisfy: frame overlap; in the range %s mag; no neighboring "
            "stars within %s arcsec.",
            len(refCatBright),
            len(refCatFull),
            "" if len(refCatFull) == 1 else "s",
            self.config.magRange,
            self.config.excludeArcsecRadius,
        )

        return refCatBright

    def _associateFootprints(
        self, inputExposure: ExposureF, pixCoords: list[Point2D], plane: str
    ) -> tuple[list[Footprint], dict[int, int]]:
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
        associations : `dict`[int, int]
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
        self.log.debug(
            "Associated %i of %i star%s to one each of the %i %s footprint%s.",
            len(associations),
            len(pixCoords),
            "" if len(pixCoords) == 1 else "s",
            len(footprints),
            plane,
            "" if len(footprints) == 1 else "s",
        )
        return footprints, associations

    def _setFootprints(self, inputExposure: ExposureF, footprints: list, maskPlane: str):
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

    def _fitPsf(self, stampMI: MaskedImageF, psfImage: ImageD | ImageF) -> dict[str, Any]:
        """Fit a scaled PSF and a pedestal to each bright star cutout.

        Parameters
        ----------
        stampMI : `~lsst.afw.image.MaskedImageF`
            The masked image of the bright star cutout.
        psfImage : `~lsst.afw.image.ImageD` | `~lsst.afw.image.ImageF`
            The PSF model to fit.

        Returns
        -------
        fitPsfResults : `dict`[`str`, `float`]
            The result of the PSF fitting, with keys:

            ``scale`` : `float`
                The scale factor.
            ``scaleErr`` : `float`
                The error on the scale factor.
            ``pedestal`` : `float`
                The pedestal value.
            ``pedestalErr`` : `float`
                The error on the pedestal value.
            ``pedestalScaleCov`` : `float`
                The covariance between the pedestal and scale factor.
            ``xGradient`` : `float`
                The gradient in the x-direction.
            ``yGradient`` : `float`
                The gradient in the y-direction.
            ``globalReducedChiSquared`` : `float`
                The global reduced chi-squared goodness-of-fit.
            ``globalDegreesOfFreedom`` : `int`
                The global number of degrees of freedom.
            ``psfReducedChiSquared`` : `float`
                The PSF BBox reduced chi-squared goodness-of-fit.
            ``psfDegreesOfFreedom`` : `int`
                The PSF BBox number of degrees of freedom.
            ``psfMaskedFluxFrac`` : `float`
                The fraction of the PSF image flux masked by bad pixels.
        """
        badMaskBitMask = stampMI.mask.getPlaneBitMask(self.config.badMaskPlanes)

        # Calculate the fraction of the PSF image flux masked by bad pixels
        psfMaskedPixels = ImageF(psfImage.getBBox())
        psfMaskedPixels.array[:, :] = (stampMI.mask[psfImage.getBBox()].array & badMaskBitMask).astype(bool)
        psfMaskedFluxFrac = (
            np.dot(psfImage.array.flat, psfMaskedPixels.array.flat).astype(np.float64) / psfImage.array.sum()
        )
        if psfMaskedFluxFrac > self.config.psfMaskedFluxFracThreshold:
            return {}  # Handle cases where the PSF image is mostly masked

        # Generating good spans for gradient-pedestal fitting (including the star DETECTED mask).
        gradientGoodSpans = self.generate_gradient_spans(stampMI, badMaskBitMask)
        varianceData = gradientGoodSpans.flatten(stampMI.variance.array, stampMI.getXY0())
        if self.config.useMedianVariance:
            varianceData = np.median(varianceData)
        sigmaData = np.sqrt(varianceData)

        for i in range(self.config.fitIterations):
            # Gradient-pedestal fitting:
            if i:
                # if i > 0, there should be scale factor from the previous fit iteration. Therefore, we can
                # remove the star using the scale factor.
                stamp = self.remove_star(stampMI, scale, paddedPsfImage)    # noqa: F821
            else:
                stamp = deepcopy(stampMI.image.array)

            imageDataGr = gradientGoodSpans.flatten(stamp, stampMI.getXY0()) / sigmaData  # B
            nData = len(imageDataGr)
            coefficientMatrix = np.ones((nData, 3), dtype=float)  # A
            coefficientMatrix[:, 0] /= sigmaData
            coefficientMatrix[:, 1:] = gradientGoodSpans.indices().T
            coefficientMatrix[:, 1] /= sigmaData
            coefficientMatrix[:, 2] /= sigmaData

            try:
                grSolutions, grSumSquaredResiduals, *_ = np.linalg.lstsq(
                    coefficientMatrix, imageDataGr, rcond=None
                )
                covarianceMatrix = np.linalg.inv(
                    np.dot(coefficientMatrix.transpose(), coefficientMatrix)
                )  # C
            except np.linalg.LinAlgError:
                return {}  # Handle singular matrix errors
            if grSumSquaredResiduals.size == 0:
                return {}  # Handle cases where sum of the squared residuals are empty

            pedestal = grSolutions[0]
            pedestalErr = np.sqrt(covarianceMatrix[0, 0])
            scalePedestalCov = None
            xGradient = grSolutions[2]
            yGradient = grSolutions[1]

            # Scale fitting:
            updatedStampMI = deepcopy(stampMI)
            self._removePedestalAndGradient(updatedStampMI, pedestal, xGradient, yGradient)

            # Create a padded version of the input constant PSF image
            paddedPsfImage = ImageF(updatedStampMI.getBBox())
            paddedPsfImage[psfImage.getBBox()] = psfImage.convertF()

            # Generating a mask plane while considering bad pixels in the psf model.
            mask = self.add_psf_mask(paddedPsfImage, updatedStampMI)
            # Create consistently masked data
            scaleGoodSpans = self.generate_good_spans(mask, updatedStampMI.getBBox(), badMaskBitMask)

            imageData = scaleGoodSpans.flatten(updatedStampMI.image.array, updatedStampMI.getXY0())
            psfData = scaleGoodSpans.flatten(paddedPsfImage.array, paddedPsfImage.getXY0())
            scaleCoefficientMatrix = psfData.reshape(psfData.shape[0], 1)

            try:
                scaleSolution, scaleSumSquaredResiduals, *_ = np.linalg.lstsq(
                    scaleCoefficientMatrix, imageData, rcond=None
                )
            except np.linalg.LinAlgError:
                return {}  # Handle singular matrix errors
            if scaleSumSquaredResiduals.size == 0:
                return {}  # Handle cases where sum of the squared residuals are empty
            scale = scaleSolution[0]
            if scale <= 0:
                return {}  # Handle cases where the PSF scale fit has failed
            # TODO: calculate scale error and store it.
            scaleErr = None

        scale *= self.modelScale  # ####### model scale correction ########
        nData = len(imageData)

        # Calculate global (whole image) reduced chi-squared (scaling fit is assumed as the main fitting
        # process here.)
        globalChiSquared = np.sum(scaleSumSquaredResiduals)
        globalDegreesOfFreedom = nData - 1
        globalReducedChiSquared = np.float64(globalChiSquared / globalDegreesOfFreedom)

        # Calculate PSF BBox reduced chi-squared
        psfBBoxscaleGoodSpans = scaleGoodSpans.clippedTo(psfImage.getBBox())
        psfBBoxscaleGoodSpansX, psfBBoxscaleGoodSpansY = psfBBoxscaleGoodSpans.indices()
        psfBBoxData = psfBBoxscaleGoodSpans.flatten(stampMI.image.array, stampMI.getXY0())
        paddedPsfImage.array /= self.modelScale  # ####### model scale correction ########
        psfBBoxModel = (
            psfBBoxscaleGoodSpans.flatten(paddedPsfImage.array, stampMI.getXY0()) * scale
            + pedestal
            + psfBBoxscaleGoodSpansX * xGradient
            + psfBBoxscaleGoodSpansY * yGradient
        )
        psfBBoxResiduals = (psfBBoxData - psfBBoxModel) ** 2  # / psfBBoxVariance
        psfBBoxChiSquared = np.sum(psfBBoxResiduals)
        psfBBoxDegreesOfFreedom = len(psfBBoxData) - 1
        psfBBoxReducedChiSquared = psfBBoxChiSquared / psfBBoxDegreesOfFreedom
        return dict(
            scale=scale,
            scaleErr=scaleErr,
            pedestal=pedestal,
            pedestalErr=pedestalErr,
            xGradient=xGradient,
            yGradient=yGradient,
            pedestalScaleCov=scalePedestalCov,
            globalReducedChiSquared=globalReducedChiSquared,
            globalDegreesOfFreedom=globalDegreesOfFreedom,
            psfReducedChiSquared=psfBBoxReducedChiSquared,
            psfDegreesOfFreedom=psfBBoxDegreesOfFreedom,
            psfMaskedFluxFrac=psfMaskedFluxFrac,
        )

    def add_psf_mask(self, psfImage, stampMI, maskZeros=True):
        """
        Creates a new mask by adding PSF bad pixels to an existing stamp mask.

        This method identifies "bad" pixels in the PSF image (NaNs and
        optionally zeros/non-positives) and adds them to a deep copy
        of the input stamp's mask.

        Args:
            psfImage : `~lsst.afw.image.ImageD` | `~lsst.afw.image.ImageF`
                The PSF image object.
            stampMI: `~lsst.afw.image.MaskedImageF`
                The masked image of the bright star cutout.
            maskZeros (bool, optional): If True (default), mask pixels
                where the PSF is <= 0. If False, only mask pixels < 0.

        Returns:
            Any: A new mask object (deep copy) with the PSF mask planes added.
        """
        cond = np.isnan(psfImage.array)
        if maskZeros:
            cond |= psfImage.array <= 0
        else:
            cond |= psfImage.array < 0
        mask = deepcopy(stampMI.mask)
        mask.array[cond] = np.bitwise_or(mask.array[cond], 1)
        return mask

    def _removePedestalAndGradient(self, stampMI, pedestal, xGradient, yGradient):
        """Apply fitted pedestal and gradients to a single bright star stamp."""
        stampBBox = stampMI.getBBox()
        xGrid, yGrid = np.meshgrid(stampBBox.getX().arange(), stampBBox.getY().arange())
        xPlane = ImageF((xGrid * xGradient).astype(np.float32), xy0=stampMI.getXY0())
        yPlane = ImageF((yGrid * yGradient).astype(np.float32), xy0=stampMI.getXY0())
        stampMI -= pedestal
        stampMI -= xPlane
        stampMI -= yPlane

    def remove_star(self, stampMI, scale, psfImage):
        """
        Subtracts a scaled PSF model from a star image.

        This performs a simple subtraction: `image - (psf * scale)`.

        Args:
            stampMI: `~lsst.afw.image.MaskedImageF`
                The masked image of the bright star cutout.
            scale (float): The scaling factor to apply to the PSF.
            psfImage: `~lsst.afw.image.ImageD` | `~lsst.afw.image.ImageF`
                The PSF image object.

        Returns:
            np.ndarray: A new 2D numpy array containing the star-subtracted
            image.
        """
        star_removed_cutout = stampMI.image.array - psfImage.array * scale
        return star_removed_cutout

    def computeModelScale(self, stampMI, psfImage):
        """
        Computes the scaling factor of the given model against a star.

        Args:
            stampMI : `~lsst.afw.image.MaskedImageF`
                The masked image of the bright star cutout.
            psfImage : `~lsst.afw.image.ImageD` | `~lsst.afw.image.ImageF`
                The given PSF model.
        """
        cond = stampMI.mask.array == 0
        self.starMedianValue = np.median(stampMI.image.array[cond]).astype(np.float64)

        psfPos = psfImage.array > 0

        imageArray = stampMI.image.array - self.starMedianValue
        imageArrayPos = imageArray > 0
        self.modelScale = np.nanmean(imageArray[imageArrayPos]) / np.nanmean(psfImage.array[psfPos])

    def generate_gradient_spans(self, stampMI, badMaskBitMask):
        """
        Generates spans of "good" pixels for gradient fitting.

        This method creates a combined bitmask by OR-ing the provided
        `badMaskBitMask` with the "DETECTED" plane from the stamp's mask.
        It then calls `self.generate_good_spans` to find all pixel spans
        not covered by this combined mask.

        Args:
            stampMI: `~lsst.afw.image.MaskedImageF`
                The masked image of the bright star cutout.
            badMaskBitMask (int): A bitmask representing planes to be
                considered "bad" for gradient fitting.

        Returns:
            gradientGoodSpans: A SpanSet object containing the "good" spans.
        """
        detectedMaskBitMask = stampMI.mask.getPlaneBitMask("DETECTED")
        gradientBitMask = np.bitwise_or(badMaskBitMask, detectedMaskBitMask)

        gradientGoodSpans = self.generate_good_spans(stampMI.mask, stampMI.getBBox(), gradientBitMask)
        return gradientGoodSpans

    def generate_good_spans(self, mask, bBox, badBitMask):
        """
        Generates a SpanSet of "good" pixels from a mask.

        This method identifies all spans within a given bounding box (`bBox`)
        that are *not* flagged by the `badBitMask` in the provided `mask`.

        Args:
            mask (lsst.afw.image.MaskedImageF.mask): The mask object (e.g., `stampMI.mask`).
            bBox (lsst.geom.Box2I): The bounding box of the image (e.g., `stampMI.getBBox()`).
            badBitMask (int): The combined bitmask of planes to exclude.

        Returns:
            goodSpans: A SpanSet object representing all "good" spans.
        """
        badSpans = SpanSet.fromMask(mask, badBitMask)
        goodSpans = SpanSet(bBox).intersectNot(badSpans)
        return goodSpans
