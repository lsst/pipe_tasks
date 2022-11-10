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

"""
Insert fakes into deepCoadds
"""

__all__ = ["InsertFakesConfig", "InsertFakesTask"]

import galsim
import numpy as np
from astropy import units as u

import lsst.geom as geom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.exceptions import LogicError, InvalidParameterError
from lsst.geom import SpherePoint, radians, Box2D, Point2D


def _add_fake_sources(exposure, objects, calibFluxRadius=12.0, logger=None):
    """Add fake sources to the given exposure

    Parameters
    ----------
    exposure : `lsst.afw.image.exposure.exposure.ExposureF`
        The exposure into which the fake sources should be added
    objects : `typing.Iterator` [`tuple` ['lsst.geom.SpherePoint`, `galsim.GSObject`]]
        An iterator of tuples that contains (or generates) locations and object
        surface brightness profiles to inject.
    calibFluxRadius : `float`, optional
        Aperture radius (in pixels) used to define the calibration for this
        exposure+catalog.  This is used to produce the correct instrumental fluxes
        within the radius.  The value should match that of the field defined in
        slot_CalibFlux_instFlux.
    logger : `lsst.log.log.log.Log` or `logging.Logger`, optional
        Logger.
    """
    exposure.mask.addMaskPlane("FAKE")
    bitmask = exposure.mask.getPlaneBitMask("FAKE")
    if logger:
        logger.info(f"Adding mask plane with bitmask {bitmask}")

    wcs = exposure.getWcs()
    psf = exposure.getPsf()

    bbox = exposure.getBBox()
    fullBounds = galsim.BoundsI(bbox.minX, bbox.maxX, bbox.minY, bbox.maxY)
    gsImg = galsim.Image(exposure.image.array, bounds=fullBounds)

    pixScale = wcs.getPixelScale(bbox.getCenter()).asArcseconds()

    for spt, gsObj in objects:
        pt = wcs.skyToPixel(spt)
        posd = galsim.PositionD(pt.x, pt.y)
        posi = galsim.PositionI(pt.x//1, pt.y//1)
        if logger:
            logger.debug("Adding fake source %s at %s", gsObj, pt)

        mat = wcs.linearizePixelToSky(spt, geom.arcseconds).getMatrix()
        gsWCS = galsim.JacobianWCS(mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1])

        # This check is here because sometimes the WCS
        # is multivalued and objects that should not be
        # were being included.
        gsPixScale = np.sqrt(gsWCS.pixelArea())
        if gsPixScale < pixScale/2 or gsPixScale > pixScale*2:
            continue

        try:
            psfArr = psf.computeKernelImage(pt).array
        except InvalidParameterError:
            # Try mapping to nearest point contained in bbox.
            contained_pt = Point2D(
                np.clip(pt.x, bbox.minX, bbox.maxX),
                np.clip(pt.y, bbox.minY, bbox.maxY)
            )
            if pt == contained_pt:  # no difference, so skip immediately
                if logger:
                    logger.infof(
                        "Cannot compute Psf for object at {}; skipping",
                        pt
                    )
                continue
            # otherwise, try again with new point
            try:
                psfArr = psf.computeKernelImage(contained_pt).array
            except InvalidParameterError:
                if logger:
                    logger.infof(
                        "Cannot compute Psf for object at {}; skipping",
                        pt
                    )
                continue

        apCorr = psf.computeApertureFlux(calibFluxRadius)
        psfArr /= apCorr
        gsPSF = galsim.InterpolatedImage(galsim.Image(psfArr), wcs=gsWCS)

        conv = galsim.Convolve(gsObj, gsPSF)
        stampSize = conv.getGoodImageSize(gsWCS.minLinearScale())
        subBounds = galsim.BoundsI(posi).withBorder(stampSize//2)
        subBounds &= fullBounds

        if subBounds.area() > 0:
            subImg = gsImg[subBounds]
            offset = posd - subBounds.true_center
            # Note, for calexp injection, pixel is already part of the PSF and
            # for coadd injection, it's incorrect to include the output pixel.
            # So for both cases, we draw using method='no_pixel'.

            conv.drawImage(
                subImg,
                add_to_image=True,
                offset=offset,
                wcs=gsWCS,
                method='no_pixel'
            )

            subBox = geom.Box2I(
                geom.Point2I(subBounds.xmin, subBounds.ymin),
                geom.Point2I(subBounds.xmax, subBounds.ymax)
            )
            exposure[subBox].mask.array |= bitmask


def _isWCSGalsimDefault(wcs, hdr):
    """Decide if wcs = galsim.PixelScale(1.0) is explicitly present in header,
    or if it's just the galsim default.

    Parameters
    ----------
    wcs : galsim.BaseWCS
        Potentially default WCS.
    hdr : galsim.fits.FitsHeader
        Header as read in by galsim.

    Returns
    -------
    isDefault : bool
        True if default, False if explicitly set in header.
    """
    if wcs != galsim.PixelScale(1.0):
        return False
    if hdr.get('GS_WCS') is not None:
        return False
    if hdr.get('CTYPE1', 'LINEAR') == 'LINEAR':
        return not any(k in hdr for k in ['CD1_1', 'CDELT1'])
    for wcs_type in galsim.fitswcs.fits_wcs_types:
        # If one of these succeeds, then assume result is explicit
        try:
            wcs_type._readHeader(hdr)
            return False
        except Exception:
            pass
    else:
        return not any(k in hdr for k in ['CD1_1', 'CDELT1'])


class InsertFakesConnections(PipelineTaskConnections,
                             defaultTemplates={"coaddName": "deep",
                                               "fakesType": "fakes_"},
                             dimensions=("tract", "patch", "band", "skymap")):

    image = cT.Input(
        doc="Image into which fakes are to be added.",
        name="{coaddName}Coadd",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap")
    )

    fakeCat = cT.Input(
        doc="Catalog of fake sources to draw inputs from.",
        name="{fakesType}fakeSourceCat",
        storageClass="DataFrame",
        dimensions=("tract", "skymap")
    )

    imageWithFakes = cT.Output(
        doc="Image with fake sources added.",
        name="{fakesType}{coaddName}Coadd",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "band", "skymap")
    )


class InsertFakesConfig(PipelineTaskConfig,
                        pipelineConnections=InsertFakesConnections):
    """Config for inserting fake sources
    """

    # Unchanged

    doCleanCat = pexConfig.Field(
        doc="If true removes bad sources from the catalog.",
        dtype=bool,
        default=True,
    )

    fakeType = pexConfig.Field(
        doc="What type of fake catalog to use, snapshot (includes variability in the magnitudes calculated "
            "from the MJD of the image), static (no variability) or filename for a user defined fits"
            "catalog.",
        dtype=str,
        default="static",
    )

    calibFluxRadius = pexConfig.Field(
        doc="Aperture radius (in pixels) that was used to define the calibration for this image+catalog. "
        "This will be used to produce the correct instrumental fluxes within the radius. "
        "This value should match that of the field defined in slot_CalibFlux_instFlux.",
        dtype=float,
        default=12.0,
    )

    coaddName = pexConfig.Field(
        doc="The name of the type of coadd used",
        dtype=str,
        default="deep",
    )

    doSubSelectSources = pexConfig.Field(
        doc="Set to True if you wish to sub select sources to be input based on the value in the column"
            "set in the sourceSelectionColName config option.",
        dtype=bool,
        default=False
    )

    insertImages = pexConfig.Field(
        doc="Insert images directly? True or False.",
        dtype=bool,
        default=False,
    )

    insertOnlyStars = pexConfig.Field(
        doc="Insert only stars? True or False.",
        dtype=bool,
        default=False,
    )

    doProcessAllDataIds = pexConfig.Field(
        doc="If True, all input data IDs will be processed, even those containing no fake sources.",
        dtype=bool,
        default=False,
    )

    trimBuffer = pexConfig.Field(
        doc="Size of the pixel buffer surrounding the image. Only those fake sources with a centroid"
        "falling within the image+buffer region will be considered for fake source injection.",
        dtype=int,
        default=100,
    )

    sourceType = pexConfig.Field(
        doc="The column name for the source type used in the fake source catalog.",
        dtype=str,
        default="sourceType",
    )

    fits_alignment = pexConfig.ChoiceField(
        doc="How should injections from FITS files be aligned?",
        dtype=str,
        allowed={
            "wcs": (
                "Input image will be transformed such that the local WCS in "
                "the FITS header matches the local WCS in the target image. "
                "I.e., North, East, and angular distances in the input image "
                "will match North, East, and angular distances in the target "
                "image."
            ),
            "pixel": (
                "Input image will _not_ be transformed.  Up, right, and pixel "
                "distances in the input image will match up, right and pixel "
                "distances in the target image."
            )
        },
        default="pixel"
    )

    # New source catalog config variables

    ra_col = pexConfig.Field(
        doc="Source catalog column name for RA (in radians).",
        dtype=str,
        default="ra",
    )

    dec_col = pexConfig.Field(
        doc="Source catalog column name for dec (in radians).",
        dtype=str,
        default="dec",
    )

    bulge_semimajor_col = pexConfig.Field(
        doc="Source catalog column name for the semimajor axis (in arcseconds) "
            "of the bulge half-light ellipse.",
        dtype=str,
        default="bulge_semimajor",
    )

    bulge_axis_ratio_col = pexConfig.Field(
        doc="Source catalog column name for the axis ratio of the bulge "
            "half-light ellipse.",
        dtype=str,
        default="bulge_axis_ratio",
    )

    bulge_pa_col = pexConfig.Field(
        doc="Source catalog column name for the position angle (measured from "
            "North through East in degrees) of the semimajor axis of the bulge "
            "half-light ellipse.",
        dtype=str,
        default="bulge_pa",
    )

    bulge_n_col = pexConfig.Field(
        doc="Source catalog column name for the Sersic index of the bulge.",
        dtype=str,
        default="bulge_n",
    )

    disk_semimajor_col = pexConfig.Field(
        doc="Source catalog column name for the semimajor axis (in arcseconds) "
            "of the disk half-light ellipse.",
        dtype=str,
        default="disk_semimajor",
    )

    disk_axis_ratio_col = pexConfig.Field(
        doc="Source catalog column name for the axis ratio of the disk "
            "half-light ellipse.",
        dtype=str,
        default="disk_axis_ratio",
    )

    disk_pa_col = pexConfig.Field(
        doc="Source catalog column name for the position angle (measured from "
            "North through East in degrees) of the semimajor axis of the disk "
            "half-light ellipse.",
        dtype=str,
        default="disk_pa",
    )

    disk_n_col = pexConfig.Field(
        doc="Source catalog column name for the Sersic index of the disk.",
        dtype=str,
        default="disk_n",
    )

    bulge_disk_flux_ratio_col = pexConfig.Field(
        doc="Source catalog column name for the bulge/disk flux ratio.  See "
            "also: ``bulge_flux_fraction_col``.",
        dtype=str,
        default="bulge_disk_flux_ratio",
    )

    mag_col = pexConfig.Field(
        doc="Source catalog column name template for magnitudes, in the format "
            "``filter name``_mag_col.  E.g., if this config variable is set to "
            "``%s_mag``, then the i-band magnitude will be searched for in the "
            "``i_mag`` column of the source catalog.",
        dtype=str,
        default="%s_mag"
    )

    bulge_flux_fraction_col = pexConfig.Field(
        doc="Source catalog column name for fraction of flux in bulge "
            "component, in the format ``filter name``_bulge_flux_fraction. "
            "E.g., if this config variable is set to ``%s_bulge_flux_fraction,"
            "then the i-band bulge flux fraction will be search for in the "
            "``i_bulge_flux_fraction`` column of the source catalog. "
            "Note that if the source catalog contains both the config values "
            "for bulge_flux_fraction_col and bulge_disk_flux_ratio_col, then "
            "the fluxes will be determined from bulge_flux_fraction_col and "
            "the bulge_disk_flux_ratio_col column will be ignored.",
        dtype=str,
        default="%s_bulge_flux_fraction"
    )

    select_col = pexConfig.Field(
        doc="Source catalog column name to be used to select which sources to "
            "add.",
        dtype=str,
        default="select",
    )

    length_col = pexConfig.Field(
        doc="Source catalog column name for trail length (in pixels).",
        dtype=str,
        default="trail_length",
    )

    angle_col = pexConfig.Field(
        doc="Source catalog column name for trail angle (in radians).",
        dtype=str,
        default="trail_angle",
    )

    # Deprecated config variables

    raColName = pexConfig.Field(
        doc="RA column name used in the fake source catalog.",
        dtype=str,
        default="raJ2000",
        deprecated="Use `ra_col` instead."
    )

    decColName = pexConfig.Field(
        doc="Dec. column name used in the fake source catalog.",
        dtype=str,
        default="decJ2000",
        deprecated="Use `dec_col` instead."
    )

    diskHLR = pexConfig.Field(
        doc="Column name for the disk half light radius used in the fake source catalog.",
        dtype=str,
        default="DiskHalfLightRadius",
        deprecated=(
            "Use `disk_semimajor_col`, `disk_axis_ratio_col`, and `disk_pa_col`"
            " to specify disk half-light ellipse."
        )
    )

    aDisk = pexConfig.Field(
        doc="The column name for the semi major axis length of the disk component used in the fake source"
            "catalog.",
        dtype=str,
        default="a_d",
        deprecated=(
            "Use `disk_semimajor_col`, `disk_axis_ratio_col`, and `disk_pa_col`"
            " to specify disk half-light ellipse."
        )
    )

    bDisk = pexConfig.Field(
        doc="The column name for the semi minor axis length of the disk component.",
        dtype=str,
        default="b_d",
        deprecated=(
            "Use `disk_semimajor_col`, `disk_axis_ratio_col`, and `disk_pa_col`"
            " to specify disk half-light ellipse."
        )
    )

    paDisk = pexConfig.Field(
        doc="The column name for the PA of the disk component used in the fake source catalog.",
        dtype=str,
        default="pa_disk",
        deprecated=(
            "Use `disk_semimajor_col`, `disk_axis_ratio_col`, and `disk_pa_col`"
            " to specify disk half-light ellipse."
        )
    )

    nDisk = pexConfig.Field(
        doc="The column name for the sersic index of the disk component used in the fake source catalog.",
        dtype=str,
        default="disk_n",
        deprecated="Use `disk_n_col` instead."
    )

    bulgeHLR = pexConfig.Field(
        doc="Column name for the bulge half light radius used in the fake source catalog.",
        dtype=str,
        default="BulgeHalfLightRadius",
        deprecated=(
            "Use `bulge_semimajor_col`, `bulge_axis_ratio_col`, and "
            "`bulge_pa_col` to specify disk half-light ellipse."
        )
    )

    aBulge = pexConfig.Field(
        doc="The column name for the semi major axis length of the bulge component.",
        dtype=str,
        default="a_b",
        deprecated=(
            "Use `bulge_semimajor_col`, `bulge_axis_ratio_col`, and "
            "`bulge_pa_col` to specify disk half-light ellipse."
        )
    )

    bBulge = pexConfig.Field(
        doc="The column name for the semi minor axis length of the bulge component used in the fake source "
            "catalog.",
        dtype=str,
        default="b_b",
        deprecated=(
            "Use `bulge_semimajor_col`, `bulge_axis_ratio_col`, and "
            "`bulge_pa_col` to specify disk half-light ellipse."
        )
    )

    paBulge = pexConfig.Field(
        doc="The column name for the PA of the bulge component used in the fake source catalog.",
        dtype=str,
        default="pa_bulge",
        deprecated=(
            "Use `bulge_semimajor_col`, `bulge_axis_ratio_col`, and "
            "`bulge_pa_col` to specify disk half-light ellipse."
        )
    )

    nBulge = pexConfig.Field(
        doc="The column name for the sersic index of the bulge component used in the fake source catalog.",
        dtype=str,
        default="bulge_n",
        deprecated="Use `bulge_n_col` instead."
    )

    magVar = pexConfig.Field(
        doc="The column name for the magnitude calculated taking variability into account. In the format "
            "``filter name``magVar, e.g. imagVar for the magnitude in the i band.",
        dtype=str,
        default="%smagVar",
        deprecated="Use `mag_col` instead."
    )

    sourceSelectionColName = pexConfig.Field(
        doc="The name of the column in the input fakes catalogue to be used to determine which sources to"
            "add, default is none and when this is used all sources are added.",
        dtype=str,
        default="templateSource",
        deprecated="Use `select_col` instead."
    )


class InsertFakesTask(PipelineTask):
    """Insert fake objects into images.

    Add fake stars and galaxies to the given image, read in through the dataRef. Galaxy parameters are read in
    from the specified file and then modelled using galsim.

    `InsertFakesTask` has five functions that make images of the fake sources and then add them to the
    image.

    `addPixCoords`
        Use the WCS information to add the pixel coordinates of each source.
    `mkFakeGalsimGalaxies`
        Use Galsim to make fake double sersic galaxies for each set of galaxy parameters in the input file.
    `mkFakeStars`
        Use the PSF information from the image to make a fake star using the magnitude information from the
        input file.
    `cleanCat`
        Remove rows of the input fake catalog which have half light radius, of either the bulge or the disk,
        that are 0. Also removes rows that have Sersic index outside of galsim's allowed paramters. If
        the config option sourceSelectionColName is set then this function limits the catalog of input fakes
        to only those which are True in this column.
    `addFakeSources`
        Add the fake sources to the image.

    """

    _DefaultName = "insertFakes"
    ConfigClass = InsertFakesConfig

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["wcs"] = inputs["image"].getWcs()
        inputs["photoCalib"] = inputs["image"].getPhotoCalib()

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, fakeCat, image, wcs, photoCalib):
        """Add fake sources to an image.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to be input
        image : `lsst.afw.image.exposure.exposure.ExposureF`
                    The image into which the fake sources should be added
        wcs : `lsst.afw.geom.SkyWcs`
                    WCS to use to add fake sources
        photoCalib : `lsst.afw.image.photoCalib.PhotoCalib`
                    Photometric calibration to be used to calibrate the fake sources

        Returns
        -------
        resultStruct : `lsst.pipe.base.struct.Struct`
            contains : image : `lsst.afw.image.exposure.exposure.ExposureF`

        Notes
        -----
        Adds pixel coordinates for each source to the fakeCat and removes objects with bulge or disk half
        light radius = 0 (if ``config.doCleanCat = True``).

        Adds the ``Fake`` mask plane to the image which is then set by `addFakeSources` to mark where fake
        sources have been added. Uses the information in the ``fakeCat`` to make fake galaxies (using galsim)
        and fake stars, using the PSF models from the PSF information for the image. These are then added to
        the image and the image with fakes included returned.

        The galsim galaxies are made using a double sersic profile, one for the bulge and one for the disk,
        this is then convolved with the PSF at that point.
        """
        # Attach overriding wcs and photoCalib to image, but retain originals
        # so we can reset at the end.
        origWcs = image.getWcs()
        origPhotoCalib = image.getPhotoCalib()
        image.setWcs(wcs)
        image.setPhotoCalib(photoCalib)

        band = image.getFilter().bandLabel
        fakeCat = self._standardizeColumns(fakeCat, band)

        fakeCat = self.addPixCoords(fakeCat, image)
        fakeCat = self.trimFakeCat(fakeCat, image)

        if len(fakeCat) > 0:
            if not self.config.insertImages:
                if isinstance(fakeCat[self.config.sourceType].iloc[0], str):
                    galCheckVal = "galaxy"
                    starCheckVal = "star"
                    trailCheckVal = "trail"
                elif isinstance(fakeCat[self.config.sourceType].iloc[0], bytes):
                    galCheckVal = b"galaxy"
                    starCheckVal = b"star"
                    trailCheckVal = b"trail"
                elif isinstance(fakeCat[self.config.sourceType].iloc[0], (int, float)):
                    galCheckVal = 1
                    starCheckVal = 0
                    trailCheckVal = 2
                else:
                    raise TypeError(
                        "sourceType column does not have required type, should be str, bytes or int"
                    )
                if self.config.doCleanCat:
                    fakeCat = self.cleanCat(fakeCat, starCheckVal)

                generator = self._generateGSObjectsFromCatalog(image, fakeCat, galCheckVal, starCheckVal,
                                                               trailCheckVal)
            else:
                generator = self._generateGSObjectsFromImages(image, fakeCat)
            _add_fake_sources(image, generator, calibFluxRadius=self.config.calibFluxRadius, logger=self.log)
        elif len(fakeCat) == 0 and self.config.doProcessAllDataIds:
            self.log.warning("No fakes found for this dataRef; processing anyway.")
            image.mask.addMaskPlane("FAKE")
        else:
            raise RuntimeError("No fakes found for this dataRef.")

        # restore original exposure WCS and photoCalib
        image.setWcs(origWcs)
        image.setPhotoCalib(origPhotoCalib)

        resultStruct = pipeBase.Struct(imageWithFakes=image)

        return resultStruct

    def _standardizeColumns(self, fakeCat, band):
        """Use config variables to 'standardize' the expected columns and column
        names in the input catalog.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
            The catalog of fake sources to be input
        band : `str`
            Label for the current band being processed.

        Returns
        -------
        outCat : `pandas.core.frame.DataFrame`
            The standardized catalog of fake sources
        """
        cfg = self.config
        replace_dict = {}

        def add_to_replace_dict(new_name, depr_name, std_name):
            if new_name in fakeCat.columns:
                replace_dict[new_name] = std_name
            elif depr_name in fakeCat.columns:
                replace_dict[depr_name] = std_name
            else:
                raise ValueError(f"Could not determine column for {std_name}.")

        # Prefer new config variables over deprecated config variables.
        # RA, dec, and mag are always required.  Do these first
        for new_name, depr_name, std_name in [
            (cfg.ra_col, cfg.raColName, 'ra'),
            (cfg.dec_col, cfg.decColName, 'dec'),
            (cfg.mag_col%band, cfg.magVar%band, 'mag')
        ]:
            add_to_replace_dict(new_name, depr_name, std_name)
        # Only handle bulge/disk params if not injecting images
        if not cfg.insertImages and not cfg.insertOnlyStars:
            for new_name, depr_name, std_name in [
                (cfg.bulge_n_col, cfg.nBulge, 'bulge_n'),
                (cfg.bulge_pa_col, cfg.paBulge, 'bulge_pa'),
                (cfg.disk_n_col, cfg.nDisk, 'disk_n'),
                (cfg.disk_pa_col, cfg.paDisk, 'disk_pa'),
            ]:
                add_to_replace_dict(new_name, depr_name, std_name)

        if cfg.doSubSelectSources:
            add_to_replace_dict(
                cfg.select_col,
                cfg.sourceSelectionColName,
                'select'
            )
        if replace_dict:
            self.log.debug("Replacing columns: %s", replace_dict)
        fakeCat = fakeCat.rename(columns=replace_dict, copy=False)

        # Handling the half-light radius and axis-ratio are trickier, since we
        # moved from expecting (HLR, a, b) to expecting (semimajor, axis_ratio).
        # Just handle these manually.
        if not cfg.insertImages and not cfg.insertOnlyStars:
            if (
                cfg.bulge_semimajor_col in fakeCat.columns
                and cfg.bulge_axis_ratio_col in fakeCat.columns
            ):
                replace_dict = {
                    cfg.bulge_semimajor_col: 'bulge_semimajor',
                    cfg.bulge_axis_ratio_col: 'bulge_axis_ratio',
                    cfg.disk_semimajor_col: 'disk_semimajor',
                    cfg.disk_axis_ratio_col: 'disk_axis_ratio',
                }
                fakeCat = fakeCat.rename(
                    columns=replace_dict,
                    copy=False
                )
                if replace_dict:
                    self.log.debug("Replacing columns: %s", replace_dict)
            elif (
                cfg.bulgeHLR in fakeCat.columns
                and cfg.aBulge in fakeCat.columns
                and cfg.bBulge in fakeCat.columns
            ):
                fakeCat['bulge_axis_ratio'] = (
                    fakeCat[cfg.bBulge]/fakeCat[cfg.aBulge]
                )
                fakeCat['bulge_semimajor'] = (
                    fakeCat[cfg.bulgeHLR]/np.sqrt(fakeCat['bulge_axis_ratio'])
                )
                fakeCat['disk_axis_ratio'] = (
                    fakeCat[cfg.bDisk]/fakeCat[cfg.aDisk]
                )
                fakeCat['disk_semimajor'] = (
                    fakeCat[cfg.diskHLR]/np.sqrt(fakeCat['disk_axis_ratio'])
                )
                self.log.debug(
                    "Replacing (%s, %s, %s, %s, %s, %s) with "
                    "(bulge_axis_ratio, bulge_semimajor, disk_axis_ratio, "
                    "disk_semimajor)",
                    cfg.bBulge, cfg.aBulge, cfg.bulgeHLR,
                    cfg.bDisk, cfg.aDisk, cfg.diskHLR
                )
            else:
                raise ValueError(
                    "Could not determine columns for half-light radius and "
                    "axis ratio."
                )

            # Standardize flux apportionment between bulge and disk using
            #  `bulge_flux_fraction`.  Prefer, in order:
            #   - `bulge_flux_fraction_col`
            #   - `bulge_disk_flux_ratio_col`
            #   - bd_flux_ratio = 1.0, which is equivalent to bulge_flux_fraction=0.5
            if cfg.bulge_flux_fraction_col%band in fakeCat.columns:
                fakeCat = fakeCat.rename(
                    columns={
                        cfg.bulge_flux_fraction_col%band: 'bulge_flux_fraction'
                    },
                    copy=False
                )
                self.log.debug(
                    "Replacing %s with bulge_flux_fraction.",
                    cfg.bulge_flux_fraction_col%band
                )
            elif cfg.bulge_disk_flux_ratio_col in fakeCat.columns:
                bdfr = cfg.bulge_disk_flux_ratio_col
                fakeCat['bulge_flux_fraction'] = (
                    fakeCat[bdfr] / (1 + fakeCat[bdfr])
                )
                self.log.debug("Replacing %s with bulge_flux_fraction.", bdfr)
            else:
                fakeCat['bulge_flux_fraction'] = 0.5
                self.log.debug("Asserting bulge_flux_fraction = 0.5")

        return fakeCat

    def _generateGSObjectsFromCatalog(self, exposure, fakeCat, galCheckVal, starCheckVal, trailCheckVal):
        """Process catalog to generate `galsim.GSObject` s.

        Parameters
        ----------
        exposure : `lsst.afw.image.exposure.exposure.ExposureF`
            The exposure into which the fake sources should be added
        fakeCat : `pandas.core.frame.DataFrame`
            The catalog of fake sources to be input
        galCheckVal : `str`, `bytes` or `int`
            The value that is set in the sourceType column to specify an object is a galaxy.
        starCheckVal : `str`, `bytes` or `int`
            The value that is set in the sourceType column to specify an object is a star.
        trailCheckVal : `str`, `bytes` or `int`
            The value that is set in the sourceType column to specify an object is a star

        Yields
        ------
        gsObjects : `generator`
            A generator of tuples of `lsst.geom.SpherePoint` and `galsim.GSObject`.
        """
        wcs = exposure.getWcs()
        photoCalib = exposure.getPhotoCalib()

        self.log.info("Making %d objects for insertion", len(fakeCat))

        for (index, row) in fakeCat.iterrows():
            ra = row['ra']
            dec = row['dec']
            skyCoord = SpherePoint(ra, dec, radians)
            xy = wcs.skyToPixel(skyCoord)

            try:
                flux = photoCalib.magnitudeToInstFlux(row['mag'], xy)
            except LogicError:
                continue

            sourceType = row[self.config.sourceType]
            if sourceType == galCheckVal:
                # GalSim convention: HLR = sqrt(a * b) = a * sqrt(b / a)
                bulge_gs_HLR = row['bulge_semimajor']*np.sqrt(row['bulge_axis_ratio'])
                bulge = galsim.Sersic(n=row['bulge_n'], half_light_radius=bulge_gs_HLR)
                bulge = bulge.shear(q=row['bulge_axis_ratio'], beta=((90 - row['bulge_pa'])*galsim.degrees))

                disk_gs_HLR = row['disk_semimajor']*np.sqrt(row['disk_axis_ratio'])
                disk = galsim.Sersic(n=row['disk_n'], half_light_radius=disk_gs_HLR)
                disk = disk.shear(q=row['disk_axis_ratio'], beta=((90 - row['disk_pa'])*galsim.degrees))

                gal = bulge*row['bulge_flux_fraction'] + disk*(1-row['bulge_flux_fraction'])
                gal = gal.withFlux(flux)

                yield skyCoord, gal
            elif sourceType == starCheckVal:
                star = galsim.DeltaFunction()
                star = star.withFlux(flux)
                yield skyCoord, star
            elif sourceType == trailCheckVal:
                length = row['trail_length']
                angle = row['trail_angle']

                # Make a 'thin' box to mimic a line surface brightness profile
                thickness = 1e-6  # Make the box much thinner than a pixel
                theta = galsim.Angle(angle*galsim.radians)
                trail = galsim.Box(length, thickness)
                trail = trail.rotate(theta)
                trail = trail.withFlux(flux*length)

                # Galsim objects are assumed to be in sky-coordinates. Since
                # we want the trail to appear as defined above in image-
                # coordinates, we must transform the trail here.
                mat = wcs.linearizePixelToSky(skyCoord, geom.arcseconds).getMatrix()
                trail = trail.transform(mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1])

                yield skyCoord, trail
            else:
                raise TypeError(f"Unknown sourceType {sourceType}")

    def _generateGSObjectsFromImages(self, exposure, fakeCat):
        """Process catalog to generate `galsim.GSObject` s.

        Parameters
        ----------
        exposure : `lsst.afw.image.exposure.exposure.ExposureF`
            The exposure into which the fake sources should be added
        fakeCat : `pandas.core.frame.DataFrame`
            The catalog of fake sources to be input

        Yields
        ------
        gsObjects : `generator`
            A generator of tuples of `lsst.geom.SpherePoint` and `galsim.GSObject`.
        """
        band = exposure.getFilter().bandLabel
        wcs = exposure.getWcs()
        photoCalib = exposure.getPhotoCalib()

        self.log.info("Processing %d fake images", len(fakeCat))

        for (index, row) in fakeCat.iterrows():
            ra = row['ra']
            dec = row['dec']
            skyCoord = SpherePoint(ra, dec, radians)
            xy = wcs.skyToPixel(skyCoord)

            try:
                flux = photoCalib.magnitudeToInstFlux(row['mag'], xy)
            except LogicError:
                continue

            imFile = row[band+"imFilename"]
            try:
                imFile = imFile.decode("utf-8")
            except AttributeError:
                pass
            imFile = imFile.strip()
            im = galsim.fits.read(imFile, read_header=True)

            if self.config.fits_alignment == "wcs":
                # galsim.fits.read will always attach a WCS to its output. If it
                # can't find a WCS in the FITS header, then it defaults to
                # scale = 1.0 arcsec / pix.  So if that's the scale, then we
                # need to check if it was explicitly set or if it's just the
                # default.  If it's just the default then we should raise an
                # exception.
                if _isWCSGalsimDefault(im.wcs, im.header):
                    raise RuntimeError(
                        f"Cannot find WCS in input FITS file {imFile}"
                    )
            elif self.config.fits_alignment == "pixel":
                # Here we need to set im.wcs to the local WCS at the target
                # position.
                linWcs = wcs.linearizePixelToSky(skyCoord, geom.arcseconds)
                mat = linWcs.getMatrix()
                im.wcs = galsim.JacobianWCS(
                    mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1]
                )
            else:
                raise ValueError(
                    f"Unknown fits_alignment type {self.config.fits_alignment}"
                )

            obj = galsim.InterpolatedImage(im, calculate_stepk=False)
            obj = obj.withFlux(flux)
            yield skyCoord, obj

    def processImagesForInsertion(self, fakeCat, wcs, psf, photoCalib, band, pixelScale):
        """Process images from files into the format needed for insertion.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to be input
        wcs : `lsst.afw.geom.skyWcs.skyWcs.SkyWc`
                    WCS to use to add fake sources
        psf : `lsst.meas.algorithms.coaddPsf.coaddPsf.CoaddPsf` or
              `lsst.meas.extensions.psfex.psfexPsf.PsfexPsf`
                    The PSF information to use to make the PSF images
        photoCalib : `lsst.afw.image.photoCalib.PhotoCalib`
                    Photometric calibration to be used to calibrate the fake sources
        band : `str`
                    The filter band that the observation was taken in.
        pixelScale : `float`
                    The pixel scale of the image the sources are to be added to.

        Returns
        -------
        galImages : `list`
                    A list of tuples of `lsst.afw.image.exposure.exposure.ExposureF` and
                    `lsst.geom.Point2D` of their locations.
                    For sources labelled as galaxy.
        starImages : `list`
                    A list of tuples of `lsst.afw.image.exposure.exposure.ExposureF` and
                    `lsst.geom.Point2D` of their locations.
                    For sources labelled as star.

        Notes
        -----
        The input fakes catalog needs to contain the absolute path to the image in the
        band that is being used to add images to. It also needs to have the R.A. and
        declination of the fake source in radians and the sourceType of the object.
        """
        galImages = []
        starImages = []

        self.log.info("Processing %d fake images", len(fakeCat))

        for (imFile, sourceType, mag, x, y) in zip(fakeCat[band + "imFilename"].array,
                                                   fakeCat["sourceType"].array,
                                                   fakeCat['mag'].array,
                                                   fakeCat["x"].array, fakeCat["y"].array):

            im = afwImage.ImageF.readFits(imFile)

            xy = geom.Point2D(x, y)

            # We put these two PSF calculations within this same try block so that we catch cases
            # where the object's position is outside of the image.
            try:
                correctedFlux = psf.computeApertureFlux(self.config.calibFluxRadius, xy)
                psfKernel = psf.computeKernelImage(xy).getArray()
                psfKernel /= correctedFlux

            except InvalidParameterError:
                self.log.info("%s at %0.4f, %0.4f outside of image", sourceType, x, y)
                continue

            psfIm = galsim.InterpolatedImage(galsim.Image(psfKernel), scale=pixelScale)
            galsimIm = galsim.InterpolatedImage(galsim.Image(im.array), scale=pixelScale)
            convIm = galsim.Convolve([galsimIm, psfIm])

            try:
                outIm = convIm.drawImage(scale=pixelScale, method="real_space").array
            except (galsim.errors.GalSimFFTSizeError, MemoryError):
                continue

            imSum = np.sum(outIm)
            divIm = outIm/imSum

            try:
                flux = photoCalib.magnitudeToInstFlux(mag, xy)
            except LogicError:
                flux = 0

            imWithFlux = flux*divIm

            if sourceType == b"galaxy":
                galImages.append((afwImage.ImageF(imWithFlux), xy))
            if sourceType == b"star":
                starImages.append((afwImage.ImageF(imWithFlux), xy))

        return galImages, starImages

    def addPixCoords(self, fakeCat, image):

        """Add pixel coordinates to the catalog of fakes.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to be input
        image : `lsst.afw.image.exposure.exposure.ExposureF`
                    The image into which the fake sources should be added

        Returns
        -------
        fakeCat : `pandas.core.frame.DataFrame`
        """
        wcs = image.getWcs()
        ras = fakeCat['ra'].values
        decs = fakeCat['dec'].values
        xs, ys = wcs.skyToPixelArray(ras, decs)
        fakeCat["x"] = xs
        fakeCat["y"] = ys

        return fakeCat

    def trimFakeCat(self, fakeCat, image):
        """Trim the fake cat to the size of the input image plus trimBuffer padding.

        `fakeCat` must be processed with addPixCoords before using this method.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to be input
        image : `lsst.afw.image.exposure.exposure.ExposureF`
                    The image into which the fake sources should be added

        Returns
        -------
        fakeCat : `pandas.core.frame.DataFrame`
                    The original fakeCat trimmed to the area of the image
        """
        wideBbox = Box2D(image.getBBox()).dilatedBy(self.config.trimBuffer)

        # prefilter in ra/dec to avoid cases where the wcs incorrectly maps
        # input fakes which are really off the chip onto it.
        ras = fakeCat[self.config.ra_col].values * u.rad
        decs = fakeCat[self.config.dec_col].values * u.rad

        isContainedRaDec = image.containsSkyCoords(ras, decs, padding=self.config.trimBuffer)

        # also filter on the image BBox in pixel space
        xs = fakeCat["x"].values
        ys = fakeCat["y"].values

        isContainedXy = xs >= wideBbox.minX
        isContainedXy &= xs <= wideBbox.maxX
        isContainedXy &= ys >= wideBbox.minY
        isContainedXy &= ys <= wideBbox.maxY

        return fakeCat[isContainedRaDec & isContainedXy]

    def mkFakeGalsimGalaxies(self, fakeCat, band, photoCalib, pixelScale, psf, image):
        """Make images of fake galaxies using GalSim.

        Parameters
        ----------
        band : `str`
        pixelScale : `float`
        psf : `lsst.meas.extensions.psfex.psfexPsf.PsfexPsf`
                    The PSF information to use to make the PSF images
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to be input
        photoCalib : `lsst.afw.image.photoCalib.PhotoCalib`
                    Photometric calibration to be used to calibrate the fake sources

        Yields
        ------
        galImages : `generator`
                    A generator of tuples of `lsst.afw.image.exposure.exposure.ExposureF` and
                    `lsst.geom.Point2D` of their locations.

        Notes
        -----

        Fake galaxies are made by combining two sersic profiles, one for the bulge and one for the disk. Each
        component has an individual sersic index (n), a, b and position angle (PA). The combined profile is
        then convolved with the PSF at the specified x, y position on the image.

        The names of the columns in the ``fakeCat`` are configurable and are the column names from the
        University of Washington simulations database as default. For more information see the doc strings
        attached to the config options.

        See mkFakeStars doc string for an explanation of calibration to instrumental flux.
        """

        self.log.info("Making %d fake galaxy images", len(fakeCat))

        for (index, row) in fakeCat.iterrows():
            xy = geom.Point2D(row["x"], row["y"])

            # We put these two PSF calculations within this same try block so that we catch cases
            # where the object's position is outside of the image.
            try:
                correctedFlux = psf.computeApertureFlux(self.config.calibFluxRadius, xy)
                psfKernel = psf.computeKernelImage(xy).getArray()
                psfKernel /= correctedFlux

            except InvalidParameterError:
                self.log.info("Galaxy at %0.4f, %0.4f outside of image", row["x"], row["y"])
                continue

            try:
                flux = photoCalib.magnitudeToInstFlux(row['mag'], xy)
            except LogicError:
                flux = 0

            # GalSim convention: HLR = sqrt(a * b) = a * sqrt(b / a)
            bulge_gs_HLR = row['bulge_semimajor']*np.sqrt(row['bulge_axis_ratio'])
            bulge = galsim.Sersic(n=row['bulge_n'], half_light_radius=bulge_gs_HLR)
            bulge = bulge.shear(q=row['bulge_axis_ratio'], beta=((90 - row['bulge_pa'])*galsim.degrees))

            disk_gs_HLR = row['disk_semimajor']*np.sqrt(row['disk_axis_ratio'])
            disk = galsim.Sersic(n=row['disk_n'], half_light_radius=disk_gs_HLR)
            disk = disk.shear(q=row['disk_axis_ratio'], beta=((90 - row['disk_pa'])*galsim.degrees))

            gal = bulge*row['bulge_flux_fraction'] + disk*(1-row['bulge_flux_fraction'])
            gal = gal.withFlux(flux)

            psfIm = galsim.InterpolatedImage(galsim.Image(psfKernel), scale=pixelScale)
            gal = galsim.Convolve([gal, psfIm])
            try:
                galIm = gal.drawImage(scale=pixelScale, method="real_space").array
            except (galsim.errors.GalSimFFTSizeError, MemoryError):
                continue

            yield (afwImage.ImageF(galIm), xy)

    def mkFakeStars(self, fakeCat, band, photoCalib, psf, image):

        """Make fake stars based off the properties in the fakeCat.

        Parameters
        ----------
        band : `str`
        psf : `lsst.meas.extensions.psfex.psfexPsf.PsfexPsf`
                    The PSF information to use to make the PSF images
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to be input
        image : `lsst.afw.image.exposure.exposure.ExposureF`
                    The image into which the fake sources should be added
        photoCalib : `lsst.afw.image.photoCalib.PhotoCalib`
                    Photometric calibration to be used to calibrate the fake sources

        Yields
        ------
        starImages : `generator`
                    A generator of tuples of `lsst.afw.image.ImageF` of fake stars and
                    `lsst.geom.Point2D` of their locations.

        Notes
        -----
        To take a given magnitude and translate to the number of counts in the image
        we use photoCalib.magnitudeToInstFlux, which returns the instrumental flux for the
        given calibration radius used in the photometric calibration step.
        Thus `calibFluxRadius` should be set to this same radius so that we can normalize
        the PSF model to the correct instrumental flux within calibFluxRadius.
        """

        self.log.info("Making %d fake star images", len(fakeCat))

        for (index, row) in fakeCat.iterrows():
            xy = geom.Point2D(row["x"], row["y"])

            # We put these two PSF calculations within this same try block so that we catch cases
            # where the object's position is outside of the image.
            try:
                correctedFlux = psf.computeApertureFlux(self.config.calibFluxRadius, xy)
                starIm = psf.computeImage(xy)
                starIm /= correctedFlux

            except InvalidParameterError:
                self.log.info("Star at %0.4f, %0.4f outside of image", row["x"], row["y"])
                continue

            try:
                flux = photoCalib.magnitudeToInstFlux(row['mag'], xy)
            except LogicError:
                flux = 0

            starIm *= flux
            yield ((starIm.convertF(), xy))

    def cleanCat(self, fakeCat, starCheckVal):
        """Remove rows from the fakes catalog which have HLR = 0 for either the buldge or disk component,
           also remove galaxies that have Sersic index outside the galsim min and max
           allowed (0.3 <= n <= 6.2).

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to be input
        starCheckVal : `str`, `bytes` or `int`
                    The value that is set in the sourceType column to specifiy an object is a star.

        Returns
        -------
        fakeCat : `pandas.core.frame.DataFrame`
                    The input catalog of fake sources but with the bad objects removed
        """

        rowsToKeep = (((fakeCat['bulge_semimajor'] != 0.0) & (fakeCat['disk_semimajor'] != 0.0))
                      | (fakeCat[self.config.sourceType] == starCheckVal))
        numRowsNotUsed = len(fakeCat) - len(np.where(rowsToKeep)[0])
        self.log.info("Removing %d rows with HLR = 0 for either the bulge or disk", numRowsNotUsed)
        fakeCat = fakeCat[rowsToKeep]

        minN = galsim.Sersic._minimum_n
        maxN = galsim.Sersic._maximum_n
        rowsWithGoodSersic = (((fakeCat['bulge_n'] >= minN) & (fakeCat['bulge_n'] <= maxN)
                              & (fakeCat['disk_n'] >= minN) & (fakeCat['disk_n'] <= maxN))
                              | (fakeCat[self.config.sourceType] == starCheckVal))
        numRowsNotUsed = len(fakeCat) - len(np.where(rowsWithGoodSersic)[0])
        self.log.info("Removing %d rows of galaxies with nBulge or nDisk outside of %0.2f <= n <= %0.2f",
                      numRowsNotUsed, minN, maxN)
        fakeCat = fakeCat[rowsWithGoodSersic]

        if self.config.doSubSelectSources:
            numRowsNotUsed = len(fakeCat) - len(fakeCat['select'])
            self.log.info("Removing %d rows which were not designated as template sources", numRowsNotUsed)
            fakeCat = fakeCat[fakeCat['select']]

        return fakeCat

    def addFakeSources(self, image, fakeImages, sourceType):
        """Add the fake sources to the given image

        Parameters
        ----------
        image : `lsst.afw.image.exposure.exposure.ExposureF`
                    The image into which the fake sources should be added
        fakeImages : `typing.Iterator` [`tuple` ['lsst.afw.image.ImageF`, `lsst.geom.Point2d`]]
                    An iterator of tuples that contains (or generates) images of fake sources,
                    and the locations they are to be inserted at.
        sourceType : `str`
                    The type (star/galaxy) of fake sources input

        Returns
        -------
        image : `lsst.afw.image.exposure.exposure.ExposureF`

        Notes
        -----
        Uses the x, y information in the ``fakeCat`` to position an image of the fake interpolated onto the
        pixel grid of the image. Sets the ``FAKE`` mask plane for the pixels added with the fake source.
        """

        imageBBox = image.getBBox()
        imageMI = image.maskedImage

        for (fakeImage, xy) in fakeImages:
            X0 = xy.getX() - fakeImage.getWidth()/2 + 0.5
            Y0 = xy.getY() - fakeImage.getHeight()/2 + 0.5
            self.log.debug("Adding fake source at %d, %d", xy.getX(), xy.getY())
            if sourceType == "galaxy":
                interpFakeImage = afwMath.offsetImage(fakeImage, X0, Y0, "lanczos3")
            else:
                interpFakeImage = fakeImage

            interpFakeImBBox = interpFakeImage.getBBox()
            interpFakeImBBox.clip(imageBBox)

            if interpFakeImBBox.getArea() > 0:
                imageMIView = imageMI[interpFakeImBBox]
                clippedFakeImage = interpFakeImage[interpFakeImBBox]
                clippedFakeImageMI = afwImage.MaskedImageF(clippedFakeImage)
                clippedFakeImageMI.mask.set(self.bitmask)
                imageMIView += clippedFakeImageMI

        return image
