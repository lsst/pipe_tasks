# This file is part of pipe tasks
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
Insert fakes into deepCoadds
"""
import galsim
from astropy.table import Table
import numpy as np

import lsst.geom as geom
import lsst.afw.image as afwImage
import lsst.afw.math as afwMath
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase

from lsst.pipe.base import CmdLineTask, PipelineTask, PipelineTaskConfig, PipelineTaskConnections
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.exceptions import LogicError, InvalidParameterError
from lsst.coadd.utils.coaddDataIdContainer import ExistingCoaddDataIdContainer
from lsst.geom import SpherePoint, radians, Box2D

__all__ = ["InsertFakesConfig", "InsertFakesTask"]


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

    Notes
    -----
    The default column names are those from the University of Washington sims database.
    """

    raColName = pexConfig.Field(
        doc="RA column name used in the fake source catalog.",
        dtype=str,
        default="raJ2000",
    )

    decColName = pexConfig.Field(
        doc="Dec. column name used in the fake source catalog.",
        dtype=str,
        default="decJ2000",
    )

    doCleanCat = pexConfig.Field(
        doc="If true removes bad sources from the catalog.",
        dtype=bool,
        default=True,
    )

    diskHLR = pexConfig.Field(
        doc="Column name for the disk half light radius used in the fake source catalog.",
        dtype=str,
        default="DiskHalfLightRadius",
    )

    bulgeHLR = pexConfig.Field(
        doc="Column name for the bulge half light radius used in the fake source catalog.",
        dtype=str,
        default="BulgeHalfLightRadius",
    )

    magVar = pexConfig.Field(
        doc="The column name for the magnitude calculated taking variability into account. In the format "
            "``filter name``magVar, e.g. imagVar for the magnitude in the i band.",
        dtype=str,
        default="%smagVar",
    )

    nDisk = pexConfig.Field(
        doc="The column name for the sersic index of the disk component used in the fake source catalog.",
        dtype=str,
        default="disk_n",
    )

    nBulge = pexConfig.Field(
        doc="The column name for the sersic index of the bulge component used in the fake source catalog.",
        dtype=str,
        default="bulge_n",
    )

    aDisk = pexConfig.Field(
        doc="The column name for the semi major axis length of the disk component used in the fake source"
            "catalog.",
        dtype=str,
        default="a_d",
    )

    aBulge = pexConfig.Field(
        doc="The column name for the semi major axis length of the bulge component.",
        dtype=str,
        default="a_b",
    )

    bDisk = pexConfig.Field(
        doc="The column name for the semi minor axis length of the disk component.",
        dtype=str,
        default="b_d",
    )

    bBulge = pexConfig.Field(
        doc="The column name for the semi minor axis length of the bulge component used in the fake source "
            "catalog.",
        dtype=str,
        default="b_b",
    )

    paDisk = pexConfig.Field(
        doc="The column name for the PA of the disk component used in the fake source catalog.",
        dtype=str,
        default="pa_disk",
    )

    paBulge = pexConfig.Field(
        doc="The column name for the PA of the bulge component used in the fake source catalog.",
        dtype=str,
        default="pa_bulge",
    )

    sourceType = pexConfig.Field(
        doc="The column name for the source type used in the fake source catalog.",
        dtype=str,
        default="sourceType",
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

    sourceSelectionColName = pexConfig.Field(
        doc="The name of the column in the input fakes catalogue to be used to determine which sources to"
            "add, default is none and when this is used all sources are added.",
        dtype=str,
        default="templateSource"
    )

    insertImages = pexConfig.Field(
        doc="Insert images directly? True or False.",
        dtype=bool,
        default=False,
    )

    doProcessAllDataIds = pexConfig.Field(
        doc="If True, all input data IDs will be processed, even those containing no fake sources.",
        dtype=bool,
        default=False,
    )


class InsertFakesTask(PipelineTask, CmdLineTask):
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

    def runDataRef(self, dataRef):
        """Read in/write out the required data products and add fake sources to the deepCoadd.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
            Data reference defining the image to have fakes added to it
            Used to access the following data products:
                deepCoadd
        """

        infoStr = "Adding fakes to: tract: %d, patch: %s, filter: %s" % (dataRef.dataId["tract"],
                                                                         dataRef.dataId["patch"],
                                                                         dataRef.dataId["filter"])
        self.log.info(infoStr)

        # To do: should it warn when asked to insert variable sources into the coadd

        if self.config.fakeType == "static":
            fakeCat = dataRef.get("deepCoadd_fakeSourceCat").toDataFrame()
            # To do: DM-16254, the read and write of the fake catalogs will be changed once the new pipeline
            # task structure for ref cats is in place.
            self.fakeSourceCatType = "deepCoadd_fakeSourceCat"
        else:
            fakeCat = Table.read(self.config.fakeType).to_pandas()

        coadd = dataRef.get("deepCoadd")
        wcs = coadd.getWcs()
        photoCalib = coadd.getPhotoCalib()

        imageWithFakes = self.run(fakeCat, coadd, wcs, photoCalib)

        dataRef.put(imageWithFakes.imageWithFakes, "fakes_deepCoadd")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        inputs["wcs"] = inputs["image"].getWcs()
        inputs["photoCalib"] = inputs["image"].getPhotoCalib()

        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    @classmethod
    def _makeArgumentParser(cls):
        parser = pipeBase.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="deepCoadd",
                               help="data IDs for the deepCoadd, e.g. --id tract=12345 patch=1,2 filter=r",
                               ContainerClass=ExistingCoaddDataIdContainer)
        return parser

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

        image.mask.addMaskPlane("FAKE")
        self.bitmask = image.mask.getPlaneBitMask("FAKE")
        self.log.info("Adding mask plane with bitmask %d" % self.bitmask)

        fakeCat = self.addPixCoords(fakeCat, wcs)
        fakeCat = self.trimFakeCat(fakeCat, image, wcs)
        band = image.getFilterLabel().bandLabel
        psf = image.getPsf()
        pixelScale = wcs.getPixelScale().asArcseconds()

        if len(fakeCat) > 0:
            if isinstance(fakeCat[self.config.sourceType].iloc[0], str):
                galCheckVal = "galaxy"
                starCheckVal = "star"
            elif isinstance(fakeCat[self.config.sourceType].iloc[0], bytes):
                galCheckVal = b"galaxy"
                starCheckVal = b"star"
            elif isinstance(fakeCat[self.config.sourceType].iloc[0], (int, float)):
                galCheckVal = 1
                starCheckVal = 0
            else:
                raise TypeError("sourceType column does not have required type, should be str, bytes or int")

            if not self.config.insertImages:
                if self.config.doCleanCat:
                    fakeCat = self.cleanCat(fakeCat, starCheckVal)

                galaxies = (fakeCat[self.config.sourceType] == galCheckVal)
                galImages = self.mkFakeGalsimGalaxies(fakeCat[galaxies], band, photoCalib, pixelScale, psf,
                                                      image)

                stars = (fakeCat[self.config.sourceType] == starCheckVal)
                starImages = self.mkFakeStars(fakeCat[stars], band, photoCalib, psf, image)
            else:
                galImages, starImages = self.processImagesForInsertion(fakeCat, wcs, psf, photoCalib, band,
                                                                       pixelScale)

            image = self.addFakeSources(image, galImages, "galaxy")
            image = self.addFakeSources(image, starImages, "star")
        elif len(fakeCat) == 0 and self.config.doProcessAllDataIds:
            self.log.warn("No fakes found for this dataRef; processing anyway.")
        else:
            raise RuntimeError("No fakes found for this dataRef.")

        resultStruct = pipeBase.Struct(imageWithFakes=image)

        return resultStruct

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

        self.log.info("Processing %d fake images" % len(fakeCat))

        for (imFile, sourceType, mag, x, y) in zip(fakeCat[band + "imFilename"].array,
                                                   fakeCat["sourceType"].array,
                                                   fakeCat[self.config.magVar % band].array,
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
                self.log.info("%s at %0.4f, %0.4f outside of image" % (sourceType, x, y))
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

    def addPixCoords(self, fakeCat, wcs):

        """Add pixel coordinates to the catalog of fakes.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to be input
        wcs : `lsst.afw.geom.SkyWcs`
                    WCS to use to add fake sources

        Returns
        -------
        fakeCat : `pandas.core.frame.DataFrame`

        Notes
        -----
        The default option is to use the WCS information from the image. If the ``useUpdatedCalibs`` config
        option is set then it will use the updated WCS from jointCal.
        """

        ras = fakeCat[self.config.raColName].values
        decs = fakeCat[self.config.decColName].values
        skyCoords = [SpherePoint(ra, dec, radians) for (ra, dec) in zip(ras, decs)]
        pixCoords = wcs.skyToPixel(skyCoords)
        xs = [coord.getX() for coord in pixCoords]
        ys = [coord.getY() for coord in pixCoords]
        fakeCat["x"] = xs
        fakeCat["y"] = ys

        return fakeCat

    def trimFakeCat(self, fakeCat, image, wcs):
        """Trim the fake cat to about the size of the input image.

        `fakeCat` must be processed with addPixCoords before using this method.

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to be input
        image : `lsst.afw.image.exposure.exposure.ExposureF`
                    The image into which the fake sources should be added
        wcs : `lsst.afw.geom.SkyWcs`
                    WCS to use to add fake sources

        Returns
        -------
        fakeCat : `pandas.core.frame.DataFrame`
                    The original fakeCat trimmed to the area of the image
        """

        bbox = Box2D(image.getBBox())

        def trim(row):
            return bbox.contains(row["x"], row["y"])

        return fakeCat[fakeCat.apply(trim, axis=1)]

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
        -------
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

        self.log.info("Making %d fake galaxy images" % len(fakeCat))

        for (index, row) in fakeCat.iterrows():
            xy = geom.Point2D(row["x"], row["y"])

            # We put these two PSF calculations within this same try block so that we catch cases
            # where the object's position is outside of the image.
            try:
                correctedFlux = psf.computeApertureFlux(self.config.calibFluxRadius, xy)
                psfKernel = psf.computeKernelImage(xy).getArray()
                psfKernel /= correctedFlux

            except InvalidParameterError:
                self.log.info("Galaxy at %0.4f, %0.4f outside of image" % (row["x"], row["y"]))
                continue

            try:
                flux = photoCalib.magnitudeToInstFlux(row[self.config.magVar % band], xy)
            except LogicError:
                flux = 0

            bulge = galsim.Sersic(row[self.config.nBulge], half_light_radius=row[self.config.bulgeHLR])
            axisRatioBulge = row[self.config.bBulge]/row[self.config.aBulge]
            bulge = bulge.shear(q=axisRatioBulge, beta=((90 - row[self.config.paBulge])*galsim.degrees))

            disk = galsim.Sersic(row[self.config.nDisk], half_light_radius=row[self.config.diskHLR])
            axisRatioDisk = row[self.config.bDisk]/row[self.config.aDisk]
            disk = disk.shear(q=axisRatioDisk, beta=((90 - row[self.config.paDisk])*galsim.degrees))

            gal = disk + bulge
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
        -------
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

        self.log.info("Making %d fake star images" % len(fakeCat))

        for (index, row) in fakeCat.iterrows():
            xy = geom.Point2D(row["x"], row["y"])

            # We put these two PSF calculations within this same try block so that we catch cases
            # where the object's position is outside of the image.
            try:
                correctedFlux = psf.computeApertureFlux(self.config.calibFluxRadius, xy)
                starIm = psf.computeImage(xy)
                starIm /= correctedFlux

            except InvalidParameterError:
                self.log.info("Star at %0.4f, %0.4f outside of image" % (row["x"], row["y"]))
                continue

            try:
                flux = photoCalib.magnitudeToInstFlux(row[self.config.magVar % band], xy)
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

        Notes
        -----
        If the config option sourceSelectionColName is set then only objects with this column set to True
        will be added.
        """

        rowsToKeep = (((fakeCat[self.config.bulgeHLR] != 0.0) & (fakeCat[self.config.diskHLR] != 0.0))
                      | (fakeCat[self.config.sourceType] == starCheckVal))
        numRowsNotUsed = len(fakeCat) - len(np.where(rowsToKeep)[0])
        self.log.info("Removing %d rows with HLR = 0 for either the bulge or disk" % numRowsNotUsed)
        fakeCat = fakeCat[rowsToKeep]

        minN = galsim.Sersic._minimum_n
        maxN = galsim.Sersic._maximum_n
        rowsWithGoodSersic = (((fakeCat[self.config.nBulge] >= minN) & (fakeCat[self.config.nBulge] <= maxN)
                              & (fakeCat[self.config.nDisk] >= minN) & (fakeCat[self.config.nDisk] <= maxN))
                              | (fakeCat[self.config.sourceType] == starCheckVal))
        numRowsNotUsed = len(fakeCat) - len(np.where(rowsWithGoodSersic)[0])
        self.log.info("Removing %d rows of galaxies with nBulge or nDisk outside of %0.2f <= n <= %0.2f" %
                      (numRowsNotUsed, minN, maxN))
        fakeCat = fakeCat[rowsWithGoodSersic]

        if self.config.doSubSelectSources:
            try:
                rowsSelected = (fakeCat[self.config.sourceSelectionColName])
            except KeyError:
                raise KeyError("Given column, %s, for source selection not found." %
                               self.config.sourceSelectionColName)
            numRowsNotUsed = len(fakeCat) - len(rowsSelected)
            self.log.info("Removing %d rows which were not designated as template sources" % numRowsNotUsed)
            fakeCat = fakeCat[rowsSelected]

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
            self.log.debug("Adding fake source at %d, %d" % (xy.getX(), xy.getY()))
            if sourceType == "galaxy":
                interpFakeImage = afwMath.offsetImage(fakeImage, X0, Y0, "lanczos3")
                interpFakeImBBox = interpFakeImage.getBBox()
            else:
                interpFakeImage = fakeImage
                interpFakeImBBox = fakeImage.getBBox()

            interpFakeImBBox.clip(imageBBox)
            imageMIView = imageMI.Factory(imageMI, interpFakeImBBox)

            if interpFakeImBBox.getArea() > 0:
                clippedFakeImage = interpFakeImage.Factory(interpFakeImage, interpFakeImBBox)
                clippedFakeImageMI = afwImage.MaskedImageF(clippedFakeImage)
                clippedFakeImageMI.mask.set(self.bitmask)
                imageMIView += clippedFakeImageMI

        return image

    def _getMetadataName(self):
        """Disable metadata writing"""
        return None
