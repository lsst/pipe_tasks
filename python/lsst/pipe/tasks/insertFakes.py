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
from lsst.sphgeom import ConvexPolygon

__all__ = ["InsertFakesConfig", "InsertFakesTask"]


class InsertFakesConnections(PipelineTaskConnections, defaultTemplates={"CoaddName": "deep"},
                             dimensions=("tract", "patch", "abstract_filter", "skymap")):

    image = cT.Input(
        doc="Image into which fakes are to be added.",
        name="{CoaddName}Coadd",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "abstract_filter", "skymap")
    )

    fakeCat = cT.Input(
        doc="Catalog of fake sources to draw inputs from.",
        name="{CoaddName}Coadd_fakeSourceCat",
        storageClass="Parquet",
        dimensions=("tract", "skymap")
    )

    imageWithFakes = cT.Output(
        doc="Image with fake sources added.",
        name="fakes_{CoaddName}Coadd",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "abstract_filter", "skymap")
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
        doc="Radius for the calib flux (in pixels).",
        dtype=float,
        default=12.0,
    )

    coaddName = pexConfig.Field(
        doc="The name of the type of coadd used",
        dtype=str,
        default="deep",
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
        that are 0.
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
        if self.config.doCleanCat:
            fakeCat = self.cleanCat(fakeCat)
        fakeCat = self.trimFakeCat(fakeCat, image, wcs)

        band = image.getFilter().getName()
        pixelScale = wcs.getPixelScale().asArcseconds()
        psf = image.getPsf()

        galaxies = (fakeCat[self.config.sourceType] == "galaxy")
        galImages = self.mkFakeGalsimGalaxies(fakeCat[galaxies], band, photoCalib, pixelScale, psf, image)
        image = self.addFakeSources(image, galImages, "galaxy")

        stars = (fakeCat[self.config.sourceType] == "star")
        starImages = self.mkFakeStars(fakeCat[stars], band, photoCalib, psf, image)
        image = self.addFakeSources(image, starImages, "star")
        resultStruct = pipeBase.Struct(imageWithFakes=image)

        return resultStruct

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
        corners = bbox.getCorners()

        skyCorners = wcs.pixelToSky(corners)
        region = ConvexPolygon([s.getVector() for s in skyCorners])

        def trim(row):
            coord = SpherePoint(row[self.config.raColName], row[self.config.decColName], radians)
            return region.contains(coord.getVector())

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
        """

        self.log.info("Making %d fake galaxy images" % len(fakeCat))

        for (index, row) in fakeCat.iterrows():
            xy = geom.Point2D(row["x"], row["y"])

            try:
                # Due to the different radii used for calibration and measurement a correction factor is
                # needed to prevent there being an offset in the final processed output.
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
        """

        self.log.info("Making %d fake star images" % len(fakeCat))

        for (index, row) in fakeCat.iterrows():
            xy = geom.Point2D(row["x"], row["y"])

            try:
                # Due to the different radii used for calibration and measurement a correction factor is
                # needed to prevent there being an offset in the final processed output.
                correctedFlux = psf.computeApertureFlux(self.config.calibFluxRadius, xy)
                starIm = psf.computeImage(xy)
                starIm /= correctedFlux

            except InvalidParameterError:
                self.log.info("Star at %0.4f, %0.4f outside of image" % (row["x"], row["y"]))
                continue

            try:
                flux = photoCalib.magnitudeToInstFlux(row[band + "magVar"], xy)
            except LogicError:
                flux = 0

            starIm *= flux
            yield ((starIm.convertF(), xy))

    def cleanCat(self, fakeCat):
        """Remove rows from the fakes catalog which have HLR = 0 for either the buldge or disk component

        Parameters
        ----------
        fakeCat : `pandas.core.frame.DataFrame`
                    The catalog of fake sources to be input

        Returns
        -------
        fakeCat : `pandas.core.frame.DataFrame`
                    The input catalog of fake sources but with the bad objects removed
        """

        goodRows = ((fakeCat[self.config.bulgeHLR] != 0.0) & (fakeCat[self.config.diskHLR] != 0.0))

        badRows = len(fakeCat) - len(goodRows)
        self.log.info("Removing %d rows with HLR = 0 for either the bulge or disk" % badRows)

        return fakeCat[goodRows]

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
