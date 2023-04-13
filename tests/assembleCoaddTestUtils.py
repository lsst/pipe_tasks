# This file is part of pipe_tasks.
#
# LSST Data Management System
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
# See COPYRIGHT file at the top of the source tree.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#
"""Set up simulated test data and simplified APIs for AssembleCoaddTask
and its derived classes.

This is not intended to test accessing data with the Butler and instead uses
mock Butler data references to pass in the simulated data.
"""
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, Angle
import numpy as np

from lsst.afw.cameraGeom.testUtils import DetectorWrapper
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.geom as geom
from lsst.geom import arcseconds, degrees
from lsst.meas.algorithms.testUtils import plantSources
from lsst.obs.base import MakeRawVisitInfoViaObsInfo
import lsst.pipe.base as pipeBase
from lsst.pipe.tasks.coaddInputRecorder import CoaddInputRecorderTask, CoaddInputRecorderConfig

from astro_metadata_translator import makeObservationInfo

__all__ = ["MockWarpReference", "makeMockSkyInfo", "MockCoaddTestData"]


class MockWarpReference(pipeBase.InMemoryDatasetHandle):
    """Very simple object that looks like a Gen 3 data reference to a warped
    exposure.

    Parameters
    ----------
    exposure : `lsst.afw.image.Exposure`
        The exposure to be retrieved by the data reference.
    """
    def get(self, *, component=None, parameters=None):
        """Retrieve the specified dataset using the API of the Gen 3 Butler.

        Parameters
        ----------
        component : `str`, optional
            If supplied, return the named metadata of the exposure.
        parameters : `dict`, optional
            If supplied, use the parameters to modify the exposure,
            typically by taking a subset.

        Returns
        -------
        `lsst.afw.image.Exposure` or `lsst.afw.image.VisitInfo`
        or `lsst.meas.algorithms.SingleGaussianPsf`
            Either the exposure or its metadata, depending on the component
            requested.
        """
        exp = super().get(component=component, parameters=parameters)
        if isinstance(exp, afwImage.ExposureF):
            exp = exp.clone()
        return exp


def makeMockSkyInfo(bbox, wcs, patch):
    """Construct a `Struct` containing the geometry of the patch to be coadded.

    Parameters
    ----------
    bbox : `lsst.geom.Box`
        Bounding box of the patch to be coadded.
    wcs : `lsst.afw.geom.SkyWcs`
        Coordinate system definition (wcs) for the exposure.

    Returns
    -------
    skyInfo : `lsst.pipe.base.Struct`
        Patch geometry information.
    """
    def getIndex():
        return patch
    patchInfo = pipeBase.Struct(getIndex=getIndex)
    skyInfo = pipeBase.Struct(bbox=bbox, wcs=wcs, patchInfo=patchInfo)
    return skyInfo


class MockCoaddTestData:
    """Generate repeatable simulated exposures with consistent metadata that
    are realistic enough to test the image coaddition algorithms.

    Notes
    -----
    The simple GaussianPsf used by lsst.meas.algorithms.testUtils.plantSources
    will always return an average position of (0, 0).
    The bounding box of the exposures MUST include (0, 0), or else the PSF will
    not be valid and `AssembleCoaddTask` will fail with the error
    'Could not find a valid average position for CoaddPsf'.

    Parameters
    ----------
    shape : `lsst.geom.Extent2I`, optional
        Size of the bounding box of the exposures to be simulated, in pixels.
    offset : `lsst.geom.Point2I`, optional
        Pixel coordinate of the lower left corner of the bounding box.
    backgroundLevel : `float`, optional
        Background value added to all pixels in the simulated images.
    seed : `int`, optional
        Seed value to initialize the random number generator.
    nSrc : `int`, optional
        Number of sources to simulate.
    fluxRange : `float`, optional
        Range in flux amplitude of the simulated sources.
    noiseLevel : `float`, optional
        Standard deviation of the noise to add to each pixel.
    sourceSigma : `float`, optional
        Average amplitude of the simulated sources,
        relative to ``noiseLevel``
    minPsfSize : `float`, optional
        The smallest PSF width (sigma) to use, in pixels.
    maxPsfSize : `float`, optional
        The largest PSF width (sigma) to use, in pixels.
    pixelScale : `lsst.geom.Angle`, optional
        The plate scale of the simulated images.
    ra : `lsst.geom.Angle`, optional
        Right Ascension of the boresight of the camera for the observation.
    dec : `lsst.geom.Angle`, optional
        Declination of the boresight of the camera for the observation.
    ccd : `int`, optional
        CCD number to put in the metadata of the exposure.
    patch : `int`, optional
        Unique identifier for a subdivision of a tract.
    tract : `int`, optional
        Unique identifier for a tract of a skyMap.

    Raises
    ------
    ValueError
        If the bounding box does not contain the pixel coordinate (0, 0).
        This is due to `GaussianPsf` that is used by `lsst.meas.algorithms.testUtils.plantSources`
        lacking the option to specify the pixel origin.
    """
    rotAngle = 0.*degrees
    "Rotation of the pixel grid on the sky, East from North (`lsst.geom.Angle`)."
    filterLabel = None
    """The filter definition, usually set in the current instruments' obs package.
    For these tests, a simple filter is defined without using an obs package (`lsst.afw.image.FilterLabel`).
    """
    rngData = None
    """Pre-initialized random number generator for constructing the test images
    repeatably (`numpy.random.Generator`).
    """
    rngMods = None
    """Pre-initialized random number generator for applying modifications to
    the test images for only some test cases (`numpy.random.Generator`).
    """
    kernelSize = None
    "Width of the kernel used for simulating sources, in pixels."
    exposures = {}
    "The simulated test data, with variable PSF sizes (`dict` of `lsst.afw.image.Exposure`)"
    matchedExposures = {}
    """The simulated exposures, all with PSF width set to `maxPsfSize`
    (`dict` of `lsst.afw.image.Exposure`).
    """
    photoCalib = afwImage.makePhotoCalibFromCalibZeroPoint(27, 10)
    "The photometric zero point to use for converting counts to flux units (`lsst.afw.image.PhotoCalib`)."
    badMaskPlanes = ["NO_DATA", "BAD"]
    "Mask planes that, if set, the associated pixel should not be included in the coaddTempExp."
    detector = None
    "Properties of the CCD for the exposure (`lsst.afw.cameraGeom.Detector`)."

    def __init__(self, shape=geom.Extent2I(201, 301), offset=geom.Point2I(-123, -45),
                 backgroundLevel=314.592, seed=42, nSrc=37,
                 fluxRange=2., noiseLevel=5, sourceSigma=200.,
                 minPsfSize=1.5, maxPsfSize=3.,
                 pixelScale=0.2*arcseconds, ra=209.*degrees, dec=-20.25*degrees,
                 ccd=37, patch=42, tract=0):
        self.ra = ra
        self.dec = dec
        self.pixelScale = pixelScale
        self.patch = patch
        self.tract = tract
        self.filterLabel = afwImage.FilterLabel(band="gTest", physical="gTest")
        self.rngData = np.random.default_rng(seed)
        self.rngMods = np.random.default_rng(seed + 1)
        self.bbox = geom.Box2I(offset, shape)
        if not self.bbox.contains(0, 0):
            raise ValueError(f"The bounding box must contain the coordinate (0, 0). {repr(self.bbox)}")
        self.wcs = self.makeDummyWcs()

        # Set up properties of the simulations
        nSigmaForKernel = 5
        self.kernelSize = (int(maxPsfSize*nSigmaForKernel + 0.5)//2)*2 + 1  # make sure it is odd

        bufferSize = self.kernelSize//2
        x0, y0 = self.bbox.getBegin()
        xSize, ySize = self.bbox.getDimensions()
        # Set the pixel coordinates and fluxes of the simulated sources.
        self.xLoc = self.rngData.random(nSrc)*(xSize - 2*bufferSize) + bufferSize + x0
        self.yLoc = self.rngData.random(nSrc)*(ySize - 2*bufferSize) + bufferSize + y0
        self.flux = (self.rngData.random(nSrc)*(fluxRange - 1.) + 1.)*sourceSigma*noiseLevel

        self.backgroundLevel = backgroundLevel
        self.noiseLevel = noiseLevel
        self.minPsfSize = minPsfSize
        self.maxPsfSize = maxPsfSize
        self.detector = DetectorWrapper(name=f"detector {ccd}", id=ccd).detector

    def setDummyCoaddInputs(self, exposure, expId):
        """Generate an `ExposureCatalog` as though the exposures had been
        processed using `warpAndPsfMatch`.

        Parameters
        ----------
        exposure : `lsst.afw.image.Exposure`
            The exposure to construct a `CoaddInputs` `ExposureCatalog` for.
        expId : `int`
            A unique identifier for the visit.
        """
        badPixelMask = afwImage.Mask.getPlaneBitMask(self.badMaskPlanes)
        nGoodPix = np.sum(exposure.getMask().getArray() & badPixelMask == 0)

        config = CoaddInputRecorderConfig()
        inputRecorder = CoaddInputRecorderTask(config=config, name="inputRecorder")
        tempExpInputRecorder = inputRecorder.makeCoaddTempExpRecorder(expId, num=1)
        tempExpInputRecorder.addCalExp(exposure, expId, nGoodPix)
        tempExpInputRecorder.finish(exposure, nGoodPix=nGoodPix)

    def makeCoaddTempExp(self, rawExposure, visitInfo, expId):
        """Add the metadata required by `AssembleCoaddTask` to an exposure.

        Parameters
        ----------
        rawExposure : `lsst.afw.image.Exposure`
            The simulated exposure.
        visitInfo : `lsst.afw.image.VisitInfo`
            VisitInfo containing metadata for the exposure.
        expId : `int`
            A unique identifier for the visit.

        Returns
        -------
        tempExp : `lsst.afw.image.Exposure`
            The exposure, with all of the metadata needed for coaddition.
        """
        tempExp = rawExposure.clone()
        tempExp.setWcs(self.wcs)

        tempExp.setFilter(self.filterLabel)
        tempExp.setPhotoCalib(self.photoCalib)
        tempExp.getInfo().setVisitInfo(visitInfo)
        tempExp.getInfo().setDetector(self.detector)
        self.setDummyCoaddInputs(tempExp, expId)
        return tempExp

    def makeDummyWcs(self, rotAngle=None, pixelScale=None, crval=None, flipX=True):
        """Make a World Coordinate System object for testing.

        Parameters
        ----------
        rotAngle : `lsst.geom.Angle`
            Rotation of the CD matrix, East from North
        pixelScale : `lsst.geom.Angle`
            Pixel scale of the projection.
        crval : `lsst.afw.geom.SpherePoint`
            Coordinates of the reference pixel of the wcs.
        flipX : `bool`, optional
            Flip the direction of increasing Right Ascension.

        Returns
        -------
        wcs : `lsst.afw.geom.skyWcs.SkyWcs`
            A wcs that matches the inputs.
        """
        if rotAngle is None:
            rotAngle = self.rotAngle
        if pixelScale is None:
            pixelScale = self.pixelScale
        if crval is None:
            crval = geom.SpherePoint(self.ra, self.dec)
        crpix = geom.Box2D(self.bbox).getCenter()
        cdMatrix = afwGeom.makeCdMatrix(scale=pixelScale, orientation=rotAngle, flipX=flipX)
        wcs = afwGeom.makeSkyWcs(crpix=crpix, crval=crval, cdMatrix=cdMatrix)
        return wcs

    def makeDummyVisitInfo(self, exposureId, randomizeTime=False):
        """Make a self-consistent visitInfo object for testing.

        Parameters
        ----------
        exposureId : `int`, optional
            Unique integer identifier for this observation.
        randomizeTime : `bool`, optional
            Add a random offset within a 6 hour window to the observation time.

        Returns
        -------
        visitInfo : `lsst.afw.image.VisitInfo`
            VisitInfo for the exposure.
        """
        lsstLat = -30.244639*u.degree
        lsstLon = -70.749417*u.degree
        lsstAlt = 2663.*u.m
        lsstTemperature = 20.*u.Celsius
        lsstHumidity = 40.  # in percent
        lsstPressure = 73892.*u.pascal
        loc = EarthLocation(lat=lsstLat,
                            lon=lsstLon,
                            height=lsstAlt)

        time = Time(2000.0, format="jyear", scale="tt")
        if randomizeTime:
            # Pick a random time within a 6 hour window
            time += 6*u.hour*(self.rngMods.random() - 0.5)
        radec = SkyCoord(dec=self.dec.asDegrees(), ra=self.ra.asDegrees(),
                         unit='deg', obstime=time, frame='icrs', location=loc)
        airmass = float(1.0/np.sin(radec.altaz.alt))
        obsInfo = makeObservationInfo(location=loc,
                                      detector_exposure_id=exposureId,
                                      datetime_begin=time,
                                      datetime_end=time,
                                      boresight_airmass=airmass,
                                      boresight_rotation_angle=Angle(0.*u.degree),
                                      boresight_rotation_coord='sky',
                                      temperature=lsstTemperature,
                                      pressure=lsstPressure,
                                      relative_humidity=lsstHumidity,
                                      tracking_radec=radec,
                                      altaz_begin=radec.altaz,
                                      observation_type='science',
                                      )
        visitInfo = MakeRawVisitInfoViaObsInfo.observationInfo2visitInfo(obsInfo)
        return visitInfo

    def makeTestImage(self, expId, noiseLevel=None, psfSize=None, backgroundLevel=None,
                      detectionSigma=5., badRegionBox=None):
        """Make a reproduceable PSF-convolved masked image for testing.

        Parameters
        ----------
        expId : `int`
            A unique identifier to use to refer to the visit.
        noiseLevel : `float`, optional
            Standard deviation of the noise to add to each pixel.
        psfSize : `float`, optional
            Width of the PSF of the simulated sources, in pixels.
        backgroundLevel : `float`, optional
            Background value added to all pixels in the simulated images.
        detectionSigma : `float`, optional
            Threshold amplitude of the image to set the "DETECTED" mask.
        badRegionBox : `lsst.geom.Box2I`, optional
            Add a bad region bounding box (set to "BAD").
        """
        if backgroundLevel is None:
            backgroundLevel = self.backgroundLevel
        if noiseLevel is None:
            noiseLevel = 5.
        visitInfo = self.makeDummyVisitInfo(expId, randomizeTime=True)

        if psfSize is None:
            psfSize = self.rngMods.random()*(self.maxPsfSize - self.minPsfSize) + self.minPsfSize
        nSrc = len(self.flux)
        sigmas = [psfSize for src in range(nSrc)]
        sigmasPsfMatched = [self.maxPsfSize for src in range(nSrc)]
        coordList = list(zip(self.xLoc, self.yLoc, self.flux, sigmas))
        coordListPsfMatched = list(zip(self.xLoc, self.yLoc, self.flux, sigmasPsfMatched))
        xSize, ySize = self.bbox.getDimensions()
        model = plantSources(self.bbox, self.kernelSize, self.backgroundLevel,
                             coordList, addPoissonNoise=False)
        modelPsfMatched = plantSources(self.bbox, self.kernelSize, self.backgroundLevel,
                                       coordListPsfMatched, addPoissonNoise=False)
        model.variance.array = np.abs(model.image.array) + noiseLevel
        modelPsfMatched.variance.array = np.abs(modelPsfMatched.image.array) + noiseLevel
        noise = self.rngData.random((ySize, xSize))*noiseLevel
        noise -= np.median(noise)
        model.image.array += noise
        modelPsfMatched.image.array += noise
        detectedMask = afwImage.Mask.getPlaneBitMask("DETECTED")
        detectionThreshold = self.backgroundLevel + detectionSigma*noiseLevel
        model.mask.array[model.image.array > detectionThreshold] += detectedMask

        if badRegionBox is not None:
            model.mask[badRegionBox] = afwImage.Mask.getPlaneBitMask("BAD")

        exposure = self.makeCoaddTempExp(model, visitInfo, expId)
        matchedExposure = self.makeCoaddTempExp(modelPsfMatched, visitInfo, expId)
        return exposure, matchedExposure

    @staticmethod
    def makeDataRefList(exposures, matchedExposures, warpType, tract=0, patch=42, coaddName="deep"):
        """Make data references from the simulated exposures that can be
        retrieved using the Gen 3 Butler API.

        Parameters
        ----------
        warpType : `str`
            Either 'direct' or 'psfMatched'.
        tract : `int`, optional
            Unique identifier for a tract of a skyMap.
        patch : `int`, optional
            Unique identifier for a subdivision of a tract.
        coaddName : `str`, optional
            The type of coadd being produced. Typically 'deep'.

        Returns
        -------
        dataRefList : `list` of `MockWarpReference`
            The data references.

        Raises
        ------
        ValueError
            If an unknown `warpType` is supplied.
        """
        dataRefList = []
        for expId in exposures:
            if warpType == 'direct':
                exposure = exposures[expId]
            elif warpType == 'psfMatched':
                exposure = matchedExposures[expId]
            else:
                raise ValueError("warpType must be one of 'direct' or 'psfMatched'")
            dataRef = MockWarpReference(exposure, storageClass="ExposureF",
                                        tract=tract, patch=patch, visit=expId, coaddName=coaddName)
            dataRefList.append(dataRef)
        return dataRefList
