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
#
"""Extract small cutouts around bright stars, normalize and warp them to the
same arbitrary pixel grid.
"""

__all__ = ["ReprocessBrightStarsTask"]

import numpy as np

from lsst.afw import image as afwImage
import lsst.pex.config as pexConfig
from lsst.pipe import base as pipeBase
from lsst.pipe.base import connectionTypes as cT
from lsst.meas.algorithms import brightStarStamps as bSS


def replaceMaskedPixels(maskedIm, maskPlane, val=0, inPlace=False, verbose=False):
    mask = maskedIm.mask
    mpd = mask.getMaskPlaneDict()
    mpValues = set(list(mask.array.flatten()))
    bitNb = mpd[maskPlane]
    badVals = []
    for mpv in mpValues:
        binPv = bin(mpv)[2:]
        if len(binPv) >= bitNb + 1:
            if int(binPv[-(bitNb + 1)]):
                badVals += [mpv]
    if not badVals:
        if verbose:
            print(f'Mask plane {maskPlane} seems absent from image; returning it unchanged')
        return maskedIm
    if inPlace:
        newIm = maskedIm
    else:
        newIm = maskedIm.clone()
    arr = newIm.image.array
    maskArr = maskedIm.mask.array
    for bv in badVals:
        arr[maskArr==bv] = val
    return newIm


class ProcessBrightStarsConnections(pipeBase.PipelineTaskConnections, dimensions=("visit", "detector")):
    inputExposure = cT.Input(
        doc="Input exposure from which to extract bright star stamps",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("visit", "detector")
    )
    brightStarStamps = cT.Output(
        doc="Set of preprocessed postage stamps, each centered on a single bright star.",
        name="brightStarStamps",
        storageClass="BrightStarStamps",
        dimensions=("visit", "detector")
    )


class ReprocessBrightStarsConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=ProcessBrightStarsConnections):
    """Configuration parameters for ProcessBrightStarsTask
    """
    modelFilename = pexConfig.Field(
        dtype=str,
        doc="Absolute path to saved extended PSF model (temporary)",
    )
    badMaskPlanes = pexConfig.ListField(
        dtype=str,
        doc="Mask planes that, if set, lead to associated pixels not being included in the computation of the"
            " annular flux.",
        default=('BAD', 'CR', 'CROSSTALK', 'EDGE', 'NO_DATA', 'SAT', 'SUSPECT', 'UNMASKEDNAN')
    )
    erosionFactor = pexConfig.Field(
        dtype=int,
        doc="Erosion factor to apply to get small insert centered on objects for hackyFlux computation",
        default=0,
    )
    centerFew = pexConfig.Field(
        dtype=int,
        doc="Number of pixels around center (in each direction) to ignore even if not saturated",
        default=6,
    )


class ReprocessBrightStarsTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """The description of the parameters for this Task are detailed in
    :lsst-task:`~lsst.pipe.base.PipelineTask`.

    Notes
    -----
    `ProcessBrightStarsTask` is used to extract, process, and store small
    image cut-outs (or "postage stamps") around bright stars. It relies on
    three methods, called in succession:

    `extractStamps`
        Find bright stars within the exposure using a reference catalog and
        extract a stamp centered on each.
    `warpStamps`
        Shift and warp each stamp to remove optical distortions and sample all
        stars on the same pixel grid.
    `measureAndNormalize`
        Compute the flux of an object in an annulus and normalize it. This is
        required to normalize each bright star stamp as their central pixels
        are likely saturated and/or contain ghosts, and cannot be used.
    """
    ConfigClass = ReprocessBrightStarsConfig
    _DefaultName = "reprocessBrightStars"
    RunnerClass = pipeBase.ButlerInitializedTaskRunner

    def __init__(self, butler=None, initInputs=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = afwImage.MaskedImageF(self.config.modelFilename)
        self.modelStampSize = np.array(self.model.getDimensions())
        self.modelCenter = self.modelStampSize[0]//2, self.modelStampSize[1]//2

    def leastSquareFlux(self, starIm, normalize=False):
        bbox = starIm.getBBox().erodedBy(self.config.erosionFactor)
        croppedStar = starIm.clone()[bbox]
        croppedModel = self.model.clone()[bbox]
        for bm in self.config.badMaskPlanes:
            croppedStar = replaceMaskedPixels(croppedStar, bm, val=np.nan, inPlace=True)
        if self.config.centerFew:
            croppedCenter = np.array(croppedStar.getDimensions()) // 2
            croppedStar.image.array[croppedCenter[0] - self.config.centerFew:
                                        croppedCenter[0] + self.config.centerFew,
                                    croppedCenter[1] - self.config.centerFew:
                                        croppedCenter[1] + self.config.centerFew] = np.isnan
        croppedModel.image.array[np.isnan(croppedStar.image.array)] = np.nan
        X = croppedModel.image.array.flatten()
        Y = croppedStar.image.array.flatten()
        scaling = np.nansum(X*Y) / np.nansum(X*X)
        if normalize:
            starIm.image.array /= scaling
        return float(scaling)

    @pipeBase.timeMethod
    def run(self, bss, dataId=None):
        """Identify bright stars within an exposure using a reference catalog,
        extract stamps around each, then preprocess them. The preprocessing
        steps are: shifting, warping and potentially rotating them to the same
        pixel grid; computing their annular flux and normalizing them.

        Parameters
        ----------
        inputExposure : `afwImage.exposure.exposure.ExposureF`
            The image from which bright star stamps should be extracted.
        refObjLoader : `LoadIndexedReferenceObjectsTask`, optional
            Loader to find objects within a reference catalog.
        dataId : `dict`
            The dataId of the exposure (and detector) bright stars should be
            extracted from.

        Returns
        -------
        result :  `lsst.pipe.base.Struct`
            Result struct with component:

            - ``brightStarStamps``: ``bSS.BrightStarStamps``
        """
        self.log.info("Computing least square fit flux for exposure %s", dataId)
        # Unnormalize 1st pass bss
        for j, (starIm, flux) in enumerate(zip(bss.getMaskedImages(), bss.getAnnularFluxes())):
            starIm.image.array *= flux
        # compute hacky flux
        fluxes = [self.leastSquareFlux(starIm, normalize=True) for starIm in bss.getMaskedImages()]
        brightStarList = [bSS.BrightStarStamp(starStamp=warp,
                                              gaiaGMag=bss.getMagnitudes()[j],
                                              gaiaId=bss.getGaiaIds()[j],
                                              XY0=bss.getXY0s()[j],
                                              annularFlux=fluxes[j],
                                              transform=bss.getTransforms()[j])
                          for j, warp in enumerate(bss.getMaskedImages())]
        brightStarStamps = bSS.BrightStarStamps(brightStarList, -1, -1,
                                                nb90Rots=bss.nb90Rots)
        return pipeBase.Struct(brightStarStamps=brightStarStamps)

    def runDataRef(self, dataRef):
        """ Read in required calexp, extract and process stamps around bright
        stars and write them to disk.

        Parameters
        ----------
        dataRef : `lsst.daf.persistence.butlerSubset.ButlerDataRef`
            Data reference to the calexp to extract bright stars from.
        """
        bss1stpass = dataRef.get("brightStarStamps")
        output = self.run(bss1stpass, dataId=dataRef.dataId)
        # Save processed bright star stamps
        dataRef.put(output.brightStarStamps, "brightStarStamps")
        return pipeBase.Struct(brightStarStamps=output.brightStarStamps)
