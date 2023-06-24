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


import logging
import numpy as np

from lsst.afw.detection import Psf
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
from lsst.meas.algorithms import SourceDetectionTask
from lsst.meas.base import SkyMapIdGeneratorConfig
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.utils as utils


log = logging.getLogger(__name__)


def calculateKernelSize(sigma: float, nSigmaForKernel: float = 7) -> int:
    """Calculate the size of the smoothing kernel.

    Parameters
    ----------
    sigma:
        Gaussian sigma of smoothing kernel.
    nSigmaForKernel:
        The multiple of `sigma` to use to set the size of the kernel.
        Note that that is the full width of the kernel bounding box
        (so a value of 7 means 3.5 sigma on either side of center).
        The value will be rounded up to the nearest odd integer.

    Returns
    -------
    size:
        Size of the smoothing kernel.
    """
    return (int(sigma * nSigmaForKernel + 0.5)//2)*2 + 1  # make sure it is odd


def convolveImage(image: afwImage.Image, psf: Psf) -> afwImage.Image:
    """Convolve an image with a psf

    This methodm and the docstring, is based off the method in
    `~lsst.meas.algorithms.detection.SourceDetectionTask`.

    We convolve the image with a Gaussian approximation to the PSF,
    because this is separable and therefore fast. It's technically a
    correlation rather than a convolution, but since we use a symmetric
    Gaussian there's no difference.

    Parameters
    ----------
    image:
        The image to convovle.
    psf:
        The PSF to convolve the `image` with.

    Returns
    -------
    convolved:
        The result of convolving `image` with the `psf`.
    """
    sigma = psf.computeShape(psf.getAveragePosition()).getDeterminantRadius()
    bbox = image.getBBox()

    # Smooth using a Gaussian (which is separable, hence fast) of width sigma
    # Make a SingleGaussian (separable) kernel with the 'sigma'
    kWidth = calculateKernelSize(sigma)
    gaussFunc = afwMath.GaussianFunction1D(sigma)
    gaussKernel = afwMath.SeparableKernel(kWidth, kWidth, gaussFunc, gaussFunc)

    convolvedImage = image.Factory(bbox)

    afwMath.convolve(convolvedImage, image, gaussKernel, afwMath.ConvolutionControl())

    return convolvedImage.Factory(convolvedImage, bbox, afwImage.PARENT, False)


class AssembleChi2CoaddConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=("tract", "patch", "skymap"),
                                   defaultTemplates={"inputCoaddName": "deep",
                                                     "outputCoaddName": "deepChi2"}):
    inputCoadds = cT.Input(
        doc="Exposure on which to run deblending",
        name="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        multiple=True,
        dimensions=("tract", "patch", "band", "skymap")
    )
    chi2Coadd = cT.Output(
        doc="Chi^2 exposure, produced by merging multiband coadds",
        name="{outputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap"),
    )


class AssembleChi2CoaddConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=AssembleChi2CoaddConnections):
    outputPixelatedVariance = pexConfig.Field(
        dtype=bool,
        default=False,
        doc="Whether to output a pixelated variance map for the generated "
            "chi^2 coadd, or to have a flat variance map defined by combining "
            "the inverse variance maps of the coadds that were combined."
    )

    useUnionForMask = pexConfig.Field(
        dtype=bool,
        default=True,
        doc="Whether to calculate the union of the mask plane in each band, "
            "or the intersection of the mask plane in each band."
    )


class AssembleChi2CoaddTask(pipeBase.PipelineTask):
    """Assemble a chi^2 coadd from a collection of multi-band coadds

    References
    ----------
    .. [1] Szalay, A. S., Connolly, A. J., and Szokoly, G. P., “Simultaneous
    Multicolor Detection of Faint Galaxies in the Hubble Deep Field”,
    The Astronomical Journal, vol. 117, no. 1, pp. 68–74,
    1999. doi:10.1086/300689.

    .. [2] Kaiser 2001 whitepaper,
    http://pan-starrs.ifa.hawaii.edu/project/people/kaiser/imageprocessing/im%2B%2B.pdf  # noqa: E501

    .. [3] https://dmtn-015.lsst.io/

    .. [4] https://project.lsst.org/meetings/law/sites/lsst.org.meetings.law/files/Building%20and%20using%20coadds.pdf
    """  # noqa: E501
    ConfigClass = AssembleChi2CoaddConfig
    _DefaultName = "assembleChi2Coadd"

    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)

    def combinedMasks(self, masks: list[afwImage.MaskX]) -> afwImage.MaskX:
        """Combine the mask plane in each input coadd

        Parameters
        ----------
        mMask:
            The MultibandMask in each band.

        Returns
        -------
        result:
            The resulting single band mask.
        """
        refMask = masks[0]
        bbox = refMask.getBBox()
        mask = refMask.array
        for _mask in masks[1:]:
            if self.config.useUnionForMask:
                mask = mask | _mask.array
            else:
                mask = mask & _mask.array
        result = refMask.Factory(bbox)
        result.array[:] = mask
        return result

    @utils.inheritDoc(pipeBase.PipelineTask)
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, inputCoadds: list[afwImage.Exposure]) -> pipeBase.Struct:
        """Assemble the chi2 coadd from the multiband coadds

        Parameters
        ----------
        inputCoadds:
            The coadds to combine into a single chi2 coadd.

        Returns
        -------
        result:
            The chi2 coadd created from the input coadds.
        """
        convControl = afwMath.ConvolutionControl()
        convControl.setDoNormalize(False)
        convControl.setDoCopyEdge(False)

        # Set a reference exposure to use for creating the new coadd.
        # It doesn't matter which exposure we use, since we just need the
        # bounding box information and Factory to create a new expsure with
        # the same dtype.
        refExp = inputCoadds[0]
        bbox = refExp.getBBox()

        image = refExp.image.Factory(bbox)
        variance_list = []
        # Convovle the image in each band and weight by the median variance
        for calexp in inputCoadds:
            convolved = convolveImage(calexp.image, calexp.getPsf())
            _variance = np.median(calexp.variance.array)
            convolved.array[:] /= _variance
            image += convolved
            variance_list.append(_variance)

        variance = refExp.variance.Factory(bbox)
        if self.config.outputPixelatedVariance:
            # Write the per pixel variance to the output coadd
            variance.array[:] = np.sum([1/coadd.variance for coadd in inputCoadds], axis=0)
        else:
            # Use a flat variance in each band
            variance.array[:] = np.sum(1/np.array(variance_list))
        # Combine the masks planes to calculate the mask plae of the new coadd
        mask = self.combinedMasks([coadd.mask for coadd in inputCoadds])
        # Create the exposure
        maskedImage = refExp.maskedImage.Factory(image, mask=mask, variance=variance)
        chi2coadd = refExp.Factory(maskedImage, exposureInfo=refExp.getInfo())
        chi2coadd.info.setFilter(None)
        return pipeBase.Struct(chi2Coadd=chi2coadd)


class DetectChi2SourcesConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap"),
    defaultTemplates={
        "inputCoaddName": "deepChi2",
        "outputCoaddName": "deepChi2"
    }
):
    detectionSchema = cT.InitOutput(
        doc="Schema of the detection catalog",
        name="{outputCoaddName}Coadd_det_schema",
        storageClass="SourceCatalog",
    )
    exposure = cT.Input(
        doc="Exposure on which detections are to be performed",
        name="{inputCoaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap"),
    )
    outputSources = cT.Output(
        doc="Detected sources catalog",
        name="{outputCoaddName}Coadd_det",
        storageClass="SourceCatalog",
        dimensions=("tract", "patch", "skymap"),
    )


class DetectChi2SourcesConfig(pipeBase.PipelineTaskConfig, pipelineConnections=DetectChi2SourcesConnections):
    detection = pexConfig.ConfigurableField(
        target=SourceDetectionTask,
        doc="Detect sources in chi2 coadd"
    )

    idGenerator = SkyMapIdGeneratorConfig.make_field()

    def setDefaults(self):
        super().setDefaults()
        self.detection.reEstimateBackground = False
        self.detection.thresholdValue = 3


class DetectChi2SourcesTask(pipeBase.PipelineTask):
    _DefaultName = "detectChi2Sources"
    ConfigClass = DetectChi2SourcesConfig

    def __init__(self, schema=None, **kwargs):
        # N.B. Super is used here to handle the multiple inheritance of PipelineTasks, the init tree
        # call structure has been reviewed carefully to be sure super will work as intended.
        super().__init__(**kwargs)
        if schema is None:
            schema = afwTable.SourceTable.makeMinimalSchema()
        self.schema = schema
        self.makeSubtask("detection", schema=self.schema)
        self.detectionSchema = afwTable.SourceCatalog(self.schema)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)
        idGenerator = self.config.idGenerator.apply(butlerQC.quantum.dataId)
        inputs["idFactory"] = idGenerator.make_table_id_factory()
        inputs["expId"] = idGenerator.catalog_id
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self, exposure: afwImage.Exposure, idFactory: afwTable.IdFactory, expId: int) -> pipeBase.Struct:
        """Run detection on a chi2 exposure.

        Parameters
        ----------
        exposure :
            Exposure on which to detect (may be backround-subtracted and scaled,
            depending on configuration).
        idFactory :
            IdFactory to set source identifiers.
        expId :
            Exposure identifier (integer) for RNG seed.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:
            ``outputSources``
                Catalog of detections (`lsst.afw.table.SourceCatalog`).
        """
        table = afwTable.SourceTable.make(self.schema, idFactory)
        # We override `doSmooth` since the chi2 coadd has already had an
        # extra PSF convolution applied to decorrelate the images
        # accross bands.
        detections = self.detection.run(table, exposure, expId=expId, doSmooth=False)
        sources = detections.sources
        return pipeBase.Struct(outputSources=sources)
