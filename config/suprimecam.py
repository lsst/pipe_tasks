root.doWriteIsr = False
root.isr.methodList=["doConversionForIsr", "doSaturationDetection",
                     "doOverscanCorrection", "doVariance", "doFlatCorrection"]
root.isr.doWrite = False

root.calibrate.repair.doCosmicRay = True
root.calibrate.repair.cosmicray.nCrPixelMax = 100000
root.calibrate.repair.cosmicray.keepCRs = False
root.calibrate.background.binSize = 1024

# crosstalk coefficients for SuprimeCam, as crudely measured by RHL
root.calibrate.repair.crosstalkCoeffs.values = [
     0.00e+00, -8.93e-05, -1.11e-04, -1.18e-04,
    -8.09e-05,  0.00e+00, -7.15e-06, -1.12e-04,
    -9.90e-05, -2.28e-05,  0.00e+00, -9.64e-05,
    -9.59e-05, -9.85e-05, -8.77e-05,  0.00e+00,
    ]
# nonlinearity for SuprimeCam
root.calibrate.repair.linearizationCoefficient = 2.5e-7

# correct photometry for known radial distortions leading to unreliable flats
root.calibrate.measurement.doCorrectDistortion = True
root.measurement.doCorrectDistortion = True

# PSF determination
root.calibrate.measurePsf.starSelector.name = "secondMoment"
root.calibrate.measurePsf.psfDeterminer.name = "pca"
root.calibrate.measurePsf.starSelector["secondMoment"].clumpNSigma = 2.0
root.calibrate.measurePsf.psfDeterminer["pca"].nEigenComponents = 4
root.calibrate.measurePsf.psfDeterminer["pca"].kernelSize = 7
root.calibrate.measurePsf.psfDeterminer["pca"].spatialOrder = 2
root.calibrate.measurePsf.psfDeterminer["pca"].kernelSizeMin = 25

# Final photometry
root.detection.thresholdValue = 5.0
root.detection.includeThresholdMultiplier = 1.0

root.measurement.algorithms["flux.gaussian"].shiftmax = 10.0

# Initial photometry
root.calibrate.detection.thresholdValue = 5.0
root.calibrate.detection.includeThresholdMultiplier = 10.0

root.calibrate.initialMeasurement.algorithms = root.measurement.algorithms
root.calibrate.measurement.algorithms = root.measurement.algorithms

from lsst.meas.photocal.colorterms import Colorterm
from lsst.obs.suprimecam.colorterms import colortermsData
Colorterm.setColorterms(colortermsData)

Colorterm.setActiveDevice("Hamamatsu")
