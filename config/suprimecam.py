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

# PSF determination
root.calibrate.measurePsf.starSelector.name = "secondMoment"
root.calibrate.measurePsf.psfDeterminer.name = "pca"
root.calibrate.measurePsf.starSelector["secondMoment"].clumpNSigma = 2.0
root.calibrate.measurePsf.psfDeterminer["pca"].nEigenComponents = 4
root.calibrate.measurePsf.psfDeterminer["pca"].kernelSize = 7.0
root.calibrate.measurePsf.psfDeterminer["pca"].spatialOrder = 2
root.calibrate.measurePsf.psfDeterminer["pca"].kernelSizeMin = 25

# Final photometry
root.photometry.detect.thresholdValue = 5.0
root.photometry.detect.includeThresholdMultiplier = 1.0
root.photometry.measure.source.astrom = "SDSS"
root.photometry.measure.source.apFlux = "SINC"
root.photometry.measure.source.modelFlux = "GAUSSIAN"
root.photometry.measure.source.psfFlux = "PSF"
root.photometry.measure.source.shape = "SDSS"
root.photometry.measure.astrometry.names = ["GAUSSIAN", "NAIVE", "SDSS"]
root.photometry.measure.shape.names = ["SDSS"]
root.photometry.measure.photometry.names = ["NAIVE", "GAUSSIAN", "PSF", "SINC"]
root.photometry.measure.photometry["GAUSSIAN"].shiftmax = 10
root.photometry.measure.photometry["SINC"].radius = 7.0
root.photometry.measure.photometry["NAIVE"].radius = root.photometry.measure.photometry["SINC"].radius

# Initial photometry
root.calibrate.photometry.detect.thresholdValue = 5.0
root.calibrate.photometry.detect.includeThresholdMultiplier = 10.0
root.calibrate.photometry.measure = root.photometry.measure

# Aperture correction
root.calibrate.apCorr.alg1.name = root.photometry.measure.source.psfFlux
root.calibrate.apCorr.alg2.name = root.photometry.measure.source.apFlux
root.calibrate.apCorr.alg1[root.calibrate.apCorr.alg1.name] = \
    root.photometry.measure.photometry[root.calibrate.apCorr.alg1.name]
root.calibrate.apCorr.alg2[root.calibrate.apCorr.alg2.name] = \
    root.photometry.measure.photometry[root.calibrate.apCorr.alg2.name]

# Astrometry
root.calibrate.astrometry.distortion.name = "null" # comes from cameraGeom
