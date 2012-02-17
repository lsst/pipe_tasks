root.doWriteIsr = False
root.isr.methodList=["doConversionForIsr", "doSaturationDetection",
                     "doOverscanCorrection", "doVariance", "doFlatCorrection"]
root.isr.doWrite = False

root.calibrate.repair.doCosmicRay = True
root.calibrate.repair.cosmicray.nCrPixelMax = 100000
root.calibrate.repair.cosmicray.keepCRs = False
root.calibrate.background.binSize = 1024

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
root.photometry.measure.source.astrom = "NAIVE"
root.photometry.measure.source.apFlux = "NAIVE"
root.photometry.measure.source.modelFlux = "GAUSSIAN"
root.photometry.measure.source.psfFlux = "PSF"
root.photometry.measure.source.shape = "SDSS"
root.photometry.measure.astrometry.names = ["GAUSSIAN", "NAIVE", "SDSS"]
root.photometry.measure.shape.names = ["SDSS"]
root.photometry.measure.photometry.names = ["NAIVE", "GAUSSIAN", "PSF", "SINC"]
root.photometry.measure.photometry["NAIVE"].radius = 7.0
root.photometry.measure.photometry["GAUSSIAN"].shiftmax = 10
root.photometry.measure.photometry["SINC"].radius = 7.0

# Initial photometry
root.calibrate.photometry.detect.thresholdValue = 5.0
root.calibrate.photometry.detect.includeThresholdMultiplier = 10.0
root.calibrate.photometry.measure = root.photometry.measure

# Aperture correction
root.calibrate.apCorr.alg1.name = "PSF"
root.calibrate.apCorr.alg2.name = "SINC"
root.calibrate.apCorr.alg1[root.calibrate.apCorr.alg1.name] = root.photometry.measure.photometry[root.calibrate.apCorr.alg1.name]
root.calibrate.apCorr.alg2[root.calibrate.apCorr.alg2.name] = root.photometry.measure.photometry[root.calibrate.apCorr.alg2.name]

# Astrometry
root.calibrate.astrometry.distortion.name = "radial"
root.calibrate.astrometry.distortion["radial"].coefficients = [0.0, 1.0, 7.16417e-08, 3.03146e-10, 5.69338e-14, -6.61572e-18]
root.calibrate.astrometry.distortion["radial"].observedToCorrected = True
