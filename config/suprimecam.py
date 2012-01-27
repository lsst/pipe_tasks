from lsst.pipe.tasks.processCcd import ProcessCcdConfig

root = ProcessCcdConfig()
root.doWriteIsr = False
root.isr.methodList=["doConversionForIsr", "doSaturationDetection",
                     "doOverscanCorrection", "doFlatCorrection"]
root.isr.doWrite = False
