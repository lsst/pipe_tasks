from contextlib import contextmanager
from lsst.pex.config import Config, ConfigurableField
from lsst.pipe.base import CmdLineTask, Struct, ArgumentParser
from lsst.meas.algorithms.trail import TrailTask, MomentLimitConfig

class TrailsConfig(Config):
    satellites = ConfigurableField(target=TrailTask, doc="Find satellite (narrow) trails")
    aircraft = ConfigurableField(target=TrailTask, doc="Find aircraft (broad) trails")

    def setDefaults(self):
        # Defaults for TrailTask are appropriate for satellites; set defaults for aircraft
        self.aircraft.widths = [40.0, 70.0, 100.0]
        self.aircraft.sigmaSmooth = 2.0
        self.aircraft.kernelSigma = 9.0
        self.aircraft.kernelWidth = 15
        self.aircraft.bins = 8
        self.aircraft.thetaTolerance = 0.25
        self.aircraft.hough.bins = 100
        self.aircraft.growKernel = 1.0
        self.aircraft.selection = {}
        for name, limit, style in (
                                   ("center", 1.0, "center"),
                                   ("skew", 50.0, "center"),
#                                   ("ellip", 0.2, "center"),
#                                   ("b", 1.5, "center"),
                                   ):
            self.aircraft.selection[name] = MomentLimitConfig()
            self.aircraft.selection[name].limit = limit
            self.aircraft.selection[name].style = style


class TrailsTask(CmdLineTask):
    ConfigClass = TrailsConfig
    _DefaultName = "trails"

    def __init__(self, *args, **kwargs):
        CmdLineTask.__init__(self, *args, **kwargs)
        self.makeSubtask("satellites")
        self.makeSubtask("aircraft")

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument(name="--id", datasetType="calexp",
                               help="data IDs, e.g. --id visit=12345 ccd=1,2^0,3")
        return parser

    def run(self, dataRef):
        calexp = dataRef.get()
        with self.debugging():  # Because we have no other output
            self.runExposure(calexp)

    @contextmanager
    def debugging(self):
        import lsst.log
        import lsst.afw.display
        logLevel = self.log.getLevel()
        self.log.setLevel(lsst.log.DEBUG)
        backend = lsst.afw.display.getDefaultBackend()
        lsst.afw.display.setDefaultBackend("ds9")
        try:
            yield
        finally:
            self.log.setLevel(logLevel)
            lsst.afw.display.setDefaultBackend(backend)

    def runExposure(self, exposure):
        satellites = self.satellites.run(exposure)
        aircraft = self.aircraft.run(exposure)
        return Struct(satellites=satellites, aircraft=aircraft)

    def _getConfigName(self):
        return None

    def _getMetadataName(self):
        return None
