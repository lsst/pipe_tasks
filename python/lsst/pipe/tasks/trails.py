from lsst.pex.config import Config, ConfigurableField
from lsst.pipe.base import Task, Struct
from lsst.meas.algorithms.trail import TrailTask

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

class TrailsTask(Task):
    ConfigClass = TrailsConfig

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.makeSubtask("satellites")
        self.makeSubtask("aircraft")

    def run(self, exposure):
        satellites = self.satellites.run(exposure)
        aircraft = self.satellites.run(exposure)
        return Struct(satellites=satellites, aircraft=aircraft)

        