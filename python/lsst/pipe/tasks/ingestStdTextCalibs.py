from .ingestCalibs import IngestCalibsTask, IngestCalibsConfig
from .read_stdText_calibs import read_all
from lsst.pipe.base import InputOnlyArgumentParser

import tempfile
import shutil
import os


class IngestStdTextCalibsArgumentParser(InputOnlyArgumentParser):
    """Argument parser to support ingesting human curated calibration
       products in a standardized text file format into the repository"""

    def __init__(self, *args, **kwargs):
        InputOnlyArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("-n", "--dry-run", dest="dryrun", action="store_true",
                          default=False, help="Don't perform any action?")
        self.add_argument("--create", action="store_true", help="Create new registry?")
        self.add_argument("--ignore-ingested", dest="ignoreIngested", action="store_true",
                          help="Don't register files that have already been registered")
        self.add_argument("root", help="Root directory to scan for calibs.")


class IngestStdTextCalibsConfig(IngestCalibsConfig):
    def setDefaults(self):
        if "filter" in self.register.columns:
            self.parse.defaults["filter"] = "NONE"


class IngestStdTextCalibsTask(IngestCalibsTask):
    """Task that generates registry for human curated calibration products
       in a standardized text file format"""
    ArgumentParser = IngestStdTextCalibsArgumentParser
    _DefaultName = "ingestStdTextCalibs"
    ConfigClass = IngestStdTextCalibsConfig

    def run(self, args):
        """Ingest all defect files and add them to the registry"""

        try:
            camera = args.butler.get('camera')
            temp_dir = tempfile.mkdtemp()
            calibs, calib_type = read_all(args.root, camera)
            file_names = []
            for d in calibs:
                for s in calibs[d]:
                    file_name = f'{calib_type}_{d}_{s.isoformat()}.fits'
                    full_file_name = os.path.join(temp_dir, file_name)
                    self.log.info('%s written for sensor: %s and calibDate: %s' %
                                  (calib_type, d, s.isoformat()))
                    calibs[d][s].writeFits(full_file_name)
                    file_names.append(full_file_name)
            args.files = file_names
            args.mode = 'move'
            args.validity = None  # Validity range is determined from the files
            IngestCalibsTask.run(self, args)
        finally:
            shutil.rmtree(temp_dir)
