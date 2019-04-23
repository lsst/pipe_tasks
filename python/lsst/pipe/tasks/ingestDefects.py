from .ingestCalibs import IngestCalibsTask
from lsst.obs.base.read_defects import read_all_defects
from lsst.pipe.base import InputOnlyArgumentParser

from dateutil import parser


class IngestDefectsArgumentParser(InputOnlyArgumentParser):
    """Argument parser to support ingesting calibration images into the repository"""

    def __init__(self, *args, **kwargs):
        InputOnlyArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("-n", "--dry-run", dest="dryrun", action="store_true",
                          default=False, help="Don't perform any action?")
        self.add_argument("--create", action="store_true", help="Create new registry?")
        self.add_argument("--ignore-ingested", dest="ignoreIngested", action="store_true",
                          help="Don't register files that have already been registered")
        self.add_argument("root", help="Root directory to scan for defects.")


class IngestDefectsTask(IngestCalibsTask):
    """Task that generates registry for calibration images"""
    ArgumentParser = IngestDefectsArgumentParser
    _DefaultName = "ingestDefects"

    def run(self, args):
        """Ingest all defect files and add them to the registry"""
        calibRoot = args.calib if args.calib is not None else args.output
        camera = args.butler.get('camera')
        with self.register.openRegistry(calibRoot, create=args.create, dryrun=args.dryrun,
                                        name="calibRegistry.sqlite3") as registry:
            calibType = 'deflects'
            defects = read_all_defects(args.root)
            if calibType not in self.register.config.tables:
                raise RuntimeError(str("Skipped adding %s of observation type '%s' to registry "
                                       "(must be one of %s)" %
                                       (args.root, calibType, ", ".join(self.register.config.tables))))
            for ccd_name in defects:
                time_keys = [k for k in defects[ccd_name]]
                if len(time_keys) == 0:
                    continue  # no defects for this chip
                time_keys.sort()
                for valid_start, valid_end in zip(time_keys[:-1], time_keys[1:]):
                    defectList = defects[ccd_name][valid_start]
                    dataId = {'ccd': camera[ccd_name].getId(), 'calibDate': valid_start}
                    if not args.dryrun:
                        try:
                            args.butler.put(defectList, calibType, ccd=camera[ccd_name].getId(), calibDate=valid_start.isoformat())
                        except Exception:
                            self.log.warn(str("Failed to ingest %s at %s of observation type '%s'" %
                                          (ccd_name, valid_start, calibType)))
                            continue
                    if self.register.check(registry, dataId, table=calibType):
                        if args.ignoreIngested:
                            continue

                        self.log.warn("%s--%s: already ingested: %s" % (ccd_name, valid_start, dataId))
                    dataId['validStart'] = valid_start
                    dataId['validEnd'] = valid_end
                    self.register.addRow(registry, dataId, dryrun=args.dryrun,
                                         create=args.create, table=calibType)

                dataId = {'ccd': camera[ccd_name].getId(), 'calibDate': time_keys[-1],
                          'validStart': time_keys[-1],
                          'validEnd': parser.parse('20371231235959')}  # set end to near end of UNIX time
                self.register.addRow(registry, dataId, dryrun=args.dryrun,
                                     create=args.create, table=calibType)
