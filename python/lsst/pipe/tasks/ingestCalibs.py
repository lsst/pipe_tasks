import collections
import datetime
import itertools
import sqlite3
import lsst.afw.image as afwImage
from lsst.pex.config import Config, ListField, ConfigurableField
from lsst.pipe.base import ArgumentParser
from lsst.pipe.tasks.ingest import RegisterTask, ParseTask, RegisterConfig, IngestTask


def _convertToDate(dateString):
    """Convert a string into a date object, or return None
    when the date string cannot be converted with format %Y-%m-%d
    """
    try:
        return datetime.datetime.strptime(dateString, "%Y-%m-%d").date()
    except ValueError:
        return None


class CalibsParseTask(ParseTask):
    """Task that will parse the filename and/or its contents to get the
    required information to populate the calibration registry."""
    def getCalibType(self, filename):
        """Return a a known calibration dataset type using
        the observation type in the header keyword OBSTYPE

        @param filename: Input filename
        """
        md = afwImage.readMetadata(filename, self.config.hdu)
        if not md.exists("OBSTYPE"):
            raise RuntimeError("Unable to find the required header keyword OBSTYPE")
        obstype = md.get("OBSTYPE").strip().lower()
        if "flat" in obstype:
            obstype = "flat"
        elif "zero" in obstype or "bias" in obstype:
            obstype = "bias"
        elif "dark" in obstype:
            obstype = "dark"
        elif "fringe" in obstype:
            obstype = "fringe"
        return obstype


class CalibsRegisterConfig(RegisterConfig):
    """Configuration for the CalibsRegisterTask"""
    tables = ListField(dtype=str, default=["bias", "dark", "flat", "fringe"],
                       doc="Name of tables")


class CalibsRegisterTask(RegisterTask):
    """Task that will generate the calibration registry for the Mapper"""
    ConfigClass = CalibsRegisterConfig

    def openRegistry(self, directory, create=False, dryrun=False, name="calibRegistry.sqlite3"):
        """Open the registry and return the connection handle"""
        return RegisterTask.openRegistry(self, directory, create, dryrun, name)

    def createTable(self, conn):
        """Create the registry tables"""
        for table in self.config.tables:
            RegisterTask.createTable(self, conn, table=table)

    def updateValidityRanges(self, conn):
        """Loop over all tables, filters, and ccdnums,
        and update the validity ranges in the registry.

        @param conn: Database connection
        """
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        for table in self.config.tables:
            sql = "SELECT DISTINCT filter,ccdnum FROM %s" % table
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                self.fixSubsetValidity(conn, table, str(row["filter"]), row["ccdnum"])

    def fixSubsetValidity(self, conn, table, filterName, ccdnum):
        """Update the validity ranges among selected rows in the registry.

        For defects, the products are valid from their start date until
        they are superseded by subsequent defect data.
        For other calibration products, the validity ranges are checked and
        if there are overlaps, a midpoint is used to fix the overlaps,
        so that the calibration data with whose date is nearest the date
        of the observation is used.

        @param conn: Database connection
        @param table: Name of table to be selected
        @param filterName: Select condition for column filter
        @param ccdnum: Select condition for column ccdnum
        """
        sql = "SELECT id, calibDate, validStart, validEnd FROM %s" % table
        sql += " WHERE filter='%s' AND ccdnum=%s" % (filterName, ccdnum)
        sql += " ORDER BY calibDate"
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        valids = collections.OrderedDict([(_convertToDate(row["calibDate"]),
                                           [_convertToDate(row["validStart"]),
                                            _convertToDate(row["validEnd"])]) for row in rows])
        dates = valids.keys()
        if None in dates:
            self.log.warn("Skipped fixing the validity overlaps for %s filter=%s"
                          " ccdnum=%s because of missing calibration dates" %
                          (table, filterName, ccdnum))
            return
        if table == "defect":
            for thisDate, nextDate in itertools.izip(dates[:-1], dates[1:]):
                valids[thisDate][0] = thisDate
                valids[thisDate][1] = nextDate - datetime.timedelta(1)
            valids[dates[-1]][1] = _convertToDate("2037-12-31")  # End of UNIX time
        else:
            midpoints = [t1 + (t2 - t1)//2 for t1, t2 in itertools.izip(dates[:-1], dates[1:])]
            for i, (date, midpoint) in enumerate(itertools.izip(dates[:-1], midpoints)):
                if valids[date][1] > midpoint:
                    nextDate = dates[i + 1]
                    valids[nextDate][0] = midpoint + datetime.timedelta(1)
                    valids[date][1] = midpoint
            del midpoints
        del dates
        for row in rows:
            calibDate = _convertToDate(row["calibDate"])
            validStart = valids[calibDate][0].isoformat()
            validEnd = valids[calibDate][1].isoformat()
            sql = "UPDATE %s" % table
            sql += " SET validStart='%s', validEnd='%s'" % (validStart, validEnd)
            sql += " WHERE id=%s" % row["id"]
            conn.execute(sql)


class IngestCalibsArgumentParser(ArgumentParser):
    """Argument parser to support ingesting calibration images into the repository"""
    def __init__(self, *args, **kwargs):
        ArgumentParser.__init__(self, *args, **kwargs)
        self.add_argument("-n", "--dry-run", dest="dryrun", action="store_true",
                          default=False, help="Don't perform any action?")
        self.add_argument("--create", action="store_true", help="Create new registry?")
        self.add_argument("--validity", type=int, help="Calibration validity period (days)")
        self.add_argument("--calibType", type=str, default=None,
                          choices=[None, "bias", "dark", "flat", "fringe", "defect"],
                          help="Type of the calibration data to be ingested;" +
                               " if omitted, the type is determined from" +
                               " the file header information")
        self.add_argument("files", nargs="+", help="Names of file")


class IngestCalibsConfig(Config):
    """Configuration for IngestCalibsTask"""
    parse = ConfigurableField(target=CalibsParseTask, doc="File parsing")
    register = ConfigurableField(target=CalibsRegisterTask, doc="Registry entry")


class IngestCalibsTask(IngestTask):
    """Task that generates registry for calibration images"""
    ConfigClass = IngestCalibsConfig
    ArgumentParser = IngestCalibsArgumentParser
    _DefaultName = "ingestCalibs"

    def run(self, args):
        """Ingest all specified files and add them to the registry"""
        calibRoot = args.calib if args.calib is not None else "."
        with self.register.openRegistry(calibRoot, create=args.create, dryrun=args.dryrun) as registry:
            for infile in args.files:
                fileInfo, hduInfoList = self.parse.getInfo(infile)
                if args.calibType is None:
                    calibType = self.parse.getCalibType(infile)
                else:
                    calibType = args.calibType
                if calibType not in self.register.config.tables:
                    self.log.warn("Skipped adding %s of observation type %s to registry" %
                                  (infile, calibType))
                    continue
                for info in hduInfoList:
                    info['path'] = infile
                    if args.validity is not None:
                        try:
                            info['validStart'] = (_convertToDate(info['calibDate']) -
                                                  datetime.timedelta(args.validity)).isoformat()
                            info['validEnd'] = (_convertToDate(info['calibDate']) +
                                                datetime.timedelta(args.validity)).isoformat()
                        except TypeError:
                            self.log.warn("Skipped setting validity period of %s" %
                                          args.validity)
                    self.register.addRow(registry, info, dryrun=args.dryrun,
                                         create=args.create, table=calibType)
            if args.dryrun:
                self.log.info("Would update validity ranges")
            else:
                self.register.updateValidityRanges(registry)
