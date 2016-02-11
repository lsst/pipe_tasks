import collections
import datetime
import itertools
import sqlite3
import lsst.afw.image as afwImage
from lsst.pex.config import Config, Field, ListField, ConfigurableField
from lsst.pipe.base import ArgumentParser
from lsst.pipe.tasks.ingest import RegisterTask, ParseTask, RegisterConfig, IngestTask


def _convertToDate(dateString):
    """Convert a string into a date object"""
    return datetime.datetime.strptime(dateString, "%Y-%m-%d").date()


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
    tables = ListField(dtype=str, default=["bias", "dark", "flat", "fringe"], doc="Names of tables")
    calibDate = Field(dtype=str, default="calibDate", doc="Name of column for calibration date")
    validStart = Field(dtype=str, default="validStart", doc="Name of column for validity start")
    validEnd = Field(dtype=str, default="validEnd", doc="Name of column for validity stop")
    detector = ListField(dtype=str, default=["filter", "ccd"],
                         doc="Columns that identify individual detectors")
    validityUntilSuperseded = ListField(dtype=str, default=["defect"],
                                        doc="Tables for which to set validity for a calib from when it is "
                                        "taken until it is superseded by the next; validity in other tables "
                                        "is calculated by applying the validity range.")

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

    def addRow(self, conn, info, *args, **kwargs):
        """Add a row to the file table"""
        info[self.config.validStart] = None
        info[self.config.validEnd] = None
        RegisterTask.addRow(self, conn, info, *args, **kwargs)

    def updateValidityRanges(self, conn, validity):
        """Loop over all tables, filters, and ccdnums,
        and update the validity ranges in the registry.

        @param conn: Database connection
        @param validity: Validity range (days)
        """
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        for table in self.config.tables:
            sql = "SELECT DISTINCT %s FROM %s" % (", ".join(self.config.detector), table)
            cursor.execute(sql)
            rows = cursor.fetchall()
            for row in rows:
                self.fixSubsetValidity(conn, table, row, validity)

    def fixSubsetValidity(self, conn, table, detectorData, validity):
        """Update the validity ranges among selected rows in the registry.

        For defects, the products are valid from their start date until
        they are superseded by subsequent defect data.
        For other calibration products, the validity ranges are checked and
        if there are overlaps, a midpoint is used to fix the overlaps,
        so that the calibration data with whose date is nearest the date
        of the observation is used.

        @param conn: Database connection
        @param table: Name of table to be selected
        @param detectorData: Values identifying a detector (from columns in self.config.detector)
        @param validity: Validity range (days)
        """
        columns = ", ".join([self.config.calibDate, self.config.validStart, self.config.validEnd])
        sql = "SELECT id, %s FROM %s" % (columns, table)
        sql += " WHERE " + " AND ".join(col + "=?" for col in self.config.detector)
        sql += " ORDER BY " + self.config.calibDate
        cursor = conn.cursor()
        cursor.execute(sql, detectorData)
        rows = cursor.fetchall()

        try:
            valids = collections.OrderedDict([(_convertToDate(row[self.config.calibDate]), [None, None]) for
                                              row in rows])
        except Exception as e:
            det = " ".join("%s=%s" % (k, v) for k, v in zip(self.config.detector, detectorData))
            self.log.warn("Skipped setting the validity overlaps for %s %s: missing calibration dates" %
                          (table, det))
            return
        dates = valids.keys()
        if table in self.config.validityUntilSuperseded:
            # A calib is valid until it is superseded
            for thisDate, nextDate in itertools.izip(dates[:-1], dates[1:]):
                valids[thisDate][0] = thisDate
                valids[thisDate][1] = nextDate - datetime.timedelta(1)
            valids[dates[-1]][0] = dates[-1]
            valids[dates[-1]][1] = _convertToDate("2037-12-31")  # End of UNIX time
        else:
            # A calib is valid within the validity range (in days) specified.
            for dd in dates:
                valids[dd] = [dd - datetime.timedelta(validity), dd + datetime.timedelta(validity)]
            # Fix the dates so that they do not overlap, which can cause the butler to find a
            # non-unique calib.
            midpoints = [t1 + (t2 - t1)//2 for t1, t2 in itertools.izip(dates[:-1], dates[1:])]
            for i, (date, midpoint) in enumerate(itertools.izip(dates[:-1], midpoints)):
                if valids[date][1] > midpoint:
                    nextDate = dates[i + 1]
                    valids[nextDate][0] = midpoint + datetime.timedelta(1)
                    valids[date][1] = midpoint
            del midpoints
        del dates
        # Update the validity data in the registry
        for row in rows:
            calibDate = _convertToDate(row[self.config.calibDate])
            validStart = valids[calibDate][0].isoformat()
            validEnd = valids[calibDate][1].isoformat()
            sql = "UPDATE %s" % table
            sql += " SET %s=?, %s=?" % (self.config.validStart, self.config.validEnd)
            sql += " WHERE id=?"
            conn.execute(sql, (validStart, validEnd, row["id"]))


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
                    self.log.warn("Skipped adding %s of observation type '%s' to registry" %
                                  (infile, calibType))
                    continue
                for info in hduInfoList:
                    self.register.addRow(registry, info, dryrun=args.dryrun,
                                         create=args.create, table=calibType)
            if not args.dryrun:
                self.register.updateValidityRanges(registry, args.validity)
            else:
                self.log.info("Would update validity ranges here, but dryrun")
