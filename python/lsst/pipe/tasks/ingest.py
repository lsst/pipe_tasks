import os
import shutil
import tempfile
try:
    import sqlite3
except ImportError:
    # try external pysqlite package; deprecated
    import sqlite as sqlite3
from fnmatch import fnmatch
from glob import glob

from lsst.pex.config import Config, Field, DictField, ListField, ConfigurableField
import lsst.pex.exceptions
from lsst.pipe.base import Task, ArgumentParser
import lsst.afw.image as afwImage

class IngestArgumentParser(ArgumentParser):
    """Argument parser to support ingesting images into the image repository"""
    def __init__(self, *args, **kwargs):
        super(IngestArgumentParser, self).__init__(*args, **kwargs)
        self.add_argument("-n", "--dry-run", dest="dryrun", action="store_true", default=False,
                          help="Don't perform any action?")
        self.add_argument("--mode", choices=["move", "copy", "link", "skip"], default="link",
                          help="Mode of delivering the files to their destination")
        self.add_argument("--create", action="store_true", help="Create new registry (clobber old)?")
        self.add_id_argument("--badId", "raw", "Data identifier for bad data", doMakeDataRefList=False)
        self.add_argument("--badFile", nargs="*", default=[],
                          help="Names of bad files (no path; wildcards allowed)")
        self.add_argument("files", nargs="+", help="Names of file")

class ParseConfig(Config):
    """Configuration for ParseTask"""
    translation = DictField(keytype=str, itemtype=str, default={},
                            doc="Translation table for property --> header")
    translators = DictField(keytype=str, itemtype=str, default={},
                            doc="Properties and name of translator method")
    defaults = DictField(keytype=str, itemtype=str, default={},
                         doc="Default values if header is not present")
    hdu = Field(dtype=int, default=0, doc="HDU to read for metadata")
    extnames = ListField(dtype=str, default=[], doc="Extension names to search for")

class ParseTask(Task):
    """Task that will parse the filename and/or its contents to get the required information
    for putting the file in the correct location and populating the registry."""
    ConfigClass = ParseConfig

    def getInfo(self, filename):
        """Get information about the image from the filename and its contents

        Here, we open the image and parse the header, but one could also look at the filename itself
        and derive information from that, or set values from the configuration.

        @param filename    Name of file to inspect
        @return File properties; list of file properties for each extension
        """
        md = afwImage.readMetadata(filename, self.config.hdu)
        phuInfo = self.getInfoFromMetadata(md)
        if len(self.config.extnames) == 0:
            # No extensions to worry about
            return phuInfo, [phuInfo]
        # Look in the provided extensions
        extnames = set(self.config.extnames)
        extnum = 1
        infoList = []
        while len(extnames) > 0:
            extnum += 1
            try:
                md = afwImage.readMetadata(filename, extnum)
            except:
                self.log.warn("Error reading %s extensions %s" % (filename, extnames))
                break
            ext = self.getExtensionName(md)
            if ext in extnames:
                infoList.append(self.getInfoFromMetadata(md, info=phuInfo.copy()))
                extnames.discard(ext)
        return phuInfo, infoList

    @staticmethod
    def getExtensionName(md):
        """ Get the name of an extension.
        @param md: PropertySet like one obtained from afwImage.readMetadata)
        @return Name of the extension if it exists.  None otherwise.
        """
        try:
            # This returns a tuple
            ext = md.get("EXTNAME")
            return ext[1]
        except lsst.pex.exceptions.Exception:
            return None

    def getInfoFromMetadata(self, md, info={}):
        """Attempt to pull the desired information out of the header

        This is done through two mechanisms:
        * translation: a property is set directly from the relevant header keyword
        * translator: a property is set with the result of calling a method

        The translator methods receive the header metadata and should return the
        appropriate value, or None if the value cannot be determined.

        @param md      FITS header
        @param info    File properties, to be supplemented
        @return info
        """
        for p, h in self.config.translation.iteritems():
            if md.exists(h):
                value = md.get(h)
                if isinstance(value, basestring):
                    value = value.strip()
                info[p] = value
            elif p in self.config.defaults:
                info[p] = self.config.defaults[p]
            else:
                self.log.warn("Unable to find value for %s (derived from %s)" % (p, h))
        for p, t in self.config.translators.iteritems():
            func = getattr(self, t)
            try:
                value = func(md)
            except:
                value = None
            if value is not None:
                info[p] = value
        return info

    def translate_date(self, md):
        """Convert a full DATE-OBS to a mere date

        Besides being an example of a translator, this is also generally useful.
        It will only be used if listed as a translator in the configuration.
        """
        date = md.get("DATE-OBS").strip()
        c = date.find("T")
        if c > 0:
            date = date[:c]
        return date

    def translate_filter(self, md):
        """Translate a full filter description into a mere filter name

        Besides being an example of a translator, this is also generally useful.
        It will only be used if listed as a translator in the configuration.
        """
        filterName = md.get("FILTER").strip()
        filterName = filterName.strip()
        c = filterName.find(" ")
        if c > 0:
            filterName = filterName[:c]
        return filterName

    def getDestination(self, butler, info, filename):
        """Get destination for the file

        @param butler      Data butler
        @param info        File properties, used as dataId for the butler
        @param filename    Input filename
        @return Destination filename
        """
        raw = butler.get("raw_filename", info)[0]
        # Ensure filename is devoid of cfitsio directions about HDUs
        c = raw.find("[")
        if c > 0:
            raw = raw[:c]
        return raw

class RegisterConfig(Config):
    """Configuration for the RegisterTask"""
    table = Field(dtype=str, default="raw", doc="Name of table")
    columns = DictField(keytype=str, itemtype=str, doc="List of columns for raw table, with their types",
                        itemCheck=lambda x: x in ("text", "int", "double"),
                        default={'object':  'text',
                                 'visit':   'int',
                                 'ccd':     'int',
                                 'filter':  'text',
                                 'date':    'text',
                                 'taiObs':  'text',
                                 'expTime': 'double',
                                 },
                        )
    unique = ListField(dtype=str, doc="List of columns to be declared unique for the table",
                       default=["visit", "ccd"])
    visit = ListField(dtype=str, default=["visit", "object", "date", "filter"],
                      doc="List of columns for raw_visit table")
    ignore = Field(dtype=bool, default=False, doc="Ignore duplicates in the table?")
    permissions = Field(dtype=int, default=0664, doc="Permissions mode for registry") # octal 664 = rw-rw-r--

class RegistryContext(object):
    """Context manager to provide a registry

    An existing registry is copied, so that it may continue
    to be used while we add to this new registry.  Finally,
    the new registry is moved into the right place.
    """
    def __init__(self, registryName, createTableFunc, forceCreateTables, permissions):
        """Construct a context manager

        @param registryName: Name of registry file
        @param createTableFunc: Function to create tables
        """
        self.registryName = registryName
        self.permissions = permissions

        updateFile = tempfile.NamedTemporaryFile(prefix=registryName, dir=os.path.dirname(self.registryName),
                                                 delete=False)
        self.updateName = updateFile.name

        haveTable = False
        if os.path.exists(registryName):
            assertCanCopy(registryName, self.updateName)
            os.chmod(self.updateName, os.stat(registryName).st_mode)
            shutil.copyfile(registryName, self.updateName)
            haveTable = True

        self.conn = sqlite3.connect(self.updateName)
        if not haveTable or forceCreateTables:
            createTableFunc(self.conn)
        os.chmod(self.updateName, self.permissions)

    def __enter__(self):
        """Provide the 'as' value"""
        return self.conn

    def __exit__(self, excType, excValue, traceback):
        self.conn.commit()
        self.conn.close()
        if excType is None:
            assertCanCopy(self.updateName, self.registryName)
            if os.path.exists(self.registryName):
                os.unlink(self.registryName)
            os.rename(self.updateName, self.registryName)
            os.chmod(self.registryName, self.permissions)
        return False # Don't suppress any exceptions

class RegisterTask(Task):
    """Task that will generate the registry for the Mapper"""
    ConfigClass = RegisterConfig

    def openRegistry(self, directory, create=False, dryrun=False, name="registry.sqlite3"):
        """Open the registry and return the connection handle.

        @param directory  Directory in which the registry file will be placed
        @param create  Clobber any existing registry and create a new one?
        @param dryrun  Don't do anything permanent?
        @param name    Filename of the registry
        @return Database connection
        """
        if dryrun:
            from contextlib import contextmanager
            @contextmanager
            def fakeContext():
                yield
            return fakeContext()

        registryName = os.path.join(directory, name)
        context = RegistryContext(registryName, self.createTable, create, self.config.permissions)
        return context

    def createTable(self, conn, table=None):
        """Create the registry tables

        One table (typically 'raw') contains information on all files, and the
        other (typically 'raw_visit') contains information on all visits.

        @param conn    Database connection
        @param table   Name of table to create in database
        """
        if table is None:
            table = self.config.table
        cmd = "create table %s (id integer primary key autoincrement, " % table
        cmd += ",".join([("%s %s" % (col, colType)) for col,colType in self.config.columns.items()])
        if len(self.config.unique) > 0:
            cmd += ", unique(" + ",".join(self.config.unique) + ")"
        cmd += ")"
        conn.execute(cmd)

        cmd = "create table %s_visit (" % table
        cmd += ",".join([("%s %s" % (col, self.config.columns[col])) for col in self.config.visit])
        cmd += ", unique(" + ",".join(set(self.config.visit).intersection(set(self.config.unique))) + ")"
        cmd += ")"
        conn.execute(cmd)

        conn.commit()

    def check(self, conn, info, table=None):
        """Check for the presence of a row already

        Not sure this is required, given the 'ignore' configuration option.
        """
        if table is None:
            table = self.config.table
        if self.config.ignore or len(self.config.unique) == 0:
            return False # Our entry could already be there, but we don't care
        cursor = conn.cursor()
        sql = "SELECT COUNT(*) FROM %s WHERE " % table
        sql += " AND ".join(["%s=?" % col for col in self.config.unique])
        values = [info[col] for col in self.config.unique]

        cursor.execute(sql, values)
        if cursor.fetchone()[0] > 0:
            return True
        return False

    def addRow(self, conn, info, dryrun=False, create=False, table=None):
        """Add a row to the file table (typically 'raw').

        @param conn    Database connection
        @param info    File properties to add to database
        @param table   Name of table in database
        """
        if table is None:
            table = self.config.table
        sql = "INSERT"
        if self.config.ignore:
            sql += " OR IGNORE"
        sql += " INTO %s VALUES (NULL" % table
        sql += ", ?" * len(self.config.columns)
        sql += ")"
        values = [info[col] for col in self.config.columns]
        if dryrun:
            print "Would execute: '%s' with %s" % (sql, ",".join([str(value) for value in values]))
        else:
            conn.execute(sql, values)

    def addVisits(self, conn, dryrun=False, table=None):
        """Generate the visits table (typically 'raw_visits') from the
        file table (typically 'raw').

        @param conn    Database connection
        @param table   Name of table in database
        """
        if table is None:
            table = self.config.table
        sql = "INSERT OR IGNORE INTO %s_visit SELECT DISTINCT " % table
        sql += ",".join(self.config.visit)
        sql += " FROM %s" % table
        if dryrun:
            print "Would execute: %s" % sql
        else:
            conn.execute(sql)


class IngestConfig(Config):
    """Configuration for IngestTask"""
    parse = ConfigurableField(target=ParseTask, doc="File parsing")
    register = ConfigurableField(target=RegisterTask, doc="Registry entry")
    allowError = Field(dtype=bool, default=False, doc="Allow error in ingestion?")
    clobber = Field(dtype=bool, default=False, doc="Clobber existing file?")

class IngestTask(Task):
    """Task that will ingest images into the data repository"""
    ConfigClass = IngestConfig
    ArgumentParser = IngestArgumentParser
    _DefaultName = "ingest"

    def __init__(self, *args, **kwargs):
        super(IngestTask, self).__init__(*args, **kwargs)
        self.makeSubtask("parse")
        self.makeSubtask("register")

    @classmethod
    def parseAndRun(cls):
        """Parse the command-line arguments and run the Task"""
        config = cls.ConfigClass()
        parser = cls.ArgumentParser(name=cls._DefaultName)
        args = parser.parse_args(config)
        task = cls(config=args.config)
        task.run(args)

    def ingest(self, infile, outfile, mode="move", dryrun=False):
        """Ingest a file into the image repository.

        @param infile  Name of input file
        @param outfile Name of output file (file in repository)
        @param mode    Mode of ingest (copy/link/move/skip)
        @param dryrun  Only report what would occur?
        @param Success boolean
        """
        if mode == "skip":
            return True
        if dryrun:
            self.log.info("Would %s from %s to %s" % (mode, infile, outfile))
            return True
        try:
            outdir = os.path.dirname(outfile)
            if not os.path.isdir(outdir):
                try:
                    os.makedirs(outdir)
                except:
                    # Silently ignore mkdir failures due to race conditions
                    if not os.path.isdir(outdir):
                        raise
            if self.config.clobber and os.path.lexists(outfile):
                os.unlink(outfile)
            if mode == "copy":
                assertCanCopy(infile, outfile)
                shutil.copyfile(infile, outfile)
            elif mode == "link":
                os.symlink(os.path.abspath(infile), outfile)
            elif mode == "move":
                assertCanCopy(infile, outfile)
                os.rename(infile, outfile)
            else:
                raise AssertionError("Unknown mode: %s" % mode)
            print "%s --<%s>--> %s" % (infile, mode, outfile)
        except Exception, e:
            self.log.warn("Failed to %s %s to %s: %s" % (mode, infile, outfile, e))
            if not self.config.allowError:
                raise
            return False
        return True

    def isBadFile(self, filename, badFileList):
        """Return whether the file qualifies as bad

        We match against the list of bad file patterns.
        """
        filename = os.path.basename(filename)
        if not badFileList:
            return False
        for badFile in badFileList:
            if fnmatch(filename, badFile):
                return True
        return False

    def isBadId(self, info, badIdList):
        """Return whether the file information qualifies as bad

        We match against the list of bad data identifiers.
        """
        if not badIdList:
            return False
        for badId in badIdList:
            if all(info[key] == value for key, value in badId.iteritems()):
                return True
        return False

    def run(self, args):
        """Ingest all specified files and add them to the registry"""
        filenameList = sum([glob(filename) for filename in args.files], [])
        root = args.butler.repository._mapper.root
        context = self.register.openRegistry(root, create=args.create, dryrun=args.dryrun)
        with context as registry:
            for infile in filenameList:
                if self.isBadFile(infile, args.badFile):
                    self.log.info("Skipping declared bad file %s" % infile)
                    continue
                fileInfo, hduInfoList = self.parse.getInfo(infile)
                if self.isBadId(fileInfo, args.badId.idList):
                    self.log.info("Skipping declared bad file %s: %s" % (infile, fileInfo))
                    continue
                if self.register.check(registry, fileInfo):
                    self.log.warn("%s: already ingested: %s" % (infile, fileInfo))
                outfile = self.parse.getDestination(args.butler, fileInfo, infile)
                ingested = self.ingest(infile, outfile, mode=args.mode, dryrun=args.dryrun)
                if not ingested:
                    continue
                for info in hduInfoList:
                    self.register.addRow(registry, info, dryrun=args.dryrun, create=args.create)
            self.register.addVisits(registry, dryrun=args.dryrun)

def assertCanCopy(fromPath, toPath):
    """Can I copy a file?  Raise an exception is space constraints not met.

    @param fromPath    Path from which the file is being copied
    @param toPath      Path to which the file is being copied
    """
    req = os.stat(fromPath).st_size
    st = os.statvfs(os.path.dirname(toPath))
    avail = st.f_bavail * st.f_frsize
    if avail < req:
        raise RuntimeError("Insufficient space: %d vs %d" % (req, avail))
