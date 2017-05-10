from __future__ import absolute_import, division, print_function

import os

from lsst.pex.config import ConfigurableField
from lsst.pipe.tasks.ingest import IngestTask, IngestConfig, RegisterTask, RegistryContext, fakeContext
from lsst.daf.persistence.registries import PgsqlRegistry

try:
    import psycopg2 as pgsql
    havePgSql = True
except ImportError:
    havePgSql = False


class PgsqlRegistryContext(RegistryContext):
    """Context manager to provide a pgsql registry
    """
    def __init__(self, registryName, createTableFunc, forceCreateTables):
        """Construct a context manager

        @param registryName: Name of registry file
        @param createTableFunc: Function to create tables
        @param forceCreateTables: Force the (re-)creation of tables?
        """
        self.registryName = registryName
        data = PgsqlRegistry.readYaml(registryName)
        self.conn = pgsql.connect(host=data["host"], port=data["port"], user=data["user"],
                                  password=data["password"], database=data["database"])
        cur = self.conn.cursor()

        # Check for existence of tables
        cur.execute("SELECT relname FROM pg_class WHERE relkind='r' AND relname='raw'")
        rows = cur.fetchall()

        if forceCreateTables or len(rows) == 0:
            # Delete all tables and start over.
            # Not simply doing "DROP SCHEMA" and "CREATE SCHEMA" because of permissions.
            cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            tables = cur.fetchall()
            for tt in tables:
                cur.execute("DROP TABLE %s CASCADE" % tt)
            createTableFunc(self.conn)

    def __exit__(self, excType, excValue, traceback):
        self.conn.commit()
        self.conn.close()
        return False  # Don't suppress any exceptions


class PgsqlRegisterTask(RegisterTask):
    placeHolder = "%s"

    def openRegistry(self, directory, create=False, dryrun=False):
        """Open the registry and return the connection handle.

        @param directory  Directory in which the registry file will be placed
        @param create  Clobber any existing registry and create a new one?
        @param dryrun  Don't do anything permanent?
        @return Database connection
        """
        if dryrun:
            return fakeContext()
        registryName = os.path.join(directory, "registry.pgsql")
        return PgsqlRegistryContext(registryName, self.createTable, create)

    def createTable(self, conn, table=None):
        """Create the registry tables

        One table (typically 'raw') contains information on all files, and the
        other (typically 'raw_visit') contains information on all visits.

        This method is required because there's a slightly different syntax
        compared to SQLite (FLOAT instead of DOUBLE, SERIAL instead of
        AUTOINCREMENT).

        @param conn    Database connection
        @param table   Name of table to create in database
        """
        if table is None:
            table = self.config.table

        typeMap = {'int': 'INT',
                   'double': 'FLOAT',  # Defaults to double precision
                   }

        cur = conn.cursor()
        cmd = "CREATE TABLE %s (id SERIAL NOT NULL PRIMARY KEY, " % table
        cmd += ",".join(["%s %s" % (col, typeMap.get(colType.lower(), 'text')) for
                         col, colType in self.config.columns.items()])
        if len(self.config.unique) > 0:
            cmd += ", UNIQUE(" + ",".join(self.config.unique) + ")"
        cmd += ")"
        cur.execute(cmd)

        cmd = "CREATE TABLE %s_visit (" % self.config.table
        cmd += ",".join(["%s %s" % (col, typeMap.get(self.config.columns[col].lower(), 'TEXT')) for
                         col in self.config.visit])
        cmd += ", UNIQUE(" + ",".join(set(self.config.visit).intersection(set(self.config.unique))) + ")"
        cmd += ")"
        cur.execute(cmd)
        del cur
        conn.commit()


class PgsqlIngestConfig(IngestConfig):
    register = ConfigurableField(target=PgsqlRegisterTask, doc="Registry entry")


class PgsqlIngestTask(IngestTask):
    ConfigClass = PgsqlIngestConfig
