"""Command-line task and associated config for consolidating QA tables.


The deepCoadd_qa table is a table with QA columns of interest computed
for all filters for which the deepCoadd_obj tables are written.
"""
import os
import pandas as pd

from lsst.pex.config import Config, Field
from lsst.pipe.base import CmdLineTask, ArgumentParser

from .parquetTable import ParquetTable
from .tractQADataIdContainer import TractQADataIdContainer

# Question: is there a way that LSST packages store data files?
ROOT = os.path.abspath(os.path.dirname(__file__))


class ConsolidateQATableConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")


class ConsolidateQATableTask(CmdLineTask):
    """Write patch-merged source tables to a tract-level parquet file
    """
    _DefaultName = "consolidateQATable"
    ConfigClass = ConsolidateQATableConfig

    inputDataset = 'deepCoadd_qa'
    outputDataset = 'deepCoadd_qa_tract'

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)

        parser.add_id_argument("--id", cls.inputDataset,
                               help="data ID, e.g. --id tract=12345",
                               ContainerClass=TractQADataIdContainer)
        return parser

    def runDataRef(self, patchRefList):
        df = pd.concat([patchRef.get().toDataFrame() for patchRef in patchRefList])
        patchRefList[0].put(ParquetTable(dataFrame=df), self.outputDataset)

    def writeMetadata(self, dataRef):
        """No metadata to write.
        """
        pass


class ConsolidateObjectTableConfig(ConsolidateQATableConfig):
    pass


class ConsolidateObjectTableTask(ConsolidateQATableTask):
    _DefaultName = "consolidateObjectTable"
    ConfigClass = ConsolidateObjectTableConfig

    inputDataset = 'objectTable'
    outputDataset = 'objectTable_tract'
