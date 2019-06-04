# This file is part of qa_explorer.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Command-line task and associated config for writing deepCoadd_obj table.

The deepCoadd_obj table is a merged catalog of deepCoadd_meas, deepCoadd_forced_src and deepCoadd_ref
catalogs for multiple bands.
"""
import functools
import pandas as pd

from .parquetTable import ParquetTable

from lsst.pex.config import Config, Field, ListField
from lsst.pipe.base import CmdLineTask
from lsst.pipe.tasks.multiBandUtils import makeMergeArgumentParser, MergeSourcesRunner


class WriteObjectTableConfig(Config):
    priorityList = ListField(dtype=str, default=['HSC-G', 'HSC-R', 'HSC-I', 'HSC-Z', 'HSC-Y'],
                             doc="Priority-ordered list of bands for the merge.")
    engine = Field(dtype=str, default="pyarrow", doc="Parquet engine for writing (pyarrow or fastparquet)")
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")

    def validate(self):
        Config.validate(self)
        if len(self.priorityList) == 0:
            raise RuntimeError("No priority list provided")


class WriteObjectTableTask(CmdLineTask):
    """Write filter-merged source tables to parquet
    """
    _DefaultName = "writeObjectTable"
    ConfigClass = WriteObjectTableConfig
    RunnerClass = MergeSourcesRunner

    # Names of table datasets to be merged
    inputDatasets = ('forced_src', 'meas', 'ref')

    # Tag of output dataset written by `MergeSourcesTask.write`
    outputDataset = 'obj'

    def __init__(self, butler=None, schema=None, **kwargs):
        # It is a shame that this class can't use the default init for CmdLineTask
        # But to do so would require its own special task runner, which is many
        # more lines of specialization, so this is how it is for now
        CmdLineTask.__init__(self, **kwargs)

    def runDataRef(self, patchRefList):
        """!
        @brief Merge coadd sources from multiple bands. Calls @ref `run` which must be defined in
        subclasses that inherit from MergeSourcesTask.
        @param[in] patchRefList list of data references for each filter
        """
        catalogs = dict(self.readCatalog(patchRef) for patchRef in patchRefList)
        mergedCatalog = self.run(catalogs, patchRefList[0])
        self.write(patchRefList[0], mergedCatalog)

    def getSchemaCatalogs(self, *args, **kwargs):
        """Not using this function, but it must return a dictionary.
        """
        return {}

    @classmethod
    def _makeArgumentParser(cls):
        """Create a suitable ArgumentParser.

        We will use the ArgumentParser to get a list of data
        references for patches; the RunnerClass will sort them into lists
        of data references for the same patch.

        References first of self.inputDatasets, rather than
        self.inputDataset
        """
        return makeMergeArgumentParser(cls._DefaultName, cls.inputDatasets[0])

    def readCatalog(self, patchRef):
        """Read input catalogs

        Read all the input datasets given by the 'inputDatasets'
        attribute.

        Parameters
        ----------
        patchRef :
            Data reference for patch

        Returns
        -------
        Tuple consisting of filter name and a dict of catalogs, keyed by dataset
        name
        """
        filterName = patchRef.dataId["filter"]
        catalogDict = {}
        for dataset in self.inputDatasets:
            catalog = patchRef.get(self.config.coaddName + "Coadd_" + dataset, immediate=True)
            self.log.info("Read %d sources from %s for filter %s: %s" %
                          (len(catalog), dataset, filterName, patchRef.dataId))
            catalogDict[dataset] = catalog
        return filterName, catalogDict

    def run(self, catalogs, patchRef):
        """Merge multiple catalogs.

        Parameters
        ----------
        catalogs : `dict`
            Mapping from filter names to dict of catalogs.

        Returns
        -------
        catalog : `lsst.qa.explorer.table.ParquetTable`
            Merged dataframe, with each column prefixed by
            `filter_tag(filt)`, wrapped in the parquet writer shim class.
        """

        dfs = []
        for filt, tableDict in catalogs.items():
            for dataset, table in tableDict.items():
                # Convert afwTable to pandas DataFrame
                df = table.asAstropy().to_pandas().set_index('id', drop=True)

                # Sort columns by name, to ensure matching schema among patches
                df = df.reindex(sorted(df.columns), axis=1)
                df['tractId'] = patchRef.dataId['tract']
                df['patchId'] = patchRef.dataId['patch']

                # Make columns a 3-level MultiIndex
                df.columns = pd.MultiIndex.from_tuples([(dataset, filt, c) for c in df.columns],
                                                       names=('dataset', 'filter', 'column'))
                dfs.append(df)

        catalog = functools.reduce(lambda d1, d2: d1.join(d2), dfs)
        return ParquetTable(dataFrame=catalog)

    def write(self, patchRef, catalog):
        """!
        @brief Write the output.
        @param[in]  patchRef   data reference for patch
        @param[in]  catalog    catalog
        We write as the dataset provided by the 'outputDataset'
        class variable.
        """
        patchRef.put(catalog, self.config.coaddName + "Coadd_" + self.outputDataset)
        # since the filter isn't actually part of the data ID for the dataset we're saving,
        # it's confusing to see it in the log message, even if the butler simply ignores it.
        mergeDataId = patchRef.dataId.copy()
        del mergeDataId["filter"]
        self.log.info("Wrote merged catalog: %s" % (mergeDataId,))

    def writeMetadata(self, dataRefList):
        """!
        @brief No metadata to write, and not sure how to write it for a list of dataRefs.
        """
        pass
