# This file is part of pipe_tasks.
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

__all__ = ['TableColumnSummaryConnections',
           'TableColumnSummaryConfig',
           'TableColumnSummaryTask']

import numpy as np
import pandas as pd
import lsst.pipe.base as pipeBase

from lsst.verify import Measurement
from astropy import units as u

from lsst.pex.config import Field

import datetime
import logging

from lsst.analysis.tools.interfaces import MetricMeasurementBundle


def _timestampValidator(value: str) -> bool:
    if value in ("reference_package_timestamp", "run_timestamp", "current_timestamp", "dataset_timestamp"):
        return True
    elif "explicit_timestamp" in value:
        try:
            _, splitTime = value.split(":")
        except ValueError:
            logging.error(
                "Explicit timestamp must be given in the format 'explicit_timestamp:datetime', "
                r"where datetime is given in the form '%Y%m%dT%H%M%S%z"
            )
            return False
        try:
            datetime.datetime.strptime(splitTime, r"%Y%m%dT%H%M%S%z")
        except ValueError:
            # This is explicitly chosen to be an f string as the string
            # contains control characters.
            logging.error(
                f"The supplied datetime {splitTime} could not be parsed correctly into "
                r"%Y%m%dT%H%M%S%z format"
            )
            return False
        return True
    else:
        return False


class TableColumnSummaryConnections(pipeBase.PipelineTaskConnections,
                                    dimensions=('skymap', 'tract',),
                                    defaultTemplates={}):
    object = pipeBase.connectionTypes.Input(
        doc='object table in parquet format, per tract',
        name='object',
        storageClass='DataFrame',
        dimensions=('skymap', 'tract'),
        deferLoad=True,
        multiple=True,
    )
    table_column_summary = pipeBase.connectionTypes.Output(
        doc='Summary of columns in object table',
        name='object_summary_metrics',
        storageClass='MetricMeasurementBundle',
        dimensions=('skymap', 'tract'),
    )


class TableColumnSummaryConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=TableColumnSummaryConnections):
    """Configuration for TableColumnSummaryTask."""

    dataset_identifier = Field[str](doc="An identifier to be associated with output Metrics", optional=True)
    reference_package = Field[str](
        doc="A package whose version, at the time of metric upload to a "
        "time series database, will be converted to a timestamp of when "
        "that version was produced",
        default="lsst_distrib",
    )
    timestamp_version = Field[str](
        doc="Which time stamp should be used as the reference timestamp for a "
        "metric in a time series database, valid values are; "
        "reference_package_timestamp, run_timestamp, current_timestamp, "
        "dataset_timestamp and explicit_timestamp:datetime where datetime is "
        "given in the form %Y%m%dT%H%M%S%z",
        default="run_timestamp",
        check=_timestampValidator,
    )


class TableColumnSummaryTask(pipeBase.PipelineTask):
    """Create a summary of the columns for a given table.
    """
    ConfigClass = TableColumnSummaryConfig
    _DefaultName = 'tableColumnSummary'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        input_ref_dict = butlerQC.get(inputRefs)

        tract = butlerQC.quantum.dataId['tract']

        object_refs = input_ref_dict['object']

        self.log.info('Running with %d object dataRefs',
                      len(object_refs))

        object_ref_dict_temp = {object_ref.dataId['tract']: object_ref for
                                object_ref in object_refs}

        # TODO: Sort by tract until DM-31701 is done and we have deterministic dataset ordering.
        object_ref_dict = {tract: object_ref_dict_temp[tract] for
                           tract in sorted(object_ref_dict_temp.keys())}

        bundle = self.run(tract, object_ref_dict)

        butlerQC.put(bundle, outputRefs.table_column_summary)

    def run(self, tract, object_ref_dict):
        """Run the summarize table columns task.

        Parameters
        ----------
        tract : `int`
            tract number.
        object_ref_dict : `dict`
            Dictionary of object refs.  Key is tract, value is dataref.

        Returns
        -------
        bundle : `lsst.analysis.tools.interfaces.MetricMeasurementBundle`
            Metric measurement bundle containing table summary information.
        """

        df_summary = self._make_stuff(object_ref_dict)

        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', None):
            print(df_summary)

        df_summary_temp = df_summary.copy()
        df_summary_temp['colname'] = df_summary.index
        outputFileName = """objectSummary.%d.csv""" % (tract)
        df_summary_temp.to_csv(outputFileName, index=True)

        # change variable from index_list, column_list to something less confusing:
        summary_column_list = df_summary.index.tolist()
        summary_stats_list = df_summary.columns.tolist()

        ###########################################
        # Resolve dimensions
        # do this for object table
        ###########################################

        metricsList = []
        for summary_column_name in summary_column_list:
            for summary_stats_name in summary_stats_list:
                metric_name = f"{summary_column_name}_{summary_stats_name}"
                metric_value = df_summary.loc[summary_column_name, summary_stats_name]
                # Eventually want to add dimensions, once dimensions are included in the
                # butlerQC parquet files...
                metric = Measurement(metric_name, metric_value*u.dimensionless_unscaled)
                metricsList.append(metric)

        bundle = MetricMeasurementBundle(
            dataset_identifier=self.config.dataset_identifier,
            reference_package=self.config.reference_package,
            timestamp_version=self.config.timestamp_version,
        )

        bundle['object'] = metricsList

        print(bundle)

        return bundle

    def _make_stuff(self, object_ref_dict):
        """Make a catalog of all the stuff.

        Parameters
        ----------
        object_ref_dict : `dict`
            Dictionary of object refs.  Key is tract, value is dataref.

        Returns
        -------
        stuff_cat : `np.ndarray`
            Catalog of stuff.
        """

        # Following thanks to Claude-3.5-Sonnet via Poe.com...

        # Retrieve object for this tract & patch...

        # Combine object refs into a concatenated pandas DataFrame.
        #  (This might not be necessary, since we would generally considering a single tract.)
        dfs = []
        for tract in object_ref_dict:
            object_ref = object_ref_dict[tract]
            df = object_ref.get()
            # Only consider entries for which detect_isPrimary is True...
            try:
                df = df[(df['detect_isPrimary'] is True)]
            except Exception:
                print('detect_isPrimary is not an avaiable column.  Ignoring and moving on.')
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        df.reset_index(inplace=True)

        # Get list of numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns

        # Create summary statistics
        summary_stats = {
            'percent_01': df[numeric_cols].quantile(0.01),
            'percent_50': df[numeric_cols].quantile(0.50),
            'percent_99': df[numeric_cols].quantile(0.99),
            'total_rows': df[numeric_cols].count() + df[numeric_cols].isna().sum(),
            'nan_count': df[numeric_cols].isna().sum(),
            'nan_fraction': df[numeric_cols].isna().sum()
            / (df[numeric_cols].count() + df[numeric_cols].isna().sum())
        }

        # Create summary DataFrame
        df_summary = pd.DataFrame(summary_stats).round(3)

        # Return summary DataFrame
        return df_summary
