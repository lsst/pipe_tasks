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

__all__ = ["PropagateSourceFlagsConfig", "PropagateSourceFlagsTask"]

import numpy as np

from smatch.matcher import Matcher

import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase


class PropagateSourceFlagsConfig(pexConfig.Config):
    """Configuration for propagating source flags to coadd objects."""
    source_flags = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        default={
            "calib_astrometry_used": 0.2,
            "calib_photometry_used": 0.2,
            "calib_photometry_reserved": 0.2
        },
        doc=("Source flags to propagate, with the threshold of relative occurrence "
             "(valid range: [0-1]). Coadd object will have flag set if fraction "
             "of input visits in which it is flagged is greater than the threshold."),
    )
    finalized_source_flags = pexConfig.DictField(
        keytype=str,
        itemtype=float,
        default={
            "calib_psf_candidate": 0.2,
            "calib_psf_used": 0.2,
            "calib_psf_reserved": 0.2
        },
        doc=("Finalized source flags to propagate, with the threshold of relative "
             "occurrence (valid range: [0-1]). Coadd object will have flag set if "
             "fraction of input visits in which it is flagged is greater than the "
             "threshold."),
    )
    x_column = pexConfig.Field(
        doc="Name of column with source x position (sourceTable_visit).",
        dtype=str,
        default="x",
    )
    y_column = pexConfig.Field(
        doc="Name of column with source y position (sourceTable_visit).",
        dtype=str,
        default="y",
    )
    finalized_x_column = pexConfig.Field(
        doc="Name of column with source x position (finalized_src_table).",
        dtype=str,
        default="slot_Centroid_x",
    )
    finalized_y_column = pexConfig.Field(
        doc="Name of column with source y position (finalized_src_table).",
        dtype=str,
        default="slot_Centroid_y",
    )
    match_radius = pexConfig.Field(
        dtype=float,
        default=0.2,
        doc="Source matching radius (arcsec)"
    )

    def validate(self):
        super().validate()

        if set(self.source_flags).intersection(set(self.finalized_source_flags)):
            source_flags = self.source_flags.keys()
            finalized_source_flags = self.finalized_source_flags.keys()
            raise ValueError(f"The set of source_flags {source_flags} must not overlap "
                             f"with the finalized_source_flags {finalized_source_flags}")


class PropagateSourceFlagsTask(pipeBase.Task):
    """Task to propagate source flags to coadd objects.

    Flagged sources may come from a mix of two different types of source catalogs.
    The source_table catalogs from ``CalibrateTask`` contain flags for the first
    round of astromety/photometry/psf fits.
    The finalized_source_table catalogs from ``FinalizeCalibrationTask`` contain
    flags from the second round of psf fitting.
    """
    ConfigClass = PropagateSourceFlagsConfig

    def __init__(self, schema, **kwargs):
        pipeBase.Task.__init__(self, **kwargs)

        self.schema = schema
        for f in self.config.source_flags:
            self.schema.addField(f, type="Flag", doc="Propagated from sources")
        for f in self.config.finalized_source_flags:
            self.schema.addField(f, type="Flag", doc="Propagated from finalized sources")

    def run(self, coadd_object_cat, ccd_inputs,
            source_table_handle_dict=None, finalized_source_table_handle_dict=None):
        """Propagate flags from single-frame sources to coadd objects.

        Flags are only propagated if a configurable percentage of the sources
        are matched to the coadd objects.  This task will match both "plain"
        source flags and "finalized" source flags.

        Parameters
        ----------
        coadd_object_cat : `lsst.afw.table.SourceCatalog`
            Table of coadd objects.
        ccd_inputs : `lsst.afw.table.ExposureCatalog`
            Table of single-frame inputs to coadd.
        source_table_handle_dict : `dict` [`int`: `lsst.daf.butler.DeferredDatasetHandle`], optional
            Dict for sourceTable_visit handles (key is visit).  May be None if
            ``config.source_flags`` has no entries.
        finalized_source_table_handle_dict : `dict` [`int`:
                                                     `lsst.daf.butler.DeferredDatasetHandle`], optional
            Dict for finalized_src_table handles (key is visit).  May be None if
            ``config.finalized_source_flags`` has no entries.
        """
        if len(self.config.source_flags) == 0 and len(self.config.finalized_source_flags) == 0:
            return

        source_columns = self._get_source_table_column_names(
            self.config.x_column,
            self.config.y_column,
            self.config.source_flags.keys()
        )
        finalized_columns = self._get_source_table_column_names(
            self.config.finalized_x_column,
            self.config.finalized_y_column,
            self.config.finalized_source_flags.keys(),
        )

        # We need the number of overlaps of individual detectors for each coadd source.
        # The following code is slow and inefficient, but can be made simpler in the future
        # case of cell-based coadds and so optimizing usage in afw is not a priority.
        num_overlaps = np.zeros(len(coadd_object_cat), dtype=np.int32)
        for i, obj in enumerate(coadd_object_cat):
            num_overlaps[i] = len(ccd_inputs.subsetContaining(obj.getCoord(), True))

        visits = np.unique(ccd_inputs["visit"])

        matcher = Matcher(np.rad2deg(coadd_object_cat["coord_ra"]),
                          np.rad2deg(coadd_object_cat["coord_dec"]))

        source_flag_counts = {f: np.zeros(len(coadd_object_cat), dtype=np.int32)
                              for f in self.config.source_flags}
        finalized_source_flag_counts = {f: np.zeros(len(coadd_object_cat), dtype=np.int32)
                                        for f in self.config.finalized_source_flags}

        handles_list = [source_table_handle_dict, finalized_source_table_handle_dict]
        columns_list = [source_columns, finalized_columns]
        counts_list = [source_flag_counts, finalized_source_flag_counts]
        x_column_list = [self.config.x_column, self.config.finalized_x_column]
        y_column_list = [self.config.y_column, self.config.finalized_y_column]
        name_list = ["sources", "finalized_sources"]

        for handle_dict, columns, flag_counts, x_col, y_col, name in zip(handles_list,
                                                                         columns_list,
                                                                         counts_list,
                                                                         x_column_list,
                                                                         y_column_list,
                                                                         name_list):
            if handle_dict is not None and len(columns) > 0:
                for visit in visits:
                    if visit not in handle_dict:
                        self.log.info("Visit %d not in input handle dict for %s", visit, name)
                        continue
                    handle = handle_dict[visit]
                    tbl = handle.get(parameters={"columns": columns})

                    # Loop over all ccd_inputs rows for this visit.
                    for row in ccd_inputs[ccd_inputs["visit"] == visit]:
                        detector = row["ccd"]
                        wcs = row.getWcs()
                        if wcs is None:
                            self.log.info("No WCS for visit %d detector %d, so can't match sources to "
                                          "propagate flags.  Skipping...", visit, detector)
                            continue

                        tbl_det = tbl[tbl["detector"] == detector]

                        if len(tbl_det) == 0:
                            continue

                        ra, dec = wcs.pixelToSkyArray(np.asarray(tbl_det[x_col]),
                                                      np.asarray(tbl_det[y_col]),
                                                      degrees=True)

                        try:
                            # The output from the matcher links
                            # coadd_object_cat[i1] <-> df_det[i2]
                            # All objects within the match radius are matched.
                            idx, i1, i2, d = matcher.query_radius(
                                ra,
                                dec,
                                self.config.match_radius/3600.,
                                return_indices=True
                            )
                        except IndexError:
                            # No matches.  Workaround a bug in older version of smatch.
                            self.log.info("Visit %d has no overlapping objects", visit)
                            continue

                        if len(i1) == 0:
                            # No matches (usually because detector does not overlap patch).
                            self.log.info("Visit %d has no overlapping objects", visit)
                            continue

                        for flag in flag_counts:
                            flag_values = np.asarray(tbl_det[flag])
                            flag_counts[flag][i1] += flag_values[i2].astype(np.int32)

        for flag in source_flag_counts:
            thresh = num_overlaps*self.config.source_flags[flag]
            object_flag = (source_flag_counts[flag] > thresh)
            coadd_object_cat[flag] = object_flag
            self.log.info("Propagated %d sources with flag %s", object_flag.sum(), flag)

        for flag in finalized_source_flag_counts:
            thresh = num_overlaps*self.config.finalized_source_flags[flag]
            object_flag = (finalized_source_flag_counts[flag] > thresh)
            coadd_object_cat[flag] = object_flag
            self.log.info("Propagated %d finalized sources with flag %s", object_flag.sum(), flag)

    def _get_source_table_column_names(self, x_column, y_column, flags):
        """Get the list of source table columns from the config.

        Parameters
        ----------
        x_column : `str`
            Name of column with x centroid.
        y_column : `str`
            Name of column with y centroid.
        flags : `list` [`str`]
            List of flags to retrieve.

        Returns
        -------
        columns : [`list`] [`str`]
            Columns to read.
        """
        columns = ["visit", "detector",
                   x_column, y_column]
        columns.extend(flags)

        return columns
