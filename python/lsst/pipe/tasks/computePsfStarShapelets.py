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

__all__ = ["ComputePsfStarShapeletsTask", "ComputePsfStarShapeletsConfig",
           "ConsolidatePsfStarShapeletsTask", "ConsolidatePsfStarShapeletsConfig"]

from astropy.table import Table, vstack
from astropy.utils.metadata import MergeConflictWarning
import numpy as np
import warnings

import lsst.afw.table as afwTable
from lsst.meas.algorithms.computeRoughPsfShapelets import (
    ComputeRoughPsfShapeletsTask,
    ComputeRoughPsfShapeletsConfig,
    NoStarsForShapeletsError,
)
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base import connectionTypes
from lsst.utils.timer import timeMethod

from .coaddBase import reorderRefs


class ComputePsfStarShapeletsConnections(pipeBase.PipelineTaskConnections,
                                         dimensions=("instrument", "visit")):
    input_catalog_handles = connectionTypes.Input(
        doc="Catalog of bright unresolved sources detected on the exposure used for PSF determination; "
            "includes source footprints.",
        name="single_visit_psf_star_footprints",
        storageClass="SourceCatalog",
        dimensions=["instrument", "visit", "detector"],
        deferLoad=True,
        multiple=True,
    )
    input_exposure_handles = connectionTypes.Input(
        doc="Exposure to measure the shapelets on.",
        name="preliminary_visit_image",
        storageClass="ExposureF",
        dimensions=["instrument", "visit", "detector"],
        deferLoad=True,
        multiple=True,
    )
    # Outputs
    shapelet_decomposition_visit = connectionTypes.Output(
        doc="Catalog of shapelet decomposition parameters",
        name="shapelet_decomposition_visit",
        storageClass="ArrowAstropy",
        dimensions=["instrument", "visit"],
    )
    # Optional outputs
    single_visit_shapelet_psf_star_footprints = connectionTypes.Output(
        doc="Catalog of sources used in the shapelet decomposition.",
        name="single_visit_shapelet_psf_star_footprints",
        storageClass="SourceCatalog",   # "ArrowAstropy",
        dimensions=["instrument", "visit"],
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if "single_visit_shapelet_psf_star_footprints" not in config.optional_outputs:
            del self.single_visit_shapelet_psf_star_footprints


class ComputePsfStarShapeletsConfig(
        pipeBase.PipelineTaskConfig, pipelineConnections=ComputePsfStarShapeletsConnections):
    optional_outputs = pexConfig.ListField(
        doc="Which optional outputs to save (as their connection name)?",
        dtype=str,
        default=["single_visit_shapelet_psf_star_footprints"],
        optional=False,
    )
    shapelet_column_base_name = pexConfig.Field(
        doc="Base name for the columns in the shapelets task",
        dtype=str,
        default="RoughPsfShapelets",
        optional=False,
    )
    shapelet_order = pexConfig.Field(
        doc="Order of the shapelet expansion fit to the stars.",
        dtype=int,
        default=6,
    )
    shapelet_circular_basis = pexConfig.Field(
        "Whether to use a circular shapelet basis with the same moments trace instead of an elliptical one.",
        dtype=bool,
        default=True,
    )
    shapelet_coefficient_name_list = pexConfig.ListField(
        doc="List of the shapelet coefficient names in association with the shapelet_order",
        dtype=str,
        default=[
            "[0,0]",
            "Re([1,0])", "Im([1,0])",
            "Re([2,0])", "Im([2,0])", "[1,1]",
            "Re([3,0])", "Im([3,0])", "Re([2,1])", "Im([2,1])",
            "Re([4,0])", "Im([4,0])", "Re([3,1])", "Im([3,1])", "[2,2]",
            "Re([5,0])", "Im([5,0])", "Re([4,1])", "Im([4,1])", "Re([3,2])", "Im([3,2])",
            "Re([6,0])", "Im([6,0])", "Re([5,1])", "Im([5,1])", "Re([4,2])", "Im([4,2])", "[3,3]"
        ],
    )
    non_gaussian_non_atmosphere_index_list = pexConfig.ListField(
        doc="Index list of non-Gaussian, non-atmospheric contributors in the list of shapelet "
        "coefficients associated with shapelet_order",
        dtype=int,
        default=list(range(6, 10)) + list(range(15, 24)),
    )

    def setDefaults(self):
        super().setDefaults()

    def validate(self):
        super().validate()
        # Check that the shapelet decomposition order matches the number of
        # coefficient names in the config.
        n_coeff = 0
        for i_offset in range(self.psf_star_compute_shapelets.shapelet_order + 1):
            n_coeff += (i_offset + 1)
        if len(self.shapelet_coefficient_name_list) != n_coeff:
            raise pexConfig.FieldValidationError(
                ComputePsfStarShapeletsConfig,
                self,
                "The number of coefficients defined in config.shapelet_coefficient_name_list "
                f"({len(self.shapelet_coefficient_name_list)}) does not match the number "
                f"expected ({n_coeff}) from the shapelet decomposition order ({self.shapelet_order})."
            )


class ComputePsfStarShapeletsTask(pipeBase.PipelineTask):
    """Compute the shapelet decomposition using a candiate catalog limited to
    sources used in the PSF model.
    """
    _DefaultName = "computePsfStarShapelets"
    ConfigClass = ComputePsfStarShapeletsConfig

    # def __init__(self, initInputs=None, **kwargs):
    def __init__(self, initInputs=None, **kwargs):
        super().__init__(**kwargs)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        visit = inputRefs.input_catalog_handles[0].dataId["visit"]
        band = inputRefs.input_catalog_handles[0].dataId["band"]
        detector_order = [ref.dataId["detector"] for ref in inputRefs.input_catalog_handles]
        detector_order.sort()
        inputRefs = reorderRefs(inputRefs, detector_order, dataIdKey="detector")
        inputs = butlerQC.get(inputRefs)

        exposure_handles = inputs.pop("input_exposure_handles")
        catalog_handles = inputs.pop("input_catalog_handles")

        # This should not happen with a properly configured execution context.
        assert not inputs, "runQuantum got more inputs than expected"

        # Specify the fields that `annotate` needs below, to ensure they
        # exist, even as None.
        result = pipeBase.Struct(
            shapelet_decomposition_visit=None,
            single_visit_shapelet_psf_star_footprints=None,
        )
        try:
            self.run(
                exposure_handles=exposure_handles,
                catalog_handles=catalog_handles,
                detector_order=detector_order,
                visit=visit,
                band=band,
                result=result,
            )
        except pipeBase.AlgorithmError as e:
            error = pipeBase.AnnotatedPartialOutputsError.annotate(
                e,
                self,
                result.shapelet_decomposition_visit,
                log=self.log
            )
            butlerQC.put(result, outputRefs)
            raise error from e

        butlerQC.put(result, outputRefs)

    @timeMethod
    def run(
        self,
        *,
        exposure_handles,
        catalog_handles,
        detector_order,
        visit,
        band,
        result=None,
    ):
        """Find stars and perform psf measurement, then do a deeper detection
        and measurement and calibrate astrometry and photometry from that.

        Parameters
        ----------
        exposure_handles : `lsst.afw.image.Exposure`
            Input exposure on which to measure the shapelet decomposition.
        result : `lsst.pipe.base.Struct`, optional
            Result struct that is modified to allow saving of partial outputs
            for some failure conditions. If the task completes successfully,
            this is also returned.

        Returns
        -------
        result : `lsst.pipe.base.Struct`
            Results as a struct with attributes:

            ``shaelet_decomposition_visit``
                Footprints of stars that were used to determine the image PSF.
                (`lsst.afw.table.SourceCatalog`)
        """
        if result is None:
            result = pipeBase.Struct()

        shapelet_config = ComputeRoughPsfShapeletsConfig()
        shapelet_config.shapelet_order = self.config.shapelet_order
        shapelet_config.shapelet_circular_basis = self.config.shapelet_circular_basis
        shapelet_base_name = self.config.shapelet_column_base_name
        shapelet_table_list = []
        shapelet_catalog_list = []
        seed = 4242
        nan = float("nan")

        for i_data, (exposure_handle, catalog_handle, detector_id) in enumerate(zip(
                exposure_handles, catalog_handles, detector_order)):
            self.log.info("Computing shapelets for visit: %d; detector: %d; band: %s",
                          visit, detector_id, band)
            n_shapelet_coeff = int(((self.config.shapelet_order + 1)*(self.config.shapelet_order + 2))/2)
            shapelet_coeffs = [nan]*n_shapelet_coeff
            shapelet_used_cat = []
            shapelets_score = nan
            shapelets_score_non_atm = nan
            centroid_diff_shapelet_vs_slot_median = nan
            shapelet_star_xx_median = nan
            shapelet_star_yy_median = nan
            shapelet_star_xy_median = nan
            shapelet_star_e1_median = nan
            shapelet_star_e2_median = nan
            shapelet_star_e_median = nan
            shapelet_star_unnormalized_e_median = nan

            have_shapelet_fit = False

            if exposure_handle is not None and catalog_handle is not None:
                exposure = exposure_handle.get()
                catalog = catalog_handle.get()

                psf_used_cat = catalog[catalog["calib_psf_used"]].copy(deep=True)

                mapper = afwTable.SchemaMapper(psf_used_cat.schema, True)
                mapper.addMinimalSchema(psf_used_cat.schema, True)
                shapelet_task = ComputeRoughPsfShapeletsTask(config=shapelet_config,
                                                             schema=mapper.editOutputSchema())
                shapelet_cat = afwTable.SourceCatalog(mapper.getOutputSchema())
                shapelet_cat.extend(psf_used_cat, mapper=mapper)

                # Compute rough shapelet expansion on the PSF stars to assess
                # focus (IQ).
                try:
                    shapelet_result = shapelet_task.run(
                        masked_image=exposure.getMaskedImage(), catalog=shapelet_cat, seed=seed
                    )
                    shapelet_coeffs = shapelet_result.shapelet.getCoefficients()
                    total_power = np.sum(shapelet_coeffs**2.0)
                    non_atm_power = np.sum(
                        shapelet_coeffs[self.config.non_gaussian_non_atmosphere_index_list]**2.0
                    )
                    shapelets_score_non_atm = np.where(total_power > 0, non_atm_power/total_power, 0.0)

                    shapelet_used_cat = shapelet_result.catalog[
                        (shapelet_result.catalog[shapelet_base_name + "_used"])
                        & (~shapelet_result.catalog[shapelet_base_name + "_flag"])].copy(deep=True)
                    centroid_diff_shapelet_vs_slot = np.sqrt(
                        (shapelet_used_cat["slot_Centroid_x"]
                         - shapelet_used_cat[shapelet_base_name + "_x"])**2.0
                        + (shapelet_used_cat["slot_Centroid_y"]
                           - shapelet_used_cat[shapelet_base_name + "_y"])**2.0
                    )
                    centroid_diff_shapelet_vs_slot_median = np.nanmedian(centroid_diff_shapelet_vs_slot)

                    shapelet_star_xx = shapelet_used_cat[shapelet_base_name + "_xx"]
                    shapelet_star_yy = shapelet_used_cat[shapelet_base_name + "_yy"]
                    shapelet_star_xy = shapelet_used_cat[shapelet_base_name + "_xy"]

                    # Use the trace radius for the star size.
                    # shapelet_star_size = np.sqrt(shapelet_star_xx/2.0 + shapelet_star_yy/2.0)
                    # shapelet_star_size_median = float(np.median(shapelet_star_size))
                    # scaled_centroid_diff = centroid_diff_shapelet_vs_slot/shapelet_star_size
                    # scaled_centroid_diff_median = np.median(scaled_centroid_diff)

                    # Used a fixed number of pixels to scale the centroid diff based on
                    # the maximum footprint size imposed in ComputeRoughPsfShapeletsTask.
                    fixed_scaling = 0.5*np.sqrt(shapelet_config.max_footprint_area)
                    centroid_diff_scaled = centroid_diff_shapelet_vs_slot/fixed_scaling
                    centroid_diff_scaled_median = float(np.median(centroid_diff_scaled))

                    # Compute a shapelets score that includes power from the centroid shift
                    # in addition to the non-atmospheric decomposition coefficients.
                    total_power_plus_scaled = (np.sum(shapelet_coeffs**2.0)
                                               + centroid_diff_scaled_median**2.0)
                    non_atm_power_plus_scaled = (
                        np.sum(shapelet_coeffs[self.config.non_gaussian_non_atmosphere_index_list]**2.0)
                        + centroid_diff_scaled_median**2.0)

                    shapelets_score = float(
                        np.where(total_power_plus_scaled > 0,
                                 non_atm_power_plus_scaled/total_power_plus_scaled, 0.0)
                    )
                    self.log.info("n_shapelet_star = %d; shapelets_score = %.5f; "
                                  "shapelets_score_non_atm = %.5f; "
                                  "centroid_diff_shapelet_vs_slot_median = %.3f (pixels), "
                                  "centroid_diff_scaled_median = %.3f", len(shapelet_used_cat),
                                  shapelets_score, shapelets_score_non_atm,
                                  centroid_diff_shapelet_vs_slot_median,
                                  centroid_diff_scaled_median)

                    shapelet_star_e1 = ((shapelet_star_xx - shapelet_star_yy)
                                        / (shapelet_star_xx + shapelet_star_yy))
                    shapelet_star_e2 = 2*shapelet_star_xy/(shapelet_star_xx + shapelet_star_yy)
                    shapelet_star_e = np.sqrt(shapelet_star_e1**2.0 + shapelet_star_e2**2.0)

                    shapelet_star_xx_median = np.median(shapelet_star_xx)
                    shapelet_star_yy_median = np.median(shapelet_star_yy)
                    shapelet_star_xy_median = np.median(shapelet_star_xy)
                    shapelet_star_e1_median = np.median(shapelet_star_e1)
                    shapelet_star_e2_median = np.median(shapelet_star_e2)
                    shapelet_star_e_median = np.median(shapelet_star_e)

                    shapelet_star_unnormalized_e = np.sqrt(
                        (shapelet_star_xx - shapelet_star_yy)**2.0 + (2.0*shapelet_star_xy)**2.0)
                    shapelet_star_unnormalized_e_median = np.median(shapelet_star_unnormalized_e)
                    have_shapelet_fit = True
                except NoStarsForShapeletsError as e:
                    self.log.warning("ComputeRoughPsfShapeletsTask failed for visit: %d; detector: %d "
                                     "band: %s with error %s.  Setting all metrics to nan.",
                                     visit, detector_id, band, e)
            else:
                self.log.warning("Either exposure_handle (%s) or catalog_handle (%s) is None for "
                                 "visit: %d; detector: %d; band %s.  Setting all metrics to nan.",
                                 exposure_handle, catalog_handle, visit, detector_id, band)

            shapelet_decomposition = Table({
                "visit": [visit],
                "detector": [detector_id],
                "band": [band],
                "shapelet_coeffs": [shapelet_coeffs],
                "n_shapelet_star": [len(shapelet_used_cat)],
                "shapelets_score": [shapelets_score],
                "shapelets_score_non_atm": [shapelets_score_non_atm],
                "centroid_diff_shapelet_vs_slot_median": [centroid_diff_shapelet_vs_slot_median],
                "shapelet_star_xx_median": [shapelet_star_xx_median],
                "shapelet_star_yy_median": [shapelet_star_yy_median],
                "shapelet_star_xy_median": [shapelet_star_xy_median],
                "shapelet_star_e1_median": [shapelet_star_e1_median],
                "shapelet_star_e2_median": [shapelet_star_e2_median],
                "shapelet_star_e_median": [shapelet_star_e_median],
                "shapelet_star_unnormalized_e_median": [shapelet_star_unnormalized_e_median],
            })
            for i_coeff, shapelet_coefficient_name in enumerate(self.config.shapelet_coefficient_name_list):
                shapelet_decomposition[shapelet_coefficient_name] = [shapelet_coeffs[i_coeff]]

            if have_shapelet_fit:
                # Add visit and detector columns to shapelets catalog
                shapelet_catalog_mapper = afwTable.SchemaMapper(shapelet_result.catalog.schema, True)
                shapelet_catalog_mapper.addMinimalSchema(shapelet_result.catalog.schema, True)
                shapelet_result_catalog_schema = shapelet_catalog_mapper.editOutputSchema()
                visit_key = shapelet_result_catalog_schema.addField(
                    "visit", type=np.int64, doc="Visit number associated with the source.")
                detector_key = shapelet_result_catalog_schema.addField(
                    "detector", type=np.int32, doc="Detector name associated with the source.")
                shapelet_result_catalog = afwTable.SourceCatalog(shapelet_result_catalog_schema)
                shapelet_result_catalog.reserve(len(shapelet_result.catalog))
                for record in shapelet_result.catalog:
                    new_record = shapelet_result_catalog.addNew()
                    new_record.assign(record, shapelet_catalog_mapper)
                    new_record.set(visit_key, visit)
                    new_record.set(detector_key, detector_id)
                shapelet_catalog_list.append(shapelet_result_catalog)

            shapelet_table_list.append(shapelet_decomposition)

        result.shapelet_decomposition_visit = vstack(shapelet_table_list)

        if len(shapelet_catalog_list) > 0:
            shapelet_catalog_visit = afwTable.SourceCatalog(shapelet_catalog_list[0].schema)
            shapelet_catalog_visit.reserve(sum([len(cat) for cat in shapelet_catalog_list]))
            for shapelet_catalog in shapelet_catalog_list:
                shapelet_catalog_visit.extend(shapelet_catalog)
        else:
            shapelet_catalog_visit = None
        result.single_visit_shapelet_psf_star_footprints = shapelet_catalog_visit
        return result


class ConsolidatePsfStarShapeletsConnections(pipeBase.PipelineTaskConnections, dimensions=("instrument",)):
    shapelet_decomposition_visit_refs = connectionTypes.Input(
        doc="Data references for per-visit shapelet decomposition parameters",
        name="shapelet_decomposition_visit",
        storageClass="ArrowAstropy",
        dimensions=("instrument", "visit"),
        multiple=True,
        deferLoad=True,
    )

    outputCatalog = connectionTypes.Output(
        doc="Per-collection visit/detector shapelet decomposition parameters",
        name="shapelet_decomposition_table",
        storageClass="ArrowAstropy",
        dimensions=("instrument",)
    )


class ConsolidatePsfStarShapeletsConfig(pipeBase.PipelineTaskConfig,
                                        pipelineConnections=ConsolidatePsfStarShapeletsConnections):
    pass


class ConsolidatePsfStarShapeletsTask(pipeBase.PipelineTask):
    """Produce a per-collection shapelet_decomposition_table from all input
    visits.
    """
    _DefaultName = "consolidatePsfStarShapelets"
    ConfigClass = ConsolidatePsfStarShapeletsConfig

    def run(self, shapelet_decomposition_visit_refs):
        """Make a table of ccd information from the shapelet_decomposition
        tables.

        Parameters
        ----------
        shapelet_decomposition_visit_refs : `list` \
                                      [`lsst.daf.butler.DeferredDatasetHandle`]
           List of DeferredDatasetHandles pointing to the
           shapelet_decomposition tables with per-visit/per-detector shapelet
           decomposition results.

        Returns
        -------
        result : `~lsst.pipe.base.Struct`
           Results struct with attribute:

           ``outputCatalog``
               Consolidated table over all input visits.
        """
        visitEntries = []
        for shapelet_decomposition_visit_ref in shapelet_decomposition_visit_refs:
            shapelet_decomposition_visit = shapelet_decomposition_visit_ref.get()
            if not shapelet_decomposition_visit:
                continue
            visitEntries.append(shapelet_decomposition_visit)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=MergeConflictWarning)
            outputCatalog = vstack(visitEntries, join_type="exact")
        return pipeBase.Struct(outputCatalog=outputCatalog)
