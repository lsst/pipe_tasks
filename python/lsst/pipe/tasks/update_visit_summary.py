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

from __future__ import annotations

__all__ = (
    "UpdateVisitSummaryConnections",
    "UpdateVisitSummaryConfig",
    "UpdateVisitSummaryTask",
    "PossiblyMultipleInput",
    "PerTractInput",
    "GlobalInput",
)

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any

import astropy.table
import lsst.pipe.base.connectionTypes as cT
from lsst.afw.geom import SkyWcs
from lsst.afw.image import ExposureSummaryStats
from lsst.afw.math import BackgroundList
from lsst.afw.table import ExposureCatalog, ExposureRecord, SchemaMapper
from lsst.daf.butler import Butler, DatasetRef, DeferredDatasetHandle
from lsst.geom import Angle, Box2I, SpherePoint, degrees
from lsst.pex.config import ChoiceField, ConfigurableField
from lsst.pipe.base import (
    ButlerQuantumContext,
    InputQuantizedConnection,
    InvalidQuantumError,
    OutputQuantizedConnection,
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
)
from lsst.skymap import BaseSkyMap, TractInfo
from lsst.skymap.detail import makeSkyPolygonFromBBox
from .computeExposureSummaryStats import ComputeExposureSummaryStatsTask


def compute_center_for_detector_record(
    record: ExposureRecord, bbox: Box2I | None = None, wcs: SkyWcs | None = None
) -> SpherePoint | None:
    """Compute the sky coordinate center for a detector to be used when
    testing distance to tract center.

    Parameters
    ----------
    record : `lsst.afw.table.ExposureRecord`
        Exposure record to obtain WCS and bbox from if not provided.
    bbox : `lsst.geom.Box2I`, optional
        Bounding box for the detector in its own pixel coordinates.
    wcs : `lsst.afw.geom.SkyWcs`, optional
        WCS that maps the detector's pixel coordinate system to celestial
        coordinates.

    Returns
    -------
    center : `lsst.geom.SpherePoint` or `None`
        Center of the detector in sky coordinates, or `None` if no WCS was
        given or present in the given record.
    """
    if bbox is None:
        bbox = record.getBBox()
    if wcs is None:
        wcs = record.getWcs()
        if wcs is None:
            return None
    region = makeSkyPolygonFromBBox(bbox, wcs)
    return SpherePoint(region.getCentroid())


class PossiblyMultipleInput(ABC):
    """A helper ABC for handling input `~lsst.afw.table.ExposureCatalog`
    datasets that may be multiple (one per tract/visit combination) or
    unique/global (one per visit).
    """

    @abstractmethod
    def best_for_detector(
        self,
        detector_id: int,
        center: SpherePoint | None = None,
        bbox: Box2I | None = None,
    ) -> tuple[int, ExposureRecord | None]:
        """Return the exposure record for this detector that is the best match
        for this detector.

        Parameters
        ----------
        detector_id : `int`
            Detector ID; used to find the right row in the catalog or catalogs.
        center : `lsst.geom.SpherePoint` or `None`
            Center of the detector in sky coordinates.  If not provided, one
            will be computed via `compute_center_for_detector_record`.
        bbox : `lsst.geom.Box2I`, optional
            Bounding box for the detector in its own pixel coordinates.

        Returns
        -------
        tract_id : `int`
            ID of the tract that supplied this record, or `-1` if ``record`` is
            `None` or if the input was not per-tract.
        record : `lsst.afw.table.ExposureRecord` or `None`
            Best record for this detector, or `None` if there either were no
            records for this detector or no WCS available to compute a center.
        """
        raise NotImplementedError()


@dataclasses.dataclass
class PerTractInput(PossiblyMultipleInput):
    """Wrapper class for input `~lsst.afw.table.ExposureCatalog` datasets
    that are per-tract.

    This selects the best tract via the minimum average distance (on the sky)
    from the detector's corners to the tract center.
    """

    catalogs_by_tract: list[tuple[TractInfo, ExposureCatalog]]
    """List of tuples of catalogs and the tracts they correspond to
    (`list` [`tuple` [`lsst.skymap.TractInfo`,
        `lsst.afw.table.ExposureCatalog`]]).
    """

    @classmethod
    def load(
        cls,
        butler: ButlerQuantumContext | Butler,
        sky_map: BaseSkyMap,
        refs: Iterable[DatasetRef],
    ) -> PerTractInput:
        """Load and wrap input catalogs.

        Parameters
        ----------
        butler : `lsst.pipe.base.ButlerQuantumContext`
            Butler proxy used in `~lsst.pipe.base.PipelineTask.runQuantum`.
        sky_map : `lsst.skymap.BaseSkyMap`
            Definition of tracts and patches.
        refs : `~collections.abc.Iterable` [`lsst.daf.butler.DatasetRef`]
            References to the catalog datasets to load.

        Returns
        -------
        wrapper : `PerTractInput`
            Wrapper object for the loaded catalogs.
        """
        catalogs_by_tract = []
        for ref in refs:
            tract_id = ref.dataId["tract"]
            tract_info = sky_map[tract_id]
            catalogs_by_tract.append(
                (
                    tract_info,
                    butler.get(ref),
                )
            )
        return cls(catalogs_by_tract)

    def best_for_detector(
        self,
        detector_id: int,
        center: SpherePoint | None = None,
        bbox: Box2I | None = None,
    ) -> tuple[int, ExposureRecord | None]:
        # Docstring inherited.
        best_result: tuple[int, ExposureRecord | None] = (-1, None)
        best_distance: Angle = float("inf") * degrees
        for tract_info, catalog in self.catalogs_by_tract:
            record = catalog.find(detector_id)
            if record is None:
                continue
            if center is None:
                center_for_record = compute_center_for_detector_record(
                    record, bbox=bbox
                )
                if center_for_record is None:
                    continue
            else:
                center_for_record = center
            center_distance = tract_info.ctr_coord.separation(center_for_record)
            if best_distance > center_distance:
                best_result = (tract_info.tract_id, record)
                best_distance = center_distance
        return best_result


@dataclasses.dataclass
class GlobalInput(PossiblyMultipleInput):
    """Wrapper class for input `~lsst.afw.table.ExposureCatalog` datasets
    that are not per-tract.
    """

    catalog: ExposureCatalog
    """Loaded per-visit catalog dataset (`lsst.afw.table.ExposureCatalog`).
    """

    def best_for_detector(
        self,
        detector_id: int,
        center: SpherePoint | None = None,
        bbox: Box2I | None = None,
    ) -> tuple[int, ExposureRecord | None]:
        # Docstring inherited.
        return -1, self.catalog.find(detector_id)


class UpdateVisitSummaryConnections(
    PipelineTaskConnections,
    dimensions=("instrument", "visit"),
    defaultTemplates={
        "skyWcsName": "jointcal",
        "photoCalibName": "fgcm",
    },
):
    sky_map = cT.Input(
        doc="Description of tract/patch geometry.",
        name=BaseSkyMap.SKYMAP_DATASET_TYPE_NAME,
        dimensions=("skymap",),
        storageClass="SkyMap",
    )
    input_summary_schema = cT.InitInput(
        doc="Schema for input_summary_catalog.",
        name="visitSummary_schema",
        storageClass="ExposureCatalog",
    )
    input_summary_catalog = cT.Input(
        doc="Visit summary table to load and modify.",
        name="visitSummary",
        dimensions=("instrument", "visit"),
        storageClass="ExposureCatalog",
    )
    input_exposures = cT.Input(
        doc=(
            "Per-detector images to obtain image, mask, and variance from "
            "(embedded summary stats and other components are ignored)."
        ),
        name="calexp",
        dimensions=("instrument", "detector", "visit"),
        storageClass="ExposureF",
        multiple=True,
        deferLoad=True,
    )
    psf_overrides = cT.Input(
        doc="Visit-level catalog of updated PSFs to use.",
        name="finalized_psf_ap_corr_catalog",
        dimensions=("instrument", "visit"),
        storageClass="ExposureCatalog",
    )
    psf_star_catalog = cT.Input(
        doc="Per-visit table of PSF reserved- and used-star measurements.",
        name="finalized_src_table",
        dimensions=("instrument", "visit"),
        storageClass="DataFrame",
    )
    ap_corr_overrides = cT.Input(
        doc="Visit-level catalog of updated aperture correction maps to use.",
        name="finalized_psf_ap_corr_catalog",
        dimensions=("instrument", "visit"),
        storageClass="ExposureCatalog",
    )
    photo_calib_overrides_tract = cT.Input(
        doc="Per-Tract visit-level catalog of updated photometric calibration objects to use.",
        name="{photoCalibName}PhotoCalibCatalog",
        dimensions=("instrument", "visit", "tract"),
        storageClass="ExposureCatalog",
        multiple=True,
    )
    photo_calib_overrides_global = cT.Input(
        doc="Global visit-level catalog of updated photometric calibration objects to use.",
        name="{photoCalibName}PhotoCalibCatalog",
        dimensions=("instrument", "visit"),
        storageClass="ExposureCatalog",
    )
    wcs_overrides_tract = cT.Input(
        doc="Per-tract visit-level catalog of updated astrometric calibration objects to use.",
        name="{skyWcsName}SkyWcsCatalog",
        dimensions=("instrument", "visit", "tract"),
        storageClass="ExposureCatalog",
        multiple=True,
    )
    wcs_overrides_global = cT.Input(
        doc="Global visit-level catalog of updated astrometric calibration objects to use.",
        name="{skyWcsName}SkyWcsCatalog",
        dimensions=("instrument", "visit"),
        storageClass="ExposureCatalog",
    )
    background_originals = cT.Input(
        doc="Per-detector original background that has already been subtracted from 'input_exposures'.",
        name="calexpBackground",
        dimensions=("instrument", "visit", "detector"),
        storageClass="Background",
        multiple=True,
        deferLoad=True,
    )
    background_overrides = cT.Input(
        doc="Per-detector background that can be subtracted directly from 'input_exposures'.",
        name="skyCorr",
        dimensions=("instrument", "visit", "detector"),
        storageClass="Background",
        multiple=True,
        deferLoad=True,
    )
    output_summary_schema = cT.InitOutput(
        doc="Schema of the output visit summary catalog.",
        name="finalVisitSummary_schema",
        storageClass="ExposureCatalog",
    )
    output_summary_catalog = cT.Output(
        doc="Visit-level catalog summarizing all image characterizations and calibrations.",
        name="finalVisitSummary",
        dimensions=("instrument", "visit"),
        storageClass="ExposureCatalog",
    )

    def __init__(self, *, config: UpdateVisitSummaryConfig | None = None):
        super().__init__(config=config)
        match self.config.wcs_provider:
            case "input_summary":
                self.inputs.remove("wcs_overrides_tract")
                self.inputs.remove("wcs_overrides_global")
            case "tract":
                self.inputs.remove("wcs_overrides_global")
            case "global":
                self.inputs.remove("wcs_overrides_tract")
            case bad:
                raise ValueError(
                    f"Invalid value wcs_provider={bad!r}; config was not validated."
                )
        match self.config.photo_calib_provider:
            case "input_summary":
                self.inputs.remove("photo_calib_overrides_tract")
                self.inputs.remove("photo_calib_overrides_global")
            case "tract":
                self.inputs.remove("photo_calib_overrides_global")
            case "global":
                self.inputs.remove("photo_calib_overrides_tract")
            case bad:
                raise ValueError(
                    f"Invalid value photo_calib_provider={bad!r}; config was not validated."
                )
        match self.config.background_provider:
            case "input_summary":
                self.inputs.remove("background_originals")
                self.inputs.remove("background_overrides")
            case "replacement":
                pass
            case bad:
                raise ValueError(
                    f"Invalid value background_provider={bad!r}; config was not validated."
                )


class UpdateVisitSummaryConfig(
    PipelineTaskConfig, pipelineConnections=UpdateVisitSummaryConnections
):
    """Configuration for UpdateVisitSummaryTask.

    Notes
    -----
    The configuration defaults for this task reflect a simple or "least common
    denominator" pipeline, not the more complete, more sophisticated pipeline
    we run on the instruments we support best.  The expectation is that the
    various full pipeline definitions will generally import the simpler
    definition, so making the defaults correspond to any full pipeline would
    just lead to the simple pipeline setting them back to the simple-pipeline
    values and the full pipeline still having to then override them to the
    full-pipeline values.
    """

    compute_summary_stats = ConfigurableField(
        doc="Subtask that computes summary statistics from Exposure components.",
        target=ComputeExposureSummaryStatsTask,
    )
    wcs_provider = ChoiceField(
        doc="Which connection and behavior to use when applying WCS overrides.",
        dtype=str,
        allowed={
            "input_summary": (
                "Propagate the WCS from the input visit summary catalog "
                "and do not recompute WCS-based summary statistics."
            ),
            "tract": {
                "Use the 'wcs_overrides_tract' connection to load an "
                "`ExposureCatalog` with {visit, tract} dimensions and per-"
                "detector rows, and recommpute WCS-based summary statistics."
            },
            "global": {
                "Use the 'wcs_overrides_global' connection to load an "
                "`ExposureCatalog` with {visit} dimensions and per-"
                "detector rows, and recommpute WCS-based summary statistics."
            },
            # If needed, we could add options here to propagate the WCS from
            # the input exposures and/or transfer WCS-based summary statistics
            # from them as well.  Right now there's no use case for that, since
            # the input visit summary is always produced after the last time we
            # write a new Exposure.
        },
        default="input_summary",
        optional=False,
    )
    photo_calib_provider = ChoiceField(
        doc="Which connection and behavior to use when applying photometric calibration overrides.",
        dtype=str,
        allowed={
            "input_summary": (
                "Propagate the PhotoCalib from the input visit summary catalog "
                "and do not recompute photometric calibration summary "
                "statistics."
            ),
            "tract": {
                "Use the 'photo_calib_overrides_tract' connection to load an "
                "`ExposureCatalog` with {visit, tract} dimensions and per-"
                "detector rows, and recommpute photometric calibration summary "
                "statistics."
            },
            "global": {
                "Use the 'photo_calib_overrides_global' connection to load an "
                "`ExposureCatalog` with {visit} dimensions and per-"
                "detector rows, and recommpute photometric calibration summary "
                "statistics."
            },
            # If needed, we could add options here to propagate the PhotoCalib
            # from the input exposures and/or transfer photometric calibration
            # summary statistics them as well.  Right now there's no use case
            # for that, since the input visit summary is always produced after
            # the last time we write a new Exposure.
        },
        default="input_summary",
        optional=False,
    )
    background_provider = ChoiceField(
        doc="Which connection(s) and behavior to use when applying background overrides.",
        dtype=str,
        allowed={
            "input_summary": (
                "The input visit summary catalog already includes summary "
                "statistics for the final backgrounds that can be used as-is."
            ),
            "replacement": {
                "The 'background_originals' connection refers to a background "
                "model that has been superseded by the model referred to by "
                "the 'background_overrides' connection."
            },
            # Could also imagine an option in which there is no original
            # background and the new one stands alone; can add later if needed.
        },
        default="input_summary",
        optional=False,
    )
    # Could imagine an option here to say that the original background has not
    # been subtracted from the input exposures, allowing postISRCCD to be used
    # as input exposures.  Can add later if needed.


class UpdateVisitSummaryTask(PipelineTask):
    """A pipeline task that creates a new visit-summary table after all
    `lsst.afw.image.Exposure` components have been finalized.

    Notes
    -----
    This task is designed to be run just prior to making warps for coaddition,
    as it aggregates all inputs other than the images and backgrounds into a
    single ``ExposureCatalog`` dataset and recomputes summary statistics that
    are useful in selecting which images should go into a coadd.  Its output
    can also be used to reconstruct a final processed visit image when combined
    with a post-ISR image, the background model, and the final mask.
    """

    # The `run` method of this task can conditionally apply overrides for PSFs
    # and aperture corrections, but its `PipelineTask` interface always applies
    # them.  We can always add the config options to make them optional later,
    # if that turns out to be useful.

    _DefaultName = "updateVisitSummary"
    ConfigClass = UpdateVisitSummaryConfig

    compute_summary_stats: ComputeExposureSummaryStatsTask

    def __init__(self, *, initInputs: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(initInputs=initInputs, **kwargs)
        self.makeSubtask("compute_summary_stats")
        if initInputs is None or "input_summary_schema" not in initInputs:
            raise RuntimeError("Task requires 'input_summary_schema' in initInputs.")
        input_summary_schema = initInputs["input_summary_schema"].schema
        self.schema_mapper = SchemaMapper(input_summary_schema)
        self.schema_mapper.addMinimalSchema(input_summary_schema)
        self.schema = self.schema_mapper.getOutputSchema()
        if self.config.wcs_provider == "tract":
            self.schema.addField(
                "wcsTractId", type="L", doc="ID of the tract that provided the WCS."
            )
        if self.config.photo_calib_provider == "tract":
            self.schema.addField(
                "photoCalibTractId",
                type="L",
                doc="ID of the tract that provided the PhotoCalib.",
            )
        self.output_summary_schema = ExposureCatalog(self.schema)

    def runQuantum(
        self,
        butlerQC: ButlerQuantumContext,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        # Docstring inherited.
        sky_map = butlerQC.get(inputRefs.sky_map)
        del inputRefs.sky_map
        inputs = {}
        # Collapse the wcs_override_ and photo_calib_override_ connection pairs
        # into individual inputs (either ExposureCatalog or PerTractInput
        # objects).
        match self.config.wcs_provider:
            case "tract":
                inputs["wcs_overrides"] = PerTractInput.load(
                    butlerQC, sky_map, inputRefs.wcs_overrides_tract
                )
                del inputRefs.wcs_overrides_tract
            case "global":
                inputs["wcs_overrides"] = GlobalInput(
                    butlerQC.get(inputRefs.wcs_overrides_global)
                )
                del inputRefs.wcs_overrides_global
            case "input_summary":
                inputs["wcs_overrides"] = None
        match self.config.photo_calib_provider:
            case "tract":
                inputs["photo_calib_overrides"] = PerTractInput.load(
                    butlerQC, sky_map, inputRefs.photo_calib_overrides_tract
                )
                del inputRefs.photo_calib_overrides_tract
            case "global":
                inputs["photo_calib_overrides"] = GlobalInput(
                    butlerQC.get(inputRefs.photo_calib_overrides_global)
                )
                del inputRefs.photo_calib_overrides_global
            case "input_summary":
                inputs["photo_calib_overrides"] = None
        # Load or make DeferredDatasetHandles for everything else.
        inputs.update(butlerQC.get(inputRefs))
        deferred_dataset_types = ["input_exposures"]
        # Handle whether to look for background originals and overrides at all.
        match self.config.background_provider:
            case "replacement":
                deferred_dataset_types.append("background_originals")
                deferred_dataset_types.append("background_overrides")
        # Transform the lists of DeferredDatasetHandles for the multiple=True,
        # deferLoad=True connections into mappings keyed by detector ID.
        for name in deferred_dataset_types:
            handles_list = inputs[name]
            inputs[name] = {
                handle.dataId["detector"]: handle for handle in handles_list
            }
            for record in inputs["input_summary_catalog"]:
                detector_id = record.getId()
                if detector_id not in inputs[name]:
                    raise InvalidQuantumError(
                        f"No {name!r} with detector {detector_id} for visit "
                        f"{butlerQC.quantum.dataId['visit']} even though this detector is present "
                        "in the input visit summary catalog. "
                        "This is most likely to occur when the QuantumGraph that includes this task "
                        "was incorrectly generated with an explicit or implicit (from datasets) tract "
                        "constraint."
                    )
        # Convert the psf_star_catalog datasets from DataFrame to Astropy so
        # they can be handled by ComputeExposureSummaryStatsTask (which was
        # actually written to work with afw.table, but Astropy is similar
        # enough that it works, too).  Ideally this would be handled by just
        # using ArrowAstropy as the storage class in the connection, but QG
        # generation apparently doesn't fully support those yet, as it leads to
        # problems in ci_hsc.
        inputs["psf_star_catalog"] = astropy.table.Table.from_pandas(inputs["psf_star_catalog"], index=True)
        # Actually run the task and write the results.
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        input_summary_catalog: ExposureCatalog,
        input_exposures: Mapping[int, DeferredDatasetHandle],
        psf_overrides: ExposureCatalog | None = None,
        psf_star_catalog: astropy.table.Table | None = None,
        ap_corr_overrides: ExposureCatalog | None = None,
        photo_calib_overrides: PossiblyMultipleInput | None = None,
        wcs_overrides: PossiblyMultipleInput | None = None,
        background_originals: Mapping[int, DeferredDatasetHandle] | None = None,
        background_overrides: Mapping[int, DeferredDatasetHandle] | None = None,
    ):
        """Build an updated version of a visit summary catalog.

        Parameters
        ----------
        input_summary_catalog : `lsst.afw.table.ExposureCatalog`
            Input catalog.  Each row in this catalog will be used to produce
            a row in the output catalog.  Any override parameter that is `None`
            will leave the corresponding values unchanged from those in this
            input catalog.
        input_exposures : `collections.abc.Mapping` [`int`,
                `lsst.daf.butler.DeferredDatasetHandle`]
            Deferred-load objects that fetch `lsst.afw.image.Exposure`
            instances.  Only the image, mask, and variance are used; all other
            components are assumed to be superceded by at least
            ``input_summary_catalog`` and probably some ``_overrides``
            arguments as well.  This usually corresponds to the ``calexp``
            dataset.
        psf_overrides : `lsst.afw.table.ExposureCatalog`, optional
            Catalog with attached `lsst.afw.detection.Psf` objects that
            supersede the input catalog's PSFs.
        psf_star_catalog : `astropy.table.Table`, optional
            Table containing PSF stars for use in computing PSF summary
            statistics.  Must be provided if ``psf_overrides`` is.
        ap_corr_overrides : `lsst.afw.table.ExposureCatalog`, optional
            Catalog with attached `lsst.afw.image.ApCorrMap` objects that
            supersede the input catalog's aperture corrections.
        photo_calib_overrides : `PossiblyMultipleInput`, optional
            Catalog wrappers with attached `lsst.afw.image.PhotoCalib`
            objects that supersede the input catalog's photometric
            calibrations.
        wcs_overrides : `PossiblyMultipleInput`, optional
            Catalog wrappers with attached `lsst.afw.geom.SkyWcs` objects
            that supersede the input catalog's astrometric calibrations.
        background_originals : `collections.abc.Mapping` [`int`,
                `lsst.daf.butler.DeferredDatasetHandle`], optional
            Deferred-load objects that fetch `lsst.afw.math.BackgroundList`
            instances.  These should correspond to the background already
            subtracted from ``input_exposures``.  If not provided and
            ``background_overrides`` is, it is assumed that the background in
            ``input_exposures`` has not been subtracted.  If provided, all keys
            in ``background_overrides`` must also be present in
            ``background_originals``.
        background_overrides : `collections.abc.Mapping` [`int`,
                `lsst.daf.butler.DeferredDatasetHandle`], optional
            Deferred-load objects that fetch `lsst.afw.math.BackgroundList`
            instances.  These should correspond to the background that should
            now be subtracted from``input_exposures`` to yield the final
            background-subtracted image.

        Returns
        -------
        output_summary_catalog : `lsst.afw.table.ExposureCatalog`
            Output visit summary catalog.

        Notes
        -----
        If any override parameter is provided but does not have a value for a
        particular detector, that component will be set to `None` in the
        returned catalog for that detector and all summary statistics derived
        from that component will be reset (usually to ``NaN``) as well.  Not
        passing an override parameter at all will instead pass through the
        original component and values from the input catalog unchanged.
        """
        output_summary_catalog = ExposureCatalog(self.schema)
        output_summary_catalog.setMetadata(input_summary_catalog.getMetadata())
        for input_record in input_summary_catalog:
            detector_id = input_record.getId()
            output_record = output_summary_catalog.addNew()

            # Make a new ExposureSummaryStats from the input record.
            summary_stats = ExposureSummaryStats.from_record(input_record)

            # Also copy the input record values to output record; this copies
            # many of the same values just copied into `summary_stats` (which
            # will be overridden later by summary_stats.update_record), but it
            # also copies fields that aren't part of summary_stats, including
            # the actual components like Psf, Wcs, etc.
            output_record.assign(input_record, self.schema_mapper)

            exposure = input_exposures[detector_id].get()
            bbox = exposure.getBBox()

            if wcs_overrides:
                wcs_tract, wcs_record = wcs_overrides.best_for_detector(
                    detector_id, bbox=bbox
                )
                if wcs_record is not None:
                    wcs = wcs_record.getWcs()
                else:
                    wcs = None
                if self.config.wcs_provider == "tract":
                    output_record["wcsTractId"] = wcs_tract
                output_record.setWcs(wcs)
                self.compute_summary_stats.update_wcs_stats(
                    summary_stats, wcs, bbox, output_record.getVisitInfo()
                )
            else:
                wcs = input_record.getWcs()

            if psf_overrides:
                if (psf_record := psf_overrides.find(detector_id)) is not None:
                    psf = psf_record.getPsf()
                else:
                    psf = None
                output_record.setPsf(psf)
                sources = psf_star_catalog[psf_star_catalog["detector"] == detector_id]
                self.compute_summary_stats.update_psf_stats(
                    summary_stats,
                    psf,
                    bbox,
                    sources,
                    image_mask=exposure.mask,
                    sources_is_astropy=True,
                )

            if ap_corr_overrides:
                if (ap_corr_record := ap_corr_overrides.find(detector_id)) is not None:
                    ap_corr = ap_corr_record.getApCorrMap()
                else:
                    ap_corr = None
                output_record.setApCorrMap(ap_corr)

            if photo_calib_overrides:
                center = compute_center_for_detector_record(output_record, bbox, wcs)
                (
                    photo_calib_tract,
                    photo_calib_record,
                ) = photo_calib_overrides.best_for_detector(detector_id, center=center)
                if photo_calib_record is not None:
                    photo_calib = photo_calib_record.getPhotoCalib()
                else:
                    photo_calib = None
                if self.config.photo_calib_provider == "tract":
                    output_record["photoCalibTractId"] = photo_calib_tract
                output_record.setPhotoCalib(photo_calib)
                self.compute_summary_stats.update_photo_calib_stats(
                    summary_stats, photo_calib
                )

            if background_overrides is not None:
                if (handle := background_overrides.get(detector_id)) is not None:
                    new_bkg = handle.get()
                    if background_originals is not None:
                        orig_bkg = background_originals[detector_id].get()
                    else:
                        orig_bkg = BackgroundList()

                    full_bkg = orig_bkg.clone()
                    for layer in new_bkg:
                        full_bkg.append(layer)
                    exposure.image -= new_bkg.getImage()
                    self.compute_summary_stats.update_background_stats(
                        summary_stats, full_bkg
                    )
                    self.compute_summary_stats.update_masked_image_stats(
                        summary_stats, exposure.getMaskedImage()
                    )

            summary_stats.update_record(output_record)
            del exposure

        return Struct(output_summary_catalog=output_summary_catalog)
