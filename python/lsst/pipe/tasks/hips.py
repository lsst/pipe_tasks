#
# LSST Data Management System
# Copyright 2022 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <http://www.lsstcorp.org/LegalNotices/>.
#
"""Tasks for making and manipulating HIPS images."""

__all__ = ["HighResolutionHipsTask", "HighResolutionHipsConfig", "HighResolutionHipsConnections"]

from collections import defaultdict
import numpy as np
import argparse
import sys

from lsst.sphgeom import RangeSet, HealpixPixelization
from lsst.utils.timer import timeMethod
from lsst.daf.butler import Butler, DatasetRef, Quantum
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.geom as geom


class HighResolutionHipsConnections(pipeBase.PipelineTaskConnections,
                                    dimensions=("healpix9", "band"),
                                    defaultTemplates={"coaddName": "deep"}):
    coadd_exposure_handles = pipeBase.connectionTypes.Input(
        doc="Coadded exposures to convert to HIPS format.",
        name="{coaddName}Coadd_calexp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band"),
        multiple=True,
        deferLoad=True,
    )
    hips_exposures = pipeBase.connectionTypes.Output(
        doc="HIPS-compatible HPX image.",
        name="{coaddName}Coadd_hpx",
        storageClass="ExposureF",
        dimensions=("healpix11", "band"),
        multiple=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        quantum_order = None
        for dim in self.dimensions:
            if 'healpix' in dim:
                quantum_order = int(dim.split('healpix')[1])
        if quantum_order is None:
            raise ValueError("Must specify a healpix dimension in quantum dimensions.")

        if quantum_order > config.hips_order:
            raise ValueError("Quantum healpix dimension order must not be greater than hips_order")

        order = None
        for dim in self.hips_exposures.dimensions:
            if 'healpix' in dim:
                order = int(dim.split('healpix')[1])
        if order is None:
            raise ValueError("Must specify a healpix dimension in hips_exposure dimensions.")

        if order != config.hips_order:
            raise ValueError("healpix dimension order must match config.hips_order.")


class HighResolutionHipsConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=HighResolutionHipsConnections):
    """Configuration parameters for HighResolutionHipsTask.

    Notes
    -----
    A HiPS image covers one HEALPix cell, with the HEALPix nside equal to
    2**hips_order. Each cell is 'shift_order' orders deeper than the HEALPix
    cell, with 2**shift_order x 2**shift_order sub-pixels on a side, which
    defines the target resolution of the HiPS image. The IVOA recommends
    shift_order=9, for 2**9=512 pixels on a side.

    Table 5 from
    https://www.ivoa.net/documents/HiPS/20170519/REC-HIPS-1.0-20170519.pdf
    shows the relationship between hips_order, number of tiles (full
    sky coverage), cell size, and sub-pixel size/image resolution (with
    the default shift_order=9):
    +------------+-----------------+--------------+------------------+
    | hips_order | Number of Tiles | Cell Size    | Image Resolution |
    +============+=================+==============+==================+
    | 0          | 12              | 58.63 deg    | 6.871 arcmin     |
    | 1          | 48              | 29.32 deg    | 3.435 arcmin     |
    | 2          | 192             | 14.66 deg    | 1.718 arcmin     |
    | 3          | 768             | 7.329 deg    | 51.53 arcsec     |
    | 4          | 3072            | 3.665 deg    | 25.77 arcsec     |
    | 5          | 12288           | 1.832 deg    | 12.88 arcsec     |
    | 6          | 49152           | 54.97 arcmin | 6.442 arcsec     |
    | 7          | 196608          | 27.48 arcmin | 3.221 arcsec     |
    | 8          | 786432          | 13.74 arcmin | 1.61 arcsec      |
    | 9          | 3145728         | 6.871 arcmin | 805.2mas         |
    | 10         | 12582912        | 3.435 arcmin | 402.6mas         |
    | 11         | 50331648        | 1.718 arcmin | 201.3mas         |
    | 12         | 201326592       | 51.53 arcsec | 100.6mas         |
    | 13         | 805306368       | 25.77 arcsec | 50.32mas         |
    +------------+-----------------+--------------+------------------+
    """
    hips_order = pexConfig.Field(
        doc="HIPS image order.",
        dtype=int,
        default=11,
    )
    shift_order = pexConfig.Field(
        doc="HIPS shift order (such that each tile is 2**shift_order pixels on a side)",
        dtype=int,
        default=9,
    )
    warp = pexConfig.ConfigField(
        dtype=afwMath.Warper.ConfigClass,
        doc="Warper configuration",
    )

    def setDefaults(self):
        self.warp.warpingKernelName = "lanczos5"


class HighResolutionHipsTask(pipeBase.PipelineTask):
    """Task for making high resolution HIPS images."""
    ConfigClass = HighResolutionHipsConfig
    # The name should include the quantum healpix order.
    _DefaultName = "highResolutionHips9"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.warper = afwMath.Warper.fromConfig(self.config.warp)

    @timeMethod
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        healpix_dim = f"healpix{self.config.hips_order}"

        pixels = [hips_exposure.dataId[healpix_dim]
                  for hips_exposure in outputRefs.hips_exposures]

        outputs = self.run(pixels, inputs["coadd_exposure_handles"])

        hips_exposure_ref_dict = {hips_exposure_ref.dataId[healpix_dim]:
                                  hips_exposure_ref for hips_exposure_ref in outputRefs.hips_exposures}
        for pixel, hips_exposure in outputs.hips_exposures.items():
            butlerQC.put(hips_exposure, hips_exposure_ref_dict[pixel])

    def run(self, pixels, coadd_exposure_handles):
        """Run the HighResolutionHipsTask.

        Parameters
        ----------
        pixels : `Iterable` [ `int` ]
            Iterable of healpix pixels (nest ordering) to warp to.
        coadd_exposure_handles : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            Handles for the coadd exposures.

        Returns
        -------
        outputs : `lsst.pipe.base.Struct`
            ``hips_exposures`` is a dict with pixel (key) and hips_exposure (value)
        """
        self.log.info("Generating HIPS images for %d pixels at order %d", len(pixels), self.config.hips_order)

        npix = 2**self.config.shift_order
        bbox_hpx = geom.Box2I(corner=geom.Point2I(0, 0),
                              dimensions=geom.Extent2I(npix, npix))

        exp_hpx_dict = {}
        warp_dict = {}
        for pixel in pixels:
            wcs_hpx = afwGeom.makeHpxWcs(self.config.hips_order, pixel, shift_order=self.config.shift_order)
            exp_hpx = afwImage.ExposureF(bbox_hpx, wcs_hpx)
            exp_hpx_dict[pixel] = exp_hpx
            warp_dict[pixel] = []

        first_handle = True
        for handle in coadd_exposure_handles:
            coadd_exp = handle.get()

            for pixel in pixels:
                warped = self.warper.warpExposure(exp_hpx_dict[pixel].getWcs(), coadd_exp, maxBBox=bbox_hpx)

                exp = afwImage.ExposureF(exp_hpx_dict[pixel].getBBox(), exp_hpx_dict[pixel].getWcs())
                exp.maskedImage.set(np.nan, afwImage.Mask.getPlaneBitMask("NO_DATA"), np.nan)

                if first_handle:
                    exp_hpx_dict[pixel].mask.conformMaskPlanes(coadd_exp.mask.getMaskPlaneDict())
                    exp_hpx_dict[pixel].setFilterLabel(coadd_exp.getFilterLabel())
                    exp_hpx_dict[pixel].setPhotoCalib(coadd_exp.getPhotoCalib())

                if warped.getBBox().getArea() == 0 or not np.any(np.isfinite(warped.getImage().array)):
                    # There is no overlap, skip.
                    self.log.debug(
                        "No overlap between output HPX %d and input exposure %s",
                        pixel,
                        handle.dataId
                    )
                    continue

                exp.maskedImage.assign(warped.maskedImage, warped.getBBox())
                warp_dict[pixel].append(exp.maskedImage)

            first_handle = False

        stats_flags = afwMath.stringToStatisticsProperty('MEAN')
        stats_ctrl = afwMath.StatisticsControl()
        stats_ctrl.setNanSafe(True)
        stats_ctrl.setWeighted(True)
        stats_ctrl.setCalcErrorFromInputVariance(True)

        for pixel in pixels:
            exp_hpx_dict[pixel].maskedImage.set(np.nan, afwImage.Mask.getPlaneBitMask("NO_DATA"), np.nan)

            if not warp_dict[pixel]:
                # Nothing in this pixel
                self.log.debug("No data in HPX pixel %d", pixel)
                # Remove the pixel from the output
                exp_hpx_dict.pop(pixel)
                continue

            exp_hpx_dict[pixel].maskedImage = afwMath.statisticsStack(
                warp_dict[pixel],
                stats_flags,
                stats_ctrl,
                [1.0]*len(warp_dict[pixel]),
                clipped=0,
                maskMap=[]
            )

        return pipeBase.Struct(hips_exposures=exp_hpx_dict)

    @classmethod
    def build_quantum_graph(
        cls,
        task_def,
        registry,
        constraint_order,
        constraint_ranges,
        where=None,
        collections=None,
    ):
        """Generate a `QuantumGraph` for running just this task.

        This is a temporary workaround for incomplete butler query support for
        HEALPix dimensions.

        Parameters
        ----------
        task_def : `lsst.pipe.base.TaskDef`
            Task definition.
        registry : `lsst.daf.butler.Registry`
            Client for the butler database.  May be read-only.
        constraint_order : `int`
            HEALPix order used to contrain which quanta are generated, via
            ``constraint_indices``.  This should be a coarser grid (smaller
            order) than the order used for the task's quantum and output data
            IDs, and ideally something between the spatial scale of a patch or
            the data repository's "common skypix" system (usually ``htm7``).
        constraint_ranges : `lsst.sphgeom.RangeSet`
            RangeSet which describes constraint pixels (HEALPix NEST, with order
            constraint_order) to constrain generated quanta.
        where : `str`, optional
            A boolean `str` expression of the form accepted by
            `Registry.queryDatasets` to constrain input datasets.  This may
            contain a constraint on tracts, patches, or bands, but not HEALPix
            indices.  Constraints on tracts and patches should usually be
            unnecessary, however - existing coadds that overlap the given
            HEALpix indices will be selected without such a constraint, and
            providing one may reject some that should normally be included.
        collections : `str` or `Iterable` [ `str` ], optional
            Collection or collections to search for input datasets, in order.
            If not provided, ``registry.defaults.collections`` will be
            searched.
        """
        config = task_def.config

        dataset_types = pipeBase.PipelineDatasetTypes.fromPipeline(pipeline=[task_def], registry=registry)
        # Since we know this is the only task in the pipeline, we know there
        # is only one overall input and one overall output.
        (input_dataset_type,) = dataset_types.inputs

        # Extract the main output dataset type (which needs multiple
        # DatasetRefs, and tells us the output HPX level), and make a set of
        # what remains for more mechanical handling later.
        output_dataset_type = dataset_types.outputs[task_def.connections.hips_exposures.name]
        incidental_output_dataset_types = dataset_types.outputs.copy()
        incidental_output_dataset_types.remove(output_dataset_type)
        (hpx_output_dimension,) = (d for d in output_dataset_type.dimensions if d.name != "band")

        constraint_hpx_pixelization = registry.dimensions[f"healpix{constraint_order}"].pixelization
        common_skypix_name = registry.dimensions.commonSkyPix.name
        common_skypix_pixelization = registry.dimensions.commonSkyPix.pixelization

        # We will need all the pixels at the quantum resolution as well
        task_dimensions = registry.dimensions.extract(task_def.connections.dimensions)
        (hpx_dimension,) = (d for d in task_dimensions if d.name != "band")
        hpx_pixelization = hpx_dimension.pixelization

        if hpx_pixelization.level < constraint_order:
            raise ValueError(f"Quantum order {hpx_pixelization.level} must be < {constraint_order}")
        hpx_ranges = constraint_ranges.scaled(4**(hpx_pixelization.level - constraint_order))

        # We can be generous in looking for pixels here, because we constraint by actual
        # patch regions below.
        common_skypix_ranges = RangeSet()
        for begin, end in constraint_ranges:
            for hpx_index in range(begin, end):
                constraint_hpx_region = constraint_hpx_pixelization.pixel(hpx_index)
                common_skypix_ranges |= common_skypix_pixelization.envelope(constraint_hpx_region)

        # To keep the query from getting out of hand (and breaking) we simplify until we have fewer
        # than 100 ranges which seems to work fine.
        for simp in range(1, 10):
            if len(common_skypix_ranges) < 100:
                break
            common_skypix_ranges.simplify(simp)

        # Use that RangeSet to assemble a WHERE constraint expression.  This
        # could definitely get too big if the "constraint healpix" order is too
        # fine.
        where_terms = []
        bind = {}
        for n, (begin, end) in enumerate(common_skypix_ranges):
            stop = end - 1  # registry range syntax is inclusive
            if begin == stop:
                where_terms.append(f"{common_skypix_name} = cpx{n}")
                bind[f"cpx{n}"] = begin
            else:
                where_terms.append(f"({common_skypix_name} >= cpx{n}a AND {common_skypix_name} <= cpx{n}b)")
                bind[f"cpx{n}a"] = begin
                bind[f"cpx{n}b"] = stop
        if where is None:
            where = " OR ".join(where_terms)
        else:
            where = f"({where}) AND ({' OR '.join(where_terms)})"
        # Query for input datasets with this constraint, and ask for expanded
        # data IDs because we want regions.  Immediately group this by patch so
        # we don't do later geometric stuff n_bands more times than we need to.
        input_refs = registry.queryDatasets(
            input_dataset_type,
            where=where,
            findFirst=True,
            collections=collections,
            bind=bind
        ).expanded()
        inputs_by_patch = defaultdict(set)
        patch_dimensions = registry.dimensions.extract(["patch"])
        for input_ref in input_refs:
            inputs_by_patch[input_ref.dataId.subset(patch_dimensions)].add(input_ref)
        if not inputs_by_patch:
            message_body = '\n'.join(input_refs.explain_no_results())
            raise RuntimeError(f"No inputs found:\n{message_body}")

        # Iterate over patches and compute the set of output healpix pixels
        # that overlap each one.  Use that to associate inputs with output
        # pixels, but only for the output pixels we've already identified.
        inputs_by_hpx = defaultdict(set)
        for patch_data_id, input_refs_for_patch in inputs_by_patch.items():
            patch_hpx_ranges = hpx_pixelization.envelope(patch_data_id.region)
            for begin, end in patch_hpx_ranges & hpx_ranges:
                for hpx_index in range(begin, end):
                    inputs_by_hpx[hpx_index].update(input_refs_for_patch)
        # Iterate over the dict we just created and create the actual quanta.
        quanta = []
        for hpx_index, input_refs_for_hpx_index in inputs_by_hpx.items():
            # Group inputs by band.
            input_refs_by_band = defaultdict(list)
            for input_ref in input_refs_for_hpx_index:
                input_refs_by_band[input_ref.dataId["band"]].append(input_ref)
            # Iterate over bands to make quanta.
            for band, input_refs_for_band in input_refs_by_band.items():
                data_id = registry.expandDataId({hpx_dimension: hpx_index, "band": band})

                hpx_pixel_ranges = RangeSet(hpx_index)
                hpx_output_ranges = hpx_pixel_ranges.scaled(4**(config.hips_order - hpx_pixelization.level))
                output_data_ids = []
                for begin, end in hpx_output_ranges:
                    for hpx_output_index in range(begin, end):
                        output_data_ids.append(
                            registry.expandDataId({hpx_output_dimension: hpx_output_index, "band": band})
                        )
                outputs = {dt: [DatasetRef(dt, data_id)] for dt in incidental_output_dataset_types}
                outputs[output_dataset_type] = [DatasetRef(output_dataset_type, data_id)
                                                for data_id in output_data_ids]
                quanta.append(
                    Quantum(
                        taskName=task_def.taskName,
                        taskClass=task_def.taskClass,
                        dataId=data_id,
                        initInputs={},
                        inputs={input_dataset_type: input_refs_for_band},
                        outputs=outputs,
                    )
                )

        if len(quanta) == 0:
            raise RuntimeError("Given constraints yielded empty quantum graph.")

        return pipeBase.QuantumGraph(quanta={task_def: quanta})
