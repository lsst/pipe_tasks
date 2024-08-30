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

"""Tasks for making and manipulating HIPS images."""

__all__ = ["HighResolutionHipsTask", "HighResolutionHipsConfig", "HighResolutionHipsConnections",
           "HighResolutionHipsQuantumGraphBuilder",
           "GenerateHipsTask", "GenerateHipsConfig", "GenerateColorHipsTask", "GenerateColorHipsConfig"]

from collections import defaultdict
import numpy as np
import argparse
import io
import sys
import re
import warnings
import math
from datetime import datetime
import hpgeom as hpg
import healsparse as hsp
from astropy.io import fits
try:
    from astropy.visualization.lupton_rgb import AsinhMapping
except ImportError:
    from ._fallback_asinhmapping import AsinhMapping
from PIL import Image

from lsst.sphgeom import RangeSet, HealpixPixelization
from lsst.utils.timer import timeMethod
from lsst.daf.butler import Butler
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
from lsst.pipe.base.quantum_graph_builder import QuantumGraphBuilder
from lsst.pipe.base.quantum_graph_skeleton import QuantumGraphSkeleton
import lsst.afw.geom as afwGeom
import lsst.afw.math as afwMath
import lsst.afw.image as afwImage
import lsst.geom as geom
from lsst.afw.geom import makeHpxWcs
from lsst.resources import ResourcePath

from .healSparseMapping import _is_power_of_two


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
        doc="HiPS-compatible HPX image.",
        name="{coaddName}Coadd_hpx",
        storageClass="ExposureF",
        dimensions=("healpix11", "band"),
        multiple=True,
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        quantum_order = None
        for dim in self.dimensions:
            if "healpix" in dim:
                if quantum_order is not None:
                    raise ValueError("Must not specify more than one quantum healpix dimension.")
                quantum_order = int(dim.split("healpix")[1])
        if quantum_order is None:
            raise ValueError("Must specify a healpix dimension in quantum dimensions.")

        if quantum_order > config.hips_order:
            raise ValueError("Quantum healpix dimension order must not be greater than hips_order")

        order = None
        for dim in self.hips_exposures.dimensions:
            if "healpix" in dim:
                if order is not None:
                    raise ValueError("Must not specify more than one healpix dimension.")
                order = int(dim.split("healpix")[1])
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


class HipsTaskNameDescriptor:
    """Descriptor used create a DefaultName that matches the order of
    the defined dimensions in the connections class.

    Parameters
    ----------
    prefix : `str`
        The prefix of the Default name, to which the order will be
        appended.
    """
    def __init__(self, prefix):
        # create a defaultName template
        self._defaultName = f"{prefix}{{}}"
        self._order = None

    def __get__(self, obj, klass=None):
        if klass is None:
            raise RuntimeError(
                "HipsTaskDescriptor was used in an unexpected context"
            )
        if self._order is None:
            klassDimensions = klass.ConfigClass.ConnectionsClass.dimensions
            for dim in klassDimensions:
                if (match := re.match(r"^healpix(\d*)$", dim)) is not None:
                    self._order = int(match.group(1))
                    break
            else:
                raise RuntimeError(
                    "Could not find healpix dimension in connections class"
                )
        return self._defaultName.format(self._order)


class HighResolutionHipsTask(pipeBase.PipelineTask):
    """Task for making high resolution HiPS images."""
    ConfigClass = HighResolutionHipsConfig
    _DefaultName = HipsTaskNameDescriptor("highResolutionHips")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.warper = afwMath.Warper.fromConfig(self.config.warp)

    @timeMethod
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        healpix_dim = f"healpix{self.config.hips_order}"

        pixels = [hips_exposure.dataId[healpix_dim]
                  for hips_exposure in outputRefs.hips_exposures]

        outputs = self.run(pixels=pixels, coadd_exposure_handles=inputs["coadd_exposure_handles"])

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
        self.log.info("Generating HPX images for %d pixels at order %d", len(pixels), self.config.hips_order)

        npix = 2**self.config.shift_order
        bbox_hpx = geom.Box2I(corner=geom.Point2I(0, 0),
                              dimensions=geom.Extent2I(npix, npix))

        # For each healpix pixel we will create an empty exposure with the
        # correct HPX WCS. We furthermore create a dict to hold each of
        # the warps that will go into each HPX exposure.
        exp_hpx_dict = {}
        warp_dict = {}
        for pixel in pixels:
            wcs_hpx = afwGeom.makeHpxWcs(self.config.hips_order, pixel, shift_order=self.config.shift_order)
            exp_hpx = afwImage.ExposureF(bbox_hpx, wcs_hpx)
            exp_hpx_dict[pixel] = exp_hpx
            warp_dict[pixel] = []

        first_handle = True
        # Loop over input coadd exposures to minimize i/o (this speeds things
        # up by ~8x to batch together pixels that overlap a given coadd).
        for handle in coadd_exposure_handles:
            coadd_exp = handle.get()

            # For each pixel, warp the coadd to the HPX WCS for the pixel.
            for pixel in pixels:
                warped = self.warper.warpExposure(exp_hpx_dict[pixel].getWcs(), coadd_exp, maxBBox=bbox_hpx)

                exp = afwImage.ExposureF(exp_hpx_dict[pixel].getBBox(), exp_hpx_dict[pixel].getWcs())
                exp.maskedImage.set(np.nan, afwImage.Mask.getPlaneBitMask("NO_DATA"), np.nan)

                if first_handle:
                    # Make sure the mask planes, filter, and photocalib of the output
                    # exposure match the (first) input exposure.
                    exp_hpx_dict[pixel].mask.conformMaskPlanes(coadd_exp.mask.getMaskPlaneDict())
                    exp_hpx_dict[pixel].setFilter(coadd_exp.getFilter())
                    exp_hpx_dict[pixel].setPhotoCalib(coadd_exp.getPhotoCalib())

                if warped.getBBox().getArea() == 0 or not np.any(np.isfinite(warped.image.array)):
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

        stats_flags = afwMath.stringToStatisticsProperty("MEAN")
        stats_ctrl = afwMath.StatisticsControl()
        stats_ctrl.setNanSafe(True)
        stats_ctrl.setWeighted(True)
        stats_ctrl.setCalcErrorFromInputVariance(True)

        # Loop over pixels and combine the warps for each pixel.
        # The combination is done with a simple mean for pixels that
        # overlap in neighboring patches.
        for pixel in pixels:
            exp_hpx_dict[pixel].maskedImage.set(np.nan, afwImage.Mask.getPlaneBitMask("NO_DATA"), np.nan)

            if not warp_dict[pixel]:
                # Nothing in this pixel
                self.log.debug("No data in HPX pixel %d", pixel)
                # Remove the pixel from the output, no need to persist an
                # empty exposure.
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
    def build_quantum_graph_cli(cls, argv):
        """A command-line interface entry point to `build_quantum_graph`.
        This method provides the implementation for the
        ``build-high-resolution-hips-qg`` script.

        Parameters
        ----------
        argv : `Sequence` [ `str` ]
            Command-line arguments (e.g. ``sys.argv[1:]``).
        """
        parser = cls._make_cli_parser()

        args = parser.parse_args(argv)

        if args.subparser_name is None:
            parser.print_help()
            sys.exit(1)

        pipeline = pipeBase.Pipeline.from_uri(args.pipeline)
        pipeline_graph = pipeline.to_graph()

        if len(pipeline_graph.tasks) != 1:
            raise RuntimeError(f"Pipeline file {args.pipeline} may only contain one task.")

        (task_node,) = pipeline_graph.tasks.values()

        butler = Butler(args.butler_config, collections=args.input)

        if args.subparser_name == "segment":
            # Do the segmentation
            hpix_pixelization = HealpixPixelization(level=args.hpix_build_order)
            dataset = task_node.inputs["coadd_exposure_handles"].dataset_type_name
            data_ids = set(butler.registry.queryDataIds("tract", datasets=dataset).expanded())
            region_pixels = []
            for data_id in data_ids:
                region = data_id.region
                pixel_range = hpix_pixelization.envelope(region)
                for r in pixel_range.ranges():
                    region_pixels.extend(range(r[0], r[1]))
            indices = np.unique(region_pixels)

            print(f"Pixels to run at HEALPix order --hpix_build_order {args.hpix_build_order}:")
            for pixel in indices:
                print(pixel)

        elif args.subparser_name == "build":
            # Build the quantum graph.

            # Figure out collection names.
            if args.output_run is None:
                if args.output is None:
                    raise ValueError("At least one of --output or --output-run options is required.")
                args.output_run = "{}/{}".format(args.output, pipeBase.Instrument.makeCollectionTimestamp())

            build_ranges = RangeSet(sorted(args.pixels))

            # Metadata includes a subset of attributes defined in CmdLineFwk.
            metadata = {
                "input": args.input,
                "butler_argument": args.butler_config,
                "output": args.output,
                "output_run": args.output_run,
                "data_query": args.where,
                "time": f"{datetime.now()}",
            }

            builder = HighResolutionHipsQuantumGraphBuilder(
                pipeline_graph,
                butler,
                input_collections=args.input,
                output_run=args.output_run,
                constraint_order=args.hpix_build_order,
                constraint_ranges=build_ranges,
                where=args.where,
            )
            qg = builder.build(metadata, attach_datastore_records=True)
            qg.saveUri(args.save_qgraph)

    @classmethod
    def _make_cli_parser(cls):
        """Make the command-line parser.

        Returns
        -------
        parser : `argparse.ArgumentParser`
        """
        parser = argparse.ArgumentParser(
            description=(
                "Build a QuantumGraph that runs HighResolutionHipsTask on existing coadd datasets."
            ),
        )
        subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")

        parser_segment = subparsers.add_parser("segment",
                                               help="Determine survey segments for workflow.")
        parser_build = subparsers.add_parser("build",
                                             help="Build quantum graph for HighResolutionHipsTask")

        for sub in [parser_segment, parser_build]:
            # These arguments are in common.
            sub.add_argument(
                "-b",
                "--butler-config",
                type=str,
                help="Path to data repository or butler configuration.",
                required=True,
            )
            sub.add_argument(
                "-p",
                "--pipeline",
                type=str,
                help="Pipeline file, limited to one task.",
                required=True,
            )
            sub.add_argument(
                "-i",
                "--input",
                type=str,
                nargs="+",
                help="Input collection(s) to search for coadd exposures.",
                required=True,
            )
            sub.add_argument(
                "-o",
                "--hpix_build_order",
                type=int,
                default=1,
                help="HEALPix order to segment sky for building quantum graph files.",
            )
            sub.add_argument(
                "-w",
                "--where",
                type=str,
                default=None,
                help="Data ID expression used when querying for input coadd datasets.",
            )

        parser_build.add_argument(
            "--output",
            type=str,
            help=(
                "Name of the output CHAINED collection. If this options is specified and "
                "--output-run is not, then a new RUN collection will be created by appending "
                "a timestamp to the value of this option."
            ),
            default=None,
            metavar="COLL",
        )
        parser_build.add_argument(
            "--output-run",
            type=str,
            help=(
                "Output RUN collection to write resulting images. If not provided "
                "then --output must be provided and a new RUN collection will be created "
                "by appending a timestamp to the value passed with --output."
            ),
            default=None,
            metavar="RUN",
        )
        parser_build.add_argument(
            "-q",
            "--save-qgraph",
            type=str,
            help="Output filename for QuantumGraph.",
            required=True,
        )
        parser_build.add_argument(
            "-P",
            "--pixels",
            type=int,
            nargs="+",
            help="Pixels at --hpix_build_order to generate quantum graph.",
            required=True,
        )

        return parser


class HighResolutionHipsQuantumGraphBuilder(QuantumGraphBuilder):
    """A custom a `lsst.pipe.base.QuantumGraphBuilder` for running
    `HighResolutionHipsTask` only.

    This is a workaround for incomplete butler query support for HEALPix
    dimensions.

    Parameters
    ----------
    pipeline_graph : `lsst.pipe.base.PipelineGraph`
        Pipeline graph with exactly one task, which must be a configuration
        of `HighResolutionHipsTask`.
    butler : `lsst.daf.butler.Butler`
        Client for the butler data repository.  May be read-only.
    input_collections : `str` or `Iterable` [ `str` ], optional
        Collection or collections to search for input datasets, in order.
        If not provided, ``butler.collections`` will be searched.
    output_run : `str`, optional
        Name of the output collection.  If not provided, ``butler.run`` will
        be used.
    constraint_order : `int`
        HEALPix order used to constrain which quanta are generated, via
        ``constraint_indices``.  This should be a coarser grid (smaller
        order) than the order used for the task's quantum and output data
        IDs, and ideally something between the spatial scale of a patch or
        the data repository's "common skypix" system (usually ``htm7``).
    constraint_ranges : `lsst.sphgeom.RangeSet`
        RangeSet that describes constraint pixels (HEALPix NEST, with order
        ``constraint_order``) to constrain generated quanta.
    where : `str`, optional
        A boolean `str` expression of the form accepted by
        `Registry.queryDatasets` to constrain input datasets.  This may
        contain a constraint on tracts, patches, or bands, but not HEALPix
        indices.  Constraints on tracts and patches should usually be
        unnecessary, however - existing coadds that overlap the given
        HEALpix indices will be selected without such a constraint, and
        providing one may reject some that should normally be included.
    """

    def __init__(
        self,
        pipeline_graph,
        butler,
        *,
        input_collections=None,
        output_run=None,
        constraint_order,
        constraint_ranges,
        where="",
    ):
        super().__init__(pipeline_graph, butler, input_collections=input_collections, output_run=output_run)
        self.constraint_order = constraint_order
        self.constraint_ranges = constraint_ranges
        self.where = where

    def process_subgraph(self, subgraph):
        # Docstring inherited.
        (task_node,) = subgraph.tasks.values()

        # Since we know this is the only task in the pipeline, we know there
        # is only one overall input and one regular output.
        (input_dataset_type_node,) = subgraph.inputs_of(task_node.label).values()
        assert input_dataset_type_node is not None, "PipelineGraph should be resolved by base class."
        (output_edge,) = task_node.outputs.values()
        output_dataset_type_node = subgraph.dataset_types[output_edge.parent_dataset_type_name]
        (hpx_output_dimension,) = (
            self.butler.dimensions.skypix_dimensions[d]
            for d in output_dataset_type_node.dimensions.skypix
        )
        constraint_hpx_pixelization = (
            self.butler.dimensions.skypix_dimensions[f"healpix{self.constraint_order}"].pixelization
        )
        common_skypix_name = self.butler.dimensions.commonSkyPix.name
        common_skypix_pixelization = self.butler.dimensions.commonSkyPix.pixelization

        # We will need all the pixels at the quantum resolution as well.
        # '4' appears here frequently because it's the number of pixels at
        # level N in a single pixel at level N.
        (hpx_dimension,) = (
            self.butler.dimensions.skypix_dimensions[d] for d in task_node.dimensions.names if d != "band"
        )
        hpx_pixelization = hpx_dimension.pixelization
        if hpx_pixelization.level < self.constraint_order:
            raise ValueError(f"Quantum order {hpx_pixelization.level} must be < {self.constraint_order}")
        hpx_ranges = self.constraint_ranges.scaled(4**(hpx_pixelization.level - self.constraint_order))

        # We can be generous in looking for pixels here, because we constrain
        # by actual patch regions below.
        common_skypix_ranges = RangeSet()
        for begin, end in self.constraint_ranges:
            for hpx_index in range(begin, end):
                constraint_hpx_region = constraint_hpx_pixelization.pixel(hpx_index)
                common_skypix_ranges |= common_skypix_pixelization.envelope(constraint_hpx_region)

        # To keep the query from getting out of hand (and breaking) we simplify
        # until we have fewer than 100 ranges which seems to work fine.
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
        where = " OR ".join(where_terms)
        if self.where:
            where = f"({self.where}) AND ({where})"
        # Query for input datasets with this constraint, and ask for expanded
        # data IDs because we want regions.  Immediately group this by patch so
        # we don't do later geometric stuff n_bands more times than we need to.
        input_refs = self.butler.registry.queryDatasets(
            input_dataset_type_node.dataset_type,
            where=where,
            findFirst=True,
            collections=self.input_collections,
            bind=bind
        ).expanded()
        inputs_by_patch = defaultdict(set)
        patch_dimensions = self.butler.dimensions.conform(["patch"])
        skeleton = QuantumGraphSkeleton([task_node.label])
        for input_ref in input_refs:
            dataset_key = skeleton.add_dataset_node(input_ref.datasetType.name, input_ref.dataId)
            skeleton.set_dataset_ref(input_ref, dataset_key)
            inputs_by_patch[input_ref.dataId.subset(patch_dimensions)].add(dataset_key)
        if not inputs_by_patch:
            message_body = "\n".join(input_refs.explain_no_results())
            raise RuntimeError(f"No inputs found:\n{message_body}")

        # Iterate over patches and compute the set of output healpix pixels
        # that overlap each one.  Use that to associate inputs with output
        # pixels, but only for the output pixels we've already identified.
        inputs_by_hpx = defaultdict(set)
        for patch_data_id, input_keys_for_patch in inputs_by_patch.items():
            patch_hpx_ranges = hpx_pixelization.envelope(patch_data_id.region)
            for begin, end in patch_hpx_ranges & hpx_ranges:
                for hpx_index in range(begin, end):
                    inputs_by_hpx[hpx_index].update(input_keys_for_patch)

        # Iterate over the dict we just created and create preliminary quanta.
        for hpx_index, input_keys_for_hpx_index in inputs_by_hpx.items():
            # Group inputs by band.
            input_keys_by_band = defaultdict(list)
            for input_key in input_keys_for_hpx_index:
                input_ref = skeleton.get_dataset_ref(input_key)
                assert input_ref is not None, "Code above adds the same nodes to the graph with refs."
                input_keys_by_band[input_ref.dataId["band"]].append(input_key)
            # Iterate over bands to make quanta.
            for band, input_keys_for_band in input_keys_by_band.items():
                data_id = self.butler.registry.expandDataId({hpx_dimension.name: hpx_index, "band": band})
                quantum_key = skeleton.add_quantum_node(task_node.label, data_id)
                # Add inputs to the skelton
                skeleton.add_input_edges(quantum_key, input_keys_for_band)
                # Add the regular outputs.
                hpx_pixel_ranges = RangeSet(hpx_index)
                hpx_output_ranges = hpx_pixel_ranges.scaled(
                    4**(task_node.config.hips_order - hpx_pixelization.level)
                )
                for begin, end in hpx_output_ranges:
                    for hpx_output_index in range(begin, end):
                        dataset_key = skeleton.add_dataset_node(
                            output_dataset_type_node.name,
                            self.butler.registry.expandDataId(
                                {hpx_output_dimension: hpx_output_index, "band": band}
                            ),
                        )
                        skeleton.add_output_edge(quantum_key, dataset_key)
                # Add auxiliary outputs (log, metadata).
                for write_edge in task_node.iter_all_outputs():
                    if write_edge.connection_name == output_edge.connection_name:
                        continue
                    dataset_key = skeleton.add_dataset_node(write_edge.parent_dataset_type_name, data_id)
                    skeleton.add_output_edge(quantum_key, dataset_key)
        return skeleton


class HipsPropertiesSpectralTerm(pexConfig.Config):
    lambda_min = pexConfig.Field(
        doc="Minimum wavelength (nm)",
        dtype=float,
    )
    lambda_max = pexConfig.Field(
        doc="Maximum wavelength (nm)",
        dtype=float,
    )


class HipsPropertiesConfig(pexConfig.Config):
    """Configuration parameters for writing a HiPS properties file."""
    creator_did_template = pexConfig.Field(
        doc=("Unique identifier of the HiPS - Format: IVOID. "
             "Use ``{band}`` to substitute the band name."),
        dtype=str,
        optional=False,
    )
    obs_collection = pexConfig.Field(
        doc="Short name of original data set - Format: one word",
        dtype=str,
        optional=True,
    )
    obs_description_template = pexConfig.Field(
        doc=("Data set description - Format: free text, longer free text "
             "description of the dataset.  Use ``{band}`` to substitute "
             "the band name."),
        dtype=str,
    )
    prov_progenitor = pexConfig.ListField(
        doc="Provenance of the original data - Format: free text",
        dtype=str,
        default=[],
    )
    obs_title_template = pexConfig.Field(
        doc=("Data set title format: free text, but should be short. "
             "Use ``{band}`` to substitute the band name."),
        dtype=str,
        optional=False,
    )
    spectral_ranges = pexConfig.ConfigDictField(
        doc=("Mapping from band to lambda_min, lamba_max (nm).  May be approximate."),
        keytype=str,
        itemtype=HipsPropertiesSpectralTerm,
        default={},
    )
    initial_ra = pexConfig.Field(
        doc="Initial RA (deg) (default for HiPS viewer).  If not set will use a point in MOC.",
        dtype=float,
        optional=True,
    )
    initial_dec = pexConfig.Field(
        doc="Initial Declination (deg) (default for HiPS viewer).  If not set will use a point in MOC.",
        dtype=float,
        optional=True,
    )
    initial_fov = pexConfig.Field(
        doc="Initial field-of-view (deg).  If not set will use ~1 healpix tile.",
        dtype=float,
        optional=True,
    )
    obs_ack = pexConfig.Field(
        doc="Observation acknowledgements (free text).",
        dtype=str,
        optional=True,
    )
    t_min = pexConfig.Field(
        doc="Time (MJD) of earliest observation included in HiPS",
        dtype=float,
        optional=True,
    )
    t_max = pexConfig.Field(
        doc="Time (MJD) of latest observation included in HiPS",
        dtype=float,
        optional=True,
    )

    def validate(self):
        super().validate()

        if self.obs_collection is not None:
            if re.search(r"\s", self.obs_collection):
                raise ValueError("obs_collection cannot contain any space characters.")

    def setDefaults(self):
        # Values here taken from
        # https://github.com/lsst-dm/dax_obscore/blob/44ac15029136e2ec15/configs/dp02.yaml#L46
        u_term = HipsPropertiesSpectralTerm()
        u_term.lambda_min = 330.
        u_term.lambda_max = 400.
        self.spectral_ranges["u"] = u_term
        g_term = HipsPropertiesSpectralTerm()
        g_term.lambda_min = 402.
        g_term.lambda_max = 552.
        self.spectral_ranges["g"] = g_term
        r_term = HipsPropertiesSpectralTerm()
        r_term.lambda_min = 552.
        r_term.lambda_max = 691.
        self.spectral_ranges["r"] = r_term
        i_term = HipsPropertiesSpectralTerm()
        i_term.lambda_min = 691.
        i_term.lambda_max = 818.
        self.spectral_ranges["i"] = i_term
        z_term = HipsPropertiesSpectralTerm()
        z_term.lambda_min = 818.
        z_term.lambda_max = 922.
        self.spectral_ranges["z"] = z_term
        y_term = HipsPropertiesSpectralTerm()
        y_term.lambda_min = 970.
        y_term.lambda_max = 1060.
        self.spectral_ranges["y"] = y_term


class GenerateHipsConnections(pipeBase.PipelineTaskConnections,
                              dimensions=("instrument", "band"),
                              defaultTemplates={"coaddName": "deep"}):
    hips_exposure_handles = pipeBase.connectionTypes.Input(
        doc="HiPS-compatible HPX images.",
        name="{coaddName}Coadd_hpx",
        storageClass="ExposureF",
        dimensions=("healpix11", "band"),
        multiple=True,
        deferLoad=True,
    )


class GenerateHipsConfig(pipeBase.PipelineTaskConfig,
                         pipelineConnections=GenerateHipsConnections):
    """Configuration parameters for GenerateHipsTask."""
    # WARNING: In general PipelineTasks are not allowed to do any outputs
    # outside of the butler.  This task has been given (temporary)
    # Special Dispensation because of the nature of HiPS outputs until
    # a more controlled solution can be found.
    hips_base_uri = pexConfig.Field(
        doc="URI to HiPS base for output.",
        dtype=str,
        optional=False,
    )
    min_order = pexConfig.Field(
        doc="Minimum healpix order for HiPS tree.",
        dtype=int,
        default=3,
    )
    properties = pexConfig.ConfigField(
        dtype=HipsPropertiesConfig,
        doc="Configuration for properties file.",
    )
    allsky_tilesize = pexConfig.Field(
        dtype=int,
        doc="Allsky tile size; must be power of 2. HiPS standard recommends 64x64 tiles.",
        default=64,
        check=_is_power_of_two,
    )
    png_gray_asinh_minimum = pexConfig.Field(
        doc="AsinhMapping intensity to be mapped to black for grayscale png scaling.",
        dtype=float,
        default=0.0,
    )
    png_gray_asinh_stretch = pexConfig.Field(
        doc="AsinhMapping linear stretch for grayscale png scaling.",
        dtype=float,
        default=2.0,
    )
    png_gray_asinh_softening = pexConfig.Field(
        doc="AsinhMapping softening parameter (Q) for grayscale png scaling.",
        dtype=float,
        default=8.0,
    )


class GenerateHipsTask(pipeBase.PipelineTask):
    """Task for making a HiPS tree with FITS and grayscale PNGs."""
    ConfigClass = GenerateHipsConfig
    _DefaultName = "generateHips"
    color_task = False

    @timeMethod
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = butlerQC.get(inputRefs)

        dims = inputRefs.hips_exposure_handles[0].dataId.dimensions.names
        order = None
        for dim in dims:
            if "healpix" in dim:
                order = int(dim.split("healpix")[1])
                healpix_dim = dim
                break
        else:
            raise RuntimeError("Could not determine healpix order for input exposures.")

        hips_exposure_handle_dict = {
            (hips_exposure_handle.dataId[healpix_dim],
             hips_exposure_handle.dataId["band"]): hips_exposure_handle
            for hips_exposure_handle in inputs["hips_exposure_handles"]
        }

        data_bands = {hips_exposure_handle.dataId["band"]
                      for hips_exposure_handle in inputs["hips_exposure_handles"]}
        bands = self._check_data_bands(data_bands)

        self.run(
            bands=bands,
            max_order=order,
            hips_exposure_handle_dict=hips_exposure_handle_dict,
            do_color=self.color_task,
        )

    def _check_data_bands(self, data_bands):
        """Check that the data has only a single band.

        Parameters
        ----------
        data_bands : `set` [`str`]
            Bands from the input data.

        Returns
        -------
        bands : `list` [`str`]
            List of single band to process.

        Raises
        ------
        RuntimeError if there is not exactly one band.
        """
        if len(data_bands) != 1:
            raise RuntimeError("GenerateHipsTask can only use data from a single band.")

        return list(data_bands)

    @timeMethod
    def run(self, bands, max_order, hips_exposure_handle_dict, do_color=False):
        """Run the GenerateHipsTask.

        Parameters
        ----------
        bands : `list [ `str` ]
            List of bands to be processed (or single band).
        max_order : `int`
            HEALPix order of the maximum (native) HPX exposures.
        hips_exposure_handle_dict : `dict` {`int`: `lsst.daf.butler.DeferredDatasetHandle`}
            Dict of handles for the HiPS high-resolution exposures.
            Key is (pixel number, ``band``).
        do_color : `bool`, optional
            Do color pngs instead of per-band grayscale.
        """
        min_order = self.config.min_order

        if not do_color:
            png_grayscale_mapping = AsinhMapping(
                self.config.png_gray_asinh_minimum,
                self.config.png_gray_asinh_stretch,
                Q=self.config.png_gray_asinh_softening,
            )
        else:
            png_color_mapping = AsinhMapping(
                self.config.png_color_asinh_minimum,
                self.config.png_color_asinh_stretch,
                Q=self.config.png_color_asinh_softening,
            )

            bcb = self.config.blue_channel_band
            gcb = self.config.green_channel_band
            rcb = self.config.red_channel_band
            colorstr = f"{bcb}{gcb}{rcb}"

        # The base path is based on the hips_base_uri.
        hips_base_path = ResourcePath(self.config.hips_base_uri, forceDirectory=True)

        # We need to unique-ify the pixels because they show up for multiple bands.
        # The output of this is a sorted array.
        pixels = np.unique(np.array([pixel for pixel, _ in hips_exposure_handle_dict.keys()]))

        # Add a "gutter" pixel at the end.  Start with 0 which maps to 0 always.
        pixels = np.append(pixels, [0])

        # Convert the pixels to each order that will be generated.
        pixels_shifted = {}
        pixels_shifted[max_order] = pixels
        for order in range(max_order - 1, min_order - 1, -1):
            pixels_shifted[order] = np.right_shift(pixels_shifted[order + 1], 2)

        # And set the gutter to an illegal pixel value.
        for order in range(min_order, max_order + 1):
            pixels_shifted[order][-1] = -1

        # Read in the first pixel for determining image properties.
        exp0 = list(hips_exposure_handle_dict.values())[0].get()
        bbox = exp0.getBBox()
        npix = bbox.getWidth()
        shift_order = int(np.round(np.log2(npix)))

        # Create blank exposures for each level, including the highest order.
        # We also make sure we create blank exposures for any bands used in the color
        # PNGs, even if they aren't available.
        exposures = {}
        for band in bands:
            for order in range(min_order, max_order + 1):
                exp = exp0.Factory(bbox=bbox)
                exp.image.array[:, :] = np.nan
                exposures[(band, order)] = exp

        # Loop over all pixels, avoiding the gutter.
        for pixel_counter, pixel in enumerate(pixels[:-1]):
            self.log.debug("Working on high resolution pixel %d", pixel)
            for band in bands:
                # Read all the exposures here for the highest order.
                # There will always be at least one band with a HiPS image available
                # at the highest order. However, for color images it is possible that
                # not all bands have coverage so we require this check.
                if (pixel, band) in hips_exposure_handle_dict:
                    exposures[(band, max_order)] = hips_exposure_handle_dict[(pixel, band)].get()

            # Go up the HiPS tree.
            # We only write pixels and rebin to fill the parent pixel when we are
            # done with a current pixel, which is determined if the next pixel
            # has a different pixel number.
            for order in range(max_order, min_order - 1, -1):
                if pixels_shifted[order][pixel_counter + 1] == pixels_shifted[order][pixel_counter]:
                    # This order is not done, and so none of the other orders will be.
                    break

                # We can now write out the images for each band.
                # Note this will always trigger at the max order where each pixel is unique.
                if not do_color:
                    for band in bands:
                        self._write_hips_image(
                            hips_base_path.join(f"band_{band}", forceDirectory=True),
                            order,
                            pixels_shifted[order][pixel_counter],
                            exposures[(band, order)].image,
                            png_grayscale_mapping,
                            shift_order=shift_order,
                        )
                else:
                    # Make a color png.
                    self._write_hips_color_png(
                        hips_base_path.join(f"color_{colorstr}", forceDirectory=True),
                        order,
                        pixels_shifted[order][pixel_counter],
                        exposures[(self.config.red_channel_band, order)].image,
                        exposures[(self.config.green_channel_band, order)].image,
                        exposures[(self.config.blue_channel_band, order)].image,
                        png_color_mapping,
                    )

                log_level = self.log.INFO if order == (max_order - 3) else self.log.DEBUG
                self.log.log(
                    log_level,
                    "Completed HiPS generation for %s, order %d, pixel %d (%d/%d)",
                    ",".join(bands),
                    order,
                    pixels_shifted[order][pixel_counter],
                    pixel_counter,
                    len(pixels) - 1,
                )

                # When we are at the top of the tree, erase top level images and continue.
                if order == min_order:
                    for band in bands:
                        exposures[(band, order)].image.array[:, :] = np.nan
                    continue

                # Now average the images for each band.
                for band in bands:
                    arr = exposures[(band, order)].image.array.reshape(npix//2, 2, npix//2, 2)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        binned_image_arr = np.nanmean(arr, axis=(1, 3))

                    # Fill the next level up.  We figure out which of the four
                    # sub-pixels the current pixel occupies.
                    sub_index = (pixels_shifted[order][pixel_counter]
                                 - np.left_shift(pixels_shifted[order - 1][pixel_counter], 2))

                    # Fill exposure at the next level up.
                    exp = exposures[(band, order - 1)]

                    # Fill the correct subregion.
                    if sub_index == 0:
                        exp.image.array[npix//2:, 0: npix//2] = binned_image_arr
                    elif sub_index == 1:
                        exp.image.array[0: npix//2, 0: npix//2] = binned_image_arr
                    elif sub_index == 2:
                        exp.image.array[npix//2:, npix//2:] = binned_image_arr
                    elif sub_index == 3:
                        exp.image.array[0: npix//2, npix//2:] = binned_image_arr
                    else:
                        # This should be impossible.
                        raise ValueError("Illegal pixel sub index")

                    # Erase the previous exposure.
                    if order < max_order:
                        exposures[(band, order)].image.array[:, :] = np.nan

        # Write the properties files and MOCs.
        if not do_color:
            for band in bands:
                band_pixels = np.array([pixel
                                        for pixel, band_ in hips_exposure_handle_dict.keys()
                                        if band_ == band])
                band_pixels = np.sort(band_pixels)

                self._write_properties_and_moc(
                    hips_base_path.join(f"band_{band}", forceDirectory=True),
                    max_order,
                    band_pixels,
                    exp0,
                    shift_order,
                    band,
                    False,
                )
                self._write_allsky_file(
                    hips_base_path.join(f"band_{band}", forceDirectory=True),
                    min_order,
                )
        else:
            self._write_properties_and_moc(
                hips_base_path.join(f"color_{colorstr}", forceDirectory=True),
                max_order,
                pixels[:-1],
                exp0,
                shift_order,
                colorstr,
                True,
            )
            self._write_allsky_file(
                hips_base_path.join(f"color_{colorstr}", forceDirectory=True),
                min_order,
            )

    def _write_hips_image(self, hips_base_path, order, pixel, image, png_mapping, shift_order=9):
        """Write a HiPS image.

        Parameters
        ----------
        hips_base_path : `lsst.resources.ResourcePath`
            Resource path to the base of the HiPS directory tree.
        order : `int`
            HEALPix order of the HiPS image to write.
        pixel : `int`
            HEALPix pixel of the HiPS image.
        image : `lsst.afw.image.Image`
            Image to write.
        png_mapping : `astropy.visualization.lupton_rgb.AsinhMapping`
            Mapping to convert image to scaled png.
        shift_order : `int`, optional
            HPX shift_order.
        """
        # WARNING: In general PipelineTasks are not allowed to do any outputs
        # outside of the butler.  This task has been given (temporary)
        # Special Dispensation because of the nature of HiPS outputs until
        # a more controlled solution can be found.

        dir_number = self._get_dir_number(pixel)
        hips_dir = hips_base_path.join(
            f"Norder{order}",
            forceDirectory=True
        ).join(
            f"Dir{dir_number}",
            forceDirectory=True
        )

        wcs = makeHpxWcs(order, pixel, shift_order=shift_order)

        uri = hips_dir.join(f"Npix{pixel}.fits")

        with ResourcePath.temporary_uri(suffix=uri.getExtension()) as temporary_uri:
            image.writeFits(temporary_uri.ospath, metadata=wcs.getFitsMetadata())

            uri.transfer_from(temporary_uri, transfer="copy", overwrite=True)

        # And make a grayscale png as well

        with np.errstate(invalid="ignore"):
            vals = 255 - png_mapping.map_intensity_to_uint8(image.array).astype(np.uint8)

        vals[~np.isfinite(image.array) | (image.array < 0)] = 0
        im = Image.fromarray(vals[::-1, :], "L")

        uri = hips_dir.join(f"Npix{pixel}.png")

        with ResourcePath.temporary_uri(suffix=uri.getExtension()) as temporary_uri:
            im.save(temporary_uri.ospath)

            uri.transfer_from(temporary_uri, transfer="copy", overwrite=True)

    def _write_hips_color_png(
            self,
            hips_base_path,
            order,
            pixel,
            image_red,
            image_green,
            image_blue,
            png_mapping,
    ):
        """Write a color png HiPS image.

        Parameters
        ----------
        hips_base_path : `lsst.resources.ResourcePath`
            Resource path to the base of the HiPS directory tree.
        order : `int`
            HEALPix order of the HiPS image to write.
        pixel : `int`
            HEALPix pixel of the HiPS image.
        image_red : `lsst.afw.image.Image`
            Input for red channel of output png.
        image_green : `lsst.afw.image.Image`
            Input for green channel of output png.
        image_blue : `lsst.afw.image.Image`
            Input for blue channel of output png.
        png_mapping : `astropy.visualization.lupton_rgb.AsinhMapping`
            Mapping to convert image to scaled png.
        """
        # WARNING: In general PipelineTasks are not allowed to do any outputs
        # outside of the butler.  This task has been given (temporary)
        # Special Dispensation because of the nature of HiPS outputs until
        # a more controlled solution can be found.

        dir_number = self._get_dir_number(pixel)
        hips_dir = hips_base_path.join(
            f"Norder{order}",
            forceDirectory=True
        ).join(
            f"Dir{dir_number}",
            forceDirectory=True
        )

        # We need to convert nans to the minimum values in the mapping.
        arr_red = image_red.array.copy()
        arr_red[np.isnan(arr_red)] = png_mapping.minimum[0]
        arr_green = image_green.array.copy()
        arr_green[np.isnan(arr_green)] = png_mapping.minimum[1]
        arr_blue = image_blue.array.copy()
        arr_blue[np.isnan(arr_blue)] = png_mapping.minimum[2]

        image_array = png_mapping.make_rgb_image(arr_red, arr_green, arr_blue)

        im = Image.fromarray(image_array[::-1, :, :], mode="RGB")

        uri = hips_dir.join(f"Npix{pixel}.png")

        with ResourcePath.temporary_uri(suffix=uri.getExtension()) as temporary_uri:
            im.save(temporary_uri.ospath)

            uri.transfer_from(temporary_uri, transfer="copy", overwrite=True)

    def _write_properties_and_moc(
            self,
            hips_base_path,
            max_order,
            pixels,
            exposure,
            shift_order,
            band,
            multiband
    ):
        """Write HiPS properties file and MOC.

        Parameters
        ----------
        hips_base_path : : `lsst.resources.ResourcePath`
            Resource path to the base of the HiPS directory tree.
        max_order : `int`
            Maximum HEALPix order.
        pixels : `np.ndarray` (N,)
            Array of pixels used.
        exposure : `lsst.afw.image.Exposure`
            Sample HPX exposure used for generating HiPS tiles.
        shift_order : `int`
            HPX shift order.
        band : `str`
            Band (or color).
        multiband : `bool`
            Is band multiband / color?
        """
        area = hpg.nside_to_pixel_area(2**max_order, degrees=True)*len(pixels)

        initial_ra = self.config.properties.initial_ra
        initial_dec = self.config.properties.initial_dec
        initial_fov = self.config.properties.initial_fov

        if initial_ra is None or initial_dec is None or initial_fov is None:
            # We want to point to an arbitrary pixel in the footprint.
            # Just take the median pixel value for simplicity.
            temp_pixels = pixels.copy()
            if temp_pixels.size % 2 == 0:
                temp_pixels = np.append(temp_pixels, [temp_pixels[0]])
            medpix = int(np.median(temp_pixels))
            _initial_ra, _initial_dec = hpg.pixel_to_angle(2**max_order, medpix)
            _initial_fov = hpg.nside_to_resolution(2**max_order, units='arcminutes')/60.

            if initial_ra is None or initial_dec is None:
                initial_ra = _initial_ra
                initial_dec = _initial_dec
            if initial_fov is None:
                initial_fov = _initial_fov

        self._write_hips_properties_file(
            hips_base_path,
            self.config.properties,
            band,
            multiband,
            exposure,
            max_order,
            shift_order,
            area,
            initial_ra,
            initial_dec,
            initial_fov,
        )

        # Write the MOC coverage
        self._write_hips_moc_file(
            hips_base_path,
            max_order,
            pixels,
        )

    def _write_hips_properties_file(
            self,
            hips_base_path,
            properties_config,
            band,
            multiband,
            exposure,
            max_order,
            shift_order,
            area,
            initial_ra,
            initial_dec,
            initial_fov
    ):
        """Write HiPS properties file.

        Parameters
        ----------
        hips_base_path : `lsst.resources.ResourcePath`
            ResourcePath at top of HiPS tree. File will be written
            to this path as ``properties``.
        properties_config : `lsst.pipe.tasks.hips.HipsPropertiesConfig`
            Configuration for properties values.
        band : `str`
            Name of band(s) for HiPS tree.
        multiband : `bool`
            Is multiband / color?
        exposure : `lsst.afw.image.Exposure`
            Sample HPX exposure used for generating HiPS tiles.
        max_order : `int`
            Maximum HEALPix order.
        shift_order : `int`
            HPX shift order.
        area : `float`
            Coverage area in square degrees.
        initial_ra : `float`
            Initial HiPS RA position (degrees).
        initial_dec : `float`
            Initial HiPS Dec position (degrees).
        initial_fov : `float`
            Initial HiPS display size (degrees).
        """
        # WARNING: In general PipelineTasks are not allowed to do any outputs
        # outside of the butler.  This task has been given (temporary)
        # Special Dispensation because of the nature of HiPS outputs until
        # a more controlled solution can be found.
        def _write_property(fh, name, value):
            """Write a property name/value to a file handle.

            Parameters
            ----------
            fh : file handle (blah)
                Open for writing.
            name : `str`
                Name of property
            value : `str`
                Value of property
            """
            # This ensures that the name has no spaces or space-like characters,
            # per the HiPS standard.
            if re.search(r"\s", name):
                raise ValueError(f"``{name}`` cannot contain any space characters.")
            if "=" in name:
                raise ValueError(f"``{name}`` cannot contain an ``=``")

            fh.write(f"{name:25}= {value}\n")

        if exposure.image.array.dtype == np.dtype("float32"):
            bitpix = -32
        elif exposure.image.array.dtype == np.dtype("float64"):
            bitpix = -64
        elif exposure.image.array.dtype == np.dtype("int32"):
            bitpix = 32

        date_iso8601 = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        pixel_scale = hpg.nside_to_resolution(2**(max_order + shift_order), units='degrees')

        uri = hips_base_path.join("properties")
        with ResourcePath.temporary_uri(suffix=uri.getExtension()) as temporary_uri:
            with open(temporary_uri.ospath, "w") as fh:
                _write_property(
                    fh,
                    "creator_did",
                    properties_config.creator_did_template.format(band=band),
                )
                if properties_config.obs_collection is not None:
                    _write_property(fh, "obs_collection", properties_config.obs_collection)
                _write_property(
                    fh,
                    "obs_title",
                    properties_config.obs_title_template.format(band=band),
                )
                if properties_config.obs_description_template is not None:
                    _write_property(
                        fh,
                        "obs_description",
                        properties_config.obs_description_template.format(band=band),
                    )
                if len(properties_config.prov_progenitor) > 0:
                    for prov_progenitor in properties_config.prov_progenitor:
                        _write_property(fh, "prov_progenitor", prov_progenitor)
                if properties_config.obs_ack is not None:
                    _write_property(fh, "obs_ack", properties_config.obs_ack)
                _write_property(fh, "obs_regime", "Optical")
                _write_property(fh, "data_pixel_bitpix", str(bitpix))
                _write_property(fh, "dataproduct_type", "image")
                _write_property(fh, "moc_sky_fraction", str(area/41253.))
                _write_property(fh, "data_ucd", "phot.flux")
                _write_property(fh, "hips_creation_date", date_iso8601)
                _write_property(fh, "hips_builder", "lsst.pipe.tasks.hips.GenerateHipsTask")
                _write_property(fh, "hips_creator", "Vera C. Rubin Observatory")
                _write_property(fh, "hips_version", "1.4")
                _write_property(fh, "hips_release_date", date_iso8601)
                _write_property(fh, "hips_frame", "equatorial")
                _write_property(fh, "hips_order", str(max_order))
                _write_property(fh, "hips_tile_width", str(exposure.getBBox().getWidth()))
                _write_property(fh, "hips_status", "private master clonableOnce")
                if multiband:
                    _write_property(fh, "hips_tile_format", "png")
                    _write_property(fh, "dataproduct_subtype", "color")
                else:
                    _write_property(fh, "hips_tile_format", "png fits")
                _write_property(fh, "hips_pixel_bitpix", str(bitpix))
                _write_property(fh, "hips_pixel_scale", str(pixel_scale))
                _write_property(fh, "hips_initial_ra", str(initial_ra))
                _write_property(fh, "hips_initial_dec", str(initial_dec))
                _write_property(fh, "hips_initial_fov", str(initial_fov))
                if multiband:
                    if self.config.blue_channel_band in properties_config.spectral_ranges:
                        em_min = properties_config.spectral_ranges[
                            self.config.blue_channel_band
                        ].lambda_min/1e9
                    else:
                        self.log.warning("blue band %s not in self.config.spectral_ranges.", band)
                        em_min = 3e-7
                    if self.config.red_channel_band in properties_config.spectral_ranges:
                        em_max = properties_config.spectral_ranges[
                            self.config.red_channel_band
                        ].lambda_max/1e9
                    else:
                        self.log.warning("red band %s not in self.config.spectral_ranges.", band)
                        em_max = 1e-6
                else:
                    if band in properties_config.spectral_ranges:
                        em_min = properties_config.spectral_ranges[band].lambda_min/1e9
                        em_max = properties_config.spectral_ranges[band].lambda_max/1e9
                    else:
                        self.log.warning("band %s not in self.config.spectral_ranges.", band)
                        em_min = 3e-7
                        em_max = 1e-6
                _write_property(fh, "em_min", str(em_min))
                _write_property(fh, "em_max", str(em_max))
                if properties_config.t_min is not None:
                    _write_property(fh, "t_min", properties_config.t_min)
                if properties_config.t_max is not None:
                    _write_property(fh, "t_max", properties_config.t_max)

            uri.transfer_from(temporary_uri, transfer="copy", overwrite=True)

    def _write_hips_moc_file(self, hips_base_path, max_order, pixels, min_uniq_order=1):
        """Write HiPS MOC file.

        Parameters
        ----------
        hips_base_path : `lsst.resources.ResourcePath`
            ResourcePath to top of HiPS tree.  File will be written as
            to this path as ``Moc.fits``.
        max_order : `int`
            Maximum HEALPix order.
        pixels : `np.ndarray`
            Array of pixels covered.
        min_uniq_order : `int`, optional
            Minimum HEALPix order for looking for fully covered pixels.
        """
        # WARNING: In general PipelineTasks are not allowed to do any outputs
        # outside of the butler.  This task has been given (temporary)
        # Special Dispensation because of the nature of HiPS outputs until
        # a more controlled solution can be found.

        # Make the initial list of UNIQ pixels
        uniq = 4*(4**max_order) + pixels

        # Make a healsparse map which provides easy degrade/comparisons.
        hspmap = hsp.HealSparseMap.make_empty(2**min_uniq_order, 2**max_order, dtype=np.float32)
        hspmap[pixels] = 1.0

        # Loop over orders, degrade each time, and look for pixels with full coverage.
        for uniq_order in range(max_order - 1, min_uniq_order - 1, -1):
            hspmap = hspmap.degrade(2**uniq_order, reduction="sum")
            pix_shift = np.right_shift(pixels, 2*(max_order - uniq_order))
            # Check if any of the pixels at uniq_order have full coverage.
            covered, = np.isclose(hspmap[pix_shift], 4**(max_order - uniq_order)).nonzero()
            if covered.size == 0:
                # No pixels at uniq_order are fully covered, we're done.
                break
            # Replace the UNIQ pixels that are fully covered.
            uniq[covered] = 4*(4**uniq_order) + pix_shift[covered]

        # Remove duplicate pixels.
        uniq = np.unique(uniq)

        # Output to fits.
        tbl = np.zeros(uniq.size, dtype=[("UNIQ", "i8")])
        tbl["UNIQ"] = uniq

        order = np.log2(tbl["UNIQ"]//4).astype(np.int32)//2
        moc_order = np.max(order)

        hdu = fits.BinTableHDU(tbl)
        hdu.header["PIXTYPE"] = "HEALPIX"
        hdu.header["ORDERING"] = "NUNIQ"
        hdu.header["COORDSYS"] = "C"
        hdu.header["MOCORDER"] = moc_order
        hdu.header["MOCTOOL"] = "lsst.pipe.tasks.hips.GenerateHipsTask"

        uri = hips_base_path.join("Moc.fits")

        with ResourcePath.temporary_uri(suffix=uri.getExtension()) as temporary_uri:
            hdu.writeto(temporary_uri.ospath)

            uri.transfer_from(temporary_uri, transfer="copy", overwrite=True)

    def _write_allsky_file(self, hips_base_path, allsky_order):
        """Write an Allsky.png file.

        Parameters
        ----------
        hips_base_path : `lsst.resources.ResourcePath`
            Resource path to the base of the HiPS directory tree.
        allsky_order : `int`
            HEALPix order of the minimum order to make allsky file.
        """
        tile_size = self.config.allsky_tilesize

        # The Allsky file format is described in
        # https://www.ivoa.net/documents/HiPS/20170519/REC-HIPS-1.0-20170519.pdf
        # From S4.3.2:
        # The Allsky file is built as an array of tiles, stored side by side in
        # the left-to-right order. The width of this array must be the square
        # root of the number of the tiles of the order. For instance, the width
        # of this array at order 3 is 27 ( (int)sqrt(768) ). To avoid having a
        # too large Allsky file, the resolution of each tile may be reduced but
        # must stay a power of two (typically 64x64 pixels rather than 512x512).

        n_tiles = hpg.nside_to_npixel(hpg.order_to_nside(allsky_order))
        n_tiles_wide = int(np.floor(np.sqrt(n_tiles)))
        n_tiles_high = int(np.ceil(n_tiles / n_tiles_wide))

        allsky_image = None

        allsky_order_uri = hips_base_path.join(f"Norder{allsky_order}", forceDirectory=True)
        pixel_regex = re.compile(r"Npix([0-9]+)\.png$")
        png_uris = list(
            ResourcePath.findFileResources(
                candidates=[allsky_order_uri],
                file_filter=pixel_regex,
            )
        )

        for png_uri in png_uris:
            matches = re.match(pixel_regex, png_uri.basename())
            pix_num = int(matches.group(1))
            tile_image = Image.open(io.BytesIO(png_uri.read()))
            row = math.floor(pix_num//n_tiles_wide)
            column = pix_num % n_tiles_wide
            box = (column*tile_size, row*tile_size, (column + 1)*tile_size, (row + 1)*tile_size)
            tile_image_shrunk = tile_image.resize((tile_size, tile_size))

            if allsky_image is None:
                allsky_image = Image.new(
                    tile_image.mode,
                    (n_tiles_wide*tile_size, n_tiles_high*tile_size),
                )
            allsky_image.paste(tile_image_shrunk, box)

        uri = allsky_order_uri.join("Allsky.png")

        with ResourcePath.temporary_uri(suffix=uri.getExtension()) as temporary_uri:
            allsky_image.save(temporary_uri.ospath)

            uri.transfer_from(temporary_uri, transfer="copy", overwrite=True)

    def _get_dir_number(self, pixel):
        """Compute the directory number from a pixel.

        Parameters
        ----------
        pixel : `int`
            HEALPix pixel number.

        Returns
        -------
        dir_number : `int`
            HiPS directory number.
        """
        return (pixel//10000)*10000


class GenerateColorHipsConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=("instrument", ),
                                   defaultTemplates={"coaddName": "deep"}):
    hips_exposure_handles = pipeBase.connectionTypes.Input(
        doc="HiPS-compatible HPX images.",
        name="{coaddName}Coadd_hpx",
        storageClass="ExposureF",
        dimensions=("healpix11", "band"),
        multiple=True,
        deferLoad=True,
    )


class GenerateColorHipsConfig(GenerateHipsConfig,
                              pipelineConnections=GenerateColorHipsConnections):
    """Configuration parameters for GenerateColorHipsTask."""
    blue_channel_band = pexConfig.Field(
        doc="Band to use for blue channel of color pngs.",
        dtype=str,
        default="g",
    )
    green_channel_band = pexConfig.Field(
        doc="Band to use for green channel of color pngs.",
        dtype=str,
        default="r",
    )
    red_channel_band = pexConfig.Field(
        doc="Band to use for red channel of color pngs.",
        dtype=str,
        default="i",
    )
    png_color_asinh_minimum = pexConfig.Field(
        doc="AsinhMapping intensity to be mapped to black for color png scaling.",
        dtype=float,
        default=0.0,
    )
    png_color_asinh_stretch = pexConfig.Field(
        doc="AsinhMapping linear stretch for color png scaling.",
        dtype=float,
        default=5.0,
    )
    png_color_asinh_softening = pexConfig.Field(
        doc="AsinhMapping softening parameter (Q) for color png scaling.",
        dtype=float,
        default=8.0,
    )


class GenerateColorHipsTask(GenerateHipsTask):
    """Task for making a HiPS tree with color pngs."""
    ConfigClass = GenerateColorHipsConfig
    _DefaultName = "generateColorHips"
    color_task = True

    def _check_data_bands(self, data_bands):
        """Check the data for configured bands.

        Warn if any color bands are missing data.

        Parameters
        ----------
        data_bands : `set` [`str`]
            Bands from the input data.

        Returns
        -------
        bands : `list` [`str`]
           List of bands in bgr color order.
        """
        if len(data_bands) == 0:
            raise RuntimeError("GenerateColorHipsTask must have data from at least one band.")

        if self.config.blue_channel_band not in data_bands:
            self.log.warning(
                "Color png blue_channel_band %s not in dataset.",
                self.config.blue_channel_band
            )
        if self.config.green_channel_band not in data_bands:
            self.log.warning(
                "Color png green_channel_band %s not in dataset.",
                self.config.green_channel_band
            )
        if self.config.red_channel_band not in data_bands:
            self.log.warning(
                "Color png red_channel_band %s not in dataset.",
                self.config.red_channel_band
            )

        bands = [
            self.config.blue_channel_band,
            self.config.green_channel_band,
            self.config.red_channel_band,
        ]

        return bands
