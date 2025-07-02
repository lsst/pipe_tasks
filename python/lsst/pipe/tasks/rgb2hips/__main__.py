import argparse
import sys
import numpy as np
from datetime import datetime

from lsst.daf.butler import Butler
from lsst.sphgeom import RangeSet, HealpixPixelization

from lsst.pipe.base import Pipeline, Instrument

from ..hips import HighResolutionHipsQuantumGraphBuilder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Build a QuantumGraph that runs HighResolutionHipsTask on existing coadd datasets."),
    )
    subparsers = parser.add_subparsers(help="sub-command help", dest="subparser_name")

    parser_segment = subparsers.add_parser("segment", help="Determine survey segments for workflow.")
    parser_build = subparsers.add_parser("build", help="Build quantum graph for HighResolutionHipsTask")

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
            default="",
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

    args = parser.parse_args(sys.argv[1:])

    if args.subparser_name is None:
        parser.print_help()
        sys.exit(1)

    pipeline = Pipeline.from_uri(args.pipeline)
    pipeline_graph = pipeline.to_graph()

    if len(pipeline_graph.tasks) != 1:
        raise RuntimeError(f"Pipeline file {args.pipeline} may only contain one task.")

    (task_node,) = pipeline_graph.tasks.values()

    butler = Butler(args.butler_config, collections=args.input)

    if args.subparser_name == "segment":
        # Do the segmentation
        hpix_pixelization = HealpixPixelization(level=args.hpix_build_order)
        dataset = task_node.inputs["input_images"].dataset_type_name
        with butler.query() as q:
            data_ids = list(q.join_dataset_search(dataset).data_ids("tract").with_dimension_records())
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
            args.output_run = "{}/{}".format(args.output, Instrument.makeCollectionTimestamp())

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
            input_collections=args.input[0].split(','),
            output_run=args.output_run,
            constraint_order=args.hpix_build_order,
            constraint_ranges=build_ranges,
            where=args.where,
        )
        qg = builder.build(metadata, attach_datastore_records=True)
        qg.saveUri(args.save_qgraph)
