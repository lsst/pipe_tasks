# This file is part of summit_utils.
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


import argparse
import atexit
import logging
import multiprocessing
import os
import shutil
import time
import warnings
from collections import namedtuple
from pathlib import Path
from typing import Any

import astropy
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning
from tabulate import tabulate
from tqdm import tqdm

import lsst.afw.display as afwDisplay
from lsst.afw.geom import ellipses
from lsst.daf.butler.datastore.cache_manager import DatastoreCacheManager
from lsst.summit.utils.bestEffort import BestEffortIsr
from lsst.summit.utils.peekExposure import PeekExposureTask

# Set logger level to higher than CRITICAL to suppress all output
silentLogger = logging.getLogger("silentLogger")
silentLogger.setLevel(logging.CRITICAL + 1)


# global vars for passing items from multiprocessing initializer to doWork.
AssessPeekExposureGlobals = namedtuple(
    "AssessPeekExposureGlobals", ["bestEffort", "pet", "display", "fig", "ax"]
)


SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))
AUXTEL_PIXEL_SCALE = 0.1  # arcsec/pixel

apeGlobals: None | AssessPeekExposureGlobals = None


# get one BestEffortIsr,afwDisplay per subprocess
def initializePoolProcess() -> None:
    """Initialize the multiprocessing pool process.

    Sets up some variables we only want to create once per process, including
    the BestEffortIsr, and afwDisplay + associated matplotlib figure objects.

    Places these into a global variable, as that's the only way I know of to
    effectively pass these to the doWork function.
    """
    global apeGlobals
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9), facecolor="k")
    display = afwDisplay.Display(fig, backend="matplotlib")
    display.scale("asinh", -1, 31)
    petConfig = PeekExposureTask.ConfigClass()
    pet = PeekExposureTask(config=petConfig, log=silentLogger, display=display)
    bestEffort = BestEffortIsr(embargo=True)
    apeGlobals = AssessPeekExposureGlobals(bestEffort, pet, display, fig, ax)


# retrieve best-effort-isr and run PET on it
def doWork(idx: int, row: astropy.table.Row, doPlot: bool) -> tuple[str, str, float, int]:
    """Run PeekExposureTask on a single exposure.

    Parameters
    ----------
    idx : `int`
        Index of the exposure in the table.
    row : `astropy.table.Row`
        Row of the table containing the exposure metadata.
    doPlot : `bool`
        Whether to plot the results.

    Returns
    -------
    inTag : `str`
        Input tag of the exposure.
    outTag : `str`
        Output tag of the exposure.
    runtime : `float`
        Runtime of the exposure in seconds.
    exposureId : `int`
        Exposure ID of the exposure.
    """
    global apeGlobals
    assert apeGlobals is not None, "initializePoolProcess must be called before doWork"
    bestEffort, pet, display, fig, ax = apeGlobals
    exposureId = row["exposureId"]
    dataId = {"instrument": "LATISS", "exposure": exposureId, "detector": 0}
    exp = bestEffort.getExposure(dataId)
    mode = "auto"
    binSize = None
    donutDiameter = None

    # Until DM-41335 is fixed, we manually set mode for following exposures.
    # Incorrectly labeled as photo
    if exposureId >= 2022101200584 and exposureId <= 2022101200588:
        mode = "donut"
        donutDiameter = 1700
    # Incorrectly labeled as giant donut
    if exposureId >= 2022101200589 and exposureId <= 2022101200672:
        mode = "photo"
    if exposureId >= 2022101200673 and exposureId <= 2022101200676:
        mode = "donut"
        donutDiameter = 200
    if exposureId >= 2022101200677 and exposureId <= 2022101200873:
        mode = "photo"
    if exposureId >= 2022101200874 and exposureId <= 2022101200875:
        mode = "donut"
        donutDiameter = 200
    if exposureId >= 2022101200876 and exposureId <= 2022101201060:
        mode = "photo"
    if exposureId >= 2022101201061 and exposureId <= 2022101201062:
        mode = "donut"
        donutDiameter = 200
    if exposureId >= 2022101201063 and exposureId <= 2022101201247:
        mode = "photo"

    t0 = time.time()
    result = pet.run(exp, mode=mode, doDisplay=doPlot, binSize=binSize, donutDiameter=donutDiameter)
    t1 = time.time()
    runtime = t1 - t0

    ax.text(
        0.01,
        0.97,
        f"Runtime: {runtime:.2f} s",
        c="w",
        transform=ax.transAxes,
        fontsize=12,
    )

    inTag = row["finalTag"]
    trueX = row["centroid_x"]
    trueY = row["centroid_y"]
    foundX = result.brightestCentroid.x
    foundY = result.brightestCentroid.y

    if np.isfinite(trueX):
        nx = (trueX - (result.binSize - 1) / 2) / result.binSize
        ny = (trueY - (result.binSize - 1) / 2) / result.binSize
        if doPlot:
            display.dot("o", nx, ny, ctype=afwDisplay.MAGENTA, size=20)

        if np.isfinite(foundX):
            dist = np.sqrt((trueX - foundX) ** 2 + (trueY - foundY) ** 2)
            if dist < 20:
                outTag = "<2"
            elif dist < 100:
                outTag = "<10"
            else:
                outTag = ">10"
        else:
            outTag = "noStar"
    else:
        if np.isfinite(foundX):
            outTag = "noTruth"
        else:
            outTag = "nothing"

    if doPlot:
        if np.isfinite(result.psfPixelShape.getIxx()):
            distortion = ellipses.SeparableDistortionDeterminantRadius(result.psfPixelShape)
            fwhm = SIGMA_TO_FWHM * distortion.getDeterminantRadius() * AUXTEL_PIXEL_SCALE
        else:
            fwhm = np.nan
        display.show_colorbar(False)
        ax.set_title(f"{exposureId=}  {inTag=}  {outTag=}  {fwhm=:.2f} arcsec", color="w")
        path = Path(args.plotdir)
        path.mkdir(parents=True, exist_ok=True)
        fn = f"test_pet_{idx:04d}_{exposureId:d}_{inTag}_{outTag}.png"
        fig.savefig(path / fn)

    return inTag, outTag, runtime, exposureId


def main(args: argparse.Namespace) -> None:
    # Set up cache directory and register cleanup
    defined, cacheDir = DatastoreCacheManager.set_fallback_cache_directory_if_unset()
    if defined:
        atexit.register(shutil.rmtree, cacheDir, ignore_errors=True)

    # Loop through Merlin's curated data
    path = Path(os.environ["SUMMIT_EXTRAS_DIR"]) / "data" / "qfm_baseline_assessment.parq"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=AstropyUserWarning)
        table = Table.read(path)
    table["exposureId"] = table["day_obs"] * 100_000 + table["sequence_number"]
    table = table[args.start : args.end]
    intags = np.unique(table["finalTag"])

    # Possible PET outcomes
    outtags = [
        "noStar",  # PET didn't find star, but catalog had one
        "noTruth",  # No catalog star to compare to, but PET found one
        "nothing",  # No catalog or PET star
        "<2",  # PET brightest star agrees with catalog to better than 2 arcsec
        "<10",  # agrees to 10 arcsec
        ">10",  # disagrees by more than 10 arcsec
    ]
    results: dict[str, Any] = {}
    for intag in intags:
        results[intag] = {}
        for outtag in outtags:
            results[intag][outtag] = []

    if args.nproc == 1:
        initializePoolProcess()
        for idx, row in enumerate(tqdm(table, smoothing=0.05)):
            inTag, outTag, runtime, exposureId = doWork(idx + args.start, row, args.plot)
            results[inTag][outTag].append((exposureId, runtime))
    with multiprocessing.Pool(args.nproc, initializer=initializePoolProcess) as pool:
        futures = []
        for idx, row in enumerate(table):
            futures.append(pool.apply_async(doWork, (idx + args.start, row, args.plot)))

        for row, future in zip(tqdm(table, smoothing=0.05), futures):
            inTag, outTag, runtime, exposureId = future.get()
            results[inTag][outTag].append((exposureId, runtime))

    table = []
    for inKey, outDict in results.items():
        row = [inKey]
        for _, expIds in outDict.items():
            row.append(f"{len(expIds)}")
        table.append(row)
    print(tabulate(table, headers=["input", *outtags]))

    table = []
    for inKey, outDict in results.items():
        row = [inKey]
        for _, expIds in outDict.items():
            row.append(f"{np.nanmedian([x[1] for x in expIds]):.2f}")
        table.append(row)
    print(tabulate(table, headers=["input", *outtags]))

    print("G >10")
    print(results["G"][">10"])
    print()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=6105)
    parser.add_argument("--nproc", type=int, default=20)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plotdir", type=str, default="./")
    args = parser.parse_args()
    main(args)
