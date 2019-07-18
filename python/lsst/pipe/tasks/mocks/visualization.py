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

from matplotlib import pyplot

import lsst.geom


def plotObservations(catalog, wcs):
    """Plot the bounding boxes of an observation catalog (see MockCoaddTask.buildObservationCatalog)
    using matplotlib, in the coordinates defined by the given Wcs (usually a skymap Wcs).
    """
    for record in catalog:
        box = lsst.geom.Box2D(record.getBBox())
        x = []
        y = []
        iWcs = record.getWcs()
        for xi, yi in box.getCorners():
            try:
                coord = iWcs.pixelToSky(xi, yi)
                xo, yo = wcs.skyToPixel(coord)
                x.append(xo)
                y.append(yo)
            except Exception as e:
                print("WARNING: point %d, %d failed: error=%s" % (xi, yi, e))
        pyplot.fill(x, y, facecolor='r', alpha=0.1, edgecolor=None)


def plotPatches(tractInfo):
    """Plot the patches in a skymap tract using matplotlib.
    """
    nPatchX, nPatchY = tractInfo.getNumPatches()
    for iPatchX in range(nPatchX):
        for iPatchY in range(nPatchY):
            patchInfo = tractInfo.getPatchInfo((iPatchX, iPatchY))
            xp1, yp1 = list(zip(*patchInfo.getOuterBBox().getCorners()))
            xp2, yp2 = list(zip(*patchInfo.getInnerBBox().getCorners()))
            pyplot.fill(xp1, yp1, fill=False, edgecolor='g', linestyle='dashed')
            pyplot.fill(xp2, yp2, fill=False, edgecolor='g')


def plotTruth(catalog, wcs):
    """Plot the objects in a truth catalog as dots using matplotlib, in the coordinate
    system defined by the given Wcs.
    """
    xp = []
    yp = []
    for record in catalog:
        x, y = wcs.skyToPixel(record.getCoord())
        xp.append(x)
        yp.append(y)
    pyplot.plot(xp, yp, 'k+')


def displayImages(root):
    """Display coadd images in different frames, with the bounding boxes of the
    observations that went into them overlayed.
    """
    import lsst.afw.display as afwDisplay
    afwDisplay.setDefaultMaskTransparency(75)

    butler = lsst.daf.persistence.Butler(root=root)
    skyMap = butler.get("deepCoadd_skyMap")
    tractInfo = skyMap[0]
    task = lsst.pipe.tasks.mocks.MockCoaddTask()
    coadds = [patchRef.get("deepCoadd", immediate=True)
              for patchRef in task.iterPatchRefs(butler, tractInfo)]
    for n, coadd in enumerate(coadds):
        afwDisplay.Display(frame=n + 1).mtv(coadd, title="displayImages: coadd")
    for n, coadd in enumerate(coadds):
        afwDisplay.utils.drawCoaddInputs(coadd, frame=n + 1)
    return butler


def makePlots(root):
    """Convenience function to make all matplotlib plots.
    """
    import lsst.pipe.tasks.mocks
    import lsst.daf.persistence
    butler = lsst.daf.persistence.Butler(root=root)
    skyMap = butler.get("deepCoadd_skyMap")
    observations = butler.get("observations", tract=0)
    truth = butler.get("truth", tract=0)
    tractInfo = skyMap[0]
    plotPatches(tractInfo)
    plotObservations(observations, tractInfo.getWcs())
    plotTruth(truth, tractInfo.getWcs())
    pyplot.axis("scaled")
    pyplot.show()
    return butler
