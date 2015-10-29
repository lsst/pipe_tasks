#!/usr/bin/env python

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
from eups import Eups
eups = Eups()
import functools

from contextlib import contextmanager
from collections import defaultdict

from lsst.daf.persistence.butler import safeMakeDir
from lsst.pex.config import Config, Field, ConfigField, ListField, DictField, ConfigDictField
from lsst.pipe.base import Struct, CmdLineTask, ArgumentParser, TaskRunner, TaskError
from lsst.pipe.tasks.dataIds import PerTractCcdDataIdContainer
from hsc.pipe.tasks.stack import TractDataIdContainer
from hsc.pipe.base.matches import matchesToCatalog, matchesFromCatalog
from lsst.meas.astrom import Astrometry, MeasAstromConfig
from lsst.meas.photocal.colorterms import ColortermLibraryConfig
from lsst.meas.mosaic.updateExposure import (applyMosaicResultsCatalog, applyCalib, getFluxFitParams,
                                             getFluxKeys, getMosaicResults)

import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable
import lsst.afw.coord as afwCoord
import lsst.afw.image as afwImage

class AllLabeller(object):
    labels = {'all': 0}
    plot = ["all"]
    def __call__(self, catalog):
        return numpy.zeros(len(catalog))

class StarGalaxyLabeller(object):
    labels = {'star': 0, 'galaxy': 1}
    plot = ["star"]
    _column = 'classification.extendedness'
    def __call__(self, catalog):
        return numpy.where(catalog[self._column] < 0.5, 0, 1)

class OverlapsStarGalaxyLabeller(StarGalaxyLabeller):
    labels = {'star': 0, 'galaxy': 1, 'split': 2}
    def __call__(self, catalog):
        first = numpy.where(catalog["first." + self._column] < 0.5, 0, 1)
        second = numpy.where(catalog["second." + self._column] < 0.5, 0, 1)
        return numpy.where(first == second, first, 2)

class MatchesStarGalaxyLabeller(StarGalaxyLabeller):
    _column = 'src.classification.extendedness'

class CosmosLabeller(StarGalaxyLabeller):
    """Do star/galaxy classification using Alexie Leauthaud's Cosmos catalog"""
    def __init__(self, filename, radius):
        original = afwTable.BaseCatalog.readFits(filename)
        good = (original["CLEAN"] == 1) & (original["MU.CLASS"] == 2)
        num = good.sum()
        cosmos = afwTable.SimpleCatalog(afwTable.SimpleTable.makeMinimalSchema())
        cosmos.reserve(num)
        for ii in range(num):
            cosmos.addNew()
        cosmos["id"][:] = original["NUMBER"][good]
        cosmos["coord.ra"][:] = original["ALPHA.J2000"][good]*(1.0*afwGeom.degrees).asRadians()
        cosmos["coord.dec"][:] = original["DELTA.J2000"][good]*(1.0*afwGeom.degrees).asRadians()
        self.cosmos = cosmos
        self.radius = radius

    def __call__(self, catalog):
        # A kdTree would be better, but this is all we have right now
        matches = afwTable.matchRaDec(self.cosmos, catalog, self.radius)
        good = set(mm.second.getId() for mm in matches)
        return numpy.array([0 if ii in good else 1 for ii in catalog["id"]])

class Filenamer(object):
    """Callable that provides a filename given a style"""
    def __init__(self, butler, dataset, dataId):
        self.butler = butler
        self.dataset = dataset
        self.dataId = dataId
    def copy(self, **kwargs):
        """Return a copy with updated dataId"""
        dataId = self.dataId.copy()
        dataId.update(kwargs)
        return Filenamer(self.butler, self.dataset, dataId)
    def __call__(self, style):
        filename = self.butler.get(self.dataset + "_filename", self.dataId, style=style)[0]
        safeMakeDir(os.path.dirname(filename))
        return filename

class Data(Struct):
    def __init__(self, catalog, quantity, mag, selection, color, plot=True):
        Struct.__init__(self, catalog=catalog[selection], quantity=quantity[selection], mag=mag[selection],
                        selection=selection, color=color, plot=plot)

class Stats(Struct):
    def __init__(self, num, total, mean, stdev, forcedMean):
        Struct.__init__(self, num=num, total=total, mean=mean, stdev=stdev, forcedMean=forcedMean)

colorList = ["blue", "red", "green", "black", "yellow", "cyan", "magenta", ]

class AnalysisConfig(Config):
    flags = ListField(dtype=str, doc="Flags of objects to ignore",
                      default=["centroid.sdss.flags", "flags.pixel.saturated.center",
                               "flags.pixel.interpolated.center", "flux.psf.flags"])
    clip = Field(dtype=float, default=4.0, doc="Rejection threshold (stdev)")
    magThreshold = Field(dtype=float, default=21.0, doc="Magnitude threshold to apply")
    magPlotMin = Field(dtype=float, default=16.0, doc="Minimum magnitude to plot")
    magPlotMax = Field(dtype=float, default=28.0, doc="Maximum magnitude to plot")
    fluxColumn = Field(dtype=str, default="flux.psf", doc="Column to use for flux/magnitude plotting")
    zp = Field(dtype=float, default=27.0, doc="Magnitude zero point to apply")

class Analysis(object):
    """Centralised base for plotting"""

    def __init__(self, catalog, func, quantityName, config, qMin=-0.2, qMax=0.2, prefix="",
                 flags=[], labeller=AllLabeller()):
        self.catalog = catalog
        self.func = func
        self.quantityName = quantityName
        self.config = config
        self.qMin = qMin
        self.qMax = qMax
        self.prefix = prefix
        self.flags = flags

        self.quantity = func(catalog)
        self.mag = self.config.zp - 2.5*numpy.log10(catalog[prefix + self.config.fluxColumn])

        self.good = numpy.isfinite(self.quantity) & numpy.isfinite(self.mag)
        for ff in list(config.flags) + flags:
            self.good &= ~catalog[prefix + ff]

        labels = labeller(catalog)
        self.data = {name: Data(catalog, self.quantity, self.mag, self.good & (labels == value),
                                colorList[value], name in labeller.plot) for name, value in
                     labeller.labels.iteritems()}

    def plotAgainstMag(self, filename):
        """Plot quantity against magnitude"""
        fig, axes = plt.subplots(1, 1)
        magMin, magMax = self.config.magPlotMin, self.config.magPlotMax
        for name, data in self.data.iteritems():
            axes.scatter(data.mag, data.quantity, s=2, marker='o', lw=0, c=data.color, label=name, alpha=0.3)
            magMin = max(magMin, data.mag.min())
            magMax = min(magMax, data.mag.max())
        axes.set_xlabel("Mag from %s" % self.config.fluxColumn)
        axes.set_ylabel(self.quantityName)
        axes.set_ylim(self.qMin, self.qMax)
        axes.set_xlim(magMin, magMax)
        axes.legend()
        fig.savefig(filename)
        plt.close(fig)

    def plotHistogram(self, filename, numBins=50):
        """Plot histogram of quantity"""
        fig, axes = plt.subplots(1, 1)
        numMax = 0
        for name, data in self.data.iteritems():
            good = numpy.isfinite(data.quantity)
            if self.config.magThreshold is not None:
                good &= data.mag < self.config.magThreshold
            if good.sum() == 0:
                continue
            num, _, _ = axes.hist(data.quantity[good], numBins, range=(self.qMin, self.qMax), normed=False,
                                  color=data.color, label=name, histtype="step")
            numMax = max(numMax, num.max()*1.1)
        axes.set_xlim(self.qMin, self.qMax)
        axes.set_ylim(0.9, numMax)
        axes.set_xlabel(self.quantityName)
        axes.set_ylabel("Number")
        axes.set_yscale('log', nonposy='clip')
        axes.legend()
        fig.savefig(filename)
        plt.close(fig)

    def plotSkyPosition(self, filename, cmap=plt.cm.brg):
        """Plot quantity as a function of position"""
        ra = numpy.rad2deg(self.catalog[self.prefix + "coord.ra"])
        dec = numpy.rad2deg(self.catalog[self.prefix + "coord.dec"])
        good = (self.mag < self.config.magThreshold if self.config.magThreshold > 0 else
                numpy.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(1, 1)
        for name, data in self.data.iteritems():
            if not data.plot:
                continue
            selection = data.selection & good
            axes.scatter(ra[selection], dec[selection], s=2, marker='o', lw=0,
                         c=data.quantity[good[data.selection]], cmap=cmap, vmin=self.qMin, vmax=self.qMax)
        axes.set_xlabel("RA (deg)")
        axes.set_ylabel("Dec (deg)")

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=self.qMin, vmax=self.qMax))
        mappable._A = []        # fake up the array of the scalar mappable. Urgh...
        cb = plt.colorbar(mappable)
        cb.set_label(self.quantityName, rotation=270)
        fig.savefig(filename)
        plt.close(fig)

    def plotRaDec(self, filename):
        """Plot quantity as a function of RA, Dec"""

        ra = numpy.rad2deg(self.catalog[self.prefix + "coord.ra"])
        dec = numpy.rad2deg(self.catalog[self.prefix + "coord.dec"])
        good = (self.mag < self.config.magThreshold if self.config.magThreshold is not None else
                numpy.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(2, 1)
        for name, data in self.data.iteritems():
            selection = data.selection & good
            kwargs = {'s': 5, 'marker': 'o', 'lw': 0, 'c': data.color, 'alpha': 0.5}
            axes[0].scatter(ra[selection], data.quantity[good[data.selection]], label=name, **kwargs)
            axes[1].scatter(dec[selection], data.quantity[good[data.selection]], **kwargs)

        axes[0].set_xlabel("RA (deg)")
        axes[0].set_ylabel(self.quantityName)
        axes[1].set_xlabel("Dec (deg)")
        axes[1].set_ylabel(self.quantityName)

        axes[0].set_ylim(self.qMin, self.qMax)
        axes[1].set_ylim(self.qMin, self.qMax)

        axes[0].legend()
        fig.savefig(filename)
        plt.close(fig)

    def plotAll(self, filenamer, log, forcedMean=None):
        """Make all plots"""
        self.plotAgainstMag(filenamer("psfMag"))
        self.plotHistogram(filenamer("hist"))
        self.plotSkyPosition(filenamer("sky"))
        self.plotRaDec(filenamer("radec"))
        self.stats(log, forcedMean=forcedMean)

    def stats(self, log, forcedMean=None):
        """Calculate statistics on quantity"""
        stats = {}
        for name, data in self.data.iteritems():
            good = data.mag < self.config.magThreshold
            total = good.sum() # Total number we're considering
            quartiles = numpy.percentile(data.quantity[good], [25, 50, 75])
            median = quartiles[1]
            clip = self.config.clip*0.74*(quartiles[2] - quartiles[0])
            good &= numpy.logical_not(numpy.abs(data.quantity - median) > clip)
            actualMean = data.quantity[good].mean()
            mean = actualMean if forcedMean is None else forcedMean
            stdev = numpy.sqrt(((data.quantity[good].astype(numpy.float64) - mean)**2).mean())
            stats[name] = Stats(num=good.sum(), total=total, mean=actualMean, stdev=stdev,
                                forcedMean=forcedMean)
        log.info("Statistics for %s: %s" % (self.quantityName, stats))


class CcdAnalysis(Analysis):
    def plotCcd(self, prefix, centroid="centroid.sdss", cmap=plt.cm.brg):
        """Plot quantity as a function of CCD x,y"""
        xx = self.catalog[self.prefix + centroid + ".x"]
        yy = self.catalog[self.prefix + centroid + ".y"]
        ccd = self.catalog["ccd"]
        good = (self.mag < self.config.magThreshold if self.config.magThreshold > 0 else
                numpy.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(2, 1)
        data = self.data[self.labeller.plot]
        selection = data.selection & good
        quantity = data.quantity[good[data.selection]]
        kwargs = {'s': 2, 'marker': 'o', 'lw': 0, 'alpha': 0.5}
        axes[0].scatter(xx[selection], quantity, c=ccd, **kwargs)
        axes[1].scatter(yy[selection], quantity, c=ccd, **kwargs)

        axes[0].set_xlabel("x_{ccd}")
        axes[0].set_ylabel(self.quantityName)
        axes[1].set_xlabel("y_{ccd}")
        axes[1].set_ylabel(self.quantityName)

        axes[0].set_ylim(self.qMin, self.qMax)
        axes[1].set_ylim(self.qMin, self.qMax)

        cb = plt.colorbar(ccd)
        cb.set_label("CCD index", rotation=270)

        fig.savefig(filename)
        plt.close(fig)


class MagDiff(object):
    """Functor to calculate magnitude difference"""
    def __init__(self, col1, col2):
        self.col1 = col1
        self.col2 = col2
    def __call__(self, catalog):
        return -2.5*numpy.log10(catalog[self.col1]/catalog[self.col2])

class MagDiffMatches(object):
    """Functor to calculate magnitude difference for match catalog"""
    def __init__(self, column, colorterm, zp=27.0):
        self.column = column
        self.colorterm = colorterm
        self.zp = zp
    def __call__(self, catalog):
        ref1 = -2.5*numpy.log10(catalog.get("ref." + self.colorterm.primary))
        ref2 = -2.5*numpy.log10(catalog.get("ref." + self.colorterm.secondary))
        ref = self.colorterm.transformMags(ref1, ref2)
        src = self.zp - 2.5*numpy.log10(catalog.get("src." + self.column))
        return src - ref

class AstrometryDiff(object):
    """Functor to calculate difference between astrometry"""
    def __init__(self, first, second, declination=None):
        self.first = first
        self.second = second
        self.declination = declination
    def __call__(self, catalog):
        first = catalog[self.first]
        second = catalog[self.second]
        cosDec = numpy.cos(catalog[self.declination]) if self.declination is not None else 1.0
        return (first - second)*cosDec*(1.0*afwGeom.radians).asArcseconds()

def deconvMom(catalog):
    """Calculate deconvolved moments"""
    hsm = catalog["shape.hsm.moments.xx"] + catalog["shape.hsm.moments.yy"]
    sdss = catalog["shape.sdss.xx"] + catalog["shape.sdss.yy"]
    psf = catalog["shape.hsm.psfMoments.xx"] + catalog["shape.hsm.psfMoments.yy"]
    return numpy.where(numpy.isfinite(hsm), hsm, sdss) - psf

def deconvMomStarGal(catalog):
    """Calculate P(star) from deconvolved moments"""
    rTrace = deconvMom(catalog)
    snr = catalog["flux.psf"]/catalog["flux.psf.err"]
    poly = (-4.2759879274 + 0.0713088756641*snr + 0.16352932561*rTrace - 4.54656639596e-05*snr*snr -
            0.0482134274008*snr*rTrace + 4.41366874902e-13*rTrace*rTrace + 7.58973714641e-09*snr*snr*snr +
            1.51008430135e-05*snr*snr*rTrace + 4.38493363998e-14*snr*rTrace*rTrace +
            1.83899834142e-20*rTrace*rTrace*rTrace)
    return 1.0/(1.0 + numpy.exp(-poly))


def concatenateCatalogs(catalogList):
    template = catalogList[0]
    catalog = type(template)(template.schema)
    catalog.reserve(sum(len(cat) for cat in catalogList))
    for cat in catalogList:
        catalog.extend(cat, True)
    return catalog

def joinMatches(matches, first="first.", second="second."):
    mapperList = afwTable.SchemaMapper.join(afwTable.SchemaVector([matches[0].first.schema,
                                                                   matches[0].second.schema]),
                                            [first, second])
    schema = mapperList[0].getOutputSchema()
    distanceKey = schema.addField("distance", type="Angle", doc="Distance between %s and %s" % (first, second))
    catalog = afwTable.BaseCatalog(schema)
    catalog.reserve(len(matches))
    for mm in matches:
        row = catalog.addNew()
        row.assign(mm.first, mapperList[0])
        row.assign(mm.second, mapperList[1])
        row.set(distanceKey, mm.distance*afwGeom.radians)
    return catalog


@contextmanager
def andCatalog(version):
    current = eups.findSetupVersion("astrometry_net_data")[0]
    eups.setup("astrometry_net_data", version, noRecursion=True)
    try:
        yield
    finally:
        eups.setup("astrometry_net_data", current, noRecursion=True)

class CoaddAnalysisConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name for coadd")
    matchRadius = Field(dtype=float, default=0.5, doc="Matching radius (arcseconds)")
    colorterms = ConfigField(dtype=ColortermLibraryConfig, doc="Library of color terms")
    analysis = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options")
    analysisMatches = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options for matches")
    matchesMaxDistance = Field(dtype=float, default=0.3, doc="Maximum plotting distance for matches")
    externalCatalogs = ListField(dtype=str, default=["sdss-dr9-fink-v5b"],
                                 doc="Additional external catalogs for matching")
    astrometry = ConfigField(dtype=MeasAstromConfig, doc="Configuration for astrometric reference")
    doMags = Field(dtype=bool, default=True, doc="Plot magnitudes?")
    doStarGalaxy = Field(dtype=bool, default=True, doc="Plot star/galaxy?")
    doOverlaps = Field(dtype=bool, default=True, doc="Plot overlaps?")
    doMatches = Field(dtype=bool, default=True, doc="Plot matches?")

    def setDefaults(self):
        self.astrometry.load(os.path.join(os.environ["OBS_SUBARU_DIR"], "config", "hsc", "filterMap.py"))


class CoaddAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs["cosmos"] = parsedCmd.cosmos
        return [(refList, kwargs) for refList in parsedCmd.id.refList]


class CoaddAnalysisTask(CmdLineTask):
    _DefaultName = "analysis"
    ConfigClass = CoaddAnalysisConfig
    RunnerClass = CoaddAnalysisRunner

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_argument("--cosmos", default=None, help="Filename for Leauthaud Cosmos catalog")
        parser.add_id_argument("--id", "deepCoadd_src",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=HSC-X",
                               ContainerClass=TractDataIdContainer)
        return parser

    def run(self, patchRefList, cosmos=None):
        filterName = patchRefList[0].dataId["filter"]
        filenamer = Filenamer(patchRefList[0].getButler(), "plotCoadd", patchRefList[0].dataId)
        if (self.config.doMags or self.config.doStarGalaxy or self.config.doOverlaps or cosmos or
            self.config.externalCatalogs):
            catalog = self.readCatalogs(patchRefList, "deepCoadd_src")
        if self.config.doMags:
            self.plotMags(catalog, filenamer)
        if self.config.doStarGalaxy:
            self.plotStarGal(catalog, filenamer)
        if cosmos:
            self.plotCosmos(catalog, filenamer, cosmos)
        if self.config.doOverlaps:
            overlaps = self.overlaps(catalog)
            self.plotOverlaps(overlaps, filenamer)
        if self.config.doMatches:
            matches = self.readCatalogs(patchRefList, "deepCoadd_srcMatchFull")
            self.plotMatches(matches, filterName, filenamer)

        for cat in self.config.externalCatalogs:
            with andCatalog(cat):
                matches = self.matchCatalog(catalog, filterName)
                self.plotMatches(matches, filterName, filenamer.copy(description=cat))

    def readCatalogs(self, patchRefList, dataset):
        catList = [patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS) for
                   patchRef in patchRefList if patchRef.datasetExists(dataset)]
        return concatenateCatalogs(catList)

    def plotMags(self, catalog, filenamer):
        for col in ["flux.sinc", "flux.kron", "cmodel.flux"]:
            if col in catalog.schema:
                Analysis(catalog, MagDiff(col, "flux.psf"), "Mag(%s) - PSFMag" % col, self.config.analysis,
                         flags=[col + ".flags"], labeller=StarGalaxyLabeller()
                         ).plotAll(filenamer.copy(description="mag_" + col), self.log)

    def plotStarGal(self, catalog, filenamer):
        Analysis(catalog, deconvMomStarGal, "pStar", self.config.analysis, qMin=0.0, qMax=1.0,
                 ).plotAll(filenamer.copy(description="pStar"), self.log)
        Analysis(catalog, deconvMom, "Deconvolved moments", self.config.analysis, qMin=-1.0, qMax=6.0,
                 labeller=StarGalaxyLabeller()).plotAll(filenamer.copy(description="deconvMom"), self.log)

    def overlaps(self, catalog):
        matches = afwTable.matchRaDec(catalog, self.config.matchRadius*afwGeom.arcseconds, False)
        return joinMatches(matches, "first.", "second.")

    def plotOverlaps(self, overlaps, filenamer):
        for col in ["flux.psf", "flux.sinc", "flux.kron", "cmodel.flux"]:
            if "first." + col in overlaps.schema:
                Analysis(overlaps, MagDiff("first." + col, "second." + col),
                         "Overlap mag difference (%s)" % col, self.config.analysis, prefix="first.",
                         flags=[col + ".flags"], labeller=OverlapsStarGalaxyLabeller()
                         ).plotAll(filenamer.copy(description="overlap_" + col), self.log)

        Analysis(overlaps, lambda cat: cat["distance"]*(1.0*afwGeom.radians).asArcseconds(),
                 "Distance (arcsec)", self.config.analysis, prefix="first.", qMin=0.0, qMax=0.15,
                 labeller=OverlapsStarGalaxyLabeller(),
                 ).plotAll(filenamer.copy(description="overlap_distance"), self.log, forcedMean=0.0)

    def plotMatches(self, matches, filterName, filenamer):
        ct = self.config.colorterms.selectColorTerm(filterName)
        Analysis(matches, MagDiffMatches("flux.psf", ct), "MagPsf - ref", self.config.analysisMatches,
                 prefix="src.", labeller=MatchesStarGalaxyLabeller(),
                 ).plotAll(filenamer.copy(description="matches_mag"), self.log)
        Analysis(matches, lambda cat: cat["distance"]*(1.0*afwGeom.radians).asArcseconds(),
                 "Distance (arcsec)", self.config.analysisMatches, prefix="src.",
                 qMin=0.0, qMax=self.config.matchesMaxDistance, labeller=MatchesStarGalaxyLabeller()
                 ).plotAll(filenamer.copy(description="matches_distance"), self.log, forcedMean=0.0)
        Analysis(matches, AstrometryDiff("src.coord.ra", "ref.coord.ra", "ref.coord.dec"),
                 "dRA*cos(Dec) (arcsec)", self.config.analysisMatches, prefix="src.",
                 qMin=-self.config.matchesMaxDistance, qMax=self.config.matchesMaxDistance,
                 labeller=MatchesStarGalaxyLabeller()
                 ).plotAll(filenamer.copy(description="matches_ra"), self.log)
        Analysis(matches, AstrometryDiff("src.coord.dec", "ref.coord.dec"),
                 "dDec (arcsec)", self.config.analysisMatches, prefix="src.",
                 qMin=-self.config.matchesMaxDistance, qMax=self.config.matchesMaxDistance,
                 labeller=MatchesStarGalaxyLabeller(),
                 ).plotAll(filenamer.copy(description="matches_dec"), self.log)

    def plotCosmos(self, catalog, filenamer, cosmos):
        labeller = CosmosLabeller(cosmos, self.config.matchRadius*afwGeom.arcseconds)
        Analysis(catalog, deconvMom, "Deconvolved moments", self.config.analysis, qMin=-1.0, qMax=6.0,
                 labeller=labeller).plotAll(filenamer.copy(description="cosmos"), self.log)

    def matchCatalog(self, catalog, filterName):
        astrometry = Astrometry(self.config.astrometry)
        average = sum((afwGeom.Extent3D(src.getCoord().getVector()) for src in catalog),
                      afwGeom.Extent3D(0, 0, 0))/len(catalog)
        center = afwCoord.IcrsCoord(afwGeom.Point3D(average))
        radius = max(center.angularSeparation(src.getCoord()) for src in catalog)
        filterName = afwImage.Filter(afwImage.Filter(filterName).getId()).getName() # Get primary name
        refs = astrometry.getReferenceSources(center.getLongitude(), center.getLatitude(), radius, filterName)
        matches = afwTable.matchRaDec(refs, catalog, self.config.matchRadius*afwGeom.arcseconds)
        return joinMatches(matches, "ref.", "src.")

    def _getConfigName(self):
        return None
    def _getMetadataName(self):
        return None
    def _getEupsVersionsName(self):
        return None


class ColorTransform(Config):
    description = Field(dtype=str, doc="Description of the color transform")
    plot = Field(dtype=bool, default=True, doc="Plot this color?")
    coeffs = DictField(keytype=str, itemtype=float, doc="Coefficients for each filter")
    requireGreater = DictField(keytype=str, itemtype=float, default={},
                               doc="Minimum values for colors so that this is useful")
    requireLess = DictField(keytype=str, itemtype=float, default={},
                            doc="Maximum values for colors so that this is useful")
    @classmethod
    def fromValues(cls, description, plot, coeffs, requireGreater={}, requireLess={}):
        self = cls()
        self.description = description
        self.plot = plot
        self.coeffs = coeffs
        self.requireGreater = requireGreater
        self.requireLess = requireLess
        return self

ivezicTransforms = {
    "wPerp": ColorTransform.fromValues("Ivezic w perpendicular", True,
                                       {"HSC-G": -0.227, "HSC-R": 0.792, "HSC-I": -0.567, "": 0.050},
                                       {"wPara": -0.2}, {"wPara": 0.6}),
    "xPerp": ColorTransform.fromValues("Ivezic x perpendicular", True,
                                       {"HSC-G": 0.707, "HSC-R": -0.707, "": -0.988},
                                       {"xPara": 0.8}, {"xPara": 1.6}),
    "yPerp": ColorTransform.fromValues("Ivezic y perpendicular", True,
                                       {"HSC-R": -0.270, "HSC-I": 0.800, "HSC-Z": -0.534, "": 0.054},
                                       {"yPara": 0.1}, {"yPara": 1.2}),
    "wPara": ColorTransform.fromValues("Ivezic w parallel", False,
                                       {"HSC-G": 0.928, "HSC-R": -0.556, "HSC-I": -0.372, "": -0.425}),
    "xPara": ColorTransform.fromValues("Ivezic x parallel", False, {"HSC-R": 1.0, "HSC-I": -1.0}),
    "yPara": ColorTransform.fromValues("Ivezic y parallel", False,
                                       {"HSC-R": 0.895, "HSC-I": -0.448, "HSC-Z": -0.447, "": -0.600}),
    }

straightTransforms = {
    "g": ColorTransform.fromValues("HSC-G", True, {"HSC-G": 1.0}),
    "r": ColorTransform.fromValues("HSC-R", True, {"HSC-R": 1.0}),
    "i": ColorTransform.fromValues("HSC-I", True, {"HSC-I": 1.0}),
    "z": ColorTransform.fromValues("HSC-Z", True, {"HSC-Z": 1.0}),
    "y": ColorTransform.fromValues("HSC-Y", True, {"HSC-Y": 1.0}),
    "n921": ColorTransform.fromValues("NB0921", True, {"NB0921": 1.0}),
}

class NumStarLabeller(object):
    labels = {'star': 0, 'maybe': 1, 'notStar': 2}
    plot = ["star"]
    def __init__(self, numBands):
        self.numBands = numBands
    def __call__(self, catalog):
        return numpy.array([0 if nn == self.numBands else 2 if nn == 0 else 1 for
                            nn in catalog["numStarFlags"]])

class ColorValueInRange(object):
    """Functor to produce color value if in the appropriate range"""
    def __init__(self, column, requireGreater, requireLess):
        self.column = column
        self.requireGreater = requireGreater
        self.requireLess = requireLess
    def __call__(self, catalog):
        good = numpy.ones(len(catalog), dtype=bool)
        for col, value in self.requireGreater.iteritems():
            good &= catalog[col] > value
        for col, value in self.requireLess.iteritems():
            good &= catalog[col] < value
        return numpy.where(good, catalog[self.column], numpy.nan)

class ColorAnalysisConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name for coadd")
    flags = ListField(dtype=str, doc="Flags of objects to ignore",
                      default=["centroid.sdss.flags", "flags.pixel.saturated.center",
                               "flags.pixel.interpolated.center", "flux.psf.flags"])
    analysis = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options")
    transforms = ConfigDictField(keytype=str, itemtype=ColorTransform, default={},
                                 doc="Color transformations to analyse")
    fluxFilter = Field(dtype=str, default="HSC-I", doc="Filter to use for plotting against magnitude")

    def setDefaults(self):
        Config.setDefaults(self)
        self.transforms = ivezicTransforms
        self.analysis.flags = [] # We remove bad source ourself

class ColorAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        FilterRefsDict = functools.partial(defaultdict, list) # Dict for filter-->dataRefs
        tractFilterRefs = defaultdict(FilterRefsDict) # tract-->filter-->dataRefs
        for patchRef in sum(parsedCmd.id.refList, []):
            if patchRef.datasetExists("deepCoadd_forced_src"):
                tract = patchRef.dataId["tract"]
                filterName = patchRef.dataId["filter"]
                tractFilterRefs[tract][filterName].append(patchRef)

        # Find tract,patch with full colour coverage (makes combining catalogs easier)
        bad = []
        for tract in tractFilterRefs:
            filterRefs = tractFilterRefs[tract]
            patchesForFilters = [set(patchRef.dataId["patch"] for patchRef in patchRefList) for
                                 patchRefList in filterRefs.itervalues()]
            if len(patchesForFilters) == 0:
                parsedCmd.log.warn("No input data found for tract %d" % tract)
                bad.append(tract)
                continue
            keep = set.intersection(*patchesForFilters) # Patches with full colour coverage
            tractFilterRefs[tract] = {ff: [patchRef for patchRef in filterRefs[ff] if
                                           patchRef.dataId["patch"] in keep] for ff in filterRefs}
        for tract in bad:
            del tractFilterRefs[tract]

        return [(filterRefs, kwargs) for filterRefs in tractFilterRefs.itervalues()]

class ColorAnalysisTask(CmdLineTask):
    ConfigClass = ColorAnalysisConfig
    RunnerClass = ColorAnalysisRunner
    _DefaultName = "colorAnalysis"


    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd_forced_src",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=HSC-X",
                               ContainerClass=TractDataIdContainer)
        return parser

    def run(self, patchRefsByFilter):
        for patchRefList in patchRefsByFilter.itervalues():
            patchRef = patchRefList[0]
            butler = patchRef.getButler()
            dataId = patchRef.dataId
            break
        filenamer = Filenamer(butler, "plotColor", dataId)
        catalogsByFilter = {ff: self.readCatalogs(patchRefList, "deepCoadd_forced_src") for
                            ff, patchRefList in patchRefsByFilter.iteritems()}
        catalog = self.transformCatalogs(catalogsByFilter, self.config.transforms)
        self.plotColors(catalog, filenamer, NumStarLabeller(len(catalogsByFilter)))
        self.plotColorColor(catalogsByFilter, filenamer.copy(description="fit"))

    def readCatalogs(self, patchRefList, dataset):
        catList = [patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS) for
                   patchRef in patchRefList if patchRef.datasetExists(dataset)]
        return concatenateCatalogs(catList)

    def transformCatalogs(self, catalogs, transforms):
        template = catalogs.values()[0]
        num = len(template)
        assert all(len(cat) == num for cat in catalogs.itervalues())

        mapper = afwTable.SchemaMapper(template.schema)
        mapper.addMinimalSchema(afwTable.SourceTable.makeMinimalSchema())
        schema = mapper.getOutputSchema()
        for col in transforms:
            schema.addField(col, float, transforms[col].description)
        schema.addField("numStarFlags", int, "Number of times source was flagged as star")
        badKey = schema.addField("bad", "Flag", "Is this a bad source?")
        schema.addField(self.config.analysis.fluxColumn, float, "Flux from filter " + self.config.fluxFilter)

        # Copy basics (id, RA, Dec)
        new = afwTable.SourceCatalog(schema)
        new.reserve(num)
        new.extend(template, mapper)

        # Set transformed colors
        for col, transform in transforms.iteritems():
            value = numpy.ones(num)*transform.coeffs[""] if "" in transform.coeffs else numpy.zeros(num)
            for ff, coeff in transform.coeffs.iteritems():
                if ff == "": # Constant: already done
                    continue
                cat = catalogs[ff]
                mag = self.config.analysis.zp - 2.5*numpy.log10(cat[self.config.analysis.fluxColumn])
                value += mag*coeff
            new[col][:] = value

        # Flag bad values
        bad = numpy.zeros(num, dtype=bool)
        for cat in catalogs.itervalues():
            for flag in self.config.flags:
                bad |= cat[flag]
        # Can't set column for flags; do row-by-row
        for row, badValue in zip(new, bad):
            row.setFlag(badKey, badValue)

        # Star/galaxy
        numStarFlags = numpy.zeros(num)
        for cat in catalogs.itervalues():
            numStarFlags += numpy.where(cat["classification.extendedness"] < 0.5, 1, 0)
        new["numStarFlags"][:] = numStarFlags

        fluxColumn = self.config.analysis.fluxColumn
        new[fluxColumn][:] = catalogs[self.config.fluxFilter][fluxColumn]

        return new

    def plotColors(self, catalog, filenamer, labeller):
        for col, transform in self.config.transforms.iteritems():
            if not transform.plot:
                continue
            Analysis(catalog, ColorValueInRange(col, transform.requireGreater, transform.requireLess), col,
                     self.config.analysis, flags=["bad"], labeller=labeller, qMin=-0.2, qMax=0.2,
                     ).plotAll(filenamer.copy(description="color_" + col), self.log)

    def plotColorColor(self, catalogs, filenamer):
        num = len(catalogs.values()[0])
        zp = self.config.analysis.zp
        mags = {ff: zp - 2.5*numpy.log10(catalogs[ff][self.config.analysis.fluxColumn]) for ff in catalogs}

        bad = numpy.zeros(num, dtype=bool)
        for cat in catalogs.itervalues():
            for flag in self.config.flags:
                bad |= cat[flag]

        bright = numpy.ones(num, dtype=bool)
        for mm in mags.itervalues():
            bright &= mm < self.config.analysis.magThreshold

        numStarFlags = numpy.zeros(num)
        for cat in catalogs.itervalues():
            numStarFlags += numpy.where(cat["classification.extendedness"] < 0.5, 1, 0)

        good = (numStarFlags == len(catalogs)) & numpy.logical_not(bad) & bright

        combined = self.transformCatalogs(catalogs, straightTransforms)[good].copy(True)
        filters = set(catalogs.keys())
        color = lambda c1, c2: (mags[c1] - mags[c2])[good]
        if filters.issuperset(set(("HSC-G", "HSC-R", "HSC-I"))):
            # Lower branch only; upper branch is noisy due to astrophysics
            poly = colorColorPlot(filenamer("gri"), color("HSC-G", "HSC-R"), color("HSC-R", "HSC-I"),
                                  "g-r", "r-i", (-0.5, 2.0), (-0.5, 2.0), order=3, xFitRange=(0.3, 1.1))
            Analysis(combined, ColorColorDistance("g", "r", "i", poly, 0.3, 1.1), "griPerp",
                     self.config.analysis, flags=["bad"], qMin=-0.1, qMax=0.1,
                     ).plotAll(filenamer.copy(description="gri"), self.log)
        if filters.issuperset(set(("HSC-R", "HSC-I", "HSC-Z"))):
            poly = colorColorPlot(filenamer("riz"), color("HSC-R", "HSC-I"), color("HSC-I", "HSC-Z"),
                                  "r-i", "i-z", (-0.5, 2.0), (-0.4, 0.8), order=3)
            Analysis(combined, ColorColorDistance("r", "i", "z", poly), "rizPerp",
                     self.config.analysis, flags=["bad"], qMin=-0.1, qMax=0.1,
                     ).plotAll(filenamer.copy(description="riz"), self.log)
        if filters.issuperset(set(("HSC-I", "HSC-Z", "HSC-Y"))):
            poly = colorColorPlot(filenamer("izy"), color("HSC-I", "HSC-Z"), color("HSC-Z", "HSC-Y"),
                                  "i-z", "z-y", (-0.4, 0.8), (-0.3, 0.5), order=3)
            Analysis(combined, ColorColorDistance("i", "z", "y", poly), "izyPerp",
                     self.config.analysis, flags=["bad"], qMin=-0.1, qMax=0.1,
                     ).plotAll(filenamer.copy(description="izy"), self.log)
        if filters.issuperset(set(("HSC-Z", "NB0921", "HSC-Y"))):
            poly = colorColorPlot(filenamer("z9y"), color("HSC-Z", "NB0921"), color("NB0921", "HSC-Y"),
                                  "z-n921", "n921-y", (-0.2, 0.2), (-0.1, 0.2), order=2,
                                  xFitRange=(-0.05, 0.15))
            Analysis(combined, ColorColorDistance("z", "n921", "y", poly), "z9yPerp",
                     self.config.analysis, flags=["bad"], qMin=-0.1, qMax=0.1,
                     ).plotAll(filenamer.copy(description="z9y"), self.log)

    def _getConfigName(self):
        return None
    def _getMetadataName(self):
        return None
    def _getEupsVersionsName(self):
        return None


def colorColorPlot(filename, xx, yy, xLabel, yLabel, xRange=None, yRange=None, order=1, iterations=1, rej=3.0,
                   xFitRange=None, numBins=51):
    fig, axes = plt.subplots(1, 2)
    if xRange:
        axes[0].set_xlim(*xRange)
    else:
        xRange = (0.9*xx.min(), 1.1*xx.max())
    if yRange:
        axes[0].set_ylim(*yRange)

    select = numpy.ones_like(xx, dtype=bool) if not xFitRange else ((xx > xFitRange[0]) & (xx < xFitRange[1]))
    keep = numpy.ones_like(xx, dtype=bool)
    for ii in range(iterations):
        keep &= select
        poly = numpy.polyfit(xx[keep], yy[keep], order)
        dy = yy - numpy.polyval(poly, xx)
        q1, q3 = numpy.percentile(dy[keep], [25, 75])
        clip = rej*0.74*(q3 - q1)
        keep = numpy.logical_not(numpy.abs(dy) > clip)

    keep &= select
    poly = numpy.polyfit(xx[keep], yy[keep], order)
    print poly
    xLine = numpy.linspace(xRange[0], xRange[1], 1000)
    yLine = numpy.polyval(poly, xLine)

    kwargs = dict(s=2, marker='o', lw=0, alpha=0.3)
    axes[0].scatter(xx[keep], yy[keep], c='blue', label="used", **kwargs)
    axes[0].scatter(xx[~keep], yy[~keep], c='black', label="other", **kwargs)
    axes[0].set_xlabel(xLabel)
    axes[0].set_ylabel(yLabel)
    axes[0].legend(loc="upper left") # usually blank in color-color plots...
    axes[0].plot(xLine, yLine, 'r-')

    # Determine quality of locus
    distance2 = []
    poly = numpy.poly1d(poly)
    polyDeriv = numpy.polyder(poly)
    calculateDistance2 = lambda x1, y1, x2: (x2 - x1)**2 + (poly(x2) - y1)**2
    for x, y in zip(xx[select], yy[select]):
        roots = numpy.roots(numpy.poly1d((1, -x)) + (poly - y)*polyDeriv)
        distance2.append(min(calculateDistance2(x, y, numpy.real(rr)) for rr in roots if numpy.real(rr) == rr))
    distance = numpy.sqrt(distance2)
    distance *= numpy.where(yy[select] >= poly(xx[select]), 1.0, -1.0)

    q1, median, q3 = numpy.percentile(distance, [25, 50, 75])
    good = numpy.logical_not(numpy.abs(distance - median) > 3.0*0.74*(q3 - q1))
    print distance[good].mean(), distance[good].std()

    axes[1].hist(distance[good], numBins, range=(-0.05, 0.05), normed=False)
    axes[1].set_xlabel("Distance to polynomial fit (mag)")
    axes[1].set_ylabel("Number")
    axes[1].set_yscale('log', nonposy='clip')

    fig.savefig(filename)
    plt.close(fig)

    return poly

class ColorColorDistance(object):
    """Functor to calculate distance from stellar locus in color-color plot"""
    def __init__(self, band1, band2, band3, poly, xMin=None, xMax=None):
        self.band1 = band1
        self.band2 = band2
        self.band3 = band3
        self.poly = poly
        self.xMin = xMin
        self.xMax = xMax
    def __call__(self, catalog):
        xx = catalog[self.band1] - catalog[self.band2]
        yy = catalog[self.band2] - catalog[self.band3]
        polyDeriv = numpy.polyder(self.poly)
        calculateDistance2 = lambda x1, y1, x2: (x2 - x1)**2 + (self.poly(x2) - y1)**2
        distance2 = numpy.ones_like(xx)*numpy.nan
        for i, (x, y) in enumerate(zip(xx, yy)):
            if (not numpy.isfinite(x) or not numpy.isfinite(y) or (self.xMin is not None and x < self.xMin) or
                (self.xMax is not None and x > self.xMax)):
                distance2[i] = numpy.nan
                continue
            roots = numpy.roots(numpy.poly1d((1, -x)) + (self.poly - y)*polyDeriv)
            distance2[i] = min(calculateDistance2(x, y, numpy.real(rr)) for
                               rr in roots if numpy.real(rr) == rr)
        return numpy.sqrt(distance2)*numpy.where(yy >= self.poly(xx), 1.0, -1.0)


class VisitAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        visits = defaultdict(list)
        for ref in parsedCmd.id.refList:
            visits[ref.dataId["visit"]].append(ref)
        return [(refs, kwargs) for refs in visits.itervalues()]

class VisitAnalysisTask(CoaddAnalysisTask):
    _DefaultName = "analysis"
    ConfigClass = CoaddAnalysisConfig
    RunnerClass = VisitAnalysisRunner

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "src", help="data ID, e.g. --id tract=0 visit=12345 ccd=6",
                               ContainerClass=PerTractCcdDataIdContainer)
        return parser

    def run(self, dataRefList):
        filterName = dataRefList[0].dataId["filter"]
        filenamer = Filenamer(dataRefList[0].getButler(), "plotVisit", dataRefList[0].dataId)
        if (self.config.doMags or self.config.doStarGalaxy or self.config.doOverlaps or cosmos or
            self.config.externalCatalogs):
            catalog = self.readCatalogs(dataRefList, "src")
        if self.config.doMags:
            self.plotMags(catalog, filenamer)
        if self.config.doStarGalaxy:
            self.plotStarGal(catalog, filenamer)
        if self.config.doMatches:
            matches = self.readMatches(dataRefList, "srcMatchFull")
            self.plotMatches(matches, filterName, filenamer)

        for cat in self.config.externalCatalogs:
            with andCatalog(cat):
                matches = self.matchCatalog(catalog, filterName)
                self.plotMatches(matches, filterName, filenamer.copy(description=cat))

    def readCatalogs(self, dataRefList, dataset):
        catList = [self.calibrateSourceCatalog(dataRef, dataRef.get(dataset, immediate=True,
                                                                    flags=afwTable.SOURCE_IO_NO_FOOTPRINTS),
                                               zp=self.config.analysis.zp)
                   for dataRef in dataRefList if dataRef.datasetExists(dataset)]
        return concatenateCatalogs(catList)

    def calibrateSourceCatalog(self, dataRef, catalog, zp=27.0):
        """Calibrate catalog with meas_mosaic results

        Requires a SourceCatalog input.
        """
        result = applyMosaicResultsCatalog(dataRef, catalog, False)
        catalog = result.catalog
        ffp = result.ffp
        factor = 10.0**(0.4*zp)/ffp.calib.getFluxMag0()[0]
        # Convert to constant zero point, as for the coadds
        fluxKeys, errKeys = getFluxKeys(catalog.schema)
        for key in fluxKeys.values() + errKeys.values():
            if len(catalog[key].shape) > 1:
                continue
            catalog[key][:] *= factor
        return catalog

    def readMatches(self, dataRefList, dataset):
        catList = []
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                continue
            catalog = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
            # Extract source catalog for calibration
            matches = matchesFromCatalog(catalog)
            schema = matches[0].second.schema
            src = afwTable.SourceCatalog(schema)
            src.reserve(len(catalog))
            for mm in matches:
                src.append(mm.second)
            matches[0].second.table.defineCentroid(schema["centroid.sdss"].asKey())
            src.table.defineCentroid(schema["centroid.sdss"].asKey())
            src = self.calibrateSourceCatalog(dataRef, src, zp=self.config.analysisMatches.zp)
            for mm, ss in zip(matches, src):
                mm.second = ss
            catalog = matchesToCatalog(matches, catalog.getTable().getMetadata())
            catList.append(catalog)
        return concatenateCatalogs(catList)
