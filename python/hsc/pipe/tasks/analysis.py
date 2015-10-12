#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy

from lsst.pex.config import Config, Field, ConfigField
from lsst.pipe.base import Struct, CmdLineTask, ArgumentParser, TaskRunner
from hsc.pipe.tasks.stack import TractDataIdContainer
from lsst.meas.photocal.colorterms import ColortermLibraryConfig

import lsst.afw.geom as afwGeom
import lsst.afw.table as afwTable

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


class Data(Struct):
    def __init__(self, catalog, quantity, mag, selection, color, plot=True):
        Struct.__init__(self, catalog=catalog[selection], quantity=quantity[selection], mag=mag[selection],
                        selection=selection, color=color, plot=plot)

class Stats(Struct):
    def __init__(self, num, total, mean, stdev, forcedMean):
        Struct.__init__(self, num=num, total=total, mean=mean, stdev=stdev, forcedMean=forcedMean)

colorList = ["blue", "red", "green", "black", "yellow", "cyan", "magenta", ]
flagsList = ["centroid.sdss.flags", "flags.pixel.saturated.center", "flags.pixel.interpolated.center"]

class Analysis(object):
    """Centralised base for plotting"""

    def __init__(self, catalog, func, quantityName, qMin=-0.2, qMax=0.2, fluxColumn="flux.psf", prefix="",
                 zp=27.0, magThreshold=None, magMaxPlot=30, clip=4.0, flags=flagsList, labeller=AllLabeller()):
        self.catalog = catalog
        self.func = func
        self.quantityName = quantityName
        self.qMin = qMin
        self.qMax = qMax
        self.fluxColumn = fluxColumn
        self.prefix = prefix
        self.zp = zp
        self.magThreshold = magThreshold
        self.magMaxPlot = magMaxPlot
        self.clip = clip
        self.flags = flags

        self.quantity = func(catalog)
        self.mag = zp - 2.5*numpy.log10(catalog[prefix + fluxColumn])

        self.good = numpy.isfinite(self.quantity) & numpy.isfinite(self.mag)
        for ff in flags:
            self.good &= ~catalog[prefix + ff]

        labels = labeller(catalog)
        self.data = {name: Data(catalog, self.quantity, self.mag, self.good & (labels == value),
                                colorList[value], name in labeller.plot) for name, value in
                     labeller.labels.iteritems()}

    def plotAgainstMag(self, filename):
        """Plot quantity against magnitude"""
        fig, axes = plt.subplots(1, 1)
        magMax = self.magMaxPlot
        for name, data in self.data.iteritems():
            axes.scatter(data.mag, data.quantity, s=2, marker='o', lw=0, c=data.color, label=name, alpha=0.3)
            magMax = min(self.magMaxPlot, data.mag.max())
        axes.set_xlabel("Mag from %s" % self.fluxColumn)
        axes.set_ylabel(self.quantityName)
        axes.set_ylim(self.qMin, self.qMax)
        axes.set_xlim(right=magMax)
        axes.legend()
        fig.savefig(filename)
        plt.close(fig)

    def plotHistogram(self, filename, numBins=50):
        """Plot histogram of quantity"""
        fig, axes = plt.subplots(1, 1)
        numMax = 0
        for name, data in self.data.iteritems():
            good = numpy.isfinite(data.quantity)
            if self.magThreshold is not None:
                good &= data.mag < self.magThreshold
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
        good = (self.mag < self.magThreshold if self.magThreshold is not None else
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
        good = (self.mag < self.magThreshold if self.magThreshold is not None else
                numpy.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(2, 1)
        for name, data in self.data.iteritems():
            selection = data.selection & good
            kwargs = {'s': 2, 'marker': 'o', 'lw': 0, 'c': data.color, 'alpha': 0.5}
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

    def plotAll(self, prefix, log, forcedMean=None):
        """Make all plots"""
        self.plotAgainstMag(prefix + "_psfMag.png")
        self.plotHistogram(prefix + "_hist.png")
        self.plotSkyPosition(prefix + "_sky.png")
        self.plotRaDec(prefix + "_radec.png")
        self.stats(log, prefix, forcedMean=forcedMean)

    def stats(self, log, title, forcedMean=None):
        """Calculate statistics on quantity"""
        stats = {}
        for name, data in self.data.iteritems():
            good = data.mag < self.magThreshold
            total = good.sum() # Total number we're considering
            quartiles = numpy.percentile(data.quantity[good], [25, 50, 75])
            median = quartiles[1]
            clip = self.clip*0.74*(quartiles[2] - quartiles[0])
            good &= numpy.logical_not(numpy.abs(data.quantity - median) > clip)
            actualMean = data.quantity[good].mean()
            mean = actualMean if forcedMean is None else forcedMean
            stdev = numpy.sqrt(((data.quantity[good].astype(numpy.float64) - mean)**2).mean())
            stats[name] = Stats(num=good.sum(), total=total, mean=actualMean, stdev=stdev,
                                forcedMean=forcedMean)
        log.info("Statistics for %s: %s" % (title, stats))

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





class CoaddAnalysisConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name for coadd")
    matchRadius = Field(dtype=float, default=0.5, doc="Matching radius (arcseconds)")
    colorterms = ConfigField(dtype=ColortermLibraryConfig, doc="Library of color terms")
    magThreshold = Field(dtype=float, default=21.0, doc="General magnitude threshold to apply")
    magThresholdMatches = Field(dtype=float, default=19.0, doc="Magnitude threshold to apply for matches")
    magMaxPlot = Field(dtype=float, default=30.0, doc="Maximum magnitude to plot")
    matchesMaxDistance = Field(dtype=float, default=0.1, doc="Maximum plotting distance for matches")

    doMags = Field(dtype=bool, default=True, doc="Plot magnitudes?")
    doStarGalaxy = Field(dtype=bool, default=True, doc="Plot star/galaxy?")
    doOverlaps = Field(dtype=bool, default=True, doc="Plot overlaps?")
    doMatches = Field(dtype=bool, default=True, doc="Plot matches?")


class CoaddAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs["filenamePrefix"] = parsedCmd.prefix
        kwargs["cosmos"] = parsedCmd.cosmos
        patchRefList = sum(parsedCmd.id.refList, [])
        return [(patchRefList, kwargs)]


class CoaddAnalysisTask(CmdLineTask):
    _DefaultName = "analysis"
    ConfigClass = CoaddAnalysisConfig
    RunnerClass = CoaddAnalysisRunner

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_argument("--prefix", default="", help="Prefix for filenames")
        parser.add_argument("--cosmos", default=None, help="Filename for Leauthaud Cosmos catalog")
        parser.add_id_argument("--id", "deepCoadd_meas",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=HSC-X",
                               ContainerClass=TractDataIdContainer)
        return parser

    def run(self, patchRefList, filenamePrefix, cosmos=None):
        if self.config.doMags or self.config.doStarGalaxy or self.config.doOverlaps or cosmos:
            catalog = self.readCatalogs(patchRefList, "deepCoadd_meas")
        if self.config.doMags:
            self.plotMags(catalog, filenamePrefix)
        if self.config.doStarGalaxy:
            self.plotStarGal(catalog, filenamePrefix)
        if cosmos:
            self.plotCosmos(catalog, filenamePrefix, cosmos)
        if self.config.doOverlaps:
            overlaps = self.overlaps(catalog)
            self.plotOverlaps(overlaps, filenamePrefix)
        if self.config.doMatches:
            matches = self.readCatalogs(patchRefList, "deepCoadd_srcMatchFull")
            self.plotMatches(matches, patchRefList[0].dataId["filter"], filenamePrefix)

    def readCatalogs(self, patchRefList, dataset):
        catList = [patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS) for
                   patchRef in patchRefList if patchRef.datasetExists(dataset)]
        return concatenateCatalogs(catList)

    def plotMags(self, catalog, filenamePrefix):
        for col in ["flux.sinc", "flux.kron", "cmodel.flux"]:
            Analysis(catalog, MagDiff(col, "flux.psf"), "Mag(%s) - PSFMag" % col,
                     magThreshold=self.config.magThreshold, magMaxPlot=self.config.magMaxPlot,
                     flags=flagsList + [col + ".flags"], labeller=StarGalaxyLabeller()
                     ).plotAll(filenamePrefix + "mag_" + col, self.log)

    def plotStarGal(self, catalog, filenamePrefix):
        Analysis(catalog, deconvMomStarGal, "pStar", qMin=0.0, qMax=1.0,
                 magThreshold=self.config.magThreshold, magMaxPlot=self.config.magMaxPlot
                 ).plotAll(filenamePrefix + "pStar", self.log)
        Analysis(catalog, deconvMom, "Deconvolved moments", qMin=-1.0, qMax=6.0,
                 magThreshold=self.config.magThreshold, magMaxPlot=self.config.magMaxPlot,
                 labeller=StarGalaxyLabeller()).plotAll(filenamePrefix + "deconvMom", self.log)

    def overlaps(self, catalog):
        matches = afwTable.matchRaDec(catalog, self.config.matchRadius*afwGeom.arcseconds, False)
        mapperList = afwTable.SchemaMapper.join(afwTable.SchemaVector([catalog.schema, catalog.schema]),
                                                ["first.", "second."])
        schema = mapperList[0].getOutputSchema()
        distanceKey = schema.addField("distance", type="Angle", doc="Distance between first and second")
        overlaps = afwTable.BaseCatalog(schema)
        overlaps.reserve(len(matches))
        for mm in matches:
            row = overlaps.addNew()
            row.assign(mm.first, mapperList[0])
            row.assign(mm.second, mapperList[1])
            row.set(distanceKey, mm.distance*afwGeom.radians)
        return overlaps

    def plotOverlaps(self, overlaps, filenamePrefix):
        for col in ["flux.psf", "flux.sinc", "flux.kron", "cmodel.flux"]:
            Analysis(overlaps, MagDiff("first." + col, "second." + col),
                     "Overlap mag difference (%s)" % col, prefix="first.",
                     magThreshold=self.config.magThreshold, magMaxPlot=self.config.magMaxPlot,
                     flags = flagsList + [col + ".flags"], labeller=OverlapsStarGalaxyLabeller()
            ).plotAll(filenamePrefix + "overlap_" + col, self.log)

        Analysis(overlaps, lambda cat: cat["distance"]*(1.0*afwGeom.radians).asArcseconds(),
                 "Distance (arcsec)", prefix="first.", qMin=0.0, qMax=0.15,
                 magThreshold=self.config.magThreshold, magMaxPlot=self.config.magMaxPlot,
                 labeller=OverlapsStarGalaxyLabeller()).plotAll(filenamePrefix + "overlap_distance", self.log,
                                                                forcedMean=0.0)

    def plotMatches(self, matches, filterName, filenamePrefix):
        ct = self.config.colorterms.selectColorTerm(filterName)
        Analysis(matches, MagDiffMatches("flux.psf", ct), "MagPsf - ref", prefix="src.",
                 magThreshold=self.config.magThresholdMatches, magMaxPlot=self.config.magMaxPlot,
                 labeller=MatchesStarGalaxyLabeller()).plotAll(filenamePrefix + "matches_mag", self.log)
        Analysis(matches, lambda cat: cat["distance"]*(1.0*afwGeom.radians).asArcseconds(),
                 "Distance (arcsec)", prefix="src.", qMin=0.0, qMax=self.config.matchesMaxDistance,
                 magThreshold=self.config.magThreshold, magMaxPlot=self.config.magMaxPlot,
                 labeller=MatchesStarGalaxyLabeller()).plotAll(filenamePrefix + "matches_distance", self.log,
                                                               forcedMean=0.0)
        Analysis(matches, AstrometryDiff("src.coord.ra", "ref.coord.ra", "ref.coord.dec"),
                 "dRA*cos(Dec) (arcsec)", prefix="src.", qMin=-self.config.matchesMaxDistance,
                 qMax=self.config.matchesMaxDistance, magThreshold=self.config.magThreshold,
                 magMaxPlot=self.config.magMaxPlot, labeller=MatchesStarGalaxyLabeller()
                 ).plotAll(filenamePrefix + "matches_ra", self.log)
        Analysis(matches, AstrometryDiff("src.coord.dec", "ref.coord.dec"),
                 "dDec (arcsec)", prefix="src.", qMin=-self.config.matchesMaxDistance,
                 qMax=self.config.matchesMaxDistance, magThreshold=self.config.magThreshold,
                 magMaxPlot=self.config.magMaxPlot, labeller=MatchesStarGalaxyLabeller()
                 ).plotAll(filenamePrefix + "matches_dec", self.log)

    def plotCosmos(self, catalog, filenamePrefix, cosmos):
        labeller = CosmosLabeller(cosmos, self.config.matchRadius*afwGeom.arcseconds)
        Analysis(catalog, deconvMom, "Deconvolved moments", qMin=-1.0, qMax=6.0,
                 magThreshold=self.config.magThreshold, magMaxPlot=self.config.magMaxPlot,
                 labeller=labeller).plotAll(filenamePrefix + "cosmos", self.log)

    def _getConfigName(self):
        return None
    def _getMetadataName(self):
        return None
    def _getEupsVersionsName(self):
        return None

