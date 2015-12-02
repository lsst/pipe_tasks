#!/usr/bin/env python

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
numpy.seterr(all="ignore")
from eups import Eups
eups = Eups()
import functools

from contextlib import contextmanager
from collections import defaultdict

from lsst.daf.persistence.butler import Butler
from lsst.daf.persistence.safeFileIo import safeMakeDir
from lsst.pex.config import Config, Field, ConfigField, ListField, DictField, ConfigDictField
from lsst.pipe.base import Struct, CmdLineTask, ArgumentParser, TaskRunner, TaskError
from lsst.coadd.utils import TractDataIdContainer
from lsst.meas.base.forcedPhotCcd import PerTractCcdDataIdContainer
from lsst.afw.table.catalogMatches import matchesToCatalog, matchesFromCatalog
from lsst.meas.astrom import AstrometryTask, AstrometryConfig, LoadAstrometryNetObjectsTask
from lsst.pipe.tasks.colorterms import ColortermLibrary
# from lsst.meas.mosaic.updateExposure import (applyMosaicResultsCatalog, applyCalib, getFluxFitParams,
#                                             getFluxKeys, getMosaicResults)

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
    _column = 'base_ClassificationExtendedness_value'
    def __call__(self, catalog):
        return numpy.where(catalog[self._column] < 0.5, 0, 1)

class OverlapsStarGalaxyLabeller(StarGalaxyLabeller):
    labels = {'star': 0, 'galaxy': 1, 'split': 2}
    def __init__(self, first="first_", second="second_"):
        self._first = first
        self._second = second
    def __call__(self, catalog):
        first = numpy.where(catalog[self._first + self._column] < 0.5, 0, 1)
        second = numpy.where(catalog[self._second + self._column] < 0.5, 0, 1)
        return numpy.where(first == second, first, 2)

class MatchesStarGalaxyLabeller(StarGalaxyLabeller):
    _column = 'src_base_ClassificationExtendedness_value'

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
        cosmos["coord_ra"][:] = original["ALPHA.J2000"][good]*(1.0*afwGeom.degrees).asRadians()
        cosmos["coord_dec"][:] = original["DELTA.J2000"][good]*(1.0*afwGeom.degrees).asRadians()
        self.cosmos = cosmos
        self.radius = radius

    def __call__(self, catalog):
        # A kdTree would be better, but this is all we have right now
        matches = afwTable.matchRaDec(self.cosmos, catalog, self.radius)
        good = set(mm.second.getId() for mm in matches)
        return numpy.array([0 if ii in good else 1 for ii in catalog["id"]])

class Filenamer(object):
    """Callable that provides a filename given a style"""
    def __init__(self, butler, dataset, dataId={}):
        self.butler = butler
        self.dataset = dataset
        self.dataId = dataId
    def __call__(self, dataId, **kwargs):
        filename = self.butler.get(self.dataset + "_filename", self.dataId, **kwargs)[0]
        safeMakeDir(os.path.dirname(filename))
        return filename

class Data(Struct):
    def __init__(self, catalog, quantity, mag, selection, color, error=None, plot=True):
        Struct.__init__(self, catalog=catalog[selection], quantity=quantity[selection], mag=mag[selection],
                        selection=selection, color=color, plot=plot,
                        error=error[selection] if error is not None else None)

class Stats(Struct):
    def __init__(self, num, total, mean, stdev, forcedMean):
        Struct.__init__(self, num=num, total=total, mean=mean, stdev=stdev, forcedMean=forcedMean)

colorList = ["blue", "red", "green", "black", "yellow", "cyan", "magenta", ]

class AnalysisConfig(Config):
    flags = ListField(dtype=str, doc="Flags of objects to ignore",
                      default=["base_SdssCentroid_flag", "base_PixelFlags_flag_saturatedCenter",
                               "base_PixelFlags_flag_interpolatedCenter", "base_PsfFlux_flag"])
    clip = Field(dtype=float, default=4.0, doc="Rejection threshold (stdev)")
    magThreshold = Field(dtype=float, default=21.0, doc="Magnitude threshold to apply")
    magPlotMin = Field(dtype=float, default=16.0, doc="Minimum magnitude to plot")
    magPlotMax = Field(dtype=float, default=28.0, doc="Maximum magnitude to plot")
    fluxColumn = Field(dtype=str, default="base_PsfFlux_flux", doc="Column to use for flux/magnitude plotting")
    zp = Field(dtype=float, default=27.0, doc="Magnitude zero point to apply")

class Analysis(object):
    """Centralised base for plotting"""

    def __init__(self, catalog, func, quantityName, shortName, config, qMin=-0.2, qMax=0.2, prefix="",
                 flags=[], errFunc=None, labeller=AllLabeller()):
        self.catalog = catalog
        self.func = func
        self.quantityName = quantityName
        self.shortName = shortName
        self.config = config
        self.qMin = qMin
        self.qMax = qMax
        self.prefix = prefix
        self.flags = flags
        self.errFunc = errFunc

        self.quantity = func(catalog)
        self.quantityError = errFunc(catalog) if errFunc is not None else None
        self.mag = self.config.zp - 2.5*numpy.log10(catalog[prefix + self.config.fluxColumn])

        self.good = numpy.isfinite(self.quantity) & numpy.isfinite(self.mag)
        if errFunc is not None:
            self.good &= numpy.isfinite(self.quantityError)
        for ff in list(config.flags) + flags:
            self.good &= ~catalog[prefix + ff]

        labels = labeller(catalog)
        self.data = {name: Data(catalog, self.quantity, self.mag, self.good & (labels == value),
                                colorList[value], self.quantityError, name in labeller.plot) for
                     name, value in labeller.labels.iteritems()}

    def plotAgainstMag(self, filename):
        """Plot quantity against magnitude"""
        fig, axes = plt.subplots(1, 1)
        magMin, magMax = self.config.magPlotMin, self.config.magPlotMax
        for name, data in self.data.iteritems():
            if len(data.mag) == 0:
                continue
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
            if len(data.mag) == 0:
                continue
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
        ra = numpy.rad2deg(self.catalog[self.prefix + "coord_ra"])
        dec = numpy.rad2deg(self.catalog[self.prefix + "coord_dec"])
        good = (self.mag < self.config.magThreshold if self.config.magThreshold > 0 else
                numpy.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(1, 1)
        for name, data in self.data.iteritems():
            if not data.plot:
                continue
            if len(data.mag) == 0:
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

        ra = numpy.rad2deg(self.catalog[self.prefix + "coord_ra"])
        dec = numpy.rad2deg(self.catalog[self.prefix + "coord_dec"])
        good = (self.mag < self.config.magThreshold if self.config.magThreshold is not None else
                numpy.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(2, 1)
        for name, data in self.data.iteritems():
            if len(data.mag) == 0:
                continue
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

    def plotAll(self, dataId, filenamer, log, enforcer=None, forcedMean=None):
        """Make all plots"""
        self.plotAgainstMag(filenamer(dataId, description=self.shortName, style="psfMag"))
        self.plotHistogram(filenamer(dataId, description=self.shortName, style="hist"))
        self.plotSkyPosition(filenamer(dataId, description=self.shortName, style="sky"))
        self.plotRaDec(filenamer(dataId, description=self.shortName, style="radec"))
        stats = self.stats(forcedMean=forcedMean)
        log.info("Statistics from %s of %s: %s" % (dataId, self.quantityName, stats))
        if enforcer:
            enforcer(stats, dataId, log, self.quantityName)
        return stats

    def stats(self, forcedMean=None):
        """Calculate statistics on quantity"""
        stats = {}
        for name, data in self.data.iteritems():
            if len(data.mag) == 0:
                continue
            good = data.mag < self.config.magThreshold
            stats[name] = self.calculateStats(data.quantity, good, forcedMean=forcedMean)
            if self.quantityError is not None:
                stats[name].sysErr = self.calculateSysError(data.quantity, data.error,
                                                            good, forcedMean=forcedMean)
        return stats

    def calculateStats(self, quantity, selection, forcedMean=None):
        total = selection.sum() # Total number we're considering
        if total == 0:
            return Stats(num=0, total=0, mean=numpy.nan, stdev=numpy.nan, forcedMean=forcedMean)
        quartiles = numpy.percentile(quantity[selection], [25, 50, 75])
        assert len(quartiles) == 3
        median = quartiles[1]
        clip = self.config.clip*0.74*(quartiles[2] - quartiles[0])
        good = selection & numpy.logical_not(numpy.abs(quantity - median) > clip)
        actualMean = quantity[good].mean()
        mean = actualMean if forcedMean is None else forcedMean
        stdev = numpy.sqrt(((quantity[good].astype(numpy.float64) - mean)**2).mean())
        return Stats(num=good.sum(), total=total, mean=actualMean, stdev=stdev, forcedMean=forcedMean)

    def calculateSysError(self, quantity, error, selection, forcedMean=None, tol=1.0e-3):
        import scipy.optimize
        def function(sysErr2):
            sigNoise = quantity/numpy.sqrt(error**2 + sysErr2)
            stats = self.calculateStats(sigNoise, selection, forcedMean=forcedMean)
            return stats.stdev - 1.0

        if True:
            result = scipy.optimize.root(function, 0.0, tol=tol)
            if not result.success:
                print "Warning: sysErr calculation failed: %s" % result.message
                answer = numpy.nan
            else:
                answer = numpy.sqrt(result.x[0])
        else:
            answer = numpy.sqrt(scipy.optimize.newton(function, 0.0, tol=tol))
        print function(answer**2), function((answer+0.001)**2), function((answer-0.001)**2)
        return answer


class Enforcer(object):
    """Functor for enforcing limits on statistics"""
    def __init__(self, requireGreater={}, requireLess={}, doRaise=False):
        self.requireGreater = requireGreater
        self.requireLess = requireLess
        self.doRaise = doRaise
    def __call__(self, stats, dataId, log, description):
        for label in self.requireGreater:
            for ss in self.requireGreater[label]:
                value = getattr(stats[label], ss)
                if value <= self.requireGreater[label][ss]:
                    text = ("%s %s = %f exceeds minimum limit of %f: %s" %
                            (description, ss, value, self.requireGreater[label][ss], dataId))
                    log.warn(text)
                    if self.doRaise:
                        raise AssertionError(text)
        for label in self.requireLess:
            for ss in self.requireLess[label]:
                value = getattr(stats[label], ss)
                if value >= self.requireLess[label][ss]:
                    text = ("%s %s = %f exceeds maximum limit of %f: %s" %
                            (description, ss, value, self.requireLess[label][ss], dataId))
                    log.warn(text)
                    if self.doRaise:
                        raise AssertionError(text)


class CcdAnalysis(Analysis):
    def plotAll(self, dataId, filenamer, log, enforcer=None, forcedMean=None):
        self.plotCcd(filenamer(dataId, description=self.shortName, style="ccd"))
        self.plotFocalPlane(filenamer(dataId, description=self.shortName, style="fpa"))
        return Analysis.plotAll(self, dataId, filenamer, log, enforcer=enforcer, forcedMean=forcedMean)

    def plotCcd(self, filename, centroid="base_SdssCentroid", cmap=plt.cm.hsv, idBits=32, visitMultiplier=200):
        """Plot quantity as a function of CCD x,y"""
        xx = self.catalog[self.prefix + centroid + "_x"]
        yy = self.catalog[self.prefix + centroid + "_y"]
        ccd = (self.catalog[self.prefix + "id"] >> idBits) % visitMultiplier
        good = (self.mag < self.config.magThreshold if self.config.magThreshold > 0 else
                numpy.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(2, 1)
        for name, data in self.data.iteritems():
            if not data.plot:
                continue
            if len(data.mag) == 0:
                continue
            selection = data.selection & good
            quantity = data.quantity[good[data.selection]]
            kwargs = {'s': 2, 'marker': 'o', 'lw': 0, 'alpha': 0.5, 'cmap': cmap}
            axes[0].scatter(xx[selection], quantity, c=ccd[selection], **kwargs)
            axes[1].scatter(yy[selection], quantity, c=ccd[selection], **kwargs)

        axes[0].set_xlabel("x_ccd")
        axes[0].set_ylabel(self.quantityName)
        axes[1].set_xlabel("y_ccd")
        axes[1].set_ylabel(self.quantityName)

        axes[0].set_xlim(-100, 2150)
        axes[1].set_xlim(-100, 4300)
        axes[0].set_ylim(self.qMin, self.qMax)
        axes[1].set_ylim(self.qMin, self.qMax)

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=ccd.min(), vmax=ccd.max()))
        mappable._A = []        # fake up the array of the scalar mappable. Urgh...
        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        cb = fig.colorbar(mappable, cax=cax)
        cb.set_label("CCD index", rotation=270)

        fig.savefig(filename)
        plt.close(fig)

    def plotFocalPlane(self, filename, cmap=plt.cm.brg):
        """Plot quantity colormaped on the focal plane"""
        xx = self.catalog[self.prefix + "base_FocalPlane_x"]
        yy = self.catalog[self.prefix + "base_FocalPlane_y"]
        good = (self.mag < self.config.magThreshold if self.config.magThreshold > 0 else
                numpy.ones(len(self.mag), dtype=bool))
        fig, axes = plt.subplots(1, 1)
        for name, data in self.data.iteritems():
            if not data.plot:
                continue
            if len(data.mag) == 0:
                continue
            selection = data.selection & good
            axes.scatter(xx[selection], yy[selection], s=2, marker='o', lw=0,
                         c=data.quantity[good[data.selection]], cmap=cmap, vmin=self.qMin, vmax=self.qMax)
        axes.set_xlabel("x_fpa (pixels)")
        axes.set_ylabel("y_fpa (pixels)")

        mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=self.qMin, vmax=self.qMax))
        mappable._A = []        # fake up the array of the scalar mappable. Urgh...
        cb = plt.colorbar(mappable)
        cb.set_label(self.quantityName, rotation=270)
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
        ref1 = -2.5*numpy.log10(catalog.get("ref_" + self.colorterm.primary + "_flux"))
        ref2 = -2.5*numpy.log10(catalog.get("ref_" + self.colorterm.secondary + "_flux"))
        ref = self.colorterm.transformMags(ref1, ref2)
        src = self.zp - 2.5*numpy.log10(catalog.get("src_" + self.column))
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
    if "ext_shapeHSM_HsmMoments" in catalog.schema:
        hsm = catalog["ext_shapeHSM_HsmMoments_xx"] + catalog["ext_shapeHSM_HsmMoments_yy"]
    else:
        hsm = numpy.ones(len(catalog))*numpy.nan
    sdss = catalog["base_SdssShape_xx"] + catalog["base_SdssShape_yy"]
    if "ext_shapeHSM_HsmPsfMoments" in catalog.schema:
        psf = catalog["ext_shapeHSM_HsmPsfMoments_xx"] + catalog["ext_shapeHSM_HsmPsfMoments_yy"]
    else:
        # LSST does not have shape.sdss.psf.  Could instead add base_PsfShape to catalog using
        # exposure.getPsf().computeShape(s.getCentroid()).getIxx()
        raise TaskError("No psf shape parameter found in catalog")
    return numpy.where(numpy.isfinite(hsm), hsm, sdss) - psf

def deconvMomStarGal(catalog):
    """Calculate P(star) from deconvolved moments"""
    rTrace = deconvMom(catalog)
    snr = catalog["base_PsfFlux_flux"]/catalog["base_PsfFlux_fluxSigma"]
    poly = (-4.2759879274 + 0.0713088756641*snr + 0.16352932561*rTrace - 4.54656639596e-05*snr*snr -
            0.0482134274008*snr*rTrace + 4.41366874902e-13*rTrace*rTrace + 7.58973714641e-09*snr*snr*snr +
            1.51008430135e-05*snr*snr*rTrace + 4.38493363998e-14*snr*rTrace*rTrace +
            1.83899834142e-20*rTrace*rTrace*rTrace)
    return 1.0/(1.0 + numpy.exp(-poly))


def concatenateCatalogs(catalogList):
    assert len(catalogList) > 0, "No catalogs to concatenate"
    template = catalogList[0]
    catalog = type(template)(template.schema)
    catalog.reserve(sum(len(cat) for cat in catalogList))
    for cat in catalogList:
        catalog.extend(cat, True)
    return catalog

def joinMatches(matches, first="first_", second="second_"):
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
    colorterms = ConfigField(dtype=ColortermLibrary, doc="Library of color terms")
    photoCatName = Field(dtype=str, default="sdss", doc="Name of photometric reference catalog; "
                         "used to select a color term dict in colorterms.""Name for coadd")
    analysis = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options")
    analysisMatches = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options for matches")
    matchesMaxDistance = Field(dtype=float, default=0.15, doc="Maximum plotting distance for matches")
    externalCatalogs = ConfigDictField(keytype=str, itemtype=AstrometryConfig, default={},
                                       doc="Additional external catalogs for matching")
    astrometry = ConfigField(dtype=AstrometryConfig, doc="Configuration for astrometric reference")
    doMags = Field(dtype=bool, default=True, doc="Plot magnitudes?")
    doStarGalaxy = Field(dtype=bool, default=True, doc="Plot star/galaxy?")
    doOverlaps = Field(dtype=bool, default=True, doc="Plot overlaps?")
    doMatches = Field(dtype=bool, default=True, doc="Plot matches?")
    doForced = Field(dtype=bool, default=True, doc="Plot difference between forced and unforced?")
    onlyReadStars = Field(dtype=bool, default=False, doc="Only read stars (to save memory)?")

    def saveToStream(self, outfile, root="root"):
        """Required for loading colorterms from a Config outside the 'lsst' namespace"""
        print >> outfile, "import lsst.meas.photocal.colorterms"
        return Config.saveToStream(self, outfile, root)

    def setDefaults(self):
        Config.setDefaults(self)
        astrom = AstrometryConfig()
        astrom.refObjLoader.filterMap["y"] = "z"
        astrom.refObjLoader.filterMap["N921"] = "z"
        # self.externalCatalogs = {"sdss-dr9-fink-v5b": astrom}
        self.analysisMatches.magThreshold = 19.0 # External catalogs like PS1 and SDSS used smaller telescopes


class CoaddAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs["cosmos"] = parsedCmd.cosmos

        # Partition all inputs by tract,filter
        FilterRefsDict = functools.partial(defaultdict, list) # Dict for filter-->dataRefs
        tractFilterRefs = defaultdict(FilterRefsDict) # tract-->filter-->dataRefs
        for patchRef in sum(parsedCmd.id.refList, []):
            if patchRef.datasetExists("deepCoadd_meas"):
                tract = patchRef.dataId["tract"]
                filterName = patchRef.dataId["filter"]
                tractFilterRefs[tract][filterName].append(patchRef)

        return [(tractFilterRefs[tract][filterName], kwargs) for tract in tractFilterRefs for
                filterName in tractFilterRefs[tract]]


class CoaddAnalysisTask(CmdLineTask):
    _DefaultName = "coaddAnalysis"
    ConfigClass = CoaddAnalysisConfig
    RunnerClass = CoaddAnalysisRunner
    AnalysisClass = Analysis
    outputDataset = "plotCoadd"

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_argument("--cosmos", default=None, help="Filename for Leauthaud Cosmos catalog")
        parser.add_id_argument("--id", "deepCoadd_meas",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=HSC-X",
                               ContainerClass=TractDataIdContainer)
        return parser

    def run(self, patchRefList, cosmos=None):
        dataId = patchRefList[0].dataId
        filterName = dataId["filter"]
        filenamer = Filenamer(patchRefList[0].getButler(), self.outputDataset, patchRefList[0].dataId)
        if (self.config.doMags or self.config.doStarGalaxy or self.config.doOverlaps or
            self.config.doForced or cosmos or self.config.externalCatalogs):
###            catalog = self.readCatalogs(patchRefList, "deepCoadd_meas")
###            catalog = catalog[catalog["deblend_nChild"] == 0].copy(True) # Don't care about blended objects
            catalog = self.readCatalogs(patchRefList, "deepCoadd_forced_src")
        if self.config.doMags:
            self.plotMags(catalog, filenamer, dataId)
        if self.config.doStarGalaxy:
            self.plotStarGal(catalog, filenamer, dataId)
        if cosmos:
            self.plotCosmos(catalog, filenamer, cosmos, dataId)
        if self.config.doForced:
            forced = self.readCatalogs(patchRefList, "deepCoadd_forced_src")
            self.plotForced(catalog, forced, filenamer, dataId)
        if self.config.doOverlaps:
            overlaps = self.overlaps(catalog)
            self.plotOverlaps(overlaps, filenamer, dataId)
        if self.config.doMatches:
            matches = self.readCatalogs(patchRefList, "deepCoadd_measMatchFull")
            self.plotMatches(matches, filterName, filenamer, dataId)

        for cat in self.config.externalCatalogs:
            with andCatalog(cat):
                matches = self.matchCatalog(catalog, filterName, self.config.externalCatalogs[cat])
                self.plotMatches(matches, filterName, filenamer, dataId, cat)

    def readCatalogs(self, patchRefList, dataset):
        catList = [patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS) for
                   patchRef in patchRefList if patchRef.datasetExists(dataset)]
        if len(catList) == 0:
            raise TaskError("No catalogs read: %s" % ([patchRef.dataId for patchRef in patchRefList]))
        if self.config.onlyReadStars and "base_ClassificationExtendedness_value" in catList[0].schema:
            catList = [cat[cat["base_ClassificationExtendedness_value"] < 0.5].copy(True) for cat in catList]
        return concatenateCatalogs(catList)

    def plotMags(self, catalog, filenamer, dataId):
        enforcer = Enforcer(requireLess={'star': {'stdev': 0.02}})
        for col in ["base_GaussianFlux", "ext_photometryKron_KronFlux", "modelfit_Cmodel"]:
            if col + "_flux" in catalog.schema:
                self.AnalysisClass(catalog, MagDiff(col + "_flux", "base_PsfFlux_flux"), "Mag(%s) - PSFMag"
                                   % col, "mag_" + col,
                                   self.config.analysis, flags=[col + "_flag"], labeller=StarGalaxyLabeller(),
                                   ).plotAll(dataId, filenamer, self.log, enforcer)

    def plotStarGal(self, catalog, filenamer, dataId):
        self.AnalysisClass(catalog, deconvMomStarGal, "pStar", "pStar", self.config.analysis,
                           qMin=0.0, qMax=1.0, ).plotAll(dataId, filenamer, self.log)
        self.AnalysisClass(catalog, deconvMom, "Deconvolved moments", "deconvMom", self.config.analysis,
                           qMin=-1.0, qMax=6.0, labeller=StarGalaxyLabeller()
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={'star': {'stdev': 0.2}}))

    def plotForced(self, unforced, forced, filenamer, dataId):
        catalog = joinMatches(afwTable.matchRaDec(unforced, forced,
                                                  self.config.matchRadius*afwGeom.arcseconds),
                              "unforced_", "forced_")
        catalog.writeFits(dataId["filter"] + ".fits")
        for col in ["base_PsfFlux", "base_GaussianFlux", "slot_CalibFlux", "ext_photometryKron_KronFlux",
                    "modelfit_Cmodel", "modelfit_Cmodel_exp_flux", "modelfit_Cmodel_dev_flux"]:
            if "forced." + col in catalog.schema:
                self.AnalysisClass(catalog, MagDiff("unforced." + col, "forced." + col),
                                   "Forced mag difference (%s)" % col, "forced_" + col, self.config.analysis,
                                   prefix="unforced.", flags=[col + ".flags"],
                                   labeller=OverlapsStarGalaxyLabeller("forced.", "unforced."),
                                   ).plotAll(dataId, filenamer, self.log)

    def overlaps(self, catalog):
        matches = afwTable.matchRaDec(catalog, self.config.matchRadius*afwGeom.arcseconds, False)
        return joinMatches(matches, "first_", "second_")

    def plotOverlaps(self, overlaps, filenamer, dataId):
        magEnforcer = Enforcer(requireLess={'star': {'stdev': 0.003}})
        for col in ["base_PsfFlux", "base_GaussianFlux", "ext_photometryKron_KronFlux", "modelfit_Cmodel"]:
            if "first_" + col + "_flux" in overlaps.schema:
                self.AnalysisClass(overlaps, MagDiff("first_" + col + "_flux", "second_" + col + "_flux"),
                                   "Overlap mag difference (%s)" % col, "overlap_" + col,
                                   self.config.analysis,
                                   prefix="first_", flags=[col + "_flag"],
                                   labeller=OverlapsStarGalaxyLabeller(),
                                   ).plotAll(dataId, filenamer, self.log, magEnforcer)

        distEnforcer = Enforcer(requireLess={'star': {'stdev': 0.005}})
        self.AnalysisClass(overlaps, lambda cat: cat["distance"]*(1.0*afwGeom.radians).asArcseconds(),
                           "Distance (arcsec)", "overlap_distance", self.config.analysis, prefix="first_",
                           qMin=0.0, qMax=0.15, labeller=OverlapsStarGalaxyLabeller(),
                           ).plotAll(dataId, filenamer, self.log, distEnforcer, forcedMean=0.0)

    def plotMatches(self, matches, filterName, filenamer, dataId, description="matches"):
        ct = self.config.colorterms.getColorterm(filterName, self.config.photoCatName)
        self.AnalysisClass(matches, MagDiffMatches("base_PsfFlux_flux", ct, zp=self.zp), "MagPsf - ref",
                           description + "_mag",
                           self.config.analysisMatches, prefix="src_", labeller=MatchesStarGalaxyLabeller(),
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={'star': {'stdev': 0.030}}))
        self.AnalysisClass(matches, lambda cat: cat["distance"]*(1.0*afwGeom.radians).asArcseconds(),
                           "Distance (arcsec)", description + "_distance", self.config.analysisMatches,
                           prefix="src_", qMin=0.0, qMax=self.config.matchesMaxDistance,
                           labeller=MatchesStarGalaxyLabeller()
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={'star': {'stdev': 0.050}}),
                                     forcedMean=0.0)
        self.AnalysisClass(matches, AstrometryDiff("src_coord_ra", "ref_coord_ra", "ref_coord_dec"),
                           "dRA*cos(Dec) (arcsec)", description + "_ra", self.config.analysisMatches,
                           prefix="src_", qMin=-self.config.matchesMaxDistance,
                           qMax=self.config.matchesMaxDistance, labeller=MatchesStarGalaxyLabeller(),
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={'star': {'stdev': 0.050}}))
        self.AnalysisClass(matches, AstrometryDiff("src_coord_dec", "ref_coord_dec"),
                           "dDec (arcsec)", description + "_dec", self.config.analysisMatches, prefix="src_",
                           qMin=-self.config.matchesMaxDistance, qMax=self.config.matchesMaxDistance,
                           labeller=MatchesStarGalaxyLabeller(),
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={'star': {'stdev': 0.050}}))

    def plotCosmos(self, catalog, filenamer, cosmos, dataId):
        labeller = CosmosLabeller(cosmos, self.config.matchRadius*afwGeom.arcseconds)
        self.AnalysisClass(catalog, deconvMom, "Deconvolved moments", "cosmos", self.config.analysis,
                           qMin=-1.0, qMax=6.0, labeller=labeller,
                           ).plotAll(dataId, filenamer, self.log,
                                     Enforcer(requireLess={'star': {'stdev': 0.2}}))

    def matchCatalog(self, catalog, filterName, astrometryConfig):
        astrometry = AstrometryTask(astrometryConfig)
        average = sum((afwGeom.Extent3D(src.getCoord().getVector()) for src in catalog),
                      afwGeom.Extent3D(0, 0, 0))/len(catalog)
        center = afwCoord.IcrsCoord(afwGeom.Point3D(average))
        radius = max(center.angularSeparation(src.getCoord()) for src in catalog)
        filterName = afwImage.Filter(afwImage.Filter(filterName).getId()).getName() # Get primary name
        refs = astrometry.refObjLoader.loadSkyCircle(center, radius, filterName).refCat
        matches = afwTable.matchRaDec(refs, catalog, self.config.matchRadius*afwGeom.arcseconds)
        return joinMatches(matches, "ref_", "src_")

    def _getConfigName(self):
        return None
    def _getMetadataName(self):
        return None
    def _getEupsVersionsName(self):
        return None


class SkyAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        kwargs["cosmos"] = parsedCmd.cosmos

        # Partition all inputs by filter
        filterRefs = defaultdict(list) # filter-->dataRefs
        for patchRef in sum(parsedCmd.id.refList, []):
            if patchRef.datasetExists("deepCoadd_meas"):
                filterName = patchRef.dataId["filter"]
                filterRefs[filterName].append(patchRef)

        return [(refList, kwargs) for refList in filterRefs.itervalues()]

class SkyAnalysisTask(CoaddAnalysisTask):
    """Version of CoaddAnalysisTask that runs on all inputs simultaneously

    This is most useful for utilising overlaps between tracts.
    """
    _DefaultName = "skyAnalysis"
    RunnerClass = SkyAnalysisRunner
    outputDataset = "plotSky"


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

class GalaxyColor(object):
    """Functor to produce difference between galaxy color calculated by different algorithms"""
    def __init__(self, alg1, alg2, prefix1, prefix2):
        self.alg1 = alg1
        self.alg2 = alg2
        self.prefix1 = prefix1
        self.prefix2 = prefix2
    def __call__(self, catalog):
        color1 = -2.5*numpy.log10(catalog[self.prefix1 + self.alg1]/catalog[self.prefix2 + self.alg1])
        color2 = -2.5*numpy.log10(catalog[self.prefix1 + self.alg2]/catalog[self.prefix2 + self.alg2])
        return color1 - color2



class ColorAnalysisConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name for coadd")
    flags = ListField(dtype=str, doc="Flags of objects to ignore",
                      default=["base_SdssCentroid_flag", "base_PixelFlags_flag_saturatedCenter",
                               "base_PixelFlags_flag_interpolatedCenter", "base_PsfFlux_flag"])
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
    AnalysisClass = Analysis
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
#        self.plotGalaxyColors(catalogsByFilter, filenamer, dataId)
        catalog = self.transformCatalogs(catalogsByFilter, self.config.transforms)
        self.plotStarColors(catalog, filenamer, NumStarLabeller(len(catalogsByFilter)), dataId)
        self.plotStarColorColor(catalogsByFilter, filenamer, dataId)

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
            if all(ff in catalogs for ff in transforms[col].coeffs):
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
            if col not in schema:
                continue
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
            numStarFlags += numpy.where(cat["base_ClassificationExtendedness_value"] < 0.5, 1, 0)
        new["numStarFlags"][:] = numStarFlags

        fluxColumn = self.config.analysis.fluxColumn
        new[fluxColumn][:] = catalogs[self.config.fluxFilter][fluxColumn]

        return new

    def plotGalaxyColors(self, catalogs, filenamer, dataId):
        filters = set(catalogs.keys())
        if filters.issuperset(set(("HSC-G", "HSC-I"))):
            gg = catalogs["HSC-G"]
            ii = catalogs["HSC-I"]
            assert len(gg) == len(ii)
            mapperList = afwTable.SchemaMapper.join(afwTable.SchemaVector([gg.schema, ii.schema]),
                                                    ["g_", "i_"])
            catalog = afwTable.BaseCatalog(mapperList[0].getOutputSchema())
            catalog.reserve(len(gg))
            for gRow, iRow in zip(gg, ii):
                row = catalog.addNew()
                row.assign(gRow, mapperList[0])
                row.assign(iRow, mapperList[1])

            catalog.writeFits("gi.fits")
            self.AnalysisClass(catalog, GalaxyColor("modelfit_CModel_flux", "slot_CalibFlux_flux", "g_", "i_"),
                               "(g-i)_cmodel - (g-i)_CalibFlux", "galaxy-TEST", self.config.analysis,
                               flags=["modelfit_CModel_flag", "slot_CalibFlux_flag"], prefix="i_",
                               labeller=OverlapsStarGalaxyLabeller("g_", "i_"),
                               qMin=-0.5, qMax=0.5,).plotAll(dataId, filenamer, self.log)


    def plotStarColors(self, catalog, filenamer, labeller, dataId):
        for col, transform in self.config.transforms.iteritems():
            if not transform.plot or col not in catalog:
                continue
            self.AnalysisClass(catalog, ColorValueInRange(col, transform.requireGreater,
                                                          transform.requireLess),
                               col, "color_" + col, self.config.analysis, flags=["bad"], labeller=labeller,
                               qMin=-0.2, qMax=0.2,).plotAll(dataId, filenamer, self.log)

    def plotStarColorColor(self, catalogs, filenamer, dataId):
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
            numStarFlags += numpy.where(cat["base_ClassificationExtendedness_value"] < 0.5, 1, 0)

        good = (numStarFlags == len(catalogs)) & numpy.logical_not(bad) & bright

        combined = self.transformCatalogs(catalogs, straightTransforms)[good].copy(True)
        filters = set(catalogs.keys())
        color = lambda c1, c2: (mags[c1] - mags[c2])[good]
        if filters.issuperset(set(("HSC-G", "HSC-R", "HSC-I"))):
            # Lower branch only; upper branch is noisy due to astrophysics
            poly = colorColorPlot(filenamer(dataId, description="gri", style="fit"),
                                  color("HSC-G", "HSC-R"), color("HSC-R", "HSC-I"), "g-r", "r-i",
                                  (-0.5, 2.0), (-0.5, 2.0), order=3, xFitRange=(0.3, 1.1))
            self.AnalysisClass(combined, ColorColorDistance("g", "r", "i", poly, 0.3, 1.1), "griPerp", "gri",
                               self.config.analysis, flags=["bad"], qMin=-0.1, qMax=0.1,
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={'all': {'stdev': 0.05}}))
        if filters.issuperset(set(("HSC-R", "HSC-I", "HSC-Z"))):
            poly = colorColorPlot(filenamer(dataId, description="riz", style="fit"),
                                  color("HSC-R", "HSC-I"), color("HSC-I", "HSC-Z"), "r-i", "i-z",
                                  (-0.5, 2.0), (-0.4, 0.8), order=3)
            self.AnalysisClass(combined, ColorColorDistance("r", "i", "z", poly), "rizPerp", "riz",
                               self.config.analysis, flags=["bad"], qMin=-0.1, qMax=0.1,
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={'all': {'stdev': 0.02}}))
        if filters.issuperset(set(("HSC-I", "HSC-Z", "HSC-Y"))):
            poly = colorColorPlot(filenamer(dataId, description="izy", style="fit"),
                                  color("HSC-I", "HSC-Z"), color("HSC-Z", "HSC-Y"), "i-z", "z-y",
                                  (-0.4, 0.8), (-0.3, 0.5), order=3)
            self.AnalysisClass(combined, ColorColorDistance("i", "z", "y", poly), "izyPerp", "izy",
                               self.config.analysis, flags=["bad"], qMin=-0.1, qMax=0.1,
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={'all': {'stdev': 0.02}}))
        if filters.issuperset(set(("HSC-Z", "NB0921", "HSC-Y"))):
            poly = colorColorPlot(filenamer(dataId, description="z9y", style="fit"),
                                  color("HSC-Z", "NB0921"), color("NB0921", "HSC-Y"), "z-n921", "n921-y",
                                  (-0.2, 0.2), (-0.1, 0.2), order=2, xFitRange=(-0.05, 0.15))
            self.AnalysisClass(combined, ColorColorDistance("z", "n921", "y", poly), "z9yPerp", "z9y",
                               self.config.analysis, flags=["bad"], qMin=-0.1, qMax=0.1,
                               ).plotAll(dataId, filenamer, self.log,
                                         Enforcer(requireLess={'all': {'stdev': 0.02}}))

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
    _DefaultName = "visitAnalysis"
    ConfigClass = CoaddAnalysisConfig
    RunnerClass = VisitAnalysisRunner
    AnalysisClass = CcdAnalysis

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "src", help="data ID, e.g. --id visit=12345 ccd=6^8..11")
        return parser

    def run(self, dataRefList):
        self.log.info("dataRefList size: %d" % len(dataRefList))
        dataId = dataRefList[0].dataId
        self.log.info("dataId: %s" % (dataId,))
        filterName = dataId["filter"]
        filenamer = Filenamer(dataRefList[0].getButler(), "plotVisit", dataRefList[0].dataId)
        if (self.config.doMags or self.config.doStarGalaxy or self.config.doOverlaps or cosmos or
            self.config.externalCatalogs):
            catalog = self.readCatalogs(dataRefList, "src")
        if self.config.doMags:
            self.plotMags(catalog, filenamer, dataId)
        if self.config.doStarGalaxy:
            self.plotStarGal(catalog, filenamer, dataId)
        if self.config.doMatches:
            matches = self.readSrcMatches(dataRefList, "src")
            self.plotMatches(matches, filterName, filenamer, dataId)

        for cat in self.config.externalCatalogs:
            with andCatalog(cat):
                matches = self.matchCatalog(catalog, filterName, self.config.externalCatalogs[cat])
                self.plotMatches(matches, filterName, filenamer, dataId, cat)

    def readCatalogs(self, dataRefList, dataset):
        catList = []
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                continue
            catalog = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
            try:
                calibrated = self.calibrateSourceCatalog(dataRef, catalog, zp=self.config.analysis.zp)
                catList.append(calibrated)
            except Exception as e:
                self.log.warn("Unable to calibrate catalog for %s: %s" % (dataRef.dataId, e))
                catList.append(catalog)
                continue

        if len(catList) == 0:
            raise TaskError("No catalogs read: %s" % ([dataRef.dataId for dataRef in dataRefList]))

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

    def readSrcMatches(self, dataRefList, dataset):
        catList = []
        for dataRef in dataRefList:
            if not dataRef.datasetExists(dataset):
                continue
            butler = dataRef.getButler()
            # Generate unnormalized match list (from normalized persisted one) with joinMatchListWithCatalog
            # (which requires a refObjLoader to be initialized).
            catalog = dataRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
            sources = butler.get(dataset, dataRef.dataId, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)
            packedMatches = butler.get(dataset + "Match", dataRef.dataId)
            refObjLoaderConfig = LoadAstrometryNetObjectsTask.ConfigClass()
            refObjLoader = LoadAstrometryNetObjectsTask(refObjLoaderConfig)
            matches = refObjLoader.joinMatchListWithCatalog(packedMatches, sources)

            if len(matches) == 0:
                self.log.warn("No matches for %s" % (dataRef.dataId,))
                continue
            self.log.info("len(matches) = %d" % len(matches))

            schema = matches[0].second.schema
            src = afwTable.SourceCatalog(schema)
            src.reserve(len(catalog))
            for mm in matches:
                src.append(mm.second)
            matches[0].second.table.defineCentroid("base_SdssCentroid")
            src.table.defineCentroid("base_SdssCentroid")
            try:
                src = self.calibrateSourceCatalog(dataRef, src, zp=self.config.analysisMatches.zp)
            except Exception as e:
                self.log.warn("Unable to calibrate catalog for %s: %s" % (dataRef.dataId, e))
                metadata = butler.get("calexp_md", dataRef.dataId)
                self.zp = 2.5*numpy.log10(metadata.get("FLUXMAG0"))
                self.log.warn("Using 2.5*log10(FLUXMAG0) = %.4f from FITS header for zeropoint" % (self.zp))

            for mm, ss in zip(matches, src):
                mm.second = ss
            catalog = matchesToCatalog(matches, catalog.getTable().getMetadata())
            catList.append(catalog)

        if len(catList) == 0:
            raise TaskError("No matches read: %s" % ([dataRef.dataId for dataRef in dataRefList]))

        return concatenateCatalogs(catList)

###class CompareAnalysis(Analysis):
###    def __init__(self, catalog, func, errFunc, quantityName, shortName, config, qMin=-0.2, qMax=0.2, prefix="",
###                 flags=[], labeller=AllLabeller()):
###        Analysis.__init__(self, catalog, func, errFunc, quantityName, shortName, config, qMin=qMin, qMax=qMax,
###                          prefix=prefix, flags=flags, labeller=labeller)
###        # Add errors
###
###    def stats(self, forcedMean=None):
###        """Calculate statistics on quantity"""
###        stats = {}
###        for name, data in self.data.iteritems():
###            if len(data.mag) == 0:
###                continue
###            good = data.mag < self.config.magThreshold
###            total = good.sum() # Total number we're considering
###            quartiles = numpy.percentile(data.quantity[good], [25, 50, 75])
###            assert len(quartiles) == 3
###            median = quartiles[1]
###            clip = self.config.clip*0.74*(quartiles[2] - quartiles[0])
###            good &= numpy.logical_not(numpy.abs(data.quantity - median) > clip)
###            actualMean = data.quantity[good].mean()
###            mean = actualMean if forcedMean is None else forcedMean
###            stdev = numpy.sqrt(((data.quantity[good].astype(numpy.float64) - mean)**2).mean())
###            stats[name] = Stats(num=good.sum(), total=total, mean=actualMean, stdev=stdev,
###                                forcedMean=forcedMean)
###        return stats


class CompareAnalysisConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name for coadd")
    matchRadius = Field(dtype=float, default=0.5, doc="Matching radius (arcseconds)")
    analysis = ConfigField(dtype=AnalysisConfig, doc="Analysis plotting options")
    doMags = Field(dtype=bool, default=True, doc="Plot magnitudes?")
    doCentroids = Field(dtype=bool, default=True, doc="Plot centroids?")
    sysErrMags = Field(dtype=float, default=0.015, doc="Systematic error in magnitudes")
    sysErrCentroids = Field(dtype=float, default=0.15, doc="Systematic error in centroids (pixels)")

class CompareAnalysisRunner(TaskRunner):
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        parentDir = parsedCmd.input
        while os.path.exists(os.path.join(parentDir, "_parent")):
            parentDir = os.path.realpath(os.path.join(parentDir, "_parent"))
        butler2 = Butler(root=os.path.join(parentDir, "rerun", parsedCmd.rerun2), calibRoot=parsedCmd.calib)
        idParser = parsedCmd.id.__class__(parsedCmd.id.level)
        idParser.idList = parsedCmd.id.idList
        butler = parsedCmd.butler
        parsedCmd.butler = butler2
        idParser.makeDataRefList(parsedCmd)
        parsedCmd.butler = butler

        return [(refList1, dict(patchRefList2=refList2, **kwargs)) for
                refList1, refList2 in zip(parsedCmd.id.refList, idParser.refList)]

class CompareAnalysisTask(CmdLineTask):
    ConfigClass = CompareAnalysisConfig
    RunnerClass = CompareAnalysisRunner
    _DefaultName = "compareAnalysis"

    @classmethod
    def _makeArgumentParser(cls):
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_argument("--rerun2", required=True, help="Second rerun, for comparison")
        parser.add_id_argument("--id", "deepCoadd_forced_src",
                               help="data ID, e.g. --id tract=12345 patch=1,2 filter=HSC-X",
                               ContainerClass=TractDataIdContainer)
        return parser

    def run(self, patchRefList1, patchRefList2):
        dataId = patchRefList1[0].dataId
        filenamer = Filenamer(patchRefList1[0].getButler(), "plotCompare", patchRefList1[0].dataId)
        catalog1 = self.readCatalogs(patchRefList1, self.config.coaddName + "Coadd_forced_src")
        catalog2 = self.readCatalogs(patchRefList2, self.config.coaddName + "Coadd_forced_src")
        catalog = self.matchCatalogs(catalog1, catalog2)
        if self.config.doMags:
            self.plotMags(catalog, filenamer, dataId)
        if self.config.doCentroids:
            self.plotCentroids(catalog, filenamer, dataId)

    def readCatalogs(self, patchRefList, dataset):
        catList = [patchRef.get(dataset, immediate=True, flags=afwTable.SOURCE_IO_NO_FOOTPRINTS) for
                   patchRef in patchRefList if patchRef.datasetExists(dataset)]
        return concatenateCatalogs(catList)

    def matchCatalogs(self, catalog1, catalog2):
        matches = afwTable.matchRaDec(catalog1, catalog2, self.config.matchRadius*afwGeom.arcseconds)
        return joinMatches(matches, "first_", "second_")

    def plotMags(self, catalog, filenamer, dataId):
        enforcer = None # Enforcer(requireLess={'star': {'stdev': 0.02}})
        # for col in ["base_PsfFlux_flux", "flux.sinc", "flux.kron", "cmodel.flux"]:
        for col in ["base_PsfFlux_flux", "base_GaussianFlux_flux", "ext_photometryKron_KronFlux_flux"]: #, "modelfit_Cmodel-flux"]:
            if "first_" + col in catalog.schema and "second_" + col in catalog.schema:
                Analysis(catalog, MagDiff("first_" + col, "second_" + col),
                         "Mag difference (%s)" % col, "diff_" + col, self.config.analysis,
                         prefix="first_", flags=[col + "_flag"], errFunc=MagDiffErr(col),
                         labeller=OverlapsStarGalaxyLabeller(),
                         ).plotAll(dataId, filenamer, self.log, enforcer)

    def plotCentroids(self, catalog, filenamer, dataId):
        distEnforcer = None # Enforcer(requireLess={'star': {'stdev': 0.005}})
        Analysis(catalog, CentroidDiff("x"), "x offset (arcsec)", "diff_x", self.config.analysis,
                 prefix="first_", qMin=-0.2, qMax=0.2, errFunc=CentroidDiffErr("x"),
                 labeller=OverlapsStarGalaxyLabeller(),
                 ).plotAll(dataId, filenamer, self.log, distEnforcer)
        Analysis(catalog, CentroidDiff("y"), "y offset (arcsec)", "diff_y", self.config.analysis,
                 prefix="first_", qMin=-0.2, qMax=0.2, errFunc=CentroidDiffErr("y"),
                 labeller=OverlapsStarGalaxyLabeller(),
                 ).plotAll(dataId, filenamer, self.log, distEnforcer)

    def _getConfigName(self):
        return None
    def _getMetadataName(self):
        return None
    def _getEupsVersionsName(self):
        return None


class MagDiffErr(object):
    """Functor to calculate magnitude difference error"""
    def __init__(self, column):
        zp = 27.0 # Exact value is not important, since we're differencing the magnitudes
        self.column = column
        self.calib = afwImage.Calib()
        self.calib.setFluxMag0(10.0**(0.4*zp))
        self.calib.setThrowOnNegativeFlux(False)
    def __call__(self, catalog):
        mag1, err1 = self.calib.getMagnitude(catalog["first_" + self.column],
                                             catalog["first_" + self.column + "Sigma"])
        mag2, err2 = self.calib.getMagnitude(catalog["second_" + self.column],
                                             catalog["second_" + self.column + "Sigma"])
        return numpy.sqrt(err1**2 + err2**2)

class CentroidDiff(object):
    """Functor to calculate difference in astrometry"""
    def __init__(self, component, first="first_", second="second_", centroid="base_SdssCentroid"):
        self.component = component
        self.first = first
        self.second = second
        self.centroid = centroid

    def __call__(self, catalog):
        first = self.first + self.centroid + "_" + self.component
        second = self.second + self.centroid + "_" + self.component
        return catalog[first] - catalog[second]

class CentroidDiffErr(CentroidDiff):
    """Functor to calculate difference error for astrometry"""
    def __call__(self, catalog):
        first = self.first + self.centroid + "Sigma"
        second = self.second + self.centroid + "Sigma"

        subkeys1 = catalog.schema[first].asKey().subkeys
        subkeys2 = catalog.schema[second].asKey().subkeys

        # subkeys are xx, xy, yy
        # Ignoring xy because it's not set
        menu = {'x': 0, 'y': 2}

        return numpy.hypot(catalog[subkeys1[menu[self.component]]], catalog[subkeys2[menu[self.component]]])
