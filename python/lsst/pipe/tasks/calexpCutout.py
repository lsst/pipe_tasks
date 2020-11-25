import lsst.pipe.base as pipeBase
import lsst.geom as geom
from lsst.pex.config import Field
from lsst.meas.algorithms import Stamp, Stamps

__all__ = ['CalexpCutoutTaskConfig', 'CalexpCutoutTask']


class CalexpCutoutTaskConnections(pipeBase.PipelineTaskConnections,
                                  dimensions=("instrument", 'visit', 'detector'),
                                  defaultTemplates={}):
    """Connections class for CalexpCutoutTask
    """
    in_cat = pipeBase.connectionTypes.Input(
        doc="Locations for cutouts",
        name="cutout_positions",
        storageClass="StructuredDataDict",
        dimensions=("instrument", 'visit', 'detector'),
    )
    calexp = pipeBase.connectionTypes.Input(
        doc="Calexp objects",
        name="calexp",
        storageClass="ExposureF",
        dimensions=['instrument', 'visit', 'detector'],
    )
    cutouts = pipeBase.connectionTypes.Output(
        doc="Cutouts",
        name="calexp_cutouts",
        storageClass="Stamps",
        dimensions=("instrument", 'visit', 'detector'),
    )


class CalexpCutoutTaskConfig(pipeBase.PipelineTaskConfig,
                             pipelineConnections=CalexpCutoutTaskConnections):
    """Configuration for CalexpCutoutTask
    """
    max_cutouts = Field(dtype=int, default=100, doc='Maximum number of entries to process. '
                                                   'The result will be the first N in the catalog.')


class CalexpCutoutTask(pipeBase.PipelineTask):
    """Task for computing cutouts on a specific calexp given
    positions and sizes of the stamps.
    """
    ConfigClass = CalexpCutoutTaskConfig
    _DefaultName = "calexpCutoutTask"

    def run(self, in_cat, calexp):
        """Compute and return the cutouts.

        Parameters
        ----------
        in_cat : `dict`
            A dictionary containing at least the following columns: ra, dec, size.
            The coordinates should be in ICRS degrees.  The size is in pixels.
        calexp : `lsst.afw.image.ExposureF`
            The calibrated exposure from which to extract cutouts

        Returns
        -------
        output : `lsst.pipe.base.Struct`
            A struct containing a container class that wraps a list of
            masked images of the cutouts and a PropertyList containing
            the metadata to be persisted with the cutouts
        """
        max_idx = self.config.max_cutouts
        cutout_list = []
        wcs = calexp.getWcs()
        mim = calexp.getMaskedImage()
        metadata = calexp.getMetadata()
        metadata['RA_DEG'] = in_cat['ra'][:max_idx]
        metadata['DEC_DEG'] = in_cat['dec'][:max_idx]
        metadata['SIZE'] = in_cat['size'][:max_idx]
        for ident, ra, dec, size in zip(in_cat['ident'][:max_idx], in_cat['ra'][:max_idx],
                                        in_cat['dec'][:max_idx], in_cat['size'][:max_idx]):
            pt = geom.SpherePoint(geom.Angle(ra, geom.degrees),
                                  geom.Angle(dec, geom.degrees))
            pix = wcs.skyToPixel(pt)
            # Clamp to LL corner of the LL pixel and draw extent from there
            box = geom.Box2I(geom.Point2I(int(pix.x-size/2), int(pix.y-size/2)),
                             geom.Extent2I(size, size))
            # I think we need to think about what we want the origin to be: LOCAL or PARENT
            sub = mim.Factory(mim, box)
            stamp = Stamp(stamp_im=sub, position=pt, size=size)
            cutout_list.append(stamp)
        # We are using this stamp container as a place holder because it already has
        # the storage class defined
        return pipeBase.Struct(cutouts=Stamps(cutout_list, metadata=metadata))
