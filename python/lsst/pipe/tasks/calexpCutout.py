import lsst.pipe.base as pipeBase
import lsst.geom as geom
from lsst.meas.algorithms import Stamp, Stamps

__all__ = ['CalexpCutoutTaskConfig', 'CalexpCutoutTask']


class CalexpCutoutTaskConnections(pipeBase.PipelineTaskConnections,
                                  dimensions=("instrument", 'visit', 'detector'),
                                  defaultTemplates={}):
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
    pass


class CalexpCutoutTask(pipeBase.PipelineTask):

    ConfigClass = CalexpCutoutTaskConfig
    _DefaultName = "calexpCutoutTask"

    def run(self, in_cat, calexp):
        cutout_list = []
        wcs = calexp.getWcs()
        mim = calexp.getMaskedImage()
        metadata = calexp.getMetadata()
        metadata['RA_DEG'] = in_cat['ra'][:100]
        metadata['DEC_DEG'] = in_cat['dec'][:100]
        metadata['SIZE'] = in_cat['size'][:100]
        for i, ra, dec, size in zip(in_cat['ident'][:100], in_cat['ra'][:100],
                                    in_cat['dec'][:100], in_cat['size'][:100]):
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
