import lsst.pipe.base as pipeBase
import lsst.geom as geom
from lsst.pex.config import Field
from lsst.meas.algorithms import Stamp, Stamps

__all__ = ['CalexpCutoutTaskConfig', 'CalexpCutoutTask']
DETECTOR_DIMENSIONS = ('instrument', 'visit', 'detector')


class CalexpCutoutTaskConnections(pipeBase.PipelineTaskConnections,
                                  dimensions=DETECTOR_DIMENSIONS,
                                  defaultTemplates={}):
    """Connections class for CalexpCutoutTask
    """
    in_table = pipeBase.connectionTypes.Input(
        doc="Locations for cutouts",
        name="cutout_positions",
        storageClass="AstropyQTable",
        dimensions=DETECTOR_DIMENSIONS,
    )
    calexp = pipeBase.connectionTypes.Input(
        doc="Calexp objects",
        name="calexp",
        storageClass="ExposureF",
        dimensions=DETECTOR_DIMENSIONS,
    )
    cutouts = pipeBase.connectionTypes.Output(
        doc="Cutouts",
        name="calexp_cutouts",
        storageClass="Stamps",
        dimensions=DETECTOR_DIMENSIONS,
    )


class CalexpCutoutTaskConfig(pipeBase.PipelineTaskConfig,
                             pipelineConnections=CalexpCutoutTaskConnections):
    """Configuration for CalexpCutoutTask
    """
    max_cutouts = Field(dtype=int, default=100, doc='Maximum number of entries to process. '
                                                    'The result will be the first N in the input table.')
    skip_bad = Field(dtype=bool, default=True, doc='Skip cutouts that do not fall completely within'
                                                   ' the calexp bounding box?  If set to False a ValueError'
                                                   ' is raised instead.')


class CalexpCutoutTask(pipeBase.PipelineTask):
    """Task for computing cutouts on a specific calexp given
    positions and sizes of the stamps.
    """
    ConfigClass = CalexpCutoutTaskConfig
    _DefaultName = "calexpCutoutTask"

    def run(self, in_table, calexp):
        """Compute and return the cutouts.

        Parameters
        ----------
        in_table : `astropy.QTable`
            A table containing at least the following columns: position, size.
            The position should be an `astropy.SkyCoord`.  The size is in pixels.
        calexp : `lsst.afw.image.ExposureF`
            The calibrated exposure from which to extract cutouts

        Returns
        -------
        output : `lsst.pipe.base.Struct`
            A struct containing a `lsst.meas.algorithms.Stamps` object
            that wraps a list of masked images of the cutouts and a
            `PropertyList` containing the metadata to be persisted
            with the cutouts.  The exposure metadata is preserved and,
            in addition, arrays holding the RA and Dec of each stamp
            in degrees are added to the metadata.

        Raises
        ------
        ValueError
            If the input catalog doesn't have the required columns,
            a ValueError is raised
        """
        if 'position' not in in_table.colnames or 'size' not in in_table.colnames:
            raise ValueError('Required column missing from the input table.  '
                             'Required columns are "position" and "size".'
                             f'The column names are: {in_table.colnames}')
        max_idx = self.config.max_cutouts
        cutout_list = []
        wcs = calexp.getWcs()
        mim = calexp.getMaskedImage()
        ras = []
        decs = []
        for rec in in_table[:max_idx]:
            ra = rec['position'].ra.degree
            dec = rec['position'].dec.degree
            ras.append(ra)
            decs.append(dec)
            pt = geom.SpherePoint(geom.Angle(ra, geom.degrees),
                                  geom.Angle(dec, geom.degrees))
            pix = wcs.skyToPixel(pt)
            size = rec['size'].value
            # Clamp to LL corner of the LL pixel and draw extent from there
            box = geom.Box2I(geom.Point2I(int(pix.x-size/2), int(pix.y-size/2)),
                             geom.Extent2I(size, size))
            if not mim.getBBox().contains(box):
                if not self.config.skip_bad:
                    raise ValueError(f'Cutout bounding box is not completely contained in the image: {box}')
                else:
                    continue
            sub = mim.Factory(mim, box)
            stamp = Stamp(stamp_im=sub, position=pt)
            cutout_list.append(stamp)
        metadata = calexp.getMetadata()
        metadata['RA_DEG'] = ras
        metadata['DEC_DEG'] = decs
        return pipeBase.Struct(cutouts=Stamps(cutout_list, metadata=metadata))
