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

import functools
import io
import os
import pathlib
import pandas as pd

import lsst.dax.apdb
import lsst.pex.config as pexConfig
import lsst.pipe.base

__all__ = ["ZooniverseCutoutsConfig", "ZooniverseCutoutsTask"]


class ZooniverseCutoutsConfig(pexConfig.Config):
    size = pexConfig.Field(
        doc="Size of cutout to extract for image from science, template, and difference exposures."
            " TODO: should we have an option to use the source footprints as the cutout size instead?",
        dtype=int,
        default=30
    )
    urlRoot = pexConfig.Field(
        doc="URL that the resulting images will be served to Zooniverse from.",
        dtype=str,
        default=None
    )
    # TODO: we'll likely need these for implementing select_sources, but
    # apdb does not yet have an API for `where` statements.
    # apdb = pexConfig.ConfigurableField(
    #     target=lsst.dax.apdb.Apdb,
    #     ConfigClass=lsst.dax.apdb.ApdbConfig,
    #     doc="Database connection for retreiving DiaSources."
    #         "Must already be initialized.",
    # )
    # query = pexConfig.Field(
    #     doc="SQL query to run on APDB to identify sources to extract cutouts of.",
    #     default="select * from DiaSource LIMIT 100",
    #     dtype=str
    # )
    outputPath = pexConfig.Field(
        doc="The full path to write the files to; images will go in `outputPath/images/`, "
            "while the manifest file will go in `outputPath/`",
        default=None,
        dtype=str
    )


class ZooniverseCutoutsTask(lsst.pipe.base.Task):
    """Generate cutouts and a manifest for upload to a Zooniverse project.
    """
    ConfigClass = ZooniverseCutoutsConfig
    _DefaultName = "zooniverseCutouts"

    def run(self, butler):
        """Select sources from apdb and generate zooniverse cutouts an a
        manifest for upload.

        This method is entirely TODO, and is a placeholder while we sort out
        select_sources and how we want the butler to be passed in.
        """
        data = self.select_sources()
        self.write_images(data, butler)

        manifest = self.make_manifest(data)
        manifest.to_csv(os.path.join(self.config.outputPath, "manifest.csv"), index=False)

    def select_sources(self):
        """Select sources from apdb to make cutouts of.

        This method is entirely TODO until we have a way to make where queries
        on apdb.
        """
        raise NotImplementedError

    @staticmethod
    def _make_path(source, base_path):
        """Return a URL or file path for this source.

        Parameters
        ----------
        source : `pandas.DataFrame`
            DataFrame containing at least a ``diaSourceId`` field.
        base_path : `str`
            Base URL or directory path, with no ending ``/``.

        Returns
        -------
        path : `str`
            Formatted URL or path.
        """
        return f"{base_path}/images/{int(source.loc['diaSourceId'])}.png"

    def make_manifest(self, data):
        """Return a Zooniverse manifest attaching image URLs to source ids.

        Parameters
        ----------
        data : `pandas.DataFrame`
            DataFrame retrieved from APDB, conntaining at least a
            ``diaSourceId`` column.

        Returns
        -------
        manifest : `pandas.DataFrame`
            The formatted URL manifest for upload to Zooniverse.
        """
        manifest = pd.DataFrame()
        manifest['external_id'] = data['diaSourceId']
        manifest['location:1'] = data.apply(self._make_path, axis=1, args=(self.config.urlRoot.rstrip('/'),))
        manifest['metadata:diaSourceId'] = data['diaSourceId']
        return manifest

    def write_images(self, data, butler):
        """Make the 3-part cutout images for each requested source and write
        them to disk.

        Creates a `images/` subdirectory in `self.config.outputPath` if one
        does not already exist; images are written there as PNG files.

        Parameters
        ----------
        data : `pandas.DataFrame`
            The DiaSources to extract cutouts for. Must contain at least these
            fields: ``ra, dec, diaSourceId, ccd, visit, instrument``.
        butler : `lsst.daf.butler.Butler`
            The butler connection to use to load the data; create it with the
            collections you wish to load images from.
        """
        @functools.lru_cache(maxsize=16)
        def get_exposures(instrument, detector, visit):
            """Return science, template, difference exposures, and use a small
            cache so we don't have to re-read files as often.

            NOTE: If we redo this all to work with BPS or other parallelized
            systems, or get good butler-side cacheing, we should remove the
            lru_cache above.
            """
            dataId = {'instrument': instrument, 'detector': detector, 'visit': visit}
            return (butler.get('calexp', dataId),
                    butler.get('deepDiff_warpedExp', dataId),
                    butler.get('deepDiff_differenceExp', dataId))

        # Create a subdirectory for the images.
        pathlib.Path(os.path.join(self.config.outputPath, "images")).mkdir(exist_ok=True)

        for index, source in data.iterrows():
            center = lsst.geom.SpherePoint(source['ra'], source['decl'], lsst.geom.degrees)
            science, template, difference = get_exposures(int(source['ccd']), int(source['visit']))
            image = self.generate_image(science, template, difference, center)
            with open(self._make_path(source, self.config.outputPath), "wb") as outfile:
                outfile.write(image.getbuffer())

    def generate_image(self, science, template, difference, center):
        """Get a 3-part cutout image to save to disk, for a single source.

        Parameters
        ----------
        science : `lsst.afw.image.ExposureF`
            Science exposure to include in the cutout.
        template : `lsst.afw.image.ExposureF`
            Matched template exposure to include in the cutout.
        difference : `lsst.afw.image.ExposureF`
             Matched science minus template exposure to include in the cutout.
        center : `lsst.geom.SpherePoint`
            Center of the source to be cut out of each image.

        Returns
        -------
        image: `io.BytesIO`
            The generated image, to be output to a file or displayed on screen.
        """
        size = lsst.geom.Extent2I(self.config.size, self.config.size)
        return self._plot_cutout(science.getCutout(center, size),
                                 template.getCutout(center, size),
                                 difference.getCutout(center, size)
                                 )

    def _plot_cutout(self, science, template, difference):
        """Plot the cutouts for a source in one image.

        Parameters
        ----------
        science : `lsst.afw.image.ExposureF`
            Cutout Science exposure to include in the image.
        template : `lsst.afw.image.ExposureF`
            Cutout template exposure to include in the image.
        difference : `lsst.afw.image.ExposureF`
             Cutout science minus template exposure to include in the image.

        Returns
        -------
        image: `io.BytesIO`
            The generated image, to be output to a file via
            `image.write(filename)` or displayed on screen.
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        import astropy.visualization as aviz

        # TODO: how do we color masked pixels (including edges)?

        def plot_one_image(ax, data, name):
            """Plot a normalized image on an axis."""
            if name == 'Difference':
                norm = aviz.ImageNormalize(data, interval=aviz.ZScaleInterval(),
                                           stretch=aviz.LinearStretch())
            else:
                norm = aviz.ImageNormalize(data, interval=aviz.MinMaxInterval(),
                                           stretch=aviz.AsinhStretch(a=0.1))
            ax.imshow(data, cmap=cm.bone, interpolation="none", norm=norm)
            ax.axis('off')
            ax.set_title(name)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        plot_one_image(ax1, template.image.array, "Template")
        plot_one_image(ax2, science.image.array, "Science")
        plot_one_image(ax3, difference.image.array, "Difference")
        plt.tight_layout()

        output = io.BytesIO()
        plt.savefig(output, bbox_inches="tight", format="png")
        output.seek(0)  # to ensure opening the image starts from the front
        plt.close()
        return output
