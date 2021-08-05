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
        doc=("Size of cutout to extract for image from science, template, and difference exposures."
             " TODO: should we have an option to use the source footprints as the cutout size instead?"),
        dtype=int,
        default=30
    )
    urlRoot = pexConfig.Field(
        doc="URL that the resulting images will be served to Zooniverse from.",
        dtype=str,
        default=None
    )
    asinhScale = pexConfig.Field(
        doc="Scale factor ('a') in AsinhStretch for template/science images.",
        dtype=float,
        default=0.1
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
        doc=("The full path to write the files to; images will go in `outputPath/images/`, "
             "while the manifest file will go in `outputPath/`"),
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
        self.generate_images(data, butler)

        manifest = self.make_manifest(data)
        manifest.to_csv(os.path.join(self.config.outputPath, "manifest.csv"), index=False)

    def select_sources(self):
        """Select sources from apdb to make cutouts of.

        This method is entirely TODO until we have a way to make where queries
        on apdb.
        """
        return None

    def make_manifest(self, data):
        """Return a Zooniverse manifest attaching image URLs to source ids.
        """
        output = pd.DataFrame([])
        output['external_id'] = data['diaSourceId']

        def make_url(source):
            """Return a URL for this source."""
            return f"{self.config.urlRoot.rstrip('/')}/images/{source['diaSourceId']}.png"

        output['location:1'] = data.apply(make_url, axis=1)
        output['metadata:diaSourceId'] = data['diaSourceId']
        return output

    def generate_images(self, data, butler, collections):
        """Make the 3-part cutout images for each requested source.

        Creates a `images/` subdirectory in `self.config.outputPath` if one
        does not already exist.

        Parameters
        ----------
        data : `pandas.DataFrame`
            The DiaSources to extract cutouts for.
        butler : `lsst.daf.butler.Butler`
            The butler connection to use to load the data.
        collections : `str` or `list`
           Gen3 collection or collections from which to load the exposures.
        """
        @functools.lru_cache(maxsize=16)
        def get_exposures(instrument, detector, visit, collections):
            """Return science, template, difference exposures, and use a small
            cache so we don't have to re-read files as often.

            NOTE: If we redo this all to work with BPS or other parallelized
            systems, or get good butler-side cacheing, we should remove the
            lru_cache above.
            """
            dataId = {'detector': detector, 'visit': visit, 'instrument': instrument}
            return (butler.get('calexp', dataId, collections=collections),
                    # goodSeeing vs. deep vs. fakes_ needs to be configurable...
                    butler.get('fakes_goodSeeingDiff_warpedExp', dataId, collections=collections),
                    butler.get('fakes_goodSeeingDiff_differenceExp', dataId, collections=collections))

        # subdirectory for the images
        image_path = os.path.join(self.config.outputPath, "images")
        pathlib.Path(image_path).mkdir(exist_ok=True)

        for index, source in data.iterrows():
            center = lsst.geom.SpherePoint(source['ra'], source['decl'], lsst.geom.degrees)
            science, template, difference = get_exposures(source['instrument'], int(source['ccd']), int(source['visit']), collections)
            image = self.generate_image(science, template, difference, center)
            filename = os.path.join(image_path, f"{int(source['diaSourceId'])}.png")
            with open(filename, "wb") as outfile:
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
             Matched science - template exposure to include in the cutout.
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
             Cutout science - template exposure to include in the image.

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

        def do_one(ax, data, name):
            """Plot a normalized image on an axis."""
            if name == 'Difference':
                norm = aviz.ImageNormalize(data, interval=aviz.ZScaleInterval(),
                                           stretch=aviz.LinearStretch())
            else:
                norm = aviz.ImageNormalize(data, interval=aviz.MinMaxInterval(),
                                           stretch=aviz.AsinhStretch(a=self.config.asinhScale))
            ax.imshow(data, cmap=cm.bone, interpolation="none", norm=norm)
            ax.axis('off')
            ax.set_title(name)

        f, (ax1, ax2, ax3) = plt.subplots(1, 3)
        do_one(ax1, template.image.array, "Template")
        do_one(ax2, science.image.array, "Science")
        do_one(ax3, difference.image.array, "Difference")
        plt.tight_layout()

        output = io.BytesIO()
        plt.savefig(output, bbox_inches="tight", format="png")
        output.seek(0)  # to ensure opening the image starts from the front
        plt.close()
        return output
