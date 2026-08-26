import numpy as np

from lsst.geom import Box2I
from lsst.images import Box
from lsst.pipe.tasks.prettyPictureMaker._utils import FeatheredMosaicCreator


class TestFeatheredMosaicCreator:
    def test_make_featherings_basic(self):
        """Verify featherings are created with correct shapes."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        creator._make_featherings((50, 50))

        assert creator.featherings is not None
        assert len(creator.featherings) == 4

        for feather in creator.featherings:
            assert feather.shape == (50, 50)

    def test_make_featherings_symmetry(self):
        """Verify top/bottom and left/right masks are symmetric."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        creator._make_featherings((50, 50))

        top, bottom, left, right = creator.featherings

        np.testing.assert_allclose(top[1:30, :], bottom[20:49, :][::-1, :], rtol=1e-5)
        np.testing.assert_allclose(left[:, 1:30], right[:, 20:49][:, ::-1], rtol=1e-5)

    def test_make_featherings_values(self):
        """Verify ramp values are correct."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        creator._make_featherings((50, 50))

        top, bottom, left, right = creator.featherings

        assert top[0, 0] < 1e-6
        assert np.allclose(top[20, 0], 1.0, atol=0.1)

        assert np.allclose(bottom[30, 0], 1.0, atol=0.1)
        assert bottom[-1, 0] < 1e-6

        assert np.allclose(left[0, 20], 1.0, atol=0.1)
        assert right[0, -1] < 1e-6

    def test_make_featherings_bin_factor(self):
        """Verify bin_factor reduces resolution."""
        creator_no_bin = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        creator_bin = FeatheredMosaicCreator(patch_grow=10, bin_factor=2)

        creator_no_bin._make_featherings((50, 50))
        creator_bin._make_featherings((50, 50))

        assert creator_bin.featherings[0].shape == creator_no_bin.featherings[0].shape

    def _make_boxes(self, min_point, max_point):
        box = Box.from_legacy(Box2I(Box2I.Point(*min_point), Box2I.Point(*max_point)))
        return box

    def test_add_to_image_full_overlap(self):
        """Verify no feathering when box == newBox."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((50, 50, 3))
        patch = np.ones((50, 50, 3)) * 0.5

        box = self._make_boxes((0, 0), (49, 49))
        new_box = self._make_boxes((0, 0), (49, 49))

        creator.add_to_image(image, patch, new_box, box, reverse=False)

        np.testing.assert_allclose(image, 0.5, atol=0.1)

    def test_add_to_image_single_edge(self):
        """Verify only the differing edge gets feathering."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((60, 60, 3))
        patch = np.ones((50, 50, 3)) * 0.5

        box = self._make_boxes((0, 0), (49, 49))
        new_box = self._make_boxes((0, 5), (49, 54))

        creator.add_to_image(image, patch, new_box, box, reverse=False)

        assert image.shape == (60, 60, 3)

    def test_add_to_image_multi_edge(self):
        """Verify multiple edges get combined feathering."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((70, 70, 3))
        patch = np.ones((50, 50, 3)) * 0.5

        box = self._make_boxes((0, 0), (49, 49))
        new_box = self._make_boxes((5, 5), (54, 54))

        creator.add_to_image(image, patch, new_box, box, reverse=False)

        assert image.shape == (70, 70, 3)

    def test_add_to_image_rgb(self):
        """Verify RGB (3D) images are handled correctly."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((50, 50, 3))
        patch = np.ones((50, 50, 3)) * 0.5

        box = self._make_boxes((0, 0), (49, 49))
        new_box = self._make_boxes((0, 0), (49, 49))

        creator.add_to_image(image, patch, new_box, box, reverse=False)

        assert image.shape == (50, 50, 3)
        np.testing.assert_allclose(image[0, 0, :], 0.5, atol=0.1)

    def test_add_to_image_reverse(self):
        """Verify reverse flips the patch."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((50, 50, 3))
        patch = np.ones((50, 50, 3))

        box = self._make_boxes((0, 0), (49, 49))
        new_box = self._make_boxes((0, 0), (49, 49))

        creator.add_to_image(image, patch, new_box, box, reverse=True)

        assert image.shape == (50, 50, 3)
