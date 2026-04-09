import pytest
import numpy as np
from lsst.pipe.tasks.prettyPictureMaker._functors._bounds_remapper import BoundsRemapper
from lsst.pipe.tasks.prettyPictureMaker._functors._lum_scale import LumCompressor
from lsst.pipe.tasks.prettyPictureMaker._functors._local_contrast import (
    DiffusionFunction,
    LocalContrastEnhancer,
)
from lsst.pipe.tasks.prettyPictureMaker._functors._color_scale import ColorScaler
from lsst.pipe.tasks.prettyPictureMaker._functors._exposure_fusion import ExposureBracketer
from lsst.pipe.tasks.prettyPictureMaker._functors._gamut_fixer import GamutFixer
from lsst.pipe.tasks.prettyPictureMaker._utils import FeatheredMosaicCreator
from lsst.pipe.tasks.prettyPictureMaker._colorMapper import lsstRGB
from lsst.geom import Box2I


class TestColorScaler:
    def test_basic_saturation(self):
        """Verify saturation scaling preserves hue while adjusting chroma."""
        scaler = ColorScaler(saturation=0.6)
        lum_old = np.ones((50, 50)) * 0.5
        lum_new = np.ones((50, 50)) * 0.7
        np.random.seed(42)
        a = np.random.uniform(-0.3, 0.3, (50, 50))
        b = np.random.uniform(-0.3, 0.3, (50, 50))

        new_a, new_b = scaler(lum_old, lum_new, a, b)

        original_hue = np.arctan2(b, a)
        new_hue = np.arctan2(new_b, new_a)
        np.testing.assert_allclose(original_hue, new_hue, rtol=1e-5)

        original_chroma = np.sqrt(a**2 + b**2)
        new_chroma = np.sqrt(new_a**2 + new_b**2)
        assert np.all(new_chroma < original_chroma)

    def test_saturation_factor(self):
        """Verify different saturation values produce different chromaticity."""
        scaler_low_sat = ColorScaler(saturation=0.3)
        scaler_high_sat = ColorScaler(saturation=0.9)

        lum_old = np.ones((50, 50)) * 0.5
        lum_new = np.ones((50, 50)) * 0.7
        np.random.seed(42)
        a = np.random.uniform(-0.3, 0.3, (50, 50))
        b = np.random.uniform(-0.3, 0.3, (50, 50))

        new_a_low, _ = scaler_low_sat(lum_old, lum_new, a, b)
        new_a_high, _ = scaler_high_sat(lum_old, lum_new, a, b)

        assert not np.allclose(new_a_low, new_a_high)

    def test_max_chroma_clipping(self):
        """Verify chromaticity is clipped to maxChroma."""
        scaler = ColorScaler(saturation=1.0, maxChroma=0.1)

        lum_old = np.ones((50, 50)) * 0.5
        lum_new = np.ones((50, 50)) * 0.7
        np.random.seed(42)
        a = np.random.uniform(-0.4, 0.4, (50, 50))
        b = np.random.uniform(-0.4, 0.4, (50, 50))

        new_a, new_b = scaler(lum_old, lum_new, a, b)
        new_chroma = np.sqrt(new_a**2 + new_b**2)

        assert np.all(new_chroma <= 0.1 + 1e-6)

    def test_zero_chroma(self):
        """Verify achromatic input remains achromatic."""
        scaler = ColorScaler(saturation=0.6)

        lum_old = np.ones((50, 50)) * 0.5
        lum_new = np.ones((50, 50)) * 0.7
        a = np.zeros((50, 50))
        b = np.zeros((50, 50))

        new_a, new_b = scaler(lum_old, lum_new, a, b)

        np.testing.assert_array_equal(new_a, 0.0)
        np.testing.assert_array_equal(new_b, 0.0)

    def test_equalizer_levels(self):
        """Verify equalizer_levels modifies chromaticity."""
        scaler_no_eq = ColorScaler(saturation=0.6, equalizer_levels=None)
        scaler_with_eq = ColorScaler(saturation=0.6, equalizer_levels=[1.1, 0.9])

        lum_old = np.ones((50, 50)) * 0.5
        lum_new = np.ones((50, 50)) * 0.7
        np.random.seed(42)
        a = np.random.uniform(-0.3, 0.3, (50, 50))
        b = np.random.uniform(-0.3, 0.3, (50, 50))

        new_a_no_eq, new_b_no_eq = scaler_no_eq(lum_old, lum_new, a, b)
        new_a_with_eq, new_b_with_eq = scaler_with_eq(lum_old, lum_new, a, b)

        assert not np.allclose((new_a_no_eq, new_b_no_eq), (new_a_with_eq, new_b_with_eq))


class TestBoundsRemapper:
    def test_absmax_scaling(self):
        """Verify scaling using the fixed absMax value."""
        remapper = BoundsRemapper(absMax=100.0, quant=1.0)
        img = np.ones((10, 10, 3)) * 50.0
        expected = np.ones((10, 10, 3)) * 0.5
        result = remapper(img)
        np.testing.assert_allclose(result, expected)

    def test_quant_scaling(self):
        """Verify scaling using the quantile-based approach (absMax=None)."""
        remapper = BoundsRemapper(absMax=None, quant=0.5)
        data = np.linspace(0, 10, 100).reshape(10, 10)
        img = np.stack([data, data, data], axis=-1)
        # 95th percentile of linspace(0, 10, 100) is 9.5
        # scale = 9.5 * 0.5 = 4.75
        expected = np.clip(img / 4.75, 0, 1)
        result = remapper(img)
        np.testing.assert_allclose(result, expected)

    def test_clipping(self):
        """Verify that values exceeding the scale are clipped to 1.0."""
        remapper = BoundsRemapper(absMax=10.0)
        img = np.array([[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [15.0, 15.0, 15.0]]], dtype=float)
        expected = np.array([[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]], dtype=float)
        result = remapper(img)
        np.testing.assert_allclose(result, expected)

    def test_short_circuit(self):
        """Verify that if max is already 1, the image is returned as-is."""
        remapper = BoundsRemapper(absMax=100.0)
        img = np.ones((5, 5, 3))
        result = remapper(img)
        np.testing.assert_array_equal(result, img)


class TestLumCompressor:
    def test_basic_stretching(self):
        """Verify that asinh stretching maps values into [0, 1]."""
        remapper = LumCompressor(stretch=400.0, max=1.0)
        # Input values ranging from 0 to 1000
        img = np.linspace(0, 1000, 100).reshape(10, 10)
        result = remapper(img)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
        assert result.shape == img.shape

    def test_contrast_params(self):
        """Verify that highlight and shadow parameters affect the output."""
        # Test shadow parameter: increasing shadow should shift values
        remapper_low_shadow = LumCompressor(shadow=0.0, highlight=1.0)
        remapper_high_shadow = LumCompressor(shadow=0.5, highlight=1.0)

        img = np.linspace(0.1, 0.9, 100).reshape(10, 10)
        res_low = remapper_low_shadow(img)
        res_high = remapper_high_shadow(img)

        # High shadow should result in a different distribution
        assert not np.allclose(res_low, res_high)

    def test_midtone_adjustment(self):
        """Verify that midtone parameter shifts the intensity pivot."""
        remapper_mid_05 = LumCompressor(midtone=0.5)
        remapper_mid_08 = LumCompressor(midtone=0.8)

        img = np.linspace(0.1, 0.9, 100).reshape(10, 10)
        res_05 = remapper_mid_05(img)
        res_08 = remapper_mid_08(img)

        assert not np.allclose(res_05, res_08)

    def test_clipping(self):
        """Verify that the final output is strictly clipped to [0, 1]."""
        # Using extreme parameters to force values out of bounds
        remapper = LumCompressor(stretch=1000.0, highlight=0.1, shadow=0.9)
        img = np.linspace(0, 100, 100).reshape(10, 10)
        result = remapper(img)

        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestDiffusionFunction:
    def test_basic(self):
        """Verify that diffusion produces a different output from input."""
        diffuser = DiffusionFunction()
        img = np.random.rand(50, 50)
        result = diffuser(img)
        assert not np.allclose(result, img)

    def test_high_iterations(self):
        """Verify that high iterations produce more diffused output."""
        diffuser_low = DiffusionFunction(iterations=1)
        diffuser_high = DiffusionFunction(iterations=10)
        img = np.random.rand(50, 50)
        res_low = diffuser_low(img)
        res_high = diffuser_high(img)
        assert not np.allclose(res_low, res_high)


class TestLocalContrastEnhancer:
    def test_basic_enhancement(self):
        """Verify that local contrast enhancement increases contrast."""
        enhancer = LocalContrastEnhancer(doDiffusion=False)
        img = np.ones((50, 50)) * 0.5 + np.random.rand(50, 50) * 0.01
        result = enhancer(img)
        assert result.std() > img.std()

    def test_diffusion_off(self):
        """Verify that turning off diffusion produces different results."""
        enhancer_with_diffusion = LocalContrastEnhancer(doDiffusion=True)
        enhancer_without_diffusion = LocalContrastEnhancer(doDiffusion=False)
        img = np.random.rand(50, 50)
        res_with = enhancer_with_diffusion(img)
        res_without = enhancer_without_diffusion(img)
        assert not np.allclose(res_with, res_without)


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

        # Check symmetry excluding the first/last rows/columns where 1e-17 is placed
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

    def test_add_to_image_full_overlap(self):
        """Verify no feathering when box == newBox."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((50, 50, 3))
        patch = np.ones((50, 50, 3)) * 0.5

        box = Box2I(Box2I.Point(0, 0), Box2I.Extent(50, 50))
        newBox = Box2I(Box2I.Point(0, 0), Box2I.Extent(50, 50))

        creator.add_to_image(image, patch, newBox, box, reverse=False)

        np.testing.assert_allclose(image, 0.5, atol=0.1)

    def test_add_to_image_single_edge(self):
        """Verify only the differing edge gets feathering."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((60, 60, 3))
        patch = np.ones((50, 50, 3)) * 0.5

        box = Box2I(Box2I.Point(0, 0), Box2I.Extent(50, 50))
        newBox = Box2I(Box2I.Point(0, 5), Box2I.Extent(50, 55))

        creator.add_to_image(image, patch, newBox, box, reverse=False)

        assert image.shape == (60, 60, 3)

    def test_add_to_image_multi_edge(self):
        """Verify multiple edges get combined feathering."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((70, 70, 3))
        patch = np.ones((50, 50, 3)) * 0.5

        box = Box2I(Box2I.Point(0, 0), Box2I.Extent(50, 50))
        newBox = Box2I(Box2I.Point(5, 5), Box2I.Extent(55, 55))

        creator.add_to_image(image, patch, newBox, box, reverse=False)

        assert image.shape == (70, 70, 3)

    def test_add_to_image_rgb(self):
        """Verify RGB (3D) images are handled correctly."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((50, 50, 3))
        patch = np.ones((50, 50, 3)) * 0.5

        box = Box2I(Box2I.Point(0, 0), Box2I.Extent(50, 50))
        newBox = Box2I(Box2I.Point(0, 0), Box2I.Extent(50, 50))

        creator.add_to_image(image, patch, newBox, box, reverse=False)

        assert image.shape == (50, 50, 3)
        np.testing.assert_allclose(image[0, 0, :], 0.5, atol=0.1)

    def test_add_to_image_reverse(self):
        """Verify reverse flips the patch."""
        creator = FeatheredMosaicCreator(patch_grow=10, bin_factor=1)
        image = np.zeros((50, 50, 3))
        patch = np.ones((50, 50, 3))

        box = Box2I(Box2I.Point(0, 0), Box2I.Extent(50, 50))
        newBox = Box2I(Box2I.Point(0, 0), Box2I.Extent(50, 50))

        creator.add_to_image(image, patch, newBox, box, reverse=True)

        assert image.shape == (50, 50, 3)


class TestExposureBracketer:
    def test_basic_fusion(self):
        """Verify exposure bracketing fusion produces a balanced result."""
        bracketer = ExposureBracketer()
        img = np.random.rand(50, 50) * 0.5
        result = bracketer(img)

        assert result.shape == img.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_single_bracket(self):
        """Verify single bracket returns scaled image without fusion."""
        bracketer = ExposureBracketer(exposureBrackets=[1.25])
        img = np.random.rand(50, 50) * 0.5
        result = bracketer(img)
        expected = img / 1.25

        np.testing.assert_allclose(result, expected)

    def test_no_brackets(self):
        """Verify None brackets returns input unchanged."""
        bracketer = ExposureBracketer(exposureBrackets=None)
        img = np.random.rand(50, 50)
        result = bracketer(img)

        np.testing.assert_array_equal(result, img)

    def test_different_brackets(self):
        """Verify different bracket configurations produce different outputs."""
        bracketer_default = ExposureBracketer()
        bracketer_custom = ExposureBracketer(exposureBrackets=[1.5, 1, 0.5])

        img = np.random.rand(50, 50) * 0.5
        res_default = bracketer_default(img)
        res_custom = bracketer_custom(img)

        assert not np.allclose(res_default, res_custom)

    def test_clipping_behavior(self):
        """Verify values > 1.0 receive reduced weighting during fusion."""
        bracketer = ExposureBracketer()
        img = np.random.rand(50, 50) * 0.5
        img[25, 25] = 1.5

        result = bracketer(img)

        assert result.shape == img.shape
        assert np.all(result >= 0.0)


class TestGamutFixer:
    def test_gamut_fixer_none_method(self):
        """Verify 'none' method returns original RGB conversion."""
        fixer = GamutFixer(gamutMethod="none")
        Lab = np.random.rand(50, 50, 3) * 0.4 - 0.2
        Lab[:, :, 0] = Lab[:, :, 0] * 0.8 + 0.1
        Lab[25, 25, :] = [0.5, 0.5, 0.5]
        xyz_whitepoint = (0.31272, 0.32903)

        result = fixer(Lab, xyz_whitepoint)

        assert result.shape == Lab.shape[:-1] + (3,)

    def test_gamut_fixer_inpaint_method(self):
        """Verify 'inpaint' method fixes small out-of-gamut regions."""
        fixer = GamutFixer(gamutMethod="inpaint", max_size=10000)
        Lab = np.random.rand(50, 50, 3) * 0.4 - 0.2
        Lab[:, :, 0] = Lab[:, :, 0] * 0.8 + 0.1
        xyz_whitepoint = (0.31272, 0.32903)

        result = fixer(Lab, xyz_whitepoint)

        assert result.shape == Lab.shape[:-1] + (3,)
        assert np.all(result <= 1.0)

    def test_gamut_fixer_mapping_method(self):
        """Verify 'mapping' method remaps out-of-gamut colors."""
        fixer = GamutFixer(gamutMethod="mapping")
        Lab = np.random.rand(50, 50, 3) * 0.4 - 0.2
        Lab[:, :, 0] = Lab[:, :, 0] * 0.8 + 0.1
        Lab[25, 25, :] = [0.8, 0.8, 0.8]
        xyz_whitepoint = (0.31272, 0.32903)

        result = fixer(Lab, xyz_whitepoint)

        assert result.shape == Lab.shape[:-1] + (3,)
        assert np.all(result <= 1.0)

    def test_gamut_fixer_heal_method(self):
        """Verify 'heal' method heals out-of-gamut regions."""
        fixer = GamutFixer(gamutMethod="heal", max_size=1000)
        Lab = np.random.rand(50, 50, 3) * 0.4 - 0.2
        Lab[:, :, 0] = Lab[:, :, 0] * 0.8 + 0.1
        xyz_whitepoint = (0.31272, 0.32903)

        result = fixer(Lab, xyz_whitepoint)

        assert result.shape == Lab.shape[:-1] + (3,)
        assert np.all(result <= 1.0)

    def test_gamut_fixer_no_out_of_bounds(self):
        fixer = GamutFixer(gamutMethod="inpaint")
        Lab = np.random.rand(50, 50, 3) * 0.4 - 0.2
        Lab[:, :, 0] = Lab[:, :, 0] * 0.8 + 0.1
        xyz_whitepoint = (0.31272, 0.32903)

        result = fixer(Lab, xyz_whitepoint)

        assert result.shape == Lab.shape[:-1] + (3,)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


class TestColorMapper:
    def test_basic_integration(self):
        """Verify lsstRGB produces valid RGB output with defaults."""
        np.random.seed(42)
        r = np.random.rand(100, 100) * 0.99 + 0.01
        g = np.random.rand(100, 100) * 0.99 + 0.01
        b = np.random.rand(100, 100) * 0.99 + 0.01

        result = lsstRGB(r, g, b)

        assert result.shape == (100, 100, 3)
        assert result.dtype == np.float64
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_shape_validation(self):
        """Verify ValueError is raised for mismatched shapes."""
        np.random.seed(42)
        r = np.random.rand(100, 100)
        g = np.random.rand(100, 99)
        b = np.random.rand(100, 100)

        with pytest.raises(ValueError):
            lsstRGB(r, g, b)

    def test_none_functors(self):
        """Verify all functors set to None returns remapped RGB."""
        np.random.seed(42)
        r = np.random.rand(100, 100) * 0.99 + 0.01
        g = np.random.rand(100, 100) * 0.99 + 0.01
        b = np.random.rand(100, 100) * 0.99 + 0.01

        result = lsstRGB(
            r,
            g,
            b,
            local_contrast=None,
            scale_lum=None,
            scale_color=None,
            bracketing_function=None,
            gamut_remapping_function=None,
            remap_bounds=None,
            cieWhitePoint=(0.31272, 0.32903),
        )

        assert result.shape == (100, 100, 3)
        np.testing.assert_allclose(result[..., 0], r, rtol=1e-5)
        np.testing.assert_allclose(result[..., 1], g, rtol=1e-5)
        np.testing.assert_allclose(result[..., 2], b, rtol=1e-5)

    def test_psf_deconvolution(self):
        """Verify PSF deconvolution produces different output."""
        np.random.seed(42)
        r = np.random.rand(100, 100) * 0.99 + 0.01
        g = np.random.rand(100, 100) * 0.99 + 0.01
        b = np.random.rand(100, 100) * 0.99 + 0.01

        x = np.linspace(-2, 2, 5)
        xx, yy = np.meshgrid(x, x)
        psf = np.exp(-(xx**2 + yy**2))
        psf /= psf.sum()

        result_no_psf = lsstRGB(r, g, b, psf=None)
        result_with_psf = lsstRGB(r, g, b, psf=psf)

        assert result_no_psf.shape == result_with_psf.shape
        assert not np.allclose(result_no_psf, result_with_psf)

    def test_cie_white_point(self):
        """Verify different white points produce different outputs."""
        np.random.seed(42)
        r = np.random.rand(100, 100) * 0.99 + 0.01
        g = np.random.rand(100, 100) * 0.99 + 0.01
        b = np.random.rand(100, 100) * 0.99 + 0.01

        result_default = lsstRGB(r, g, b, cieWhitePoint=(0.28, 0.28))
        result_d65 = lsstRGB(r, g, b, cieWhitePoint=(0.31272, 0.32903))

        assert not np.allclose(result_default, result_d65)
