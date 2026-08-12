from __future__ import annotations
from lsst.pex.config.configurableActions import ConfigurableAction

__all__ = (
    "FloatImagePlane",
    "ColorImage",
    "RGBImage",
    "LABImage",
    "LocalContrastFunction",
    "ScaleLumFunction",
    "ScaleColorFunction",
    "RemapBoundsFunction",
    "BracketingFunction",
    "GamutRemappingFunction",
    "WhitePoint",
)

import numpy as np
from typing import TypeAlias, Callable, Literal

FloatImagePlane: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.floating]]
ColorImage: TypeAlias = np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.floating]]
RGBImage: TypeAlias = ColorImage
LABImage: TypeAlias = ColorImage
WhitePoint: TypeAlias = tuple[float, float]

# Callable aliases
LocalContrastFunction: TypeAlias = Callable[[FloatImagePlane], FloatImagePlane]
ScaleLumFunction: TypeAlias = Callable[[FloatImagePlane, WhitePoint], FloatImagePlane]
ScaleColorFunction: TypeAlias = Callable[
    [FloatImagePlane, FloatImagePlane, FloatImagePlane, FloatImagePlane],
    tuple[FloatImagePlane, FloatImagePlane],
]
RemapBoundsFunction: TypeAlias = Callable[[RGBImage], RGBImage]
BracketingFunction: TypeAlias = Callable[[FloatImagePlane], FloatImagePlane]
GamutRemappingFunction: TypeAlias = Callable[[LABImage, WhitePoint, WhitePoint], RGBImage]


# Configurable Action Protocols, should correspond to preceding callable aliases
class LocalContrastProtocol(ConfigurableAction):
    def __call__(self, intensities: FloatImagePlane) -> FloatImagePlane:
        raise NotImplementedError("Base protocols should not be used directly")


class ScaleLumProtocol(ConfigurableAction):
    def __call__(self, intensities: FloatImagePlane, white_point: WhitePoint) -> FloatImagePlane:
        raise NotImplementedError("Base protocols should not be used directly")


class ScaleColorProtocol(ConfigurableAction):
    def __call__(
        self,
        old_lum: FloatImagePlane,
        new_lum: FloatImagePlane,
        a: FloatImagePlane,
        b: FloatImagePlane,
    ) -> tuple[FloatImagePlane, FloatImagePlane]:
        raise NotImplementedError("Base protocols should not be used directly")


class RemapBoundsProtocol(ConfigurableAction):
    def __call__(self, image: RGBImage) -> RGBImage:
        raise NotImplementedError("Base protocols should not be used directly")


class BracketingProtocol(ConfigurableAction):
    def __call__(self, intensities: FloatImagePlane) -> FloatImagePlane:
        raise NotImplementedError("Base protocols should not be used directly")


class GamutRemappingProtocol(ConfigurableAction):
    def __call__(
        self, Lab: LABImage, input_whitepoint: WhitePoint, output_whitepoint: WhitePoint
    ) -> RGBImage:
        raise NotImplementedError("Base protocols should not be used directly")
