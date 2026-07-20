from __future__ import annotations

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
LocalContrastFunction: TypeAlias = Callable[[FloatImagePlane], FloatImagePlane]
ScaleLumFunction: TypeAlias = Callable[[FloatImagePlane, WhitePoint], FloatImagePlane]
ScaleColorFunction: TypeAlias = Callable[
    [FloatImagePlane, FloatImagePlane, FloatImagePlane, FloatImagePlane],
    tuple[FloatImagePlane, FloatImagePlane],
]
RemapBoundsFunction: type = Callable[[RGBImage], RGBImage]
BracketingFunction: type = Callable[[FloatImagePlane], FloatImagePlane]
GamutRemappingFunction: type = Callable[[LABImage, WhitePoint, WhitePoint], RGBImage]
