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
)

import numpy as np
from typing import TypeAlias, Callable, Literal

FloatImagePlane: TypeAlias = np.ndarray[tuple[int, int], np.dtype[np.floating]]
ColorImage: TypeAlias = np.ndarray[tuple[int, int, Literal[3]], np.dtype[np.floating]]
RGBImage: TypeAlias = ColorImage
LABImage: TypeAlias = ColorImage
LocalContrastFunction: TypeAlias = Callable[[FloatImagePlane], FloatImagePlane]
ScaleLumFunction: TypeAlias = Callable[[FloatImagePlane], FloatImagePlane]
ScaleColorFunction: TypeAlias = Callable[
    [FloatImagePlane, FloatImagePlane, FloatImagePlane, FloatImagePlane],
    tuple[FloatImagePlane, FloatImagePlane],
]
RemapBoundsFunction: TypeAlias = Callable[[RGBImage], RGBImage]
BracketingFunction: TypeAlias = Callable[[FloatImagePlane], FloatImagePlane]
GamutRemappingFunction: TypeAlias = Callable[[LABImage, tuple[float, float]], RGBImage]
