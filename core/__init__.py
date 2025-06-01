from .primitives import *
from .shapes2d import *
from .shapes3d import *
from ._types import *

import numpy as np

max_dimension_for_names = 10

class nArgsPoint(Point):
    def __init__(self, *args, **kwargs):
        if isinstance(args[0], (tuple, list, np.ndarray)):
            if len(args[0]) != self.n:
                raise ValueError(f"Expected {self.n} coordinates, got {len(args)}")
        else:
            if len(args) != self.n:
                raise ValueError(f"Expected {self.n} coordinates, got {len(args)}")
        super().__init__(*args, **kwargs)

for N in range(1, max_dimension_for_names):
    globals()[f'Point{N}D'] = type(f'Point{N}D', (nArgsPoint,), {'n': N})

    globals()[f'Line{N}D'] = Line
    globals()[f'Ray{N}D'] = Ray
    globals()[f'Vector{N}D'] = Vector
    globals()[f'Segment{N}D'] = Segment

    globals()[f'Space{N}D'] = Space
    globals()[f'AffineSpace{N}D'] = AffineSpace
    globals()[f'ASpace{N}D'] = ASpace