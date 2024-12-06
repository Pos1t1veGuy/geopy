from .primitives import *
from .shapes2d import *
from .shapes3d import *
from .scene import *

max_dimension_for_names = 100

class nArgsObject:
	def __init__(self, *args):
		if len(args) != self.n:
			raise ValueError(f"Expected {self.n} coordinates, got {len(args)}")

for N in range(1, max_dimension_for_names):
    globals()[f'Point{N}D'] = type(f'Point{N}D', (nArgsObject,), {'n': N})

    globals()[f'Line{N}D'] = Line
    globals()[f'Ray{N}D'] = Ray
    globals()[f'Vector{N}D'] = Vector
    globals()[f'Segment{N}D'] = Segment

    globals()[f'Space{N}D'] = Space
    globals()[f'AffineSpace{N}D'] = AffineSpace
    globals()[f'ASpace{N}D'] = ASpace