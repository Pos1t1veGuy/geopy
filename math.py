from typing import *
from math import sqrt
import numpy as np

from .types import eq_len_axeslists


EPSILON = 9e-10

class QuadraticEq:
	def __init__(self, a: float = 1, b: float = 0, c: float = 0):
		self.a = a
		self.b = b
		self.c = c
		self.solution = None # it will contains solution if you use once method solve()
		self.discriminant = self.D

	def solve(self) -> List[float]:
		self.solution = list({
			( -self.b + sqrt(self.D) ) / ( 2 * self.a ),
			( -self.b - sqrt(self.D) ) / ( 2 * self.a ),
		}) if self.D >= 0 else []
		return self.solution

	@property
	def math_view(self) -> str:
		return f'{self.a}x^2 + {self.b}x + {self.c} = 0'
	
	@property
	def D(self) -> float:
		return self.b**2 - 4 * self.a * self.c

	def __str__(self):
		return self.math_view
	def __repr__(self):
		return f'{self.__class__.__name__}(a={self.a}, b={self.b}, c={self.c})'


# Сonverts any vectors to orthogonals
def gram_schmidt(vectors: List['Vector']) -> List['Vector']:
	from .primitives import Vector

	vectors_as_lists = eq_len_axeslists(*[vec.to_zero.normalize.pos2.axes for vec in vectors])
	ortho_vectors = []
	for vec in vectors_as_lists:
		v = np.array([float(i) for i in vec])
		w = v - sum(np.dot(v, u) * u for u in ortho_vectors)
		if not np.allclose(w, 0.0, atol=EPSILON):
			ortho_vectors.append(w / np.linalg.norm(w))

	return [Vector[v] for v in ortho_vectors]