from typing import *
from math import sqrt
import numpy as np

from ._types import eq_len_axeslists


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

# Returns rank of matrix
def matrix_rank(matrix: List['Point']) -> int:
	m = [row[:] for row in matrix]
	num_rows = len(m)
	num_cols = len(m[0])
	rank = 0
	
	for col in range(num_cols):
		pivot_row = None
		for row in range(rank, num_rows):
			if m[row][col] != 0:
				pivot_row = row
				break
		if pivot_row is None:
			continue
		m[rank], m[pivot_row] = m[pivot_row], m[rank]
		pivot = m[rank][col]
		m[rank] = [val / pivot for val in m[rank]]
		
		for row in range(num_rows):
			if row != rank and m[row][col] != 0:
				factor = m[row][col]
				m[row] = [val - factor * lead for val, lead in zip(m[row], m[rank])]
		rank += 1
	
	return rank