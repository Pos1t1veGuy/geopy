from math import sqrt
from typing import *

EPSILON = 9e-10

class QuadraticEq:
	def __init__(self, a: float = 1, b: float = 0, c: float = 0):
		self.a = a
		self.b = b
		self.c = c
		self.discriminant = self.D

	def solve(self):
		return list({
			( -self.b + sqrt(self.D) ) / ( 2 * self.a ),
			( -self.b - sqrt(self.D) ) / ( 2 * self.a ),
		}) if self.D >= 0 else []

	@property
	def math_view(self):
		return f'{self.a}x^2 + {self.b}x + {self.c} = 0'
	
	@property
	def D(self):
		return self.b**2 - 4 * self.a * self.c