from typing import *
from math import sqrt, acos, cos, sin, degrees, tan, radians
from string import ascii_lowercase

import traceback
import random as r
import numpy as np
from itertools import combinations
import pdb

from ._types import *
from .math import *
from .exceptions import *
from .base_shapes import *

# Here is geometry primitives: Point, Line, Ray, Segment, Vector and Angle classes

class Primitive:
	...

letters = 'xyz' + ''.join(list(reversed(ascii_lowercase[:-3]))) # X, Y, Z, w, v, u, ...

class PointMeta(type): # to create point by "Point[]" like Point[1,2,3]
	def __getitem__(cls, pos):
		if isinstance(pos, (tuple, list, np.ndarray)):
			return cls(*pos)
		else:
			return cls(pos)
class Point(Primitive, metaclass=PointMeta):
	def __init__(self, *args, name: str = 'Point', color: str = 'm', alpha: Union[int, float] = 1, size: Union[int, float] = 15):
		self.axes = AxesList([])
		self.color = color

		if len(args) == 1 and isinstance(args[0], (tuple, list, np.ndarray)):
			args = args[0]

		self.name = name
		self.alpha = alpha
		self.size = size

		for i, axis in enumerate(args):
			if isinstance(axis, (float, int, np.int64, Fraction)):
				self.axes.append(to_fraction(axis))
			else:
				raise ConstructError(f"Invalid initialization arguments for 'Point': {args}")

	def copy(self) -> 'Point':
		return self.__class__(*self.axes, name=self.name)

	def height_to(self, object: Primitive) -> 'Segment':
		if isinstance(object, Point):
			second_pos =  np.linalg.norm(np.array(self.axes) - np.array(object.axes))
		elif isinstance(object, Line):
			second_pos = self.project_to_line(object)
		elif isinstance(object, AffineSpace):
			second_pos = self.project_to_space(object)
		else:
			raise ConstructError(f"height_to not implemented for type {type(object)}")

		if self == second_pos:
			raise ConstructError(f'Point {self} is on line {object}, can not construct height with 0 length from point to same')
		return Segment(self, second_pos, name=height_name.format(self.name))

	# Returns a Point that makes with self Point a new Line that makes perpendicular to line
	def project_to_line(self, pr: 'Line') -> 'Point':
		p1 = np.array(pr.pos1.axes)
		axes, line_vector = eq_len_axeslists(self.axes, pr.vector.pos2.axes)

		return Point(
			*(p1 + np.array(np.dot(np.array(axes) - p1, line_vector) / np.dot(line_vector, line_vector)) * line_vector),
			name=projection_name.format(self.name)
		)
	# Returns a Point that makes with self Point a new Line that makes perpendicular to Space
	def project_to_space(self, space: 'AffineSpace') -> 'Point':
		normal = space.normal
		p, ln = eq_len_axeslists((self - space.origin).axes, normal.to_zero.pos2.axes)
		dot = float(np.dot(p, ln) / np.dot(ln, ln))
		if dot == 0:
			return self
		else:
			return self - (normal * dot).pos2

	def project_to(self, dimension: int) -> 'Point':
		return self.__class__(self.axes[:dimension], name=projection_name.format(self.name))

	@staticmethod
	def random(pos1: Union['Point', tuple, list], pos2: Union['Point', tuple, list], uniform: bool = True) -> 'Point':
		# pos1 and pos2 must be Point object, returns random point from rectangle, box etc. [pos1 x pos2]
		if isinstance(pos1, (tuple, list, np.ndarray)):
			pos1 = Point(pos1)
		if isinstance(pos2, (tuple, list, np.ndarray)):
			pos2 = Point(pos2)
		max_dimension = max(pos1.dimension, pos2.dimension)

		if uniform:
			return self.__class__([r.uniform(pos1.axes[i], pos2.axes[i]) for i in max_dimension], name=self.name)
		else:
			return self.__class__([r.randint(pos1.axes[i], pos2.axes[i]) for i in max_dimension], name=self.name)

	@property
	def to_vector(self) -> 'Vector':
		return Vector[self]

	@property
	def float_axes(self) -> List[float]:
		return AxesList([float(i) for i in self.axes])

	@property
	def dimension(self) -> int:
		return len(self.axes)

	@property
	def pos(self) -> List[Fraction]:
		return self.axes

	def __getattr__(self, i):
		global letters
		if i in letters:
			return self.axes[letters.index(i)]
		else:
			raise IndexError(f"Attribute '{i}' not found in axes or object attributes.")

	def __eq__(self, obj):
		if obj == 0:
			return all([ axis == 0 for axis in self.axes ])
		elif isinstance(obj, Point):
			max_dimension = max( len(self.axes), len(obj.axes) )
			return self.axes.as_list(length=max_dimension) == obj.axes.as_list(length=max_dimension)
		else:
			return list(self.axes) == obj

	def __add__(self, object: Union[int, float, Fraction]) -> 'Point':
		if isinstance(object, (int, float, Fraction)):
			return self.__class__([axis + object for axis in self.axes], name=self.name)
		elif isinstance(object, Point):
			return self.__class__([self.axes[i] + object.axes[i] for i in range(max(self.dimension, object.dimension))], name=self.name)
		elif isinstance(object, (tuple, list, np.ndarray)):
			object = AxesList(object)
			return self.__class__([self.axes[i] + object[i] for i in range(max(self.dimension, len(object)))], name=self.name)
		elif isinstance(object, Vector):
			return self.__class__([self.axes[i] + (object.pos2.axes[i] - object.pos1.axes[i]) for i in range(max(self.dimension, object.dimension))])
		else:
			raise ConstructError(f"Unsupported operand type(s) for +: '{self.__class__.__name__}' and '{type(object).__name__}'")

	def __radd__(self, object: Union[int, float, Fraction]) -> 'Point':
		return self.__add__(object)

	def __sub__(self, object: Union[int, float, Fraction]) -> 'Point':
		if isinstance(object, (int, float, Fraction)):
			return self.__class__([axis - object for axis in self.axes], name=self.name)
		elif isinstance(object, Point):
			return self.__class__([self.axes[i] - object.axes[i] for i in range(max(self.dimension, object.dimension))], name=self.name)
		elif isinstance(object, (tuple, list, np.ndarray)):
			object = AxesList(object)
			return self.__class__([self.axes[i] - object[i] for i in range(max(self.dimension, len(object)))], name=self.name)
		elif isinstance(object, Vector):
			return self.__class__([self.axes[i] - (object.pos2.axes[i] - object.pos1.axes[i]) for i in range(max(self.dimension, object.dimension))], name=self.name)
		else:
			raise ConstructError(f"Unsupported operand type(s) for -: '{self.__class__.__name__}' and '{type(object).__name__}'")

	def __mul__(self, object: Union[int, float, Fraction]) -> 'Point':
		if isinstance(object, (int, float, Fraction)):
			return self.__class__([axis * object for axis in self.axes], name=self.name)
		elif isinstance(object, Point):
			return self.__class__([self.axes[i] * object.axes[i] for i in range(max(self.dimension, object.dimension))], name=self.name)
		elif isinstance(object, (tuple, list, np.ndarray)):
			object = AxesList(object)
			return self.__class__([self.axes[i] * object[i] for i in range(max(self.dimension, len(object)))], name=self.name)
		else:
			raise ConstructError(f"Unsupported operand type(s) for *: '{self.__class__.__name__}' and '{type(object).__name__}'")

	def __truediv__(self, object: Union[int, float, Fraction]) -> 'Point':
		try:
			if isinstance(object, (int, float, Fraction)):
				return self.__class__([to_fraction(axis, object) for axis in self.axes], name=self.name)
			elif isinstance(object, Point):
				return self.__class__([to_fraction(self.axes[i], object.axes[i]) for i in range(max(self.dimension, object.dimension))], name=self.name)
			elif isinstance(object, (tuple, list, np.ndarray)):
				object = AxesList(object)
				return self.__class__([to_fraction(self.axes[i], object[i]) for i in range(max(self.dimension, len(object)))], name=self.name)
			else:
				raise ConstructError(f"Unsupported operand type(s) for /: '{self.__class__.__name__}' and '{type(object).__name__}'")
		except ZeroDivisionError:
			raise ZeroDivisionError(f'ZeroDivisionError: {self} / {object}')

	def __floordiv__(self, object: Union[int, float, Fraction]) -> 'Point':
		try:
			if isinstance(object, (int, float, Fraction)):
				return self.__class__([to_fraction(axis, object) for axis in self.axes], name=self.name)
			elif isinstance(object, Point):
				return self.__class__([to_fraction(self.axes[i], object.axes[i]) for i in range(max(self.dimension, object.dimension))], name=self.name)
			elif isinstance(object, (tuple, list, np.ndarray)):
				object = AxesList(object)
				return self.__class__([to_fraction(self.axes[i], object[i]) for i in range(max(self.dimension, len(object)))], name=self.name)
			else:
				raise ConstructError(f"Unsupported operand type(s) for //: '{self.__class__.__name__}' and '{type(object).__name__}'")
		except ZeroDivisionError:
			raise ZeroDivisionError(f'ZeroDivisionError: {self} // {object}')

	def __pow__(self, object: Union[int, float, Fraction]) -> 'Point':
		if isinstance(object, (int, float, Fraction)):
			return self.__class__([axis ** object for axis in self.axes], name=self.name)
		elif isinstance(object, Point):
			return self.__class__([self.axes[i] ** object.axes[i] for i in range(max(self.dimension, object.dimension))], name=self.name)
		elif isinstance(object, (tuple, list, np.ndarray)):
			object = AxesList(object)
			return self.__class__([self.axes[i] ** object[i] for i in range(max(self.dimension, len(object)))], name=self.name)
		else:
			raise ConstructError(f"Unsupported operand type(s) for **: '{self.__class__.__name__}' and '{type(object).__name__}'")

	def __contains__(self, object):
		if isinstance(object, (Primitive, Shape, AffineSpace)):
			return object.intersects(self)
		else:
			return object in list(self)

	def __len__(self):
		return len(self.axes)

	def __neg__(self):
		return self.__class__([-axis[i] for axis in self.axes], name=self.name)
	def __pos__(self):
		return self

	def __iter__(self):
		return iter(self.axes)

	def __getitem__(self, i):
		if isinstance(i, slice):
			return self.__class__(self.axes[i], name=self.name)
		else:
			return self.axes[i]

	def __str__(self):
		rounded_axes = str([ (round(float(axis), 4) if float(axis) >= 0 else round(float(axis), 3)) for axis in self.axes ])
		return self.__class__.__name__ + rounded_axes
	def __repr__(self):
		rounded_axes = str([ (round(float(axis), 4) if float(axis) >= 0 else round(float(axis), 3)) for axis in self.axes ])
		return f'{self.__class__.__name__}{self.dimension}D({rounded_axes}, name="{self.name}")'


class Line(Primitive):
	def __init__(self, pos1: Union[Point, list, tuple], pos2: Union[Point, list, tuple],
				 name: str = 'Line', color: str = 'y', alpha: Union[int, float] = 1):
		if isinstance(pos1, (tuple, list, np.ndarray)):
			pos1 = Point(*pos1)
		if isinstance(pos2, (tuple, list, np.ndarray)):
			pos2 = Point(*pos2)

		self.pos1, self.pos2 = reduse_axeslists(pos1.axes, pos2.axes)
		self.pos1, self.pos2 = Point(self.pos1, name=pos1.name), Point(self.pos2, name=pos2.name)
		self.name = name
		self.color = color
		self.alpha = alpha

		if list(self.pos1.axes) == list(self.pos2.axes):
			raise ConstructError(f'Expected 2 different Points, got equal: {pos1}; {pos2}')

		if self.dimension == 1:
			self.k = 0
			self.m = 0
		elif self.dimension == 2:
			if self.pos1.x == self.pos2.x: # if it is vertical line
				self.k = 0
			elif self.pos1.y == self.pos2.y: # if it is horizontal line
				self.k = 0
			else:
				# Every point at line has proporcional Y/X
				self.k = to_fraction(self.pos1.y - self.pos2.y, self.pos1.x - self.pos2.x)
			
			if self.pos1.x == self.pos2.x: # if it is vertical line
				self.m = self.pos1.x
			elif pos1.y == pos2.y: # if it is horizontal line
				self.m = self.pos2.y
			else:
				self.m = self.pos1.y - self.k * self.pos1.x

	def intersects(self, object: Union[Primitive, Point, list, tuple]) -> List[Point]:
		if isinstance(object, (list, tuple, np.ndarray)):
			object = Point(*object)

		if isinstance(object, Point):
			return self.intersects_point(object)
		elif isinstance(object, (Ray, Vector, Segment, Line)):
			return self.intersects_line(object)
		elif hasattr(object, 'intersects') and not isinstance(object, Primitive):
			return object.intersects(self)
		else:
			raise IntersectionError(f"Invalid argument type for 'object'. Expected types are Union[Primitive, Point, list, tuple], but received {type(object)}.")

	def intersects_point(self, object: Point) -> List[Point]:
		pos1, direction, point = eq_len_axeslists(self.pos1.axes, self.vector.pos2.axes, object.axes)
		# Lists with different lengths now has equal length due to completing a smaller list with zeros
		p1, d, p2 = [], [], []

		for i in range(max(self.dimension, object.dimension)):
			if direction[i] == 0:
				if pos1[i] != point[i]:
					return []
			else:
				p2.append(point[i])
				p1.append(pos1[i])
				d.append(direction[i])

		# Point intersection with vector
		t_values = (np.array(p2) - np.array(p1)) / np.array(d)
		# if t values is equal point is on line
		if np.all(np.abs(t_values - t_values[0]) < EPSILON):
			return [object]
		else:
			return []

		# Point intersection with vector
		t_values = (np.array(p2) - np.array(p1)) / np.array(d)
		# if t values is equal point is on line
		if np.all(np.abs(t_values - t_values[0]) < EPSILON):
			return [point]
		else:
			return []

	def intersects_line(self, object: Union['Line', 'Segment', 'Ray', 'Vector']):
		spos1,spos2, opos1,opos2 = eq_len_axeslists(self.pos1.axes, self.pos2.axes, object.pos1.axes, object.pos2.axes)
		true_dim_indexes_1 = [i for i, (p1, p2) in enumerate(zip(spos1, spos2)) if p1 != p2]
		true_dim_indexes_2 = [i for i, (p1, p2) in enumerate(zip(opos1, opos2)) if p1 != p2]
		true_dim_indexes = true_dim_indexes_1 + true_dim_indexes_2

		# If lines has only 1 varying axis
		if self.true_dimension == object.true_dimension == 1:
			# Same varying axes. Lines are parallel
			if true_dim_indexes_1 == true_dim_indexes_2:
				# Same varying axis. It is intersection on the same line
				if self.pos1 in object and self.pos2 in object:
					if any([type(self) == type(object) == cl for cl in [Segment, Vector, Ray]]):
						# If 2 equals lines/rays/segements/vectors returns itself
						return [self.copy()]
					elif isinstance(self, Line):
						# if 1 line and 1 Ray/Segment/Vector returns Ray/Segment/Vector because it is less
						return [object.copy()]
					elif isinstance(object, Line):
						# if 1 Ray/Segment/Vector and 1 line returns Ray/Segment/Vector because it is less
						return [self.copy()]
					elif isinstance(self, Ray) and isinstance(object, (Segment, Vector)):
						# if 1 ray and 1 segment/vector returns segment/vector because it is less
						return [object.copy()]
					elif isinstance(self, (Segment, Vector)) and isinstance(object, Ray):
						# if 1 segment/vector and 1 ray returns segment/vector because it is less
						return [self.copy()]

				elif object.pos1 in self and object.pos2 in self:
					if any([type(self) == type(object) == cl for cl in [Segment, Vector, Ray]]):
						# If 2 equals lines/rays/segements/vectors returns itself
						return [self.copy()]
					elif isinstance(self, Line):
						# if 1 line and 1 Ray/Segment/Vector returns Ray/Segment/Vector because it is less
						return [object.copy()]
					elif isinstance(object, Line):
						# if 1 Ray/Segment/Vector and 1 line returns Ray/Segment/Vector because it is less
						return [self.copy()]
					elif isinstance(self, Ray) and isinstance(object, (Segment, Vector)):
						# if 1 ray and 1 segment/vector returns segment/vector because it is less
						return [object.copy()]
					elif isinstance(self, (Segment, Vector)) and isinstance(object, Ray):
						# if 1 segment/vector and 1 ray returns segment/vector because it is less
						return [self.copy()]

				elif self.pos1 in object:
					if object.pos1 in self and object.pos1 != self.pos1:
						ion = object.pos1
					elif object.pos2 in self and object.pos2 != self.pos1:
						ion = object.pos2
					else:
						return [self.pos1.copy()]

					if isinstance(self, Ray) and isinstance(object, Ray):
						vec1, vec2 = self.vector, object.vector
						if vec1.compare_directions(vec2):
							return [Ray(self.pos1, ion, name=intersection_result_name.format(self.name, object.name))]
						elif vec1.is_reverse_direction(vec2):
							return [
								Segment(self.pos1, ion, name=intersection_result_name.format(self.name, object.name))]
						else:
							return []
					else:
						return [Segment(self.pos1, ion, name=intersection_result_name.format(self.name, object.name))]

				elif object.pos1 in self:
					if self.pos2 in object and object.pos1 != self.pos2:
						ion = self.pos2
					elif self.pos1 in object and object.pos1 != self.pos1:
						ion = self.pos1
					else:
						return [object.pos1.copy()]

					if isinstance(self, Ray) and isinstance(object, Ray):
						vec1, vec2 = self.vector, object.vector
						if vec1.compare_directions(vec2):
							return [Ray(self.pos1, ion, name=intersection_result_name.format(self.name, object.name))]
						elif vec1.is_reverse_direction(vec2):
							return [
								Segment(self.pos1, ion, name=intersection_result_name.format(self.name, object.name))]
						else:
							return []
					else:
						return [Segment(self.pos1, ion, name=intersection_result_name.format(self.name, object.name))]

				return []

			# Different varying axes. Lines are perpendicular
			else:
				ion = [self.pos1[i] if not i in true_dim_indexes else 0 for i in
					   range(max(self.dimension, object.dimension))]

				ion[true_dim_indexes_1[0]] = object.pos1[true_dim_indexes_1[0]]
				ion[true_dim_indexes_2[0]] = self.pos1[true_dim_indexes_2[0]]

				point = Point(ion, name=intersection_result_name.format(self.name, object.name))
				return [point] if point in self and point in object else []

		# If first line has only 1 varying axis but second has 2
		elif self.true_dimension == 1 and object.true_dimension == 2:
			x = self.pos1[true_dim_indexes_2[0]]
			y = self.pos1[true_dim_indexes_2[1]]
			second = Line(
				Point[object.pos1[true_dim_indexes_2[0]], object.pos1[true_dim_indexes_2[1]]],
				Point[object.pos2[true_dim_indexes_2[0]], object.pos2[true_dim_indexes_2[1]]],
			)
			point1 = Point([0] * true_dim_indexes_2[0] +
						   [x] +
						   [0] * (true_dim_indexes_2[1] - true_dim_indexes_2[0] - 1) +
						   [second.y_from_x(x)] +
						   [0] * (self.true_dimension - true_dim_indexes_2[1] - 1),
						   name=intersection_result_name.format(self.name, object.name)
						   )
			point2 = Point([0] * true_dim_indexes_2[0] +
						   [second.x_from_y(y)] +
						   [0] * (true_dim_indexes_2[1] - true_dim_indexes_2[0] - 1) +
						   [y] +
						   [0] * (self.true_dimension - true_dim_indexes_2[1] - 1),
						   name=intersection_result_name.format(self.name, object.name)
						   )

			if point1 in object and point1 in self:
				return [point1]
			elif point2 in object and point2 in self:
				return [point2]

			elif self.pos1 in object:
				return [self.pos1]
			elif self.pos2 in object:
				return [self.pos2]
			elif object.pos1 in self:
				return [object.pos1]
			elif object.pos2 in self:
				return [object.pos2]

			else:
				return []

		# Returns result from last IF statement by changing "self -> object" to "object -> self"
		elif self.true_dimension == 2 and object.true_dimension == 1:
			return object.intersects(self)

		# If lines has only 2 varying axes. Calculating intersection in 2D space
		elif self.true_dimension == object.true_dimension == 2:
			# We can ignore constant axes [..., axis1, ..., axis2, ...]: There is 2 axes that makes 2D space
			if true_dim_indexes_1 == true_dim_indexes_2:
				true_dim_indexes = true_dim_indexes_1
				line2D_1 = self.__class__(
					Point[self.pos1[true_dim_indexes[0]], self.pos1[true_dim_indexes[1]]],
					Point[self.pos2[true_dim_indexes[0]], self.pos2[true_dim_indexes[1]]],
				)
				line2D_2 = object.__class__(
					Point[object.pos1[true_dim_indexes[0]], object.pos1[true_dim_indexes[1]]],
					Point[object.pos2[true_dim_indexes[0]], object.pos2[true_dim_indexes[1]]],
				)
				dk = line2D_2.k - line2D_1.k

				if dk:
					x = to_fraction(line2D_1.m - line2D_2.m, dk)
					y = line2D_1.k * x + line2D_1.m

					# Returns (x,y) with first varying axes that have been ignored
					point = Point([0] * true_dim_indexes[0] +
								  [x] +
								  [0] * (true_dim_indexes[1] - true_dim_indexes[0] - 1) +
								  [y] +
								  [0] * (self.true_dimension - true_dim_indexes[1] - 1),
								  name=intersection_result_name.format(self.name, object.name)
								  )
					return [point] if point in self and point in object else []
				else:
					if isinstance(line2D_1, (Vector, Segment)):
						if isinstance(line2D_2, (Vector, Segment)):
							if line2D_1.pos1 in line2D_2:
								if line2D_1.pos2 in line2D_2:
									return [line2D_1]
								elif line2D_2.pos1 in line2D_1 and line2D_1.pos1 != line2D_2.pos1:
									return [Segment(line2D_1.pos1, line2D_2.pos1,
													name=intersection_result_name.format(self.name, object.name))]
								elif line2D_2.pos2 in line2D_1 and line2D_1.pos1 != line2D_2.pos2:
									return [Segment(line2D_1.pos1, line2D_2.pos2,
													name=intersection_result_name.format(self.name, object.name))]
								else:
									return [line2D_1.pos1]
							elif line2D_1.pos2 in line2D_2:
								if line2D_1.pos1 in line2D_2:
									return [line2D_1]
								elif line2D_2.pos1 in line2D_1 and line2D_1.pos2 != line2D_2.pos1:
									return [Segment(line2D_1.pos2, line2D_2.pos1,
													name=intersection_result_name.format(self.name, object.name))]
								elif line2D_2.pos2 in line2D_1 and line2D_1.pos2 != line2D_2.pos2:
									return [Segment(line2D_1.pos2, line2D_2.pos2,
													name=intersection_result_name.format(self.name, object.name))]
								else:
									return [line2D_1.pos2]
						elif isinstance(line2D_2, Ray):
							if line2D_1.pos1 in line2D_2 and line2D_1.pos2 in line2D_2:
								return [line2D_1]
							elif line2D_1.pos1 in line2D_2 and line2D_2.pos1 != line2D_1.pos1:
								return [Segment(line2D_2.pos1, line2D_1.pos1,
												name=intersection_result_name.format(self.name, object.name))]
							elif line2D_1.pos2 in line2D_2 and line2D_2.pos1 != line2D_1.pos2:
								return [Segment(line2D_2.pos1, line2D_1.pos2,
												name=intersection_result_name.format(self.name, object.name))]
						elif isinstance(line2D_2, Line):
							return [line2D_1]

					elif isinstance(line2D_1, Ray):
						if isinstance(line2D_2, (Segment, Vector)):
							if line2D_1.pos1 in line2D_2 and line2D_1.pos2 in line2D_2:
								return [line2D_1]
							elif line2D_1.pos1 in line2D_2:
								return [Segment(line2D_2.pos1, line2D_1.pos1,
												name=intersection_result_name.format(self.name, object.name))]
							elif line2D_1.pos2 in line2D_2:
								return [Segment(line2D_2.pos1, line2D_1.pos2,
												name=intersection_result_name.format(self.name, object.name))]
						elif isinstance(line2D_2, Ray):
							if line2D_1.pos1 in line2D_1 and line2D_1.pos1 in line2D_2:
								return [Segment(line2D_2.pos1, line2D_1.pos1,
												name=intersection_result_name.format(self.name, object.name))]
							elif line2D_1.pos1 in line2D_2:
								return [line2D_2]
							elif line2D_2.pos1 in line2D_1:
								return [line2D_1]
						elif isinstance(line2D_2, Line):
							return [line2D_1]

					elif isinstance(line2D_1, Line):
						return [line2D_2] if line2D_1.pos1 in line2D_2 or line2D_1.pos2 in line2D_2 else []

					return []

		# The intersection of multidimensional lines is the union of Points from their 2D projections
		else:
			dim = max(self.dimension, object.dimension)

			self2ds = self.projects_2D(max_dimension=dim)
			obj2ds = object.projects_2D(max_dimension=dim)

			ios = []
			for projected_line1, projected_line2 in zip(obj2ds, self2ds):
				if projected_line1 in projected_line2:
					ion = projected_line1.intersects(projected_line2)[0]
					if isinstance(ion, Point):
						ios.append(ion)
					else:
						ios.append(ion.random_point)

			point = []
			for axis in range(dim):
				axes = [pos[axis] for pos in ios]
				if [axis for axis in axes if axis == 0] != axes:  # zero list
					point.append(sorted(axes, key=lambda num: abs(str(num).find('.') - len(str(num))) - 1)[0])
				else:
					point.append(0)

			point = Point[point]
			return [point]

	# List of 2D projects of Line with any dimension >2, it is used in multidimension intersections
	def projects_2D(self, max_dimension: int = None):
		dim = max_dimension if max_dimension else self.dimension
		projects_count = dim * (dim - 1) // 2
		
		if self.dimension > 2:
			lines = []
			for i, j in combinations(range(dim), 2):
				pos1 = np.zeros(dim); pos2 = np.zeros(dim)
				pos1[i] = self.pos1[i]; pos1[j] = self.pos1[j]
				pos2[i] = self.pos2[i]; pos2[j] = self.pos2[j]
				vector = pos2 - pos1

				if Point(pos1) != Point(pos2 + vector):
					lines.append(Line(pos1, pos2 + vector, name=projection_name.format(self.name)))
				else:
					lines.append(self.copy())
			return lines

		elif self.dimension == 2:
			return [self.copy()]

		else:
			return []

	# Only for 2D lines. Returns Y of point on Line by X of point on Line
	def y_from_x(self, x: float, return_none: bool = False) -> float:
		if self.dimension == 1:
			return self.pos1.x
		elif self.dimension == 2:
			# y = kx + m; returns (y) from x
			if self.direction in ['right', 'left', 'horizontal']:
				return self.pos1.y if x == self.pos1.x or not return_none else None
			elif self.direction in ['up', 'down', 'vertical']:
				return self.pos1.y if x == self.pos1.x or not return_none else None
			elif self.direction == 'point':
				return self.pos1.y
			else:
				return self.k * x + self.m if self.intersects([x, self.k * x + self.m]) or not return_none else None

	# Only for 2D lines. Returns X of point on Line by Y of point on Line
	def x_from_y(self, y: float, return_none: bool = False) -> float:
		if self.dimension == 1:
			return self.pos1.x
		elif self.dimension == 2:
			# y = kx + m; x = (y-m)/k; returns (x) from y
			if self.direction in ['right', 'left', 'horizontal']:
				return self.pos1.x if y == self.pos1.y or not return_none else None
			elif self.direction in ['up', 'down', 'vertical']:
				return self.pos1.x if y == self.pos1.y or not return_none else None
			elif self.direction == 'point':
				return self.pos1.x
			else:
				return to_fraction(y - self.m, self.k) if self.intersects(
					[to_fraction(y - self.m, self.k), y]
				) or not return_none else None

	def is_parallel(self, object: Union['Line', 'Segment', 'Ray', 'Vector', list, tuple]) -> bool:
		return self.vector.is_parallel(object.vector)

	def is_perpendicular(self, object: Union['Line', 'Segment', 'Ray', 'Vector', list, tuple]) -> bool:
		return self.vector.is_perpendicular(object.vector)

	# Moves the Line. New Line will intersects "point"
	def at_pos(self, point: Point) -> 'Line':
		if isinstance(point, (list, tuple, np.ndarray)):
			point = Point(point)
			
		if self.pos1 != self.pos2:
			shift = point - self.pos1
			return self.__class__(self.pos1 + shift, self.pos2 + shift, name=self.name)
		else:
			raise ConstructError(f'Line from and to equal points start and end pos1: {self.pos1} and pos2: {self.pos2} can not be moved by at_pos')

	def copy(self) -> 'Line':
		return self.__class__(self.pos1, self.pos2, name=self.name)

	# Cuts from pos1 and pos2 unnecessary axes
	def project_to(self, dimension: int) -> 'Line':
		if dimension == 0:
			return Point(name=self.name)
		elif dimension == 1:
			return Line(self.pos1.x, self.pos2.x, name=self.name, color=self.color, alpha=self.alpha)
		elif dimension >= 2:
			if dimension >= self.dimension:
				return self.copy()
			else:
				return self.__class__(
					list(self.pos1.axes[:dimension]), list(self.pos2.axes[:dimension]), name=self.name
				)

	# Convertor
	@property
	def to_segment(self):
		return Segment(self.pos1, self.pos2, name=self.name, color=self.color, alpha=self.alpha)
	@property
	def to_vector(self):
		return Vector(self.pos1, self.pos2, name=self.name, color=self.color, alpha=self.alpha)
	@property
	def to_ray(self):
		return Ray(self.pos1, self.pos2, name=self.name, color=self.color, alpha=self.alpha)
	@property
	def to_line(self):
		return Line(self.pos1, self.pos2, name=self.name, color=self.color, alpha=self.alpha)

	# Only for 2D Lines. Mirror with center in (0,0)
	@property
	def mirror_y(self):
		return Line.by_func(lambda x: -self.k * x + self.m, name=self.name, color=self.color, alpha=self.alpha)
	@property
	def mirror_x(self):
		return Line.by_func(lambda x: -self.k * x - self.m, name=self.name, color=self.color, alpha=self.alpha)

	# Angle between 2 lines: self and Line(x=0)
	@property
	def angle(self) -> 'Angle':
		if self.direction != 'point':
			if self.dimension == 1:
				return 0
			elif self.dimension == 2:
				second_pos = Point(self.pos2.x + 1, self.pos1.y)
				while second_pos.pos in [self.pos1.pos, self.pos2.pos]:
					second_pos += [1, 0]
				return Angle(self.pos2, self.pos1, second_pos, name=angle_result_name.format(self.name))

	# The maximum dimension of 2 points
	@property
	def dimension(self) -> int:
		return max(len(self.pos1.axes), len(self.pos2.axes))

	# The count of varying axes
	@property
	def true_dimension(self) -> int:
		pos1, pos2 = eq_len_axeslists(self.pos1.axes, self.pos2.axes)
		return len([ (i, j) for i, j in zip(pos1, pos2) if i != j ])

	@property
	def perpendicular(self) -> Union['Space', 'Line']:
		if self.dimension >= 3:
			vector = self.vector
			dir_vector_np = [ float(i) for i in vector.pos2.axes ]
			id_matrix = np.eye(self.dimension)

			ortho_vectors = gram_schmidt(
				[vector] + [
					Vector[id_matrix[i]] for i in range(self.dimension) if not np.allclose(id_matrix[i], dir_vector_np)
				]
			)[1:]  # Exclude the direction vector itself (self.vector)

			return Space(self.center, ortho_vectors, name=perpendicular_name.format(self.name))
		else:
			segment = Point[0,0].height_to(self.at_pos([1,0]))
			# Moves the Line in ".at_pos([1,0])" because Line with intersection in [0,0] will makes Segment[(0,0),(0,0)]
			segment.name = perpendicular_name.format(self.name)
			return segment.to_line

	@property
	def direction(self) -> Union[str, 'Vector']:
		if self.dimension == 0:
			return 'point'
		elif self.dimension == 1:
			return 'horizontal'
		elif self.dimension == 2:
			if self.pos1.pos == self.pos2.pos: # from [0, 0] to [0, 0]
				return 'point'
			elif self.pos1.x == self.pos2.x: # y = num
				return 'vertical'
			elif self.pos1.y == self.pos2.y: # x = num
				return 'horizontal'
			else: # y = kx + b
				return 'normal'
		else:
			return self.vector

	@property
	def vector(self) -> 'Vector':
		return self.get_vector()

	def get_vector(self, from_zero: bool = True) -> 'Vector':
		if not isinstance(self, Vector):
			vec = Vector([0,0], self.pos2 - self.pos1, name=vector_of_object_name.format(self.name))
			return vec if from_zero else vec.at_pos(self.pos1)
		else:
			return self.copy()

	@property
	def as_geometry(self) -> str:
		if self.dimension == 0:
			return str(self.pos1)
		elif self.dimension == 1:
			return f'x = {self.pos1.x}'
		elif self.dimension == 2:
			if self.direction == 'vertical':
				return f'x = {self.pos1.x}'
			elif self.direction == 'horizontal':
				return f'y = {self.pos1.y}'
			elif self.direction == 'point':
				return f'x = {self.pos1.x}; y = {self.pos1.y}'
			else:
				k = self.k if self.k != 1 else ""
				return f'y(x) = {k}x{(f" + {self.m}" if self.m > 0 else f" - {-1*self.m}") if self.m else ""}'
		else:
			return f'f(t) = {self.pos1} + {self.vector.short_view} * t'

	@property
	def center(self) -> Point:
		return Point([to_fraction(self.pos1[i] + self.pos2[i], 2) for i in range(self.dimension)],
					 name=center_name.format(self.name))

	@staticmethod
	def by_func(func: callable, **kwargs) -> 'Line':
		x = 0
		return Line(Point(x, func(x)), Point(x+1, func(x+1)), **kwargs)

	@staticmethod
	def by_angle(angle: int, pos1: Union[tuple, list, 'Point'] = [0,0], **kwargs) -> 'Line':
		if isinstance(pos1, (tuple, list, np.ndarray)):
			pos1 = Point(*pos1)

		dx = cos(radians(angle))
		dy = sin(radians(angle))
		return Line(pos1, pos1 + [dx, dy], **kwargs)

	def __contains__(self, object):
		return self.intersects(object)

	def __add__(self, i) -> 'Line':
		if isinstance(i, (Point, list, tuple, np.ndarray)):
			return self.__class__(self.pos1 + i, self.pos2 + i, name=self.name, color=self.color, alpha=self.alpha)
	def __sub__(self, i) -> 'Line':
		if isinstance(i, (Point, list, tuple, np.ndarray)):
			return self.__class__(self.pos1 - i, self.pos2 - i, name=self.name, color=self.color, alpha=self.alpha)

	def __call__(self, t: int, return_none: bool = False) -> float:
		if self.dimension == 1:
			return self.pos1.x
		elif self.dimension >= 2:
			return self.pos1 + self.get_vector(from_zero=False).pos2 * t

	def __str__(self):
		return f'{self.__class__.__name__}[({self.pos1} -> {self.pos2})]'
	def __repr__(self):
		return f'{self.__class__.__name__}{self.dimension}D({self.pos1}, {self.pos2}, name="{self.name}")'

class Segment(Line):
	def __init__(self, pos1: Union[Point, list, tuple], pos2: Union[Point, list, tuple],
				 name: str = 'Segment', color: str = 'g', alpha: Union[int, float] = 1):
		self.color = color
		super().__init__(pos1, pos2, name=name, color=color, alpha=alpha)

	def intersects(self, object: Union[Primitive, Point, list, tuple]) -> Point:
		if isinstance(object, (list, tuple, np.ndarray)):
			object = Point(object)

		if isinstance(object, Point):
			if all([object[i] >= min(self.pos1[i], self.pos2[i]) and object[i] <= max(self.pos1[i], self.pos2[i]) for i in range(self.dimension) ]):
				return super().intersects(object)
			else:
				return []

		elif hasattr(object, 'intersects') and not isinstance(object, Primitive):
			return object.intersects(self)

		else:
			return super().intersects(object)

	@property
	def length(self) -> float:
		return round(sqrt(sum([ (self.pos2[i] - self.pos1[i])**2 for i in range(self.dimension) ])), 6)

	@staticmethod
	def by_func(func: callable, length: int, **kwargs) -> 'Segment':
		random_x = float(r.randint(-10, 10))
		center = Point(random_x, func(random_x))

		pos1 = Point[1.0,func(1)]
		line = Line(center, Point(random_x+1, func(random_x+1)))
		vector = line.vector.at_pos(pos1).normalize

		return Segment(pos1, (vector * length).pos2, **kwargs)

	@staticmethod
	def by_angle(angle: int, length: int, pos1: Union[tuple, list, 'Point'] = [0,0], **kwargs) -> 'Segment':
		if isinstance(pos1, (tuple, list, np.ndarray)):
			pos1 = Point(*pos1)

		dx = cos(radians(angle))
		dy = sin(radians(angle))

		line = Line(pos1, pos1 + [dx, dy])
		vector = line.vector.at_pos(pos1).normalize

		return Segment(pos1, (vector * length).pos2, **kwargs).at_pos(pos1)

	@property
	def random_point(self) -> 'Point':
		return self( round(r.uniform(0,1), 2) )

	def __repr__(self):
		cl = self.__class__.__name__
		return f'{cl}{self.dimension}D({self.pos1}, {self.pos2}, length={round(self.length, 3)}, name="{self.name}")'

class Ray(Line):
	def __init__(self, pos1: Union[Point, list, tuple], pos2: Union[Point, list, tuple],
				 name: str = 'Ray', color: str = 'r', alpha: Union[int, float] = 1):
		super().__init__(pos1, pos2, name=name, color=color, alpha=alpha)
	
	def intersects(self, object: Union[Primitive, Point, list, tuple]) -> Point:
		if isinstance(object, (list, tuple, np.ndarray)):
			object = Point(*object)

		if isinstance(object, Point):
			for i in range(self.dimension):
				if self.vector.pos2[i] == 0:
					if object[i] != self.pos1[i]:
						return []
				else:
					t = to_fraction(object[i] - self.pos1[i], self.vector.pos2[i])
					if t < 0:
						return []
			return super().intersects(object)

		elif hasattr(object, 'intersects') and not isinstance(object, Primitive):
			return object.intersects(self)

		else:
			return super().intersects(object)

	@property
	def as_geometry(self) -> str:
		if self.dimension == 0:
			return str(self.pos1)
		elif self.dimension == 1:
			return f'x = {self.pos1.x}'
		elif self.dimension == 2:
			if self.direction in ['up', 'down']:
				return f'x = {self.pos1.x}'
			elif self.direction in ['left', 'right']:
				return f'y = {self.pos1.y}'
			elif self.direction == 'point':
				return f'x = {self.pos1.x}; y = {self.pos1.y}'
			else:
				k = self.k if self.k != 1 else ""
				return f'y(x) = {k}x{ (f" + {self.m}" if self.m > 0 else f" - {-1*self.m}") if self.m else "" }'
		else:
			return f'r(t) = {self.pos1} + {self.direction} * t'

	@property
	def direction(self) -> Union[str, 'Vector']:
		if self.dimension:
			if self.pos1 == self.pos2: # from [0, 0] to [0, 0]
				return 'point'

			if self.pos1.x == self.pos2.x: # y = num
				if self.pos1.y > self.pos2.y:
					return 'down'
				elif self.pos1.y < self.pos2.y:
					return 'up'
			elif self.pos1.y == self.pos2.y: # x = num
				if self.pos1.x < self.pos2.x:
					return 'right'
				elif self.pos1.x > self.pos2.x:
					return 'left'
			else: # y = kx + b
				if self.pos2.x > self.pos1.x and self.pos1.y > self.pos2.y:
					return 'right-down'
				elif self.pos2.x > self.pos1.x and self.pos1.y < self.pos2.y:
					return 'right-up'
				elif self.pos2.x < self.pos1.x and self.pos1.y > self.pos2.y:
					return 'left-down'
				elif self.pos2.x < self.pos1.x and self.pos1.y < self.pos2.y:
					return 'left-up'
		else:
			return self.vector

	@staticmethod
	def by_func(func: callable, length: int, **kwargs) -> List['Ray']:
		# Returns 2 Rays that created by function that going in different directions

		random_x = float(r.randint(-10, 10))
		center = Point(random_x, func(random_x))

		pos1 = Point[1.0,func(1)]
		line = Line(center, Point(random_x+1, func(random_x+1)))
		vector = line.vector.at_pos(pos1).normalize

		return [Ray(pos1, vector.pos2, **kwargs), Ray(pos1, (vector*-1).pos2, **kwargs)]

	@staticmethod
	def by_angle(angle: int, pos1: Union[tuple, list, 'Point'] = [0,0], **kwargs) -> 'Ray':
		from .shapes2d import Circle
		if isinstance(pos1, (tuple, list, np.ndarray)):
			pos1 = Point(*pos1)

		dx = cos(radians(angle))
		dy = sin(radians(angle))
		return Ray(pos1, pos1 + [dx, dy], **kwargs)

	@property
	def pos(self) -> float:
		return self.pos1
	@property
	def x(self) -> float:
		return self.pos1.x
	@property
	def y(self) -> float:
		return self.pos1.y
	@property
	def z(self) -> float:
		return self.pos1.z
	@property
	def w(self) -> float:
		return self.pos1.w

class VectorMeta(type): # to create vector by "Vector[]" like Vector[1,2,3]
	def __getitem__(cls, pos):
		if isinstance(pos, (tuple, list, np.ndarray)):
			return cls([0] * len(pos), pos)
		else:
			return cls([0], Point[pos])
class Vector(Segment, metaclass=VectorMeta):
	def __init__(self, *args, name: str = "Vector", **kwargs):
		super().__init__(*args, name=name, **kwargs)

	def dot(self, vector) -> float:
		return sum([ vector.to_zero.pos2[i] * self.to_zero.pos2[i] for i in range(self.dimension) ])

	def cross(self, vector):
		if not isinstance(vector, Vector):
			raise TypeError("Cross product requires another Vector.")

		dim = max(self.dimension, vector.dimension)
		a = self.to_zero.pos2
		b = vector.to_zero.pos2

		if dim == 2:
			return a[0] * b[1] - a[1] * b[0]
		elif dim == 3:
			return Vector([0, 0, 0], [
				a[1] * b[2] - a[2] * b[1],
				a[2] * b[0] - a[0] * b[2],
				a[0] * b[1] - a[1] * b[0]
			], name=self.name, alpha=self.alpha, color=self.color)

		else:
			raise ValueError(f"Cross product is only defined up to 3D or 2D (got dimension {dim}).")

	# True if vectors have an equal direction else False
	def compare_directions(self, vector: Union['Vector', 'Point', tuple, list]) -> bool:
		if isinstance(vector, (tuple, list, Point, np.ndarray)):
			vector = Vector[vector]

		axes = eq_len_axeslists(self.to_zero.pos2.axes, vector.to_zero.pos2.axes)
		return (
				bool(np.dot(np.array(axes[0]), np.array(axes[1])) > 0) and self.is_parallel(vector)
		) or vector.direction == 'point' or self.direction == 'point'

	# True vector*-1 has equal direction to self else False
	def is_reverse_direction(self, vector: Union['Vector', 'Point', tuple, list]) -> bool:
		if isinstance(vector, (tuple, list, Point, np.ndarray)):
			vector = Vector[vector]

		axes = eq_len_axeslists(self.to_zero.pos2.axes, vector.to_zero.pos2.axes)
		return (not self.compare_directions(vector)) and self.is_parallel(vector)

	def is_perpendicular(self, vector: Union['Point', tuple, list, 'Line', 'Ray', 'Vector', 'Segment']) -> bool:
		if isinstance(vector, (tuple, list, Point, np.ndarray)):
			vector = Vector[vector]
		elif isinstance(vector, (Segment, Line, Ray)):
			vector = vector.vector

		vec1 = self.normalize.to_zero
		vec2 = vector.normalize.to_zero
		return Angle.between(vec1,vec2).degrees == 90 if vec1.pos2 != vec2.pos2 else False

	def is_parallel(self, vector: Union['Point', tuple, list, 'Line', 'Ray', 'Vector', 'Segment']) -> bool:
		if isinstance(vector, (tuple, list, Point, np.ndarray)):
			vector = Vector[vector]
		elif isinstance(vector, (Segment, Line, Ray)):
			vector = vector.vector

		zero_vec1, zero_vec2 = self.to_zero.pos2, vector.to_zero.pos2
		line = Line([0,0], zero_vec2)
		return (zero_vec1 in line) or vector.direction == 'point' or self.direction == 'point'

	@property
	def vector(self):
		return self.copy()

	@property
	def direction(self) -> str:
		if self.dimension == 0 or self.pos1.axes == self.pos2.axes:
			return 'point'
		elif self.dimension == 1:
			return 'horizontal'
		elif self.dimension == 2:
			if self.pos1 == self.pos2: # from [0, 0] to [0, 0]
				return 'point'

			if self.pos1.x == self.pos2.x: # y = num
				if self.pos1.y > self.pos2.y:
					return 'down'
				elif self.pos1.y < self.pos2.y:
					return 'up'
			elif self.pos1.y == self.pos2.y: # x = num
				if self.pos1.x < self.pos2.x:
					return 'right'
				elif self.pos1.x > self.pos2.x:
					return 'left'
			else: # y = kx + b
				if self.pos2.x > self.pos1.x and self.pos1.y > self.pos2.y:
					return 'right-down'
				elif self.pos2.x > self.pos1.x and self.pos1.y < self.pos2.y:
					return 'right-up'
				elif self.pos2.x < self.pos1.x and self.pos1.y > self.pos2.y:
					return 'left-down'
				elif self.pos2.x < self.pos1.x and self.pos1.y < self.pos2.y:
					return 'left-up'
		else:
			return self.as_geometry

	@property
	def as_geometry(self) -> str:
		if self.dimension == 0:
			return 'point'
		elif self.dimension == 1:
			return 'horizontal'
		elif self.dimension == 2:
			if self.direction in ['up', 'down']:
				return f'x = {self.pos1.x}'
			elif self.direction in ['left', 'right']:
				return f'y = {self.pos1.y}'
			elif self.direction == 'point':
				return f'x = {self.pos1.x}; y = {self.pos1.y}'
			else:
				k = self.k if self.k != 1 else ""
				return f'y(x) = {k}x{ (f" + {self.m}" if self.m > 0 else f" - {-1*self.m}") if self.m else "" }'
		else:
			vec = f"{self.name}[({self.pos1} -> {self.pos2})]" if self.pos1 != 0 else f"{self.name}[{self.pos2}]"
			return f'r(t) = {self.pos1} + {vec} * t'

	@property
	def to_zero(self) -> 'Vector':
		return self.at_pos([0,0])

	@property
	def short_view(self) -> str:
		return f'vec{self.dimension}{self.to_zero.pos2.axes}'

	@property
	def normalize(self) -> 'Vector':
		length = self.length
		if self.direction != 'point':
			return Vector(
				self.pos1,
				self.pos1 + (self.pos2 - self.pos1) / length,
				name=normalize_result_name.format(self.name),
				color=self.color,
				alpha=self.alpha
			)
		else:
			return self

	@staticmethod
	def by_func(func: callable, length: int, **kwargs) -> 'Vector':
		from .shapes2d import Circle

		random_x = float(r.randint(-10, 10))
		center = Point(random_x, func(random_x))

		line = Line(center, Point(random_x+1, func(random_x+1)))
		points = Circle(center, length).intersects(line)

		return Vector([1.0,func(1)], points[0], **kwargs)

	@staticmethod
	def by_angle(pos1: Union[tuple, list, 'Point'], angle: int, length: int, **kwargs) -> 'Vector':
		if isinstance(pos1, (tuple, list, np.ndarray)):
			pos1 = Point(*pos1)

		k = tan(radians(angle))
		m = pos1.y - pos1.x * k
		line = Line.by_func( lambda x: k * x + m )
		points = Circle(center, length).intersects(line)
		
		return Vector([1.0,func(1)], points[0], **kwargs)

	def __neg__(self) -> 'Vector':
		return self * -1
	def __pos__(self) -> 'Vector':
		return self.copy()

	def __add__(self, other) -> 'Vector':
		if isinstance(other, (float, int)):
			return Vector(
				self.pos1,
				Point([self.pos2[i] + other for i in range(self.dimension)]),
				name=self.name, alpha=self.alpha, color=self.color
			)
		elif isinstance(other, Vector):
			return Vector(
				self.pos1,
				Point([self.pos2[i] + other.direction[i] for i in range(self.dimension)]),
				name=self.name, alpha=self.alpha, color=self.color
			)
		elif isinstance(other, Point):
			return Vector(
				self.pos1 + other,
				self.pos2 + other,
				name=self.name, alpha=self.alpha, color=self.color
			)
		else:
			raise TypeError(f"Unsupported operand type(s) for +: 'Vector' and '{type(other).__name__}'")

	def __sub__(self, other) -> 'Vector':
		if isinstance(other, (float, int)):
			return Vector(
				self.pos1,
				Point([self.pos2[i] - other for i in range(self.dimension)]),
				name=self.name, alpha=self.alpha, color=self.color
			)
		elif isinstance(other, Vector):
			return self + (-other)
		elif isinstance(other, Point):
			return Vector(
				self.pos1 - other,
				self.pos2 - other,
				name=self.name, alpha=self.alpha, color=self.color
			)
		else:
			raise TypeError(f"Unsupported operand type(s) for -: 'Vector' and '{type(other).__name__}'")

	def __mul__(self, other):
		if isinstance(other, (float, int)):
			return Vector(
				[0],
				Point([(self.pos2[i] - self.pos1[i]) * other for i in range(self.dimension)]),
				name=self.name, alpha=self.alpha, color=self.color
			).at_pos(self.pos1)
		elif isinstance(other, Point):
			return Vector(
				self.pos1 * other,
				self.pos2 * other,
				name=self.name, alpha=self.alpha, color=self.color
			)
		else:
			raise TypeError(f"Unsupported operand type(s) for *: 'Vector' and '{type(other).__name__}'")

	def __truediv__(self, other) -> 'Vector':
		if isinstance(other, (float, int)):
			return Vector(
				self.pos1,
				Point([to_fraction(self.pos2[i], other) for i in range(self.dimension)]),
				name=self.name, alpha=self.alpha, color=self.color
			)
		elif isinstance(other, Point):
			return Vector(
				self.pos1 / other,
				self.pos2 / other,
				name=self.name, alpha=self.alpha, color=self.color
			)
		else:
			raise TypeError(f"Unsupported operand type(s) for /: 'Vector' and '{type(other).__name__}'")

	def __floordiv__(self, other) -> 'Vector':
		if isinstance(other, (float, int)):
			return Vector(
				self.pos1,
				Point([self.pos2[i] // other for i in range(self.dimension)]),
				name=self.name, alpha=self.alpha, color=self.color
			)
		elif isinstance(other, Point):
			return Vector(
				self.pos1 // other,
				self.pos2 // other,
				name=self.name, alpha=self.alpha, color=self.color
			)
		else:
			raise TypeError(f"Unsupported operand type(s) for //: 'Vector' and '{type(other).__name__}'")

	def __pow__(self, other) -> 'Vector':
		if isinstance(other, (float, int)):
			return Vector(
				self.pos1,
				Point([self.pos2[i] ** other for i in range(self.dimension)]),
				name=self.name, alpha=self.alpha, color=self.color
			)
		elif isinstance(other, Point):
			return Vector(
				self.pos1 ** other,
				self.pos2 ** other,
				name=self.name, alpha=self.alpha, color=self.color
			)
		else:
			raise TypeError(f"Unsupported operand type(s) for ** or pow(): 'Vector' and '{type(other).__name__}'")

	def __str__(self):
		if self.pos1 != 0:
			return f'{self.__class__.__name__}[({self.pos1} -> {self.pos2})]'
		else:
			return f'{self.__class__.__name__}[{self.pos2}]'
Vec = Vector


class Angle:
	def __init__(self,
		 pos1: Union[Point, list, tuple], midpos: Union[Point, list, tuple], pos2: Union[Point, list, tuple],
		 default_unit: str = 'deg', name: str = 'Angle', color: str = 'p', alpha: Union[int, float] = 1
		):

		if isinstance(pos1, (tuple, list, np.ndarray)):
			pos1 = Point(*pos1)
		if isinstance(pos2, (tuple, list, np.ndarray)):
			pos2 = Point(*pos2)
		if isinstance(midpos, (tuple, list, np.ndarray)):
			midpos = Point(*midpos)

		if pos1.pos in [pos2.pos, midpos.pos] or pos2.pos in [pos1.pos, midpos.pos] or midpos.pos in [pos2.pos, pos1.pos]:
			raise ConstructError(
				f'Constructor arguments 1, 2 and 3 must be Points with different positions, not {pos1.pos}, {midpos.pos}, {pos2.pos}'
			)

		self.pos1 = pos1
		self.midpos = midpos
		self.pos2 = pos2
		self.name = name
		self.color = color
		self.alpha = alpha

		units = ['deg', 'min', 'rad']
		if default_unit in units:
			self.default_unit = default_unit
		else:
			raise ConstructError(f'Constructor "default_unit" argument must to be one of {units}, not {default_unit}')

	@property
	def vec1(self) -> Vector:
		return Vector(self.midpos, self.pos1, name=f'{self.name}_vec1', alpha=self.alpha, color=self.color)
	@property
	def vec2(self) -> Vector:
		return Vector(self.midpos, self.pos2, name=f'{self.name}_vec2', alpha=self.alpha, color=self.color)

	# Returns True if the degree measure of Angle is zero
	@property
	def zero(self) -> bool:
		return self.pos2 in Segment(self.pos1, self.midpos) or self.pos1 in Segment(self.pos2, self.midpos)

	@property
	def cos(self) -> float:
		return 1.0 if self.zero else to_fraction(self.vec1.dot(self.vec2), self.vec1.length * self.vec2.length)
	@property
	def sin(self) -> float:
		return 0.0 if self.zero else sqrt(1 - self.cos**2)
	@property
	def tan(self) -> float:
		return 0.0 if self.zero else to_fraction(self.sin, self.cos) if self.cos != 0 else 0.0

	@property
	def radians(self) -> float:
		return 0.0 if self.zero else acos(self.cos)
	@property
	def degrees(self) -> float:
		return 0.0 if self.zero else degrees(self.radians)
	@property
	def minutes(self) -> float:
		return 0.0 if self.zero else self.degrees * 60

	@property
	def value(self):
		if self.default_unit == 'deg':
			return self.degrees
		elif self.default_unit == 'rad':
			return self.radians
		elif self.default_unit == 'min':
			return self.minutes
		else:
			return 0.0
		
	@property
	def bisector(self) -> Ray:
		return Ray(self.midpos, (self.pos1 + self.pos2) / 2, name=angle_bisector_name.format(self.name))

	@property
	def dimension(self) -> int:
		return max(len(self.pos1.axes), len(self.pos2.axes), len(self.midpos.axes))

	@property
	def type(self) -> str:
		if self.degrees == 90:
			return 'right'
		elif self.degrees > 90:
			return 'obtuse'
		elif self.degrees == 180:
			return 'straight'
		else:
			return 'acute'

	@staticmethod
	def between(pr1: 'Primitive', pr2: 'Primitive') -> 'Angle':
		int_point = pr1.intersects(pr2)[0]
		if int_point:
			if isinstance(int_point, Point):
				pos1 = pr1.pos1 if pr1.pos1 != int_point else pr1.pos2
				pos2 = pr2.pos1 if pr2.pos1 != int_point else pr2.pos2
				return Angle(pos1, int_point, pos2, name=angle_between_name.format(pr1.name, pr2.name))
			else:
				raise ConstructError(f'Two primitives {pr1} and {pr2} have infinity common points: {int_point}')
		else:
			raise ConstructError(f'Two primitives {pr1} and {pr2} do not have common points: {int_point}')

	def __add__(self, num: float) -> float:
		return float(self)+num
	def __sub__(self, num: float) -> float:
		return float(self)-num
	def __mul__(self, num: float) -> float:
		return float(self)*num
	def __truediv__(self, num: float) -> float:
		return to_fraction(float(self), num)
	def __floordiv__(self, num: float) -> float:
		return float(self)//num
	def __pow__(self, num: float) -> float:
		return float(self)**num

	def __float__(self):
		return float(self.value)
	def __int__(self):
		return int(self.value)

	def __str__(self):
		cl = self.__class__.__name__
		return f'{cl}[({self.pos1} -> {self.midpos} -> {self.pos2}), {int(self)} {self.default_unit}]'
	def __repr__(self):
		cl = self.__class__.__name__
		return f'{cl}({self.midpos}, {round(self.degrees, 3)} degrees, name="{self.name}")'


class AffineSpace:
	def __init__(self, origin: Union['Point', tuple, list], vectors: Union[List['Point'], List['Vector']],
		objects: List[Primitive] = [], name: str = 'AffineSpace'):

		if isinstance(origin, (tuple, list, np.ndarray)):
			origin = Point(origin)
		self.origin = origin
		self.global_objects = objects

		if all([ isinstance(vector, (Point, tuple, list, np.ndarray)) for vector in vectors ]):
			vectors = [ Vector([0], vector, name=affine_space_vectors_name.format(name=name, i=i)) for i, vector in enumerate(vectors) ]

		if all([ isinstance(vector, Vector) for vector in vectors ]):
			self.vectors = [ vector.to_vector.at_pos(self.origin) for vector in vectors ]
			self.points = [ vector.pos2 for vector in self.vectors ]
			self.name = name
		else:
			raise ConstructError(f'"vectors" elements must be list of points or vectors, not {vectors}')

	# converts default point to point in local coordinate system
	def point_to_local(self, point: Union['Point', list, tuple]) -> 'Point':
		if isinstance(point, (list, tuple, np.ndarray)):
			point = Point(*point)

		lists = eq_len_axeslists(point.axes, self.origin.axes, *[ vec.to_zero.pos2.axes for vec in self.vectors ])
		obj, origin, vectors = lists[0], lists[1], lists[2:]
		return Point(*np.dot(vectors, (point - origin).axes), name=affine_space_local_object_name.format(self.name), color=point.color, alpha=point.alpha)

	# converts local point to point in global coordinate system
	def point_to_global(self, point: Union['Point', list, tuple]) -> 'Point':
		if isinstance(point, (list, tuple, np.ndarray)):
			point = Point(*point)

		basis_matrix = np.array([list(vec.pos2.axes) + [0] * (self.dimension - vec.pos2.dimension) for vec in self.vectors]).T
		eq = [0] * (point.dimension - len(basis_matrix[0]))

		matrix = []
		for i, axis in enumerate(basis_matrix):
			matrix.append([*axis, *eq])

		dot = np.dot(matrix, point.axes)
		return Point(list(self.origin.axes + Point[dot]), name=affine_space_global_object_name.format(self.name), color=point.color, alpha=point.alpha)

	def get_normal(self) -> Vector:
		matrix = [v.pos2.axes + [Fraction(0)] * (self.dimension - v.pos2.dimension + 1) for v in self.vectors]
		num_rows = len(matrix)
		num_cols = len(matrix[0])
		
		pivot_cols = []
		m = [row[:] for row in matrix]
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
			pivot_cols.append(col)
			rank += 1

		free_var_index = [col for col in range(num_cols) if col not in pivot_cols][0]
		solution = [Fraction(0)] * num_cols
		solution[free_var_index] = Fraction(1)
		
		for row_idx, pivot_col in enumerate(pivot_cols):
			value = -m[row_idx][free_var_index]
			solution[pivot_col] = value

		return Vector[solution]

	def intersects(self, object: Union[Primitive, Point, list, tuple]) -> List[Point]:
		if isinstance(object, (list, tuple, np.ndarray)):
			object = Point(*object)

		if isinstance(object, Point):
			'''
			There are 2 matrices: M and N. M is a matrix of vectors of Space, and N is the same matrix but with the addition of a vector from the input point
			to the origin of the point of space. If the ranks of the matrices are equal, then the point lies in space, otherwise - no.
			M - mat1, N - mat2.
			'''
			dim = max(object.dimension, self.origin.dimension, *[v.dimension for v in self.vectors])

			mat = [v.to_zero.pos2.axes + [Fraction(0)] * (dim - v.to_zero.pos2.dimension) for v in self.vectors]
			mat1 = np.array(mat)
			last_vector = object - self.origin
			mat2 = np.array(mat + [last_vector.axes + [Fraction(0)] * (dim - last_vector.dimension)])
			return [object] if matrix_rank(mat1) == matrix_rank(mat2) else []

		elif isinstance(object, (Line, Ray, Segment, Vector)):
			if self.normal.is_perpendicular(object):
				return []
			else:
				new_object = self.primitive_projection(object)
				return [Point(point.axes, name=intersection_result_name.format(self.name,object.name)) for point in object.intersects(new_object)]

		elif isinstance(object, (AffineSpace)):
			...

		elif hasattr(object, 'intersects') and not isinstance(object, Primitive):
			return object.intersects(self)

		else:
			raise IntersectionError(f"Invalid argument type for 'object'. Expected types are Union[Primitive, Point, list, tuple], but received {type(object)}.")

	# Moves the Space. New Space will intersects "point"
	def at_pos(self, point: Point) -> 'Space':
		if isinstance(point, (list, tuple, np.ndarray)):
			point = Point(point)
			
		shift = point - self.origin
		objects = [pr + shift for pr in self.global_objects]
		return self.__class__(self.origin + shift, self.vectors, name=self.name, objects=objects)

	def primitive_projection(self, pr: Union[Line, Ray, Segment, Vector, Point]) -> Union[Line, Ray, Segment, Vector, Point]:
		if isinstance(pr, Point):
			return pr.height_to(self)
		else:
			pos1 = pr.pos1.project_to_space(self)
			pos2 = pr.pos2.project_to_space(self)

			if pos1 != pos2:
				return pr.__class__(pos1, pos2, name=pr.name, alpha=pr.alpha, color=pr.color)
			else:
				return Point(*pos1.axes, name=projection_name.format(pr.name))

	def get_local_objects(self) -> List[Primitive]:
		local_objects = []
		for obj in self.global_objects:
			if isinstance(obj, Point):
				local_objects.append(self.point_to_local(obj))
			elif isinstance(obj, Line): # it may be Line/Ray/Segment/Vector
				local_objects.append(
					obj.__class__(self.point_to_local(obj.pos1), self.point_to_local(obj.pos2), name=obj.name, alpha=obj.alpha, color=obj.color)
				)
			else:
				raise ConstructError(f"Unsupported type: '{obj.__class__.__name__}', supports only Point, Line, Ray, Segment or Vector")
		return local_objects

	def show(self):
		self.scene.show()

	@property
	def scene(self) -> 'Scene':
		from .scene import Scene3D, Scene2D
		if self.dimension in [1,2]:
			return Scene2D(*self.local_objects)
		else:
			return Scene3D(*self.local_objects)

	@property
	def normal(self) -> Vector:
		return self.get_normal()

	@property
	def local_objects(self) -> List[Primitive]:
		return self.get_local_objects()

	@property
	def dimension(self) -> int:
		return len(self.vectors)

	@property
	def as_geometry(self) -> str:
		return f'Sp = {self.origin} + ' + " + ".join([ f'{letters[i]}*{self.vectors[i].to_zero}' for i in range(self.dimension) ])

	def __contains__(self, object):
		s = self.intersects(object)
		return s

	def __str__(self):
		return f"{self.__class__.__name__}[{self.origin}, {self.dimension}D]"
	def __repr__(self):
		return f"{self.__class__.__name__}{self.dimension}D({self.origin}, {self.vectors})"

ASpace = AffineSpace

class Space(AffineSpace):
	def __init__(self, origin: 'Point', vectors: Union[List['Point'], List['Vector']], **kwargs):
		super().__init__(origin, vectors, **kwargs)

		l = round(self.vectors[0].length, 8)
		for i in range(len(self.vectors)):
			if not self.vectors[i].is_perpendicular(self.vectors[i-1]):
				raise ConstructError(f'Vectors must be perpendiculars')
			if round(self.vectors[i].length, 8) != l:
				raise ConstructError(f'Vectors must have equal lengths')