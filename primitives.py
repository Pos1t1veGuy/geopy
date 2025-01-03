from typing import *

from typing import *
from math import sqrt, acos, degrees, tan, radians
from string import ascii_lowercase as ABCD

import traceback
import random as r
import numpy as np
from itertools import combinations
import pdb

from .math import *

# Here is geometry primitives: Point, Line, Ray, Segment, Vector and Angle classes

class Shape:
	...

def eq_len_axeslists(*args: 'AxesList', dimension: int = -1) -> List[list]:
	max_dimension = max([ len(l) for l in args ]) if dimension == -1 else dimension
	return [ l.as_list(length=max_dimension) for l in args ]

def reduse_axeslists(*args: 'AxesList') -> List['AxesList']:
	max_dimension = max([ len(l) for l in args ])
	for i in range(max_dimension)[::-1]:
		if not all([ l[i] == 0 for l in args ]):
			return [ AxesList(l[:i+1]) for l in args ]
	return args


class AxesList(list):
	def __getitem__(self, i):
		try:
			return list(self)[i]
		except IndexError:
			return 0
	def as_list(self, length: int = -1) -> list:
		if length == -1 or len(self) == length:
			return list(self)
		elif len(self) < length:
			return list(self) + [0] * (length - len(self))
		else:
			return list(self)[:length]

class Primitive:
	...

letters = 'xyz' + ABCD[:-3]

class PointMeta(type):
	def __getitem__(cls, pos):
		if isinstance(pos, (tuple, list, np.ndarray)):
			return cls(*pos)
		else:
			return cls(pos)
class Point(Primitive, metaclass=PointMeta):
	def __init__(self, *args, name: str = 'Point', color: str = 'm'):
		self.axes = AxesList([])
		self.color = color

		if len(args) == 1 and isinstance(args[0], (tuple, list, np.ndarray)):
			args = args[0]

		for i, axis in enumerate(args):
			if isinstance(axis, (float, int, np.int64)):
				self.axes.append(float(axis))
			else:
				raise ValueError(f"Invalid initialization arguments for 'Point': {args}")

		self.name = name

	def copy(self) -> 'Point':
		return Point(*self.axes, name=self.name)

	def height_to(self, object: Primitive) -> 'Segment':
		if isinstance(object, Point):
			return Segment(self, np.linalg.norm(np.array(self.axes) - np.array(object.axes)), name=f'{self.name}_height')
		elif isinstance(object, Primitive):
			return Segment(self, self.project_to_line(object), name=f'{self.name}_height')
		else:
			raise NotImplementedError(f"height_to not implemented for type {type(object)}")

	# Returns a Point that makes with self Point a new Line that makes 90 degrees Angle to "pr" Line
	def project_to_line(self, pr: 'Line') -> 'Point':
		p1 = np.array(pr.pos1.axes)
		axes, line_vector = eq_len_axeslists(self.axes, pr.vector.pos2.axes)

		return Point(
			*(p1 + np.array(np.dot(np.array(axes) - p1, line_vector) / np.dot(line_vector, line_vector)) * line_vector),
			name=f'{self.name}_project'
		)

	def project_to(self, dimension: int) -> 'Point':
		return self.__class__(self.axes[:dimension], name=f'{self.name}_project')

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
	def dimension(self) -> int:
		return len(self.axes)

	@property
	def pos(self) -> List[float]:
		return self.axes

	def __getattr__(self, i):
		global letters
		if i in letters:
			return self.axes[letters.index(i)]
		else:
			return self.i

	def __getitem__(self, i):
		return self.axes[i]

	def __eq__(self, obj):
		if obj == 0:
			return all([ axis == 0 for axis in self.axes ])
		elif isinstance(obj, Point):
			max_dimension = max( len(self.axes), len(obj.axes) )
			return self.axes.as_list(length=max_dimension) == obj.axes.as_list(length=max_dimension)
		else:
			return list(self.axes) == obj

	def __add__(self, object: Union[int, float]) -> 'Point':
		if isinstance(object, (int, float)):
			return self.__class__([axis + object for axis in self.axes], name=self.name)
		elif isinstance(object, Point):
			return self.__class__([self.axes[i] + object.axes[i] for i in range(max(self.dimension, object.dimension))], name=self.name)
		elif isinstance(object, (tuple, list, np.ndarray)):
			object = AxesList(object)
			return self.__class__([self.axes[i] + object[i] for i in range(max(self.dimension, len(object)))], name=self.name)
		elif isinstance(object, Vector):
			return self.__class__([self.axes[i] + (object.pos2.axes[i] - object.pos1.axes[i]) for i in range(max(self.dimension, object.dimension))])
		else:
			raise ValueError(f"Unsupported operand type(s) for +: '{self.__class__.__name__}' and '{type(object).__name__}'")

	def __radd__(self, object: Union[int, float]) -> 'Point':
		return self.__add__(object)

	def __sub__(self, object: Union[int, float]) -> 'Point':
		if isinstance(object, (int, float)):
			return self.__class__([axis - object for axis in self.axes], name=self.name)
		elif isinstance(object, Point):
			return self.__class__([self.axes[i] - object.axes[i] for i in range(max(self.dimension, object.dimension))], name=self.name)
		elif isinstance(object, (tuple, list, np.ndarray)):
			object = AxesList(object)
			return self.__class__([self.axes[i] - object[i] for i in range(max(self.dimension, len(object)))], name=self.name)
		elif isinstance(object, Vector):
			return self.__class__([self.axes[i] - (object.pos2.axes[i] - object.pos1.axes[i]) for i in range(max(self.dimension, object.dimension))], name=self.name)
		else:
			raise ValueError(f"Unsupported operand type(s) for -: '{self.__class__.__name__}' and '{type(object).__name__}'")

	def __mul__(self, object: Union[int, float]) -> 'Point':
		if isinstance(object, (int, float)):
			return self.__class__([axis * object for axis in self.axes], name=self.name)
		elif isinstance(object, Point):
			return self.__class__([self.axes[i] * object.axes[i] for i in range(max(self.dimension, object.dimension))], name=self.name)
		elif isinstance(object, (tuple, list, np.ndarray)):
			object = AxesList(object)
			return self.__class__([self.axes[i] * object[i] for i in range(max(self.dimension, len(object)))], name=self.name)
		else:
			raise ValueError(f"Unsupported operand type(s) for *: '{self.__class__.__name__}' and '{type(object).__name__}'")

	def __truediv__(self, object: Union[int, float]) -> 'Point':
		try:
			if isinstance(object, (int, float)):
				return self.__class__([axis / object for axis in self.axes], name=self.name)
			elif isinstance(object, Point):
				return self.__class__([self.axes[i] / object.axes[i] for i in range(max(self.dimension, object.dimension))], name=self.name)
			elif isinstance(object, (tuple, list, np.ndarray)):
				object = AxesList(object)
				return self.__class__([self.axes[i] / object[i] for i in range(max(self.dimension, len(object)))], name=self.name)
			else:
				raise ValueError(f"Unsupported operand type(s) for /: '{self.__class__.__name__}' and '{type(object).__name__}'")
		except ZeroDivisionError:
			raise ZeroDivisionError(f'ZeroDivisionError: {self} / {object}')

	def __floordiv__(self, object: Union[int, float]) -> 'Point':
		try:
			if isinstance(object, (int, float)):
				return self.__class__([axis // object for axis in self.axes], name=self.name)
			elif isinstance(object, Point):
				return self.__class__([self.axes[i] // object.axes[i] for i in range(max(self.dimension, object.dimension))], name=self.name)
			elif isinstance(object, (tuple, list, np.ndarray)):
				object = AxesList(object)
				return self.__class__([self.axes[i] // object[i] for i in range(max(self.dimension, len(object)))], name=self.name)
			else:
				raise ValueError(f"Unsupported operand type(s) for //: '{self.__class__.__name__}' and '{type(object).__name__}'")
		except ZeroDivisionError:
			raise ZeroDivisionError(f'ZeroDivisionError: {self} // {object}')

	def __pow__(self, object: Union[int, float]) -> 'Point':
		if isinstance(object, (int, float)):
			return self.__class__([axis ** object for axis in self.axes], name=self.name)
		elif isinstance(object, Point):
			return self.__class__([self.axes[i] ** object.axes[i] for i in range(max(self.dimension, object.dimension))], name=self.name)
		elif isinstance(object, (tuple, list, np.ndarray)):
			object = AxesList(object)
			return self.__class__([self.axes[i] ** object[i] for i in range(max(self.dimension, len(object)))], name=self.name)
		else:
			raise ValueError(f"Unsupported operand type(s) for **: '{self.__class__.__name__}' and '{type(object).__name__}'")

	def __contains__(self, object):
		if isinstance(object, (Primitive, Shape)):
			return object.intersects(self)
		else:
			return object in list(self)

	def __len__(self):
		return len(self.axes)

	def __neg__(self):
		return self.__class__([-axis[i] for axis in self.axes], name=self.name)
	def __pos__(self):
		return self

	def __getitem__(self, i):
		return self.axes[i]

	def __str__(self):
		return self.name + str(self.axes)
	def __repr__(self):
		return f'Point{self.dimension}D({self.axes}, name="{self.name}")'


class Line(Primitive):
	def __init__(self, pos1: Union[Point, list, tuple], pos2: Union[Point, list, tuple], name: str = 'Line', color: str = 'y'):
		if isinstance(pos1, (tuple, list, np.ndarray)):
			pos1 = Point(*pos1)
		if isinstance(pos2, (tuple, list, np.ndarray)):
			pos2 = Point(*pos2)

		self.pos1, self.pos2 = reduse_axeslists(pos1.axes, pos2.axes)
		self.pos1, self.pos2 = Point(self.pos1, name=pos1.name), Point(self.pos2, name=pos2.name)
		self.name = name
		self.color = color

		if list(self.pos1.axes) == list(self.pos2.axes):
			raise ValueError(f'Expected 2 different Points, got equal: {pos1}; {pos2}')

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
				self.k = ( self.pos1.y - self.pos2.y ) / ( self.pos1.x - self.pos2.x )
			
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
				return [object.pos]
			else:
				return []

		elif isinstance(object, (Ray, Vector, Segment, Line)):
			true_dim_indexes_1 = [i for i, (p1, p2) in enumerate(zip(self.pos1.axes, self.pos2.axes)) if p1 != p2]
			true_dim_indexes_2 = [i for i, (p1, p2) in enumerate(zip(object.pos1.axes, object.pos2.axes)) if p1 != p2]
			if self.true_dimension == object.true_dimension == 1:
				# If lines has only 1 non zero axis

				if true_dim_indexes_1 == true_dim_indexes_2:
					# Same non zero axis. It is intersection on the same line. Returns intersection point, segment, ray, line or vector
					if self.pos1 in object and self.pos2 in object:
						if isinstance(self, Segment) and isinstance(object, Segment) or isinstance(self, Vector) and isinstance(object, Vector) or iisinstance(self, Ray) and isinstance(object, Ray) or isinstance(self, Line) and isinstance(object, Line):
							# If 2 equals lines/rays/segements/vectors returns itself
							return [self.copy()]
						elif isinstance(self, Line): # if 1 line and 1 Ray/Segment/Vector returns Ray/Segment/Vector because it is less
							return [object.copy()]
						elif isinstance(object, Line): # if 1 Ray/Segment/Vector and 1 line returns Ray/Segment/Vector because it is less
							return [self.copy()]
						elif isinstance(self, Ray) and isinstance(object, (Segment, Vector)): # if 1 ray and 1 segment/vector returns segment/vector because it is less
							return [object.copy()]
						elif isinstance(self, (Segment, Vector)) and isinstance(object, Ray): # if 1 segment/vector and 1 ray returns segment/vector because it is less
							return [self.copy()]

					elif object.pos1 in self and object.pos2 in self:
						if isinstance(self, Segment) and isinstance(object, Segment) or isinstance(self, Vector) and isinstance(object, Vector) or iisinstance(self, Ray) and isinstance(object, Ray) or isinstance(self, Line) and isinstance(object, Line):
							# If 2 equals lines/rays/segements/vectors returns itself
							return [self.copy()]
						elif isinstance(self, Line): # if 1 line and 1 Ray/Segment/Vector returns Ray/Segment/Vector because it is less
							return [object.copy()]
						elif isinstance(object, Line): # if 1 Ray/Segment/Vector and 1 line returns Ray/Segment/Vector because it is less
							return [self.copy()]
						elif isinstance(self, Ray) and isinstance(object, (Segment, Vector)): # if 1 ray and 1 segment/vector returns segment/vector because it is less
							return [object.copy()]
						elif isinstance(self, (Segment, Vector)) and isinstance(object, Ray): # if 1 segment/vector and 1 ray returns segment/vector because it is less
							return [self.copy()]

					elif self.pos1 in object:
						if object.pos1 in self:
							ion = object.pos1
						elif object.pos2 in self:
							ion = object.pos2
						else:
							return [self.pos1.copy]

						if isinstance(self, Ray) and isinstance(object, Ray):
							vec1, vec2 = self.vector, object.vector
							if vec1.compare_directions(vec2):
								return [Ray(self.pos1, ion, name=f'{self.name}_{object.name}_intersection')]
							elif vec1.is_reverse_direction(vec2):
								return [Segment(self.pos1, ion, name=f'{self.name}_{object.name}_intersection')]
							else:
								return []
						else:
							return [Segment(self.pos1, ion, name=f'{self.name}_{object.name}_intersection')]

					elif object.pos1 in self:
						if self.pos2 in object:
							ion = self.pos2
						elif self.pos1 in object:
							ion = self.pos1
						else:
							return [object.pos1.copy]

						if isinstance(self, Ray) and isinstance(object, Ray):
							vec1, vec2 = self.vector, object.vector
							if vec1.compare_directions(vec2):
								return [Ray(self.pos1, ion, name=f'{self.name}_{object.name}_intersection')]
							elif vec1.is_reverse_direction(vec2):
								return [Segment(self.pos1, ion, name=f'{self.name}_{object.name}_intersection')]
							else:
								return []
						else:
							return [Segment(self.pos1, ion, name=f'{self.name}_{object.name}_intersection')]

					return []

				else: # Different non zero axes. Lines are perpendicular
					ion = [0] * max(self.dimension, object.dimension)

					ion[true_dim_indexes_1[0]] = object.pos1[true_dim_indexes_1[0]]
					ion[true_dim_indexes_2[0]] = self.pos1[true_dim_indexes_2[0]]

					point = Point(ion, name=f'{self.name}_{object.name}_intersection')
					return [point] if point in self and point in object else []

			elif self.true_dimension == 1 and object.true_dimension == 2:
				# If first line has only 1 non zero axis but second has 2
				y = self.pos1[true_dim_indexes_2[1]]
				second = Line(
					Point[ object.pos1[true_dim_indexes_2[0]], object.pos1[true_dim_indexes_2[1]] ],
					Point[ object.pos2[true_dim_indexes_2[0]], object.pos2[true_dim_indexes_2[1]] ],
				)
				point = Point(second.x_from_y(y), y, name=f'{self.name}_{object.name}_intersection')
				return [point] if point in object else []
				# Here is must be "if point in self and point in object", I dont know why, self.intersects(point) returns None, that means NO INTERSECTION.
				# Point intersection can be only empty list or List[Point], but this is NoneType. WHY?! I dont know. Maybe I am blind and don't see the reason
				# Try print( self.intersects(point) ) after "elif" if you dont believe me

			elif self.true_dimension == 2 and object.true_dimension == 1:
				# Returns result from last IF statement by changing "self -> object" to "object -> self"
				return object.intersects(self)

			elif self.true_dimension == object.true_dimension == 2:
				# If lines has only 2 non zero axes
				
				if true_dim_indexes_1 == true_dim_indexes_2:
					# It means that line1 and line2 has equal 2 axes and lies on the same plane and we can calculate intersection as in 2D space
					# We can ignore first and last zero axes [0,0,0, ..., axis1, ..., axis2, ..., 0,0,0]: There is 2 axes that makes 2D space
					true_dim_indexes = true_dim_indexes_1
					line2D_1 = Line(
						Point[ self.pos1[true_dim_indexes[0]], self.pos1[true_dim_indexes[1]] ],
						Point[ self.pos2[true_dim_indexes[0]], self.pos2[true_dim_indexes[1]] ],
					)
					line2D_2 = Line(
						Point[ object.pos1[true_dim_indexes[0]], object.pos1[true_dim_indexes[1]] ],
						Point[ object.pos2[true_dim_indexes[0]], object.pos2[true_dim_indexes[1]] ],
					)
					x = (line2D_1.m - line2D_2.m) / (line2D_2.k - line2D_1.k)
					y = line2D_1.k * x + line2D_1.m

					# Returns (x,y) with first zero axes that have been ignored
					point = Point([0] * true_dim_indexes[0] +
						[x] +
						[0] * (true_dim_indexes[1] - true_dim_indexes[0] - 1) +
						[y] +
						[0] * (self.true_dimension - true_dim_indexes[1] - 1),
						name=f'{self.name}_{object.name}_intersection'
					)
					return [point] if point in self and point in object else []
			else:
				# Intersection of multidimension lines is arithmetic mean of Point list that are intersections of 2D projects
				dim = max(self.dimension, object.dimension)

				self2ds = self.projects_2D(max_dimension=dim)
				obj2ds = object.projects_2D(max_dimension=dim)

				ios = []
				for projected_line1, projected_line2 in zip(obj2ds, self2ds):
					if projected_line1 in projected_line2:
						ios.append(projected_line1.intersects(projected_line2)[0])

				mean = sum(ios)/2
				return [mean] if mean in self and mean in object else []

		elif hasattr(object, 'intersects') and not isinstance(object, Primitive):
			return object.intersects(self)

		else:
			raise ValueError(f"Invalid argument type for 'object'. Expected types are Union[Primitive, Point, list, tuple], but received {type(object)}.")

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
				
				lines.append(Line(pos1, pos2 + vector))
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
				return (y - self.m) / self.k if self.intersects([(y - self.m) / self.k, y]) or not return_none else None

	def is_parallel(self, object: Union['Line', 'Segment', 'Ray', 'Vector', list, tuple]) -> bool:
		return self.vector.is_parallel(object.vector)

	def is_perpendicular(self, object: Union['Line', 'Segment', 'Ray', 'Vector', list, tuple]) -> bool:
		return self.vector.is_perpendicular(object.vector)

	# Moves the Line. New Line will intersects "point" in argument
	def at_pos(self, point: Point) -> 'Line':
		if isinstance(point, (list, tuple, np.ndarray)):
			point = Point(point)
			
		shift = Point([ point[i] - self.pos1[i] for i in range(self.dimension) ])
		return self.__class__(self.pos1 + shift, self.pos2 + shift, name=self.name)

	def copy(self) -> 'Line':
		return self.__class__(self.pos1, self.pos2, name=self.name)

	# Cuts from pos1 and pos2 unnecessary axes
	def project_to(self, dimension: int) -> 'Line':
		if dimension == 0:
			return Point(name=self.name)
		elif dimension == 1:
			return Line(self.pos1.x, self.pos2.x, name=self.name)
		elif dimension >= 2:
			if dimension >= self.dimension:
				return self.copy()
			else:
				return self.__class__(list(self.pos1.axes[:dimension]), list(self.pos2.axes[:dimension]), name=self.name)

	# Convertor
	@property
	def to_segment(self):
		return Segment(self.pos1, self.pos2, name=self.name)
	@property
	def to_vector(self):
		return Vector(self.pos1, self.pos2, name=self.name)
	@property
	def to_ray(self):
		return Ray(self.pos1, self.pos2, name=self.name)
	@property
	def to_line(self):
		return Line(self.pos1, self.pos2, name=self.name)

	# Only for 2D Lines. Mirror with center in (0,0)
	@property
	def mirror_y(self):
		return Line.by_func(lambda x: -self.k * x + self.m)
	@property
	def mirror_x(self):
		return Line.by_func(lambda x: -self.k * x - self.m)

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
				return Angle(self.pos2, self.pos1, second_pos, name=f'{self.name}_angle')

	# The maximum dimension of 2 points
	@property
	def dimension(self) -> int:
		return max(len(self.pos1.axes), len(self.pos2.axes))

	# The count of non zero axes
	@property
	def true_dimension(self) -> int:
		return len([ (i, j) for i, j in zip(self.pos1.axes, self.pos2.axes) if i != j ])

	# To create new Space in perpendicular
	def _gram_schmidt(self, vectors: List[np.ndarray]) -> List[np.ndarray]:
		ortho_vectors = []
		for v in vectors:
			w = v - sum(np.dot(v, u) * u for u in ortho_vectors)
			if (w > 1e-10).any():
				ortho_vectors.append(w / np.linalg.norm(w))
		return ortho_vectors

	@property
	def perpendicular(self) -> Union['Space', 'Line']:
		if self.dimension >= 3:
			direction_vector_np = self.vector.pos2.axes

			identity_matrix = np.eye(self.dimension)

			vectors = [direction_vector_np] + [identity_matrix[i] for i in range(self.dimension) if not np.allclose(identity_matrix[i], direction_vector_np)]
			ortho_vectors = self._gram_schmidt(np.array(vectors))[1:]  # Exclude the direction vector itself

			return Space(self.center, [Vector(self.center, self.center + Point(v)) for v in ortho_vectors], name=f'{self.name}_perpendicular_space')
		else:
			segment = Point[0,0].height_to(self.at_pos([1,0])) # Moves the Line in ".at_pos([1,0])" because Line with intersection in [0,0] will makes Segment[(0,0), (0,0)]
			segment.name = f'{self.name}_perpendicular'
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
			vec = Vector([0,0], self.pos2 - self.pos1, name=f'{self.name}_vector')
			return vec if from_zero else vec.at_pos(self.pos1)
		else:
			self.copy()

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
				return f'y(x) = { self.k if self.k != 1 else "" }x{ (f" + {self.m}" if self.m > 0 else f" - {-1*self.m}") if self.m else "" }'
		else:
			return f'f(t) = {self.pos1} + {self.vector.short_view} * t'

	@property
	def center(self) -> Point:
		return Point([ (self.pos1[i] + self.pos2[i])/2 for i in range(self.dimension) ], name=f'{self.name}_center')

	@staticmethod
	def by_func(func: callable, name: str = 'Line') -> 'Line':
		random_x = float(r.randint(-10, 10))
		return Line(Point(random_x, func(random_x)), Point(random_x+1, func(random_x+1)), name=name)

	@staticmethod
	def by_angle(pos1: Union[tuple, list, 'Point'], angle: int, name: str = 'Line') -> 'Line':
		if isinstance(pos1, (tuple, list, np.ndarray)):
			pos1 = Point(*pos1)

		k = tan(radians(angle))
		m = pos1.y - pos1.x * k
		return Line.by_func( lambda x: k * x + m )

	def __contains__(self, object):
		return self.intersects(object)

	def __add__(self, i) -> 'Line':
		if isinstance(i, (Point, list, tuple, np.ndarray)):
			return self.__class__(self.pos1 + i, self.pos2 + i, name=self.name)
	def __sub__(self, i) -> 'Line':
		if isinstance(i, (Point, list, tuple, np.ndarray)):
			return self.__class__(self.pos1 - i, self.pos2 - i, name=self.name)

	def __call__(self, t: int, return_none: bool = False) -> float:
		if self.dimension == 1:
			return self.pos1.x
		elif self.dimension >= 2:
			return self.pos1 + self.get_vector(from_zero=False).pos2 * t

	def __str__(self):
		return f'{self.name}[({self.pos1} -> {self.pos2}), {self.as_geometry}]'
	def __repr__(self):
		return f'{self.__class__.__name__}{self.dimension}D({self.pos1}, {self.pos2}, name="{self.name}")'

class Segment(Line):
	def __init__(self, pos1: Union[Point, list, tuple], pos2: Union[Point, list, tuple], name: str = 'Segment', color: str = 'g'):
		self.color = color
		super().__init__(pos1, pos2, name=name, color=color)

	def intersects(self, object: Union[Primitive, Point, list, tuple]) -> Point:
		if isinstance(object, (list, tuple, np.ndarray)):
			object = Point(object)

		if isinstance(object, Point):
			if all([ object[i] >= min(self.pos1[i], self.pos2[i]) and object[i] <= max(self.pos1[i], self.pos2[i]) for i in range(self.dimension) ]):
				return super().intersects(object)

		elif hasattr(object, 'intersects') and not isinstance(object, Primitive):
			return object.intersects(self)

		else:
			return super().intersects(object)

	@property
	def length(self) -> float:
		return sqrt(sum([ (self.pos2[i] - self.pos1[i])**2 for i in range(self.dimension) ]))

	@staticmethod
	def by_func(func: callable, length: int, name: str = 'Segment') -> 'Segment':
		from .shapes2d import Circle

		random_x = float(r.randint(-10, 10))
		center = Point(random_x, func(random_x))

		line = Line(center, Point(random_x+1, func(random_x+1)))
		points = Circle(center, length).intersects(line)

		return Segment([1.0,func(1)], points[0], name=name)

	@staticmethod
	def by_angle(pos1: Union[tuple, list, 'Point'], angle: int, length: int, name: str = 'Segment') -> 'Segment':
		if isinstance(pos1, (tuple, list, np.ndarray)):
			pos1 = Point(*pos1)

		k = tan(radians(angle))
		m = pos1.y - pos1.x * k
		line = Line.by_func( lambda x: k * x + m )
		points = Circle(center, length).intersects(line)

		return Segment([1.0,func(1)], points[0], name=name)

	@property
	def random_point(self) -> 'Point':
		return self( round(r.uniform(0,1), 2) )

class Ray(Line):
	def __init__(self, pos1: Union[Point, list, tuple], pos2: Union[Point, list, tuple], name: str = 'Ray', color: str = 'r'):
		super().__init__(pos1, pos2, name=name, color=color)
	
	def intersects(self, object: Union[Primitive, Point, list, tuple]) -> Point:
		if isinstance(object, (list, tuple, np.ndarray)):
			object = Point(*object)

		if isinstance(object, Point):
			for i in range(self.dimension):
				if self.vector.pos2[i] == 0:
					if object[i] != self.pos1[i]:
						return []
				else:
					t = (object[i] - self.pos1[i]) / self.vector.pos2[i]
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
				return f'y(x) = { self.k if self.k != 1 else "" }x{ (f" + {self.m}" if self.m > 0 else f" - {-1*self.m}") if self.m else "" }'
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

	def by_func(self, *args, **kwargs):
		raise ValueError(f'"by_func" does not fit the {self.__class__} class')
	def by_angle(self, *args, **kwargs):
		raise ValueError(f'"by_func" does not fit the {self.__class__} class')

class VectorMeta(type):
	def __getitem__(cls, pos):
		return cls([0] * len(pos), pos)
class Vector(Segment, metaclass=VectorMeta):
	def __init__(self, *args, name: str = "Vector", color: str = 'c', **kwargs):
		super().__init__(*args, name=name, color=color, **kwargs)

	def dot(self, vector) -> float:
		return sum([ vector.to_zero.pos2[i] * self.to_zero.pos2[i] for i in range(self.dimension) ])

	# True if vectors have an equal direction else False
	def compare_directions(self, vector: Union['Vector', 'Point', tuple, list]) -> bool:
		if isinstance(vector, (tuple, list, Point, np.ndarray)):
			vector = Vector[vector]

		axes = eq_len_axeslists(self.to_zero.pos2.axes, vector.to_zero.pos2.axes)
		return (bool(np.dot(np.array(axes[0]), np.array(axes[1])) > 0) and self.is_parallel(vector)) or vector.direction == 'point' or self.direction == 'point'

	# True vector*-1 has equal direction to self else False
	def is_reverse_direction(self, vector: Union['Vector', 'Point', tuple, list]) -> bool:
		if isinstance(vector, (tuple, list, Point, np.ndarray)):
			vector = Vector[vector]

		axes = eq_len_axeslists(self.to_zero.pos2.axes, vector.to_zero.pos2.axes)
		return (not self.compare_directions(vector)) and self.is_parallel(vector)

	def is_perpendicular(self, vector: Union['Vector', 'Point', tuple, list, 'Line', 'Ray', 'Vector', 'Segment']) -> bool:
		if isinstance(vector, (tuple, list, Point, np.ndarray)):
			vector = Vector[vector]
		elif isinstance(vector, (Segment, Line, Ray)):
			vector = vector.vector

		axes = eq_len_axeslists(self.to_zero.pos2.axes, vector.to_zero.pos2.axes)
		return round(np.dot(np.array(axes[0]), np.array(axes[1])), 8) == 0

	def is_parallel(self, vector: Union['Vector', 'Point', tuple, list, 'Line', 'Ray', 'Vector', 'Segment']) -> bool:
		if isinstance(vector, (tuple, list, Point, np.ndarray)):
			vector = Vector[vector]
		elif isinstance(vector, (Segment, Line, Ray)):
			vector = vector.vector

		zero_vec1, zero_vec2 = self.to_zero.pos2, vector.to_zero.pos2
		return (zero_vec1 in Line([0,0], zero_vec2)) or vector.direction == 'point' or self.direction == 'point'

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
				return f'y(x) = { self.k if self.k != 1 else "" }x{ (f" + {self.m}" if self.m > 0 else f" - {-1*self.m}") if self.m else "" }'
		else:
			return f'r(t) = {self.pos1} + {f"{self.name}[({self.pos1} -> {self.pos2})]" if self.pos1 != 0 else f"{self.name}[{self.pos2}]"} * t'

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
			new_pos2 = Point(*[self.pos1[i] + (self.pos2[i] - self.pos1[i]) / length for i in range(self.dimension)])
			return Vector(self.pos1, new_pos2, name=f'{self.name}_normalized')
		else:
			return self

	@staticmethod
	def by_func(func: callable, length: int, name: str = 'Vector') -> 'Vector':
		from .shapes2d import Circle

		random_x = float(r.randint(-10, 10))
		center = Point(random_x, func(random_x))

		line = Line(center, Point(random_x+1, func(random_x+1)))
		points = Circle(center, length).intersects(line)

		return Vector([1.0,func(1)], points[0], name=name)

	@staticmethod
	def by_angle(pos1: Union[tuple, list, 'Point'], angle: int, length: int, name: str = 'Segment') -> 'Segment':
		if isinstance(pos1, (tuple, list, np.ndarray)):
			pos1 = Point(*pos1)

		k = tan(radians(angle))
		m = pos1.y - pos1.x * k
		line = Line.by_func( lambda x: k * x + m )
		points = Circle(center, length).intersects(line)
		
		return Vector([1.0,func(1)], points[0], name=name)

	def __neg__(self) -> 'Vector':
		return self*-1
	def __pos__(self) -> 'Vector':
		return self.copy()

	def __add__(self, vector) -> 'Vector':
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x + vector, self.pos2.y + vector), name=self.name)
		elif isinstance(vector, Vector):
			return Vector(self.pos1, vector.pos2, name=self.name)

	def __sub__(self, vector) -> 'Vector':
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x - vector, self.pos2.y - vector), name=self.name)
		elif isinstance(vector, Vector):
			return self + (-vector)

	def __mul__(self, vector):
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point([ self.pos1[i] + (self.pos2[i] - self.pos1[i]) * vector for i in range(self.dimension) ]), name=self.name)
		elif isinstance(vector, Vector):
			if self.dimension == 2 and vector.dimension == 2:
				return Vector([0,0], self.pos2[0] * vector.pos2[1] - self.pos2[1] * vector.pos2[0], name=self.name)
			elif self.dimension == 3 and vector.dimension == 3:
				x = self.pos2[1] * vector.pos2[2] - self.pos2[2] * vector.pos2[1]
				y = self.pos2[2] * vector.pos2[0] - self.pos2[0] * vector.pos2[2]
				z = self.pos2[0] * vector.pos2[1] - self.pos2[1] * vector.pos2[0]
				return Vector([0, 0, 0], [x, y, z], name=self.name)

	def __truediv__(self, vector) -> 'Vector':
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x / vector, self.pos2.y / vector), name=self.name)
		elif isinstance(vector, Vector):
			dim = max(self.dimension, vector.dimension)
			return Vector([0] * dim, self.pos1 + [ self.pos2[i] / vector.pos2[i] for i in range(dim) ], name=self.name)

	def __floordiv__(self, vector) -> 'Vector':
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x // vector, self.pos2.y // vector), name=self.name)
		elif isinstance(vector, Vector):
			dim = max(self.dimension, vector.dimension)
			return Vector([0] * dim, self.pos1 + [ self.pos2[i] // vector.pos2[i] for i in range(dim) ], name=self.name)

	def __pow__(self, vector) -> 'Vector':
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x ** vector, self.pos2.y ** vector), name=self.name)
		elif isinstance(vector, Vector):
			dim = max(self.dimension, vector.dimension)
			return Vector([0] * dim, self.pos1 + [ self.pos2[i] ** vector.pos2[i] for i in range(dim) ], name=self.name)

	def __str__(self):
		if self.pos1 != 0:
			return f'{self.name}[({self.pos1} -> {self.pos2})]'
		else:
			return f'{self.name}[{self.pos2}]'


class Angle:
	def __init__(self, pos1: Union[Point, list, tuple], midpos: Union[Point, list, tuple], pos2: Union[Point, list, tuple], name: str = 'Angle'):
		if isinstance(pos1, (tuple, list, np.ndarray)):
			pos1 = Point(*pos1)
		if isinstance(pos2, (tuple, list, np.ndarray)):
			pos2 = Point(*pos2)
		if isinstance(midpos, (tuple, list, np.ndarray)):
			midpos = Point(*midpos)

		if pos1.pos in [pos2.pos, midpos.pos] or pos2.pos in [pos1.pos, midpos.pos] or midpos.pos in [pos2.pos, pos1.pos]:
			raise ValueError(f'constructor arguments 1, 2 and 3 must be Points with different positions, not {pos1.pos}, {midpos.pos}, {pos2.pos}')

		self.pos1 = pos1
		self.midpos = midpos
		self.pos2 = pos2
		self.name = name

	@property
	def vec1(self) -> Vector:
		return Vector(self.midpos, self.pos1, name=f'{self.name}_vec1')
	@property
	def vec2(self) -> Vector:
		return Vector(self.midpos, self.pos2, name=f'{self.name}_vec2')

	@property
	def cos(self) -> float:
		return 1 if self.pos2 in Segment(self.pos1, self.midpos) or self.pos1 in Segment(self.pos2, self.midpos) else (self.vec1.dot(self.vec2)) / (self.vec1.length * self.vec2.length)
	@property
	def sin(self) -> float:
		return 0 if self.pos2 in Segment(self.pos1, self.midpos) or self.pos1 in Segment(self.pos2, self.midpos) else sqrt(1 - self.cos**2)
	@property
	def tan(self) -> float:
		return 0 if self.pos2 in Segment(self.pos1, self.midpos) or self.pos1 in Segment(self.pos2, self.midpos) else self.sin / self.cos if self.cos != 0 else 0.0

	@property
	def radians(self) -> float:
		return 0 if self.pos2 in Segment(self.pos1, self.midpos) or self.pos1 in Segment(self.pos2, self.midpos) else acos(self.cos)
	@property
	def degrees(self) -> float:
		return 0 if self.pos2 in Segment(self.pos1, self.midpos) or self.pos1 in Segment(self.pos2, self.midpos) else degrees(self.radians)
	@property
	def minutes(self) -> float:
		return 0 if self.pos2 in Segment(self.pos1, self.midpos) or self.pos1 in Segment(self.pos2, self.midpos) else self.degrees * 60
		
	@property
	def bisector(self) -> Ray:
		return Ray(self.midpos, Point([ (self.pos1[i] + self.pos2[i])/2 for i in range(self.dimension) ]), name=f'{self.name}_bisector')

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
		if any([isinstance(pr, Point) for pr in [pr1,pr2]]) or any([not isinstance(pr, Primitive) for pr in [pr1,pr2]]):
			raise ValueError(f'Can not calculate Angle between {pr1} and {pr2}')
		else:
			int_point = pr1.intersects(pr2)[0]
			if int_point:
				pos1, pos2 = pr1.random_point, pr2.random_point
				return Angle(pos1, int_point, pos2, name=f'{pr1.name}_{pr2.name}_angle')

	def __add__(self, num: float) -> float:
		return float(self)+num
	def __sub__(self, num: float) -> float:
		return float(self)-num
	def __mul__(self, num: float) -> float:
		return float(self)*num
	def __truediv__(self, num: float) -> float:
		return float(self)/num
	def __floordiv__(self, num: float) -> float:
		return float(self)//num
	def __pow__(self, num: float) -> float:
		return float(self)**num

	def __float__(self, num: float):
		return float(self.degrees)
	def __int__(self, num: float):
		return float(self.degrees)

	def __str__(self):
		return f'{self.name}[({self.pos1} -> {self.midpos} -> {self.pos2}), {int(self.degrees)} degrees]'
	def __repr__(self):
		return f'Angle({self.pos1} <- {self.midpos} -> {self.pos2}, name="{self.name}")'


class AffineSpace:
	def __init__(self, origin: Union['Point', tuple, list], vectors: List['Vector'], name: str = 'AffineSpace'):
		if isinstance(origin, (tuple, list, np.ndarray)):
			origin = Point(origin)
		self.origin = origin

		if all([ isinstance(vector, (Point, tuple, list, np.ndarray)) for vector in vectors ]):
			vectors = [ Vector(origin, vector, name=f'{name}_vector{i}').to_zero for i, vector in enumerate(vectors) ]

		if all([ isinstance(vector, Vector) for vector in vectors ]):
			self.vectors = [ vector.to_vector for vector in vectors ]
			self.name = name
		else:
			raise ValueError(f'"vectors" elements must be list of points or vectors, not {vectors}')

	def transform(self, point: Union['Point', list, tuple]) -> 'Point':
		if isinstance(point, (list, tuple, np.ndarray)):
			point = Point(*point)

		basis_matrix = np.array([vec.pos2.axes + [0] * (self.dimension-vec.pos2.dimension) for vec in self.vectors])
		return Point(self.origin.axes + np.dot(basis_matrix.T, point.axes), f'{self.name}_point')

	@property
	def dimension(self) -> int:
		return len(self.vectors)

	@property
	def as_geometry(self) -> str:
		return f'{self.vector} * (x - {self.origin})'

	def __str__(self):
		return f"{self.name}[{self.origin}, {self.dimension}D]"
	def __repr__(self):
		return f"{self.__class__.__name__}{self.dimension}D({self.origin}, {self.vectors})"

ASpace = AffineSpace

class Space(AffineSpace):
	def __init__(self, origin: 'Point', points: List['Point'], name: str = 'Space'):
		super().__init__(origin, points, name=name)

		if any([ not self.vectors[i].is_perpendicular(self.vectors[i-1]) for i in range(len(self.vectors)) ]):
			raise ValueError(f'Vectors must be perpendiculars')
		if len( set([ round(self.vectors[i].length, 8) for i in range(len(self.vectors)) ]) ) > 1:
			raise ValueError(f'Vectors must have equal lengths')