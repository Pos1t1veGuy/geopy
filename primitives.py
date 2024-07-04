from typing import *

from typing import *
from math import sqrt, acos, degrees, tan, radians
from string import ascii_lowercase as ABCD
import random as r
import numpy as np

from .math import *

# Here is geometry primitives: Point, Line, Ray, Segment, Vector and Angle classes

def line_by_function(func: str, name: str = 'Line') -> 'Line':
	return Line(Point(1.0, func(1)), Point(2.0, func(2)), name=name)

def line_by_angle(pos1, angle: int, name: str = 'Line') -> 'Line':
	if isinstance(pos1, (tuple, list)):
		pos1 = Point(*pos1)

def get_angle_between(pr1: 'Primitive', pr2: 'Primitive') -> 'Angle':
	if any([isinstance(pr, Point) for pr in [pr1,pr2]]) or any([not isinstance(pr, Primitive) for pr in [pr1,pr2]]):
		raise ValueError(f'Can not calculate Angle between {pr1} and {pr2}')
	else:
		int_point = pr1.intersects(pr2)
		if int_point:
			pos1, pos2 = pr1.random_point, pr2.random_point
			return Angle(pos1, int_point, pos2)


def eq_len_axeslists(*args: 'AxesList') -> List[list]:
	max_dimension = max([ len(l) for l in args ])
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
		if isinstance(pos, (tuple, list)):
			return cls(*pos)
		else:
			return cls(pos)
class Point(Primitive, metaclass=PointMeta):
	def __init__(self, *args, name: str = 'Point'):
		global letters
		self.axes = AxesList([])

		if len(args) == 1 and isinstance(args[0], (tuple, list, np.ndarray)):
			args = args[0]

		for i, axis in enumerate(args):
			if isinstance(axis, (float, int)):
				if not hasattr(self, letters[i]):
					setattr(self, letters[i], axis)
				else:
					c = 0
					while hasattr(self, letters[i] + str(c)):
						c += 1
					setattr(self, letters[i], letters[i] + str(c))
				self.axes.append(float(axis))
			else:
				raise ValueError(f"Invalid initialization arguments for 'Point': {args}")

		self.name = name

	def height_to(self, object: Primitive) -> 'Line':
		if object.k:
			k, m = -1/object.k, self.y - (-1/object.k * self.x)
			return line_by_function( lambda x: float(k * x + m) )
		elif object.direction in ['up', 'down', 'vertial']:
			return Line(self, self + [1,0])
		elif object.direction in ['right', 'left', 'horizontal']:
			return Line(self, self + [0,1])

	def copy(self) -> 'Point':
		return Point(*self.axes, name=self.name)

	@staticmethod
	def random(pos1: Union['Point', tuple, list], pos2: Union['Point', tuple, list], uniform: bool = True) -> 'Point':
		# pos1 and pos2 must be Point object, returns random point from rectangle, box etc. [pos1 x pos2]
		if isinstance(pos1, (tuple, list)):
			pos1 = Point(pos1)
		if isinstance(pos2, (tuple, list)):
			pos2 = Point(pos2)
		max_dimension = max(pos1.dimension, pos2.dimension)

		return Point( r.uniform(pos1.axes[i], pos2.axes[i]) for i in max_dimension ) if uniform else Point( r.randint(pos1.axes[i], pos2.axes[i]) for i in max_dimension )

	@property
	def dimension(self) -> int:
		return len(self.axes)

	@property
	def pos(self) -> List[float]:
		return self.axes

	def __getitem__(self, i):
		return self.axes[i]

	def __eq__(self, obj):
		if obj == 0:
			return all([ axis == 0 for axis in self.axes ])
		elif isinstance(obj, Point):
			max_dimension = max( len(self.axes), len(obj.axes) )
			return self.axes.as_list(length=max_dimension) == obj.axes.as_list(length=max_dimension)
		else:
			return self == obj

	def __add__(self, object: Union[int, float]) -> 'Point':
		if isinstance(object, (int, float)):
			return Point([axis + object for axis in self.axes])
		elif isinstance(object, Point):
			return Point([self.axes[i] + object.axes[i] for i in range(max(self.dimension, object.dimension))])
		elif isinstance(object, (tuple, list)):
			object = AxesList(object)
			return Point([self.axes[i] + object[i] for i in range(max(self.dimension, len(object)))])
		elif isinstance(object, Vector):
			return Point([self.axes[i] + (object.pos2.axes[i] - object.pos1.axes[i]) for i in range(max(self.dimension, object.dimension))])
		else:
			raise ValueError(f"Unsupported operand type(s) for +: 'Point' and '{type(object).__name__}'")

	def __sub__(self, object: Union[int, float]) -> 'Point':
		if isinstance(object, (int, float)):
			return Point([axis - object for axis in self.axes])
		elif isinstance(object, Point):
			return Point([self.axes[i] - object.axes[i] for i in range(max(self.dimension, object.dimension))])
		elif isinstance(object, (tuple, list)):
			object = AxesList(object)
			return Point([self.axes[i] - object[i] for i in range(max(self.dimension, len(object)))])
		elif isinstance(object, Vector):
			return Point([self.axes[i] - (object.pos2.axes[i] - object.pos1.axes[i]) for i in range(max(self.dimension, object.dimension))])
		else:
			raise ValueError(f"Unsupported operand type(s) for -: 'Point' and '{type(object).__name__}'")

	def __mul__(self, object: Union[int, float]) -> 'Point':
		if isinstance(object, (int, float)):
			return Point([axis * object for axis in self.axes])
		elif isinstance(object, Point):
			return Point([self.axes[i] * object.axes[i] for i in range(max(self.dimension, object.dimension))])
		elif isinstance(object, (tuple, list)):
			object = AxesList(object)
			return Point([self.axes[i] * object[i] for i in range(max(self.dimension, len(object)))])
		else:
			raise ValueError(f"Unsupported operand type(s) for *: 'Point' and '{type(object).__name__}'")

	def __truediv__(self, object: Union[int, float]) -> 'Point':
		try:
			if isinstance(object, (int, float)):
				return Point([axis / object for axis in self.axes])
			elif isinstance(object, Point):
				return Point([self.axes[i] / object.axes[i] for i in range(max(self.dimension, object.dimension))])
			elif isinstance(object, (tuple, list)):
				object = AxesList(object)
				return Point([self.axes[i] / object[i] for i in range(max(self.dimension, len(object)))])
			else:
				raise ValueError(f"Unsupported operand type(s) for /: 'Point' and '{type(object).__name__}'")
		except ZeroDivisionError:
			raise ZeroDivisionError(f'ZeroDivisionError: {self} / {object}')

	def __floordiv__(self, object: Union[int, float]) -> 'Point':
		try:
			if isinstance(object, (int, float)):
				return Point([axis // object for axis in self.axes])
			elif isinstance(object, Point):
				return Point([self.axes[i] // object.axes[i] for i in range(max(self.dimension, object.dimension))])
			elif isinstance(object, (tuple, list)):
				object = AxesList(object)
				return Point([self.axes[i] // object[i] for i in range(max(self.dimension, len(object)))])
			else:
				raise ValueError(f"Unsupported operand type(s) for //: 'Point' and '{type(object).__name__}'")
		except ZeroDivisionError:
			raise ZeroDivisionError(f'ZeroDivisionError: {self} // {object}')

	def __pow__(self, object: Union[int, float]) -> 'Point':
		if isinstance(object, (int, float)):
			return Point([axis ** object for axis in self.axes])
		elif isinstance(object, Point):
			return Point([self.axes[i] ** object.axes[i] for i in range(max(self.dimension, object.dimension))])
		elif isinstance(object, (tuple, list)):
			object = AxesList(object)
			return Point([self.axes[i] ** object[i] for i in range(max(self.dimension, len(object)))])
		else:
			raise ValueError(f"Unsupported operand type(s) for **: 'Point' and '{type(object).__name__}'")

	def __len__(self):
		return len(self.axes)

	def __neg__(self):
		return Point([-axis[i] for axis in self.axes])
	def __pos__(self):
		return self

	def __getitem__(self, i):
		return self.axes[i]
	def __list__(self):
		return list(self.pos)
	def __tuple__(self):
		return tuple(self.axes)

	def __str__(self):
		return self.name + str(self.axes)
	def __repr__(self):
		return f'Point{self.dimension}D({self.axes}, name="{self.name}")'


class Line(Primitive):
	def __init__(self, pos1: Union[Point, list, tuple], pos2: Union[Point, list, tuple], name: str = 'Line'):
		if isinstance(pos1, (tuple, list)):
			pos1 = Point(*pos1)
		if isinstance(pos2, (tuple, list)):
			pos2 = Point(*pos2)

		self.pos1, self.pos2 = reduse_axeslists(pos1.axes, pos2.axes)
		self.pos1, self.pos2 = Point(self.pos1, name=pos1.name), Point(self.pos2, name=pos2.name)
		self.name = name

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
				self.k = ( self.pos1.y - self.pos2.y ) / ( self.pos1.x - self.pos2.x )
			
			if self.pos1.x == self.pos2.x: # if it is vertical line
				self.m = self.pos1.x
			elif pos1.y == pos2.y: # if it is horizontal line
				self.m = self.pos2.y
			else:
				self.m = self.pos1.y - self.k * self.pos1.x

	def intersects(self, object: Union[Primitive, Point, list, tuple]) -> List[Point]:
		if isinstance(object, (list, tuple)):
			object = Point(*object)

		if isinstance(object, Point):
			pos1, direction, point = eq_len_axeslists(self.pos1.axes, self.vector.pos2.axes, object.axes)
			p1, d, p2 = [], [], []

			for i in range(self.dimension):
				if direction[i] != 0:
					p2.append(point[i])
					p1.append(pos1[i])
					d.append(direction[i])

			t_values = (np.array(p2) - np.array(p1)) / np.array(d)
			return np.all(t_values == t_values[0])

		elif isinstance(object, (Ray, Vector, Segment, Line)):
			coefficients = np.zeros((self.dimension, 2))
			constants = np.zeros(self.dimension)

			for i in range(self.dimension):
				coefficients[i] = np.array([self.vector.pos2[i], -object.vector.pos2[i]])
				constants[i] = np.array(object.pos1[i] - self.pos1[i])

			try:
				solution = np.linalg.lstsq(coefficients, constants, rcond=None)[0]
			except np.linalg.LinAlgError:
				raise ValueError(f"Can not find intersection of {self} and {object}")

			t, s = solution

			if isinstance(object, (Segment, Vector)):
				if 0 <= s <= 1:
					return Point(np.array(self.pos1.axes) + t * np.array(self.vector.pos2.axes))
			elif isinstance(object, Ray):
				if s >= 0:
					return Point(np.array(self.pos1.axes) + t * np.array(self.vector.pos2.axes))
			elif isinstance(object, Line):
				return Point(np.array(self.pos1.axes) + t * np.array(self.vector.pos2.axes))

		elif hasattr(object, 'intersects') and not isinstance(object, Primitive):
			return object.intersects(self)

		else:
			raise ValueError(f"Invalid argument type for 'object'. Expected types are Union[Primitive, Point, list, tuple], but received {type(object)}.")

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

	def at_pos(self, point: Point) -> 'Line':
		if isinstance(point, (list, tuple)):
			point = Point(point)
			
		shift = Point([ point[i] - self.pos1[i] for i in range(self.dimension) ])
		return self.__class__(self.pos1 + shift, self.pos2 + shift, name=self.name)

	def copy(self) -> 'Line':
		return self.__class__(self.pos1, self.pos2, name=self.name)

	def project_to(self, dimension: int) -> 'Line':
		if dimension == 0:
			return Point(name=self.name)
		elif dimension == 1:
			return Line([-1], [1])
		elif dimension >= 2:
			if dimension >= self.dimension:
				return self.copy()
			else:
				return Line(list(self.pos1.axes[:dimension]), list(self.pos2.axes[:dimension]), name=self.name)

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

	@property
	def random_point(self) -> 'Point':
		return self( round(r.uniform(0,1), 2) )

	@property
	def angle(self) -> 'Angle':
		if self.direction != 'point':
			if self.dimension == 1:
				return 0
			elif self.dimension == 2:
				second_pos = Point(self.pos2.x + 1, self.pos1.y)
				while second_pos.pos in [self.pos1.pos, self.pos2.pos]:
					second_pos += [1, 0]
				return Angle(self.pos2, self.pos1, second_pos)

	@property
	def dimension(self) -> int:
		return max(len(self.pos1.axes), len(self.pos2.axes))

	@property
	def perpendicular(self) -> 'Line':
		if self.dimension >= 2:
			if self.k:
				k = -1 / self.k
				m = self.center.y - k * self.center.x
				return line_by_function( lambda x: k * x + m )
			elif self.direction == 'vertical':
				return Line(self.center, self.center + [1,0])
			elif self.direction == 'horizontal':
				return Line(self.center, self.center + [0,1])

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
		if not isinstance(self, Vector):
			return Vector[self.pos2 - self.pos1]
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
			return f'r(t) = {self.pos1} + {self.vector} * t'

	def __contains__(self, object):
		return self.intersects(object)

	def __add__(self, i) -> 'Line':
		if isinstance(i, (Point, list, tuple)):
			return self.__class__(self.pos1 + i, self.pos2 + i, name=self.name)
	def __sub__(self, i) -> 'Line':
		if isinstance(i, (Point, list, tuple)):
			return self.__class__(self.pos1 - i, self.pos2 - i, name=self.name)

	def __call__(self, t: int, return_none: bool = False) -> float:
		if self.dimension == 1:
			return self.pos1.x
		elif self.dimension >= 2:
			return self.pos1 + self.vector.pos2 * t

	def __str__(self):
		return f'{self.name}[({self.pos1} -> {self.pos2}), {self.as_geometry}]'
	def __repr__(self):
		return f'{self.__class__.__name__}{self.dimension}D({self.pos1}, {self.pos2}, name="{self.name}")'

class Segment(Line):
	def __init__(self, pos1: Union[Point, list, tuple], pos2: Union[Point, list, tuple], name: str = 'Segment'):
		super().__init__(pos1, pos2, name=name)

	def intersects(self, object: Union[Primitive, Point, list, tuple]) -> Point:
		if isinstance(object, (list, tuple)):
			object = Point(object)

		if isinstance(object, Point):
			for i in range(self.dimension):
				if object[i] < min(self.pos1[i], self.pos2[i]) or object[i] > max(self.pos1[i], self.pos2[i]):
					return False
			return super().intersects(object)

		elif hasattr(object, 'intersects') and not isinstance(object, Primitive):
			return object.intersects(self)

		else:
			return super().intersects(object)

	@property
	def length(self) -> float:
		return sqrt(sum([ (self.pos2[i] - self.pos1[i])**2 for i in range(self.dimension) ]))

	@property
	def center(self) -> Point:
		return Point([ (self.pos1[i] + self.pos2[i])/2 for i in range(self.dimension) ])

class Ray(Line):
	def __init__(self, pos1: Union[Point, list, tuple], pos2: Union[Point, list, tuple], name: str = 'Ray'):
		super().__init__(pos1, pos2, name=name)
	
	def intersects(self, object: Union[Primitive, Point, list, tuple]) -> Point:
		if isinstance(object, (list, tuple)):
			object = Point(*object)

		if isinstance(object, Point):
			for i in range(self.dimension):
				if self.vector.pos2[i] == 0:
					if object[i] != self.pos1[i]:
						return False
				else:
					t = (object[i] - self.pos1[i]) / self.vector.pos2[i]
					if t < 0:
						return False
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

class VectorMeta(type):
	def __getitem__(cls, pos):
		return cls([0,0], pos)
class Vector(Segment, metaclass=VectorMeta):
	def __init__(self, *args, name="Vector", **kwargs):
		super().__init__(*args, name=name, **kwargs)

	def dot(self, vector) -> float:
		return sum([ vector.to_zero.pos2[i] * self.to_zero.pos2[i] for i in range(self.dimension) ])

	def compare_directions(self, vector: Union['Vector', 'Point', tuple, list]) -> bool:
		if isinstance(vector, (tuple, list, Point)):
			vector = Vector[vector]

		axes = eq_len_axeslists(self.to_zero.pos2.axes, vector.to_zero.pos2.axes)
		return np.dot(np.array(axes[0]), np.array(axes[1])) > 0

	def is_perpendicular(self, vector: Union['Vector', 'Point', tuple, list, 'Line', 'Ray', 'Vector', 'Segment']) -> bool:
		if isinstance(vector, (tuple, list, Point)):
			vector = Vector[vector]
		elif isinstance(vector, (Segment, Line, Ray)):
			vector = vector.vector

		axes = eq_len_axeslists(self.to_zero.pos2.axes, vector.to_zero.pos2.axes)
		return np.dot(np.array(axes[0]), np.array(axes[1])) == 0

	def is_parallel(self, vector: Union['Vector', 'Point', tuple, list, 'Line', 'Ray', 'Vector', 'Segment']) -> bool:
		if isinstance(vector, (tuple, list, Point)):
			vector = Vector[vector]
		elif isinstance(vector, (Segment, Line, Ray)):
			vector = vector.vector
		return self.compare_directions(vector) or self.compare_directions(-vector)

	@property
	def vector(self):
		return self.copy()

	@property
	def direction(self) -> str:
		if self.dimension == 0:
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
	def normalize(self) -> 'Vector':
		length = self.length
		if self.direction != 'point':
			return Vector(self.pos1, Point(self.pos1.x + (self.pos2.x - self.pos1.x) / length, self.pos1.y + (self.pos2.y - self.pos1.y) / length))
		else:
			return self

	def __neg__(self) -> 'Vector':
		return self*-1
	def __pos__(self) -> 'Vector':
		return self.copy()

	def __add__(self, vector) -> 'Vector':
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x + vector, self.pos2.y + vector))
		elif isinstance(vector, Vector):
			return Vector(self.pos1 + vector.pos1, self.pos2 + vector.pos2)

	def __sub__(self, vector) -> 'Vector':
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x - vector, self.pos2.y - vector))
		elif isinstance(vector, Vector):
			return Vector(self.pos1 * vector.pos1, self.pos2 * vector.pos2)

	def __mul__(self, vector):
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point([ (self.pos2[i] - self.pos1[i]) * vector for i in range(self.dimension) ])) # self.pos1[i] * 
		elif isinstance(vector, Vector):
			return Vector(self.pos1 * vector.pos1, self.pos2 * vector.pos2)

	def __truediv__(self, vector) -> 'Vector':
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x / vector, self.pos2.y / vector))
		elif isinstance(vector, Vector):
			return Vector(self.pos1 / vector.pos1, self.pos2 / vector.pos2)

	def __floordiv__(self, vector) -> 'Vector':
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x // vector, self.pos2.y // vector))
		elif isinstance(vector, Vector):
			return Vector(self.pos1 // vector.pos1, self.pos2 // vector.pos2)

	def __pow__(self, vector) -> 'Vector':
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x ** vector, self.pos2.y ** vector))
		elif isinstance(vector, Vector):
			return Vector(self.pos1 ** vector.pos1, self.pos2 ** vector.pos2)

	def __str__(self):
		if self.pos1 != 0:
			return f'{self.name}[({self.pos1} -> {self.pos2})]'
		else:
			return f'{self.name}[{self.pos2}]'


class Angle:
	def __init__(self, pos1: Union[Point, list, tuple], midpos: Union[Point, list, tuple], pos2: Union[Point, list, tuple], name: str = 'Angle'):
		if isinstance(pos1, (tuple, list)):
			pos1 = Point(*pos1)
		if isinstance(pos2, (tuple, list)):
			pos2 = Point(*pos2)
		if isinstance(midpos, (tuple, list)):
			midpos = Point(*midpos)

		if pos1.pos in [pos2.pos, midpos.pos] or pos2.pos in [pos1.pos, midpos.pos] or midpos.pos in [pos2.pos, pos1.pos]:
			raise ValueError(f'constructor arguments 1, 2 and 3 must be Points with different positions, not {pos1.pos}, {midpos.pos}, {pos2.pos}')

		self.pos1 = pos1
		self.midpos = midpos
		self.pos2 = pos2
		self.name = name

	@property
	def vec1(self) -> Vector:
		return Vector(self.midpos, self.pos1)
	@property
	def vec2(self) -> Vector:
		return Vector(self.midpos, self.pos2)

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
		return Ray(self.midpos, Point([ (self.pos1[i] + self.pos2[i])/2 for i in range(self.dimension) ]))

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
	def __init__(self, origin: Union['Point', tuple, list], vectors: List['Vector'], name: str = 'Affine'):
		if isinstance(origin, (tuple, list)):
			origin = Point(origin)
		self.origin = origin

		if all([ isinstance(vector, (Point, tuple, list)) for vector in vectors ]):
			vectors = [ Vector(origin, i).to_zero for i in vectors ]

		if all([ isinstance(vector, Vector) for vector in vectors ]):
			self.vectors = [ vector.to_vector.project_to(self.dimension) for vector in vectors ]
			self.name = name
		else:
			raise ValueError(f'"vectors" elements must be list of points or vectors, not {vectors}')

	def transform(self, point: Union[Point, list, tuple]) -> Point:
		if isinstance(point, (list, tuple)):
			point = Point(*point)

		basis_matrix = np.array([vec.pos2.axes + [0] * (self.dimension-vec.pos2.dimension) for vec in self.vectors])
		return Point(self.origin.axes + np.dot(basis_matrix.T, point.axes))

	@property
	def dimension(self) -> int:
		return len(self.origin)

	def __str__(self):
		return f"{self.name}[{self.origin}, {self.dimension}D]"
	def __repr__(self):
		return f"{self.__class__.__name__}{self.dimension}D({self.origin}, {self.vectors})"

ASpace = AffineSpace

class Space(AffineSpace):
	def __init__(self, origin: 'Point', points: List['Point'], name: str = 'Space'):
		super().__init__(origin, vectors, name=name)

		if any([ not self.vectors[i].is_perpendicular(self.vectors[i-1]) for i in range(len(self.vectors)) ]):
			raise ValueError(f'Vectors must be perpendiculars')