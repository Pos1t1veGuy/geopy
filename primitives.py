from typing import *

from math import sqrt, acos, degrees, tan, radians
import random as r

from .math import *

# Here is geometry primitives: Point, Line, Ray, Segment, Vector and Angle classes

def line_by_function(func: str, name: str = 'Line'):
	return Line(Point(1.0, func(1)), Point(2.0, func(2)), name=name)

def line_by_angle(pos1, angle: int, name: str = 'Line'):
	if isinstance(pos1, (tuple, list)):
		pos1 = Point(*pos1)

	k = tan(radians(angle))
	m = pos1.y - pos1.x * k
	return line_by_function( lambda x: k * x + m )


class Primitive:
	...


class PointMeta(type):
	def __getitem__(cls, pos):
		if isinstance(pos, (tuple, list)):
			return cls(*pos)
		else:
			return cls(pos)
class Point(Primitive, metaclass=PointMeta):
	def __init__(self, *args, name: str = 'Point'):
		if len(args) == 0:
			self.x = 0
			self.y = 0
		elif len(args) == 2 and all([ isinstance(num, (int, float)) for num in args ]):
			self.x = float(args[0])
			self.y = float(args[1])
		elif isinstance(args[0], (list, tuple)):
			if len(args[0]) == 2 and all([ isinstance(num, (int, float)) for num in args[0] ]):
				self.x = float(args[0][0])
				self.y = float(args[0][1])
			else:
				raise ValueError(f"Invalid initialization arguments for 'Point'. Expected either two numbers representing coordinates (x, y), or a list/tuple of two numbers, but received {args}")
		else:
			raise ValueError(f"Invalid initialization arguments for 'Point'. Expected either two numbers representing coordinates (x, y), or a list/tuple of two numbers, but received {args}")

		self.name = name

	def height_to(self, object: Primitive):
		k, m = -1/object.k, self.y - (-1/object.k * self.x)
		return line_by_function( lambda x: float(k * x + m) )

	def copy(self):
		return Point(self.x, self.y, name=self.name)

	@staticmethod
	def random(pos1, pos2):
		# pos1 and pos2 must be Point object, returns random point from rectangle [pos1 x pos2]
		return Point(r.uniform(pos1.x, pos2.x), r.uniform(pos1.y, pos2.y))

	@property
	def pos(self):
		return [float(self.x), float(self.y)]

	def __add__(self, object):
		if isinstance(object, int):
			return Point(self.x + object, self.y + object)
		elif isinstance(object, Point):
			return Point(self.x + object.x, self.y + object.y)
		elif isinstance(object, (tuple, list)):
			return Point(self.x + object[0], self.y + object[1])
		elif isinstance(vector, Vector):
			return Point(self.x + (vector.pos2.x - vector.pos1.x), self.y + (vector.pos2.y - vector.pos1.y))
		else:
			raise ValueError(f"Unsupported operand type(s) for +: 'Point' and '{type(object).__name__}'")

	def __sub__(self, object):
		if isinstance(object, int):
			return Point(self.x - object, self.y - object)
		elif isinstance(object, Point):
			return Point(self.x - object.x, self.y - object.y)
		elif isinstance(object, (tuple, list)):
			return Point(self.x - object[0], self.y - object[1])
		elif isinstance(vector, Vector):
			return Point(self.x - (vector.pos2.x - vector.pos1.x), self.y - (vector.pos2.y - vector.pos1.y))
		else:
			raise ValueError(f"Unsupported operand type(s) for -: 'Point' and '{type(object).__name__}'")

	def __mul__(self, object):
		if isinstance(object, int):
			return Point(self.x * object, self.y * object)
		elif isinstance(object, Point):
			return Point(self.x * object.x, self.y * object.y)
		elif isinstance(object, (tuple, list)):
			return Point(self.x * object[0], self.y * object[1])
		else:
			raise ValueError(f"Unsupported operand type(s) for *: 'Point' and '{type(object).__name__}'")

	def __truediv__(self, object):
		if isinstance(object, int):
			return Point(self.x / object, self.y / object)
		elif isinstance(object, Point):
			return Point(self.x / object.x, self.y / object.y)
		elif isinstance(object, (tuple, list)):
			return Point(self.x / object[0], self.y / object[1])
		else:
			raise ValueError(f"Unsupported operand type(s) for /: 'Point' and '{type(object).__name__}'")

	def __floordiv__(self, object):
		if isinstance(object, int):
			return Point(self.x // object, self.y // object)
		elif isinstance(object, Point):
			return Point(self.x // object.x, self.y // object.y)
		elif isinstance(object, (tuple, list)):
			return Point(self.x // object[0], self.y // object[1])
		else:
			raise ValueError(f"Unsupported operand type(s) for //: 'Point' and '{type(object).__name__}'")

	def __pow__(self, object):
		if isinstance(object, int):
			return Point(self.x ** object, self.y ** object)
		elif isinstance(object, Point):
			return Point(self.x ** object.x, self.y ** object.y)
		elif isinstance(object, (tuple, list)):
			return Point(self.x ** object[0], self.y ** object[1])
		else:
			raise ValueError(f"Unsupported operand type(s) for **: 'Point' and '{type(object).__name__}'")

	def __neg__(self):
		return Point(-self.x, -self.y)
	def __pos__(self):
		return self

	def __getitem__(self, i):
		return self.pos[i]
	def __list__(self):
		return self.pos
	def __tuple__(self):
		return self.pos

	def __str__(self):
		return f'{self.name}[{self.x}, {self.y}]'
	def __repr__(self):
		return f'Point({self.x}, {self.y}, name="{self.name}")'


class Line(Primitive):
	def __init__(self, pos1: Union[Point, list, tuple], pos2: Union[Point, list, tuple], name: str = 'Line'):
		if isinstance(pos1, (tuple, list)):
			pos1 = Point(*pos1)
		if isinstance(pos2, (tuple, list)):
			pos2 = Point(*pos2)

		self.pos1, self.pos2 = pos1, pos2
		self.name = name

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

	def at_pos(self, point: Union[Point, list, tuple]):
		if isinstance(point, (tuple, list)):
			point = Point(*point)
		return Line(point, Point(self.pos2.x - self.pos1.x, self.pos2.y - self.pos1.y), name=self.name)

	def intersects(self, object: Union[Primitive, Point, list, tuple]) -> List[Point]:
		if isinstance(object, (list, tuple)):
			object = Point(*object)

		if isinstance(object, Point):
			if self.k:
				return [object] if self.k * object.x + self.m == object.y else []
			elif self.direction == 'vertical':
				return [object] if self.pos1.x == object.x else []
			elif self.direction == 'horizontal':
				return [object] if self.pos1.y == object.y else []

		elif isinstance(object, (Ray, Vector, Segment, Line)):
			if self.k != object.k:
				if object.direction in ['up','down','vertical']:
					x, y = object.pos1.x, self(object.pos1.x)
				elif object.direction in ['left','right','horizontal']:
					x, y = self.x_from_y(object.pos1.y), object.pos1.y

				elif self.direction in ['up','down','vertical']:
					x, y = self.pos1.x, object(self.pos1.x)
				elif self.direction in ['left','right','horizontal']:
					x, y = object.x_from_y(self.pos1.y), self.pos1.y

				else:
					x = (object.m - self.m) / (self.k - object.k)
					y = self(x)

				if not None in [x, y]:
					x, y = float(x), float(y)
					return [Point(x, y)] if Point(x, y) in self and Point(x, y) in object else []
				else:
					return []

			elif object.direction in ['up','down','vertical'] and self.direction in ['up','down','vertical'] or object.direction in ['right','left','horizontal'] and self.direction in ['right','left','horizontal']:
				if (self.direction in ['left','right','horizontal'] and self.pos1.y == object.pos1.y) or (self.direction in ['up','down','vertical'] and self.pos1.x == object.pos1.x):
					if self.pos1 in object or self.pos2 in object:
						positions = [self.pos1, object.pos1, object.pos2]
						res = min([ Segment(positions[i], positions[i-1]) for i in range(4) ], key=lambda x: x.length)
						if res.direction == 'point':
							return [res.pos1]
						else:
							return [res]

			elif self.direction in ['right','left','horizontal'] and object.direction in ['up','down','vertical']:
				return [Point(object.pos1.x, self.pos1.y)] if Point(object.pos1.x, self.pos1.y) in self and Point(object.pos1.x, self.pos1.y) in object else []
			elif self.direction in ['up','down','vertical'] and object.direction in ['right','left','horizontal']:
				return [Point(self.pos1.x, object.pos1.y)] if Point(self.pos1.x, object.pos1.y) in self and Point(self.pos1.x, object.pos1.y) in object else []

			elif self.direction == 'point':
				return object.intersects(self.pos1)
			elif object.direction == 'point':
				return self.intersects(object.pos1)

			return []

		elif hasattr(object, 'intersects') and not isinstance(object, Primitive):
			return object.intersects(self)

		else:
			raise ValueError(f"Invalid argument type for 'object'. Expected types are Union[Primitive, Point, list, tuple], but received {type(object)}.")

	def y_from_x(self, x: float, return_none: bool = False) -> float:
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
		# y = kx + m; x = (y-m)/k; returns (x) from y
		if self.direction in ['right', 'left', 'horizontal']:
			return self.pos1.x if y == self.pos1.y or not return_none else None
		elif self.direction in ['up', 'down', 'vertical']:
			return self.pos1.x if y == self.pos1.y or not return_none else None
		elif self.direction == 'point':
			return self.pos1.x
		else:
			return (y - self.m) / self.k if self.intersects([(y - self.m) / self.k, y]) or not return_none else None

	@property
	def angle(self):
		if not self.direction == 'point':
			second_pos = Point(self.pos2.x + 1, self.pos1.y)
			while second_pos.pos in [self.pos1.pos, self.pos2.pos]:
				second_pos += [1, 0]
			return Angle(self.pos2, self.pos1, second_pos)

	@property
	def perpendicular(self):
		return line_by_function( lambda x: -1/self.k * x + self.m )

	@property
	def direction(self) -> str:
		if self.pos1.pos == self.pos2.pos: # from [0, 0] to [0, 0]
			return 'point'
		elif self.pos1.x == self.pos2.x: # y = num
			return 'vertical'
		elif self.pos1.y == self.pos2.y: # x = num
			return 'horizontal'
		else: # y = kx + b
			return 'normal'

	@property
	def as_geometry(self) -> str:
		if self.direction == 'vertical':
			return f'x = {self.pos1.x}'
		elif self.direction == 'horizontal':
			return f'y = {self.pos1.y}'
		elif self.direction == 'point':
			return f'x = {self.pos1.x}; y = {self.pos1.y}'
		else:
			return f'y = { self.k if self.k != 1 else "" }x{ (f" + {self.m}" if self.m > 0 else f" - {-1*self.m}") if self.m else "" }'

	def __contains__(self, object):
		return self.intersects(object)

	def __add__(self, i):
		if isinstance(i, Point):
			new_pos1 = Point(self.pos1.x + i.x, self.pos1.y + i.y)
			new_pos2 = Point(self.pos2.x + i.x, self.pos2.y + i.y)
			return self.__class__(new_pos1, new_pos2, name=self.name)
		elif isinstance(i, (list, tuple)):
			new_pos1 = Point(self.pos1.x + i[0], self.pos1.y + i[1])
			new_pos2 = Point(self.pos2.x + i[0], self.pos2.y + i[1])
			return self.__class__(new_pos1, new_pos2, name=self.name)
	def __sub__(self, i):
		if isinstance(i, Point):
			new_pos1 = Point(self.pos1.x - i.x, self.pos1.y - i.y)
			new_pos2 = Point(self.pos2.x - i.x, self.pos2.y - i.y)
			return self.__class__(new_pos1, new_pos2, name=self.name)
		elif isinstance(i, (list, tuple)):
			new_pos1 = Point(self.pos1.x - i[0], self.pos1.y - i[1])
			new_pos2 = Point(self.pos2.x - i[0], self.pos2.y - i[1])
			return self.__class__(new_pos1, new_pos2, name=self.name)

	def __call__(self, x: int, return_none: bool = False) -> float:
		return self.y_from_x(x, return_none=return_none)

	def __str__(self):
		return f'{self.name}[({self.pos1} -> {self.pos2}), {self.as_geometry}]'
	def __repr__(self):
		return f'{self.__class__.__name__}({self.pos1}, {self.pos2}, name="{self.name}")'

class Segment(Line):
	def __init__(self, pos1: Union[Point, list, tuple], pos2: Union[Point, list, tuple], name: str = 'Segment'):
		super().__init__(pos1, pos2, name=name)

	def intersects(self, object: Union[Primitive, Point, list, tuple]) -> Point:
		if isinstance(object, (list, tuple)):
			object = Point(*object)

		if isinstance(object, Point):
			if self.k:
				at_line = round(float(object.y), 10) == round(float(self.k * object.x + self.m), 10)
				in_rect = min(self.pos1.x, self.pos2.x) <= object.x <= max(self.pos1.x, self.pos2.x) and min(self.pos1.y, self.pos2.y) <= object.y <= max(self.pos1.y, self.pos2.y)
				return [object] if at_line and in_rect else []
			elif self.direction == 'vertical':
				in_y = min(self.pos1.y, self.pos2.y) <= object.y <= max(self.pos1.y, self.pos2.y)
				return [object] if object.x == self.pos1.x and in_y else []
			elif self.direction == 'horizontal':
				in_x = min(self.pos1.x, self.pos2.x) <= object.x <= max(self.pos1.x, self.pos2.x)
				return [object] if object.y == self.pos1.y and in_x else []

		elif hasattr(object, 'intersects') and not isinstance(object, Primitive):
			return object.intersects(self)

		else:
			return super().intersects(object)

	def at_pos(self, point: Point):
		return Segment(point, Point(self.pos2.x - self.pos1.x, self.pos2.y - self.pos1.y), name=self.name)

	@property
	def length(self) -> float:
		return sqrt( (self.pos2.x - self.pos1.x)**2 + (self.pos2.y - self.pos1.y)**2 )

	@property
	def perpendicular(self):
		if self.k:
			k = -1 / self.k
			m = self.center.y - k * self.center.x
			return line_by_function( lambda x: k * x + m )
		elif self.direction == 'vertical':
			return Line(self.center, self.center + [1,0])
		elif self.direction == 'horizontal':
			return Line(self.center, self.center + [0,1])

	@property
	def center(self) -> Point:
		return Point((self.pos1.x + self.pos2.x)/2, (self.pos1.y + self.pos2.y)/2)

class Ray(Line):
	def __init__(self, pos1: Union[Point, list, tuple], pos2: Union[Point, list, tuple], name: str = 'Ray'):
		super().__init__(pos1, pos2, name=name)
	
	def intersects(self, object: Union[Primitive, Point, list, tuple]) -> Point:
		if isinstance(object, (list, tuple)):
			object = Point(*object)

		if isinstance(object, Point):
			at_line = float(object.y) == float(self.k * object.x + self.m)
			if self.k and at_line:
				if self.direction == 'right-down':
					return [object] if object.y <= self.pos1.y and object.x >= self.pos1.x else []
				elif self.direction == 'right-up':
					return [object] if object.y >= self.pos1.y and object.x >= self.pos1.x else []
				elif self.direction == 'left-down':
					return [object] if object.y <= self.pos1.y and object.x <= self.pos1.x else []
				elif self.direction == 'left-up':
					return [object] if object.y >= self.pos1.y and object.x <= self.pos1.x else []
			else:
				if self.direction == 'up':
					return [object] if object.y >= self.pos1.y and object.x == self.pos1.x else []
				elif self.direction == 'down':
					return [object] if object.y <= self.pos1.y and object.x == self.pos1.x else []
				elif self.direction == 'left':
					return [object] if object.y == self.pos1.y and object.x <= self.pos1.x else []
				elif self.direction == 'right':
					return [object] if object.y == self.pos1.y and object.x >= self.pos1.x else []

			if self.direction == 'object':
				return self.pos1 == object

			return []

		elif hasattr(object, 'intersects') and not isinstance(object, Primitive):
			return object.intersects(self)

		else:
			return super().intersects(object)

	def at_pos(self, point: Point):
		return Ray(point, Point(self.pos2.x - self.pos1.x, self.pos2.y - self.pos1.y), name=self.name)

	@property
	def as_geometry(self) -> str:
		if self.direction in ['up', 'down']:
			return f'x = {self.pos1.x}'
		elif self.direction in ['left', 'right']:
			return f'y = {self.pos1.y}'
		elif self.direction == 'point':
			return f'x = {self.pos1.x}; y = {self.pos1.y}'
		else:
			return f'y = { self.k if self.k != 1 else "" }x{ (f" + {self.m}" if self.m > 0 else f" - {-1*self.m}") if self.m else "" }'

	@property
	def direction(self) -> str:
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

class VectorMeta(type):
	def __getitem__(cls, pos):
		return cls([0,0], pos)
class Vector(Segment, metaclass=VectorMeta):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def intersects(self, object: Union[Primitive, Point, list, tuple]) -> List[Point]:
		if isinstance(object, (list, tuple)):
			object = Point(*object)

		if isinstance(object, Point):
			at_line = float(object.y) == float(self.k * object.x + self.m)
			if self.k and at_line:
				if self.direction == 'right-down':
					return [object] if object.y <= self.pos1.y and object.x >= self.pos1.x else []
				elif self.direction == 'right-up':
					return [object] if object.y >= self.pos1.y and object.x >= self.pos1.x else []
				elif self.direction == 'left-down':
					return [object] if object.y <= self.pos1.y and object.x <= self.pos1.x else []
				elif self.direction == 'left-up':
					return [object] if object.y >= self.pos1.y and object.x <= self.pos1.x else []
			else:
				if self.direction == 'up':
					return [object] if object.y >= self.pos1.y and object.x == self.pos1.x else []
				elif self.direction == 'down':
					return [object] if object.y <= self.pos1.y and object.x == self.pos1.x else []
				elif self.direction == 'left':
					return [object] if object.y == self.pos1.y and object.x <= self.pos1.x else []
				elif self.direction == 'right':
					return [object] if object.y == self.pos1.y and object.x >= self.pos1.x else []

			if self.direction == 'object':
				return self.pos1 == object

			return []

		elif hasattr(object, 'intersects') and not isinstance(object, Primitive):
			return object.intersects(self)

		else:
			return super().intersects(object)

	def dot(self, vector) -> float:
		v1, v2 = vector.to_zero, self.to_zero
		return v1.pos2.x * v2.pos2.x + v1.pos2.y * v2.pos2.y

	@property
	def direction(self) -> str:
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

	@property
	def as_geometry(self) -> str:
		if self.direction in ['up', 'down']:
			return f'x = {self.pos1.x}'
		elif self.direction in ['left', 'right']:
			return f'y = {self.pos1.y}'
		elif self.direction == 'point':
			return f'x = {self.pos1.x}; y = {self.pos1.y}'
		else:
			return f'y = { self.k if self.k != 1 else "" }x{ (f" + {self.m}" if self.m > 0 else f" - {-1*self.m}") if self.m else "" }'

	@property
	def to_zero(self):
		return self.at_pos([0,0])

	@property
	def normalize(self):
		length = self.length
		if self.direction != 'point':
			return Vector(self.pos1, Point(self.pos1.x + (self.pos2.x - self.pos1.x) / length, self.pos1.y + (self.pos2.y - self.pos1.y) / length))
		else:
			return self

	def __add__(self, vector):
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x + vector, self.pos2.y + vector))
		elif isinstance(vector, Vector):
			return Vector(self.pos1 + vector.pos1, self.pos2 + vector.pos2)

	def __sub__(self, vector):
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x - vector, self.pos2.y - vector))
		elif isinstance(vector, Vector):
			return Vector(self.pos1 * vector.pos1, self.pos2 * vector.pos2)

	def __mul__(self, vector):
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x * (self.pos2.x - self.pos1.x) * vector, self.pos1.y * (self.pos2.y - self.pos1.y) * vector))
		elif isinstance(vector, Vector):
			return Vector(self.pos1 * vector.pos1, self.pos2 * vector.pos2)

	def __truediv__(self, vector):
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x / vector, self.pos2.y / vector))
		elif isinstance(vector, Vector):
			return Vector(self.pos1 / vector.pos1, self.pos2 / vector.pos2)

	def __floordiv__(self, vector):
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x // vector, self.pos2.y // vector))
		elif isinstance(vector, Vector):
			return Vector(self.pos1 // vector.pos1, self.pos2 // vector.pos2)

	def __pow__(self, vector):
		if isinstance(vector, (float, int)):
			return Vector(self.pos1, Point(self.pos1.x ** vector, self.pos2.y ** vector))
		elif isinstance(vector, Vector):
			return Vector(self.pos1 ** vector.pos1, self.pos2 ** vector.pos2)


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
	def minutes(self):
		return 0 if self.pos2 in Segment(self.pos1, self.midpos) or self.pos1 in Segment(self.pos2, self.midpos) else self.degrees * 60
		
	@property
	def bisector(self) -> Ray:
		return Ray(self.midpos, Point( (self.pos1.x + self.pos2.x)/2, (self.pos1.y + self.pos2.y)/2 ))

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

	def __add__(self, num: float):
		return float(self)+num
	def __sub__(self, num: float):
		return float(self)-num
	def __mul__(self, num: float):
		return float(self)*num
	def __truediv__(self, num: float):
		return float(self)/num
	def __floordiv__(self, num: float):
		return float(self)//num
	def __pow__(self, num: float):
		return float(self)**num

	def __float__(self, num: float):
		return float(self.degrees)
	def __int__(self, num: float):
		return float(self.degrees)

	def __str__(self):
		return f'{self.name}[({self.pos1} -> {self.midpos} -> {self.pos2}), {int(self.degrees)} degrees]'
	def __repr__(self):
		return f'Angle({self.pos1} -> {self.midpos} -> {self.pos2}, name="{self.name}")'


class PrimitiveGroup:
	def __init__(self, primitives: List[Primitive], name: str = 'PrimitiveGroup'):
		self.primitives = []
		self.name = name

		for i, pr in enumerate(primitives):
			if isinstance(pr, Primitive):
				self.primitives.append(pr)
			else:
				raise ValueError(f'Composite constructor arg primitives[{i}] is not a Primitive: {pr}')

	def intersects(self, object: Union[Primitive]) -> List[Point]:
		if not isinstance(object, Composite) and not isinstance(object, PrimitiveGroup):
			return [ pr.intersects(object) for pr in self.primitives if pr in object ]
		else:
			points = []
			for obj in (object.shapes if isinstance(object, Composite) else object.primitives):
				for pr in self.primitives:
					if pr in obj and pr != obj:
						res = pr.intersects(obj)
						for point in res:
							points.append(point[0])
			return points

	def plot(self):
		Scene(*self.primitives).show()

	@property
	def intersections(self) -> List[Point]:
		points = []
		for pr1 in self.primitives:
			for pr2 in self.primitives:
				if pr1 in pr2 and pr1 != pr2:
					res = pr1.intersects(pr2)
					for point in res:
						points.append(point)

		return list(set(points))
	
	def __contains__(self, object):
		return self.intersects(object)

	def __str__(self):
		return f'{self.name}({len(self.primitives)} primitives)'
	def __repr__(self):
		return f'{self.__class__.__name__}({len(self.primitives)}, name="{self.name}")'