from typing import *
from tkinter import Tk, Canvas
from math import cos, sin, pi

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as MPLEllipse

from .primitives import *
from .math import *

class Shape:
	...


class Polygon(Shape):
	def __init__(self, vertices: List[[ Point, Point, ... ]], name: str = 'Polygon', pos: Point = None, segment_object = Segment):
		self.vertices = vertices
		self.segments = []
		self.angles = []
		self.name = name

		self._setup(segment_object)

		self.pos = Point(*pos) if isinstance(pos, (tuple, list)) else (pos if isinstance(pos, Point) else self.center_of_mass)

	def _setup(self, segment_object):
		for i, vertice in enumerate(self.vertices):
			if isinstance(vertice, (list,tuple)):
				vertice = Point(*vertice)
				self.vertices[i] = vertice

			if isinstance(self.vertices[i-1], (list,tuple)):
				self.vertices[i-1] = Point(*self.vertices[i-1])

			self.segments.append( segment_object(self.vertices[i-1], vertice, name=f's{self.name}{i}' if vertice.name == 'Point' else vertice.name) )
			
		for i, vertice in enumerate(self.vertices):
			self.angles.append( Angle(self.vertices[i-2], self.vertices[i-1], vertice, name=f'a{self.name}{i}' if vertice.name == 'Point' else vertice.name) )

	def intersects(self, object: Union[Primitive, Shape, Point, tuple, list], check_inside: bool = True) -> List[Point]:
		if isinstance(object, (tuple, list)):
			object = Point(*object)

		if isinstance(object, Point):
			for segment in self.segments:
				if object in segment:
					return [object]

		elif isinstance(object, (Segment, Vector)):
			points = [ object.intersects(segment) for segment in self.segments if object in segment ]
			if points:
				return points
			# Segment and Vector may be inside polygon and do not intersect it, so it will be checked in self.inside

		elif isinstance(object, (Ray, Line, Segment, Vector)):
			return [ object.intersects(segment) for segment in self.segments if object in segment ]
			# Ray and Line can not be inside polygon and do not intersect it, so it will not be checked in self.inside

		elif isinstance(object, Polygon):
			points = []
			for segment1 in self.segments:
				for segment2 in object.segments:
					if segment1 in segment2 and segment1 != segment2:
						res = segment1.intersects(segment2)
						if res:
							points.append(res[0])

			return points

		elif isinstance(object, Circle):
			points = []
			for segment in self.segments:
				if segment in object:
					res = object.intersects(segment)
					for point in res:
						points.append(point)

			return points

		else:
			raise ValueError() # недоделал

		if self.inside(object) and check_inside:
			return [object]
		else:
			return []

	def inside(self, object: Union[Primitive, Shape, Point, tuple, list]) -> bool:
		if isinstance(object, (tuple, list)):
			point = Point(*object)
		elif isinstance(object, Point):
			point = object
		elif isinstance(object, (Segment, Vector)):
			point = object.center
		elif isinstance(object, (Ray, Line)):
			return len(self.intersects(object)) > 1
		elif isinstance(object, (Polygon, Circle)):
			point = object.center_of_mass
		else:
			raise ValueError() # недоделал

		rays = {
			Ray(point, point + [0, 1]): [],
			Ray(point, point + [0, -1]): [],
			Ray(point, point + [-1, 0]): [],
			Ray(point, point + [1, 0]): [],
		} # {up:0, down:0, right:0, left:0}
		for segment in self.segments:
			for ray in rays.keys():
				if ray in segment:
					rays[ray].append(segment.intersects(ray))

		return all([ len(positions) % 2 != 0 and len(positions) != 0 for positions in rays.values() ])
		# if every ray from "rays" dict has intersection count that % 2 == 0 and != 0 then point inside

	def scale(self, factor: float, center: Point = None):
		if not center:
			center = self.center_of_mass
		elif isinstance(center, (tuple, list)):
			center = Point(*center)

		new_vertices = []
		for vertice in self.vertices:
			new_vertices.append( Point(center.x + factor * (vertice.x - center.x), center.y + factor * (vertice.y - center.y)) )

		return Polygon(new_vertices, name=self.name, pos=Point(center.x + factor * (self.pos.x - center.x), center.y + factor * (self.pos.y - center.y)))

	def rotate(self, angle: Union[int, Angle], center: Point = None):
		if center is None:
			angle = angle.radians if isinstance(angle, Angle) else angle
		if not center:
			center = self.center_of_mass
		elif isinstance(center, (tuple, list)):
			center = Point(*center)

		cos_angle, sin_angle = cos(angle), sin(angle)

		new_vertices = []
		for vertice in self.vertices:
			translated_x = vertice.x - center.x
			translated_y = vertice.y - center.y

			new_vertices.append(Point(
				translated_x * cos_angle - translated_y * sin_angle + center.x,
				translated_x * sin_angle + translated_y * cos_angle + center.y,
			))

		return Polygon(new_vertices, name=self.name, pos=self.pos)

	def segments_by_point(self, point: Point) -> list:
		if isinstance(point, (list,tuple)):
			point = Point(*point)
		return [ segment for segment in self.segments if segment.intersects(point) ]
	def segments_by_fromto_point(self, point: Point):
		if isinstance(point, (list,tuple)):
			point = Point(*point)
		return [ segment for segment in self.segments if point.pos in [segment.pos1, segment.pos2] ]
	def segment_by_from_point(self, point: Point):
		if isinstance(point, (list,tuple)):
			point = Point(*point)
		return [ segment for segment in self.segments if point.pos == segment.pos1 ]
	def segment_by_to_point(self, point: Point):
		if isinstance(position, (list,tuple)):
			position = Point(*position)
		return [ segment for segment in self.segments if point.pos == segment.pos2 ]

	def angle_by_pos(self, point: Point):
		if isinstance(point, (list,tuple)):
			point = Point(*point)
		return [ vertice for vertice in self.vertices if point.pos == vertice.pos ]

	def view(self):
		root = Tk()
		root.title(f"{self.name} view")
		root.geometry(f"{self.width*2}x{self.height*2}")

		canvas = Canvas(root, width=self.width*2, height=self.height*2, bg='gray')
		canvas.pack()

		canvas.create_polygon(self.at_pos(Point(50,50)).as_tk_polygon, outline='white', fill='red')

		root.mainloop()

	def at_pos(self, point: Union[Point, list, tuple]):
		if isinstance(point, (list,tuple)):
			point = Point(*point)
		shift = Point(point.x - self.pos.x, point.y - self.pos.y)
		return self.__class__([vertex + shift for vertex in self.vertices], name=self.name, pos=self.pos + point)

	def to_pos(self, point: Union[Point, list, tuple]):
		if isinstance(point, (list,tuple)):
			point = Point(*point)

		shift = Point(point.x - self.pos.x, point.y - self.pos.y)
		self.pos = point
		self.vertices = [vertex + shift for vertex in self.vertices]
		self._setup(self.segments[0].__class__)

	def distane_to(object: Union[Primitive, Shape, Point, tuple, list]) -> Segment:
		if isinstance(object, (tuple, list)):
			object = Point(*object)

		if isinstance(object, Point):
			return Segment(self.center_of_mass, object)
		elif isinstance(object, Line):
			return self.center_of_mass.height_to(object)
		elif isinstance(object, Ray):
			res = self.center_of_mass.height_to(object)
			return res if res in object else Segment(self.center_of_mass, object.pos1)
		elif isinstance(object, (Segment, Vector, Circle)):
			return Segment(self.center_of_mass, object.center)
		elif isinstance(object, Polygon):
			return Segment(self.center_of_mass, object.center_of_mass)

		return []

	def plot(self):
		x = [vertice.x for vertice in self.vertices]
		y = [vertice.y for vertice in self.vertices]

		x.append(self.vertices[0].x)
		y.append(self.vertices[0].y)

		plt.plot(x, y, 'bo-')

		for i in range(len(self.vertices)):
			segment = Segment(self.vertices[i], self.vertices[(i + 1) % len(self.vertices)])
			plt.plot([segment.pos1.x, segment.pos2.x], [segment.pos1.y, segment.pos2.y], 'r-')

		plt.show()

	@property
	def random_point(self):
		point = Point.random(self.min_pos, self.max_pos)
		while not point in self:
			point = Point.random(self.min_pos, self.max_pos)
		return point

	@property
	def circum_circle(self) -> float:
		last_vertice = self.center_of_mass
		for vertice in self.vertices:
			if Segment(self.center_of_mass, vertice).length > Segment(self.center_of_mass, last_vertice).length:
				last_vertice = vertice

		return Circle(self.center_of_mass, Segment(self.center_of_mass, last_vertice).length)

	@property
	def center_of_mass(self) -> float:
		y = [ vertice.y for vertice in self.vertices ]
		x = [ vertice.x for vertice in self.vertices ]
		return Point(sum(x)/len(x), sum(y)/len(y), name=f'{self.name} center')

	@property
	def area(self) -> float:
		return abs(sum([
			segment.pos1[0] * segment.pos2[1] - segment.pos1[1] * segment.pos2[0] for segment in self.segments
		])) / 2
	@property
	def perimeter(self) -> float:
		return sum([segment.length for segment in self.segments])

	@property
	def as_tk_polygon(self):
		res = []
		for pos in self.vertices:
			for xy in pos:
				res.append(xy)
		return res

	@property
	def min_x(self):
		return min( x for x, y in self.vertices )
	@property
	def min_y(self):
		return min( y for x, y in self.vertices )
	@property
	def max_x(self):
		return max( x for x, y in self.vertices )
	@property
	def max_y(self):
		return max( y for x, y in self.vertices )

	@property
	def min_pos(self):
		return Point(self.min_y, self.min_x, name=f'{self.name}_min')
	@property
	def max_pos(self):
		return Point(self.max_y, self.max_x, name=f'{self.name}_min')

	@property
	def width(self):
		return abs(self.max_x - self.min_x)
	@property
	def height(self):
		return abs(self.max_y - self.min_y)

	@property
	def box(self):
		return Box(self.min_pos, self.max_pos)
	
	def __contains__(self, object):
		return self.intersects(object, check_inside=True)

	def __add__(self, point: Union[Point, list, tuple]):
		point = Point(point)
		return self.__class__([ vertice + point for vertice in self.vertices ], name=self.name, pos=self.pos + point)
	def __sub__(self, point: Point):
		point = Point(point)
		return self.__class__([ vertice - point for vertice in self.vertices ], name=self.name, pos=self.pos - point)
	def __mul__(self, point: Point):
		point = Point(point)
		return self.__class__([ vertice * point for vertice in self.vertices ], name=self.name, pos=self.pos * point)
	def __truediv__(self, point: Point):
		point = Point(point)
		return self.__class__([ vertice / point for vertice in self.vertices ], name=self.name, pos=self.pos / point)
	def __floordiv__(self, point: Point):
		point = Point(point)
		return self.__class__([ vertice // point for vertice in self.vertices ], name=self.name, pos=self.pos // point)
	def __pow__(self, point: Point):
		point = Point(point)
		return self.__class__([ vertice ** point for vertice in self.vertices ], name=self.name, pos=self.pos ** point)

	def __len__(self):
		return int(self.perimeter)
	def __str__(self):
		return f'{self.name}({self.pos}, {len(self.vertices)} vertices)'
	def __repr__(self):
		return f'{self.__class__.__name__}({self.vertices}, name="{self.name}", pos={self.pos})'

class Box(Polygon):
	def __init__(self, pos1: Point, pos2: Point, name: str = 'Box', pos: Point = None):
		if isinstance(pos1, (tuple, list)):
			pos1 = Point(*pos1)
		if isinstance(pos2, (tuple, list)):
			pos2 = Point(*pos2)

		super().__init__([pos1, Point(pos1.x, pos2.y), pos2, Point(pos2.x, pos1.y)], name=name, pos=pos)

class Triangle(Polygon):
	def __init__(self, pos1: Point, pos2: Point, pos3: Point, name: str = 'Triangle', pos: Point = None):
		super().__init__([pos1, pos2, pos3], name=name, pos=pos)
		self.side1, self.side2, self.side3 = self.segments

	@property
	def hypotenuse(self):
		if self.angle_type == 'right':
			return max(self.segments, key=lambda segment: segment.lenth)
	@property
	def legs(self):
		if self.angle_type == 'right':
			return [ segment for segment in self.segments if segment != self.hypotenuse ]

	@property
	def angle_type(self) -> str:
		if 90 in [ angle.degree for angle in self.angles ]:
			return 'right'
		elif len([ angle for angle in self.angles if angle.degree > 90 ]) > 0:
			return 'obtuse'
		elif len([ angle for angle in self.angles if angle.degree < 90 ]) == 3:
			return 'acute'
		else:
			return 'scalene'

	@property
	def side_type(self) -> str:
		if self.segments[0].length == self.segments[1].length == self.segments[2].length:
			return 'equilateral'
		elif len([ segment for i, segment in enumerate(self.segments) if segment.length == self.segments[i-1].length ]) > 0:
			return 'isosceles'
		else:
			return 'scalene'

	@property
	def type(self) -> str:
		if self.angle_type == self.side_type == 'scalene':
			return 'scalene'
		else:
			return f'{self.angle_type} {self.side_type}'

class Rhombus(Polygon):
	def __init__(self, center: Point, diagonal_x: int, diagonal_y: int, name: str = 'Rhombus', pos: Point = None):
		if isinstance(center, (tuple, list)):
			center = Point(*center)

		self.center = center
		self.diagonal_x = diagonal_x
		self.diagonal_y = diagonal_y

		self.segment_x = Segement(self.center - [diagonal_x/2, 0], self.center + [diagonal_x/2, 0])
		self.segment_y = Segement(self.center + [0, diagonal_y/2], self.center + [diagonal_x/2, 0])

		super().__init__([
			self.center + [0, diagonal_y/2],
			self.center - [diagonal_x/2, 0],
			self.center - [0, diagonal_y/2],
			self.center + [diagonal_x/2, 0],
		], name=name, pos=pos)

class Circle(Shape):
	def __init__(self, center: Point, radius: int, name: str = 'Circle'):
		if isinstance(center, (tuple, list)):
			center = Point(*center)

		self.radius = radius
		self.diameter = 2 * radius
		self.center = center
		self.center_of_mass = center
		self.name = name

	def intersects(self, object: Union[Primitive, Shape, Point], check_inside: bool = True) -> List[Point]:
		if isinstance(object, (tuple, list)):
			object = Point(*object)

		if isinstance(object, Point):
			if check_inside:
				return [object] if round(Segment(self.center, object).length, 10) <= self.radius + EPSILON else []
			else:
				return [object] if round(Segment(self.center, object).length - self.radius, 10) <= EPSILON else []

		elif isinstance(object, (Segment, Ray, Line, Vector)):
			if object.k:
				# (1+k**2)x**2 + 2(km−k⋅self.x−self.y)x + (m**2+self.y**2−2mb−r**2+b**2)=0
				eq = QuadraticEq(
					a = 1 + object.k**2,
					b = 2 * (object.k * object.m - object.k * self.y - self.x),
					c = object.m**2 + self.x**2 - 2 * object.m * self.y + self.y**2 - self.radius**2,
				)
				return [ Point(x, object(x)) for x in eq.solve() if Point(x, object(x)) in object ]
			elif object.direction in ['horizontal', 'left', 'right']:
				return [ Point(x, object.pos1.y) for x in self.x_from_y(object.pos1.y) if Point(x, object.pos1.y) in object ]
			elif object.direction in ['vertical', 'up', 'down']:
				return [ Point(object.pos1.x, y) for y in self.y_from_x(object.pos1.x) if Point(object.pos1.x, y) in object ]
			elif object.direction == 'point':
				return object.pos1 if self.intersects(object.pos1) else []

		elif isinstance(object, Polygon):
			res = [ self.intersects(segment) for segment in object.segments if segment in self ]
			if res:
				return res

		elif isinstance(object, Circle):
			distance = Segment(self.center, object.center)
			if distance.length > self.radius + object.radius:
				return []

			if distance == 0:
				return [Circle(self.center, self.radius, name=self.name)]

			else:
				distance = Segment(self.center, object.center)

				if distance.length == self.radius + object.radius:
					res = self.intersects(distance)
					return res[0] if res[0] in self and res[0] in object else []

				elif distance.length > self.radius + object.radius:
					a = (self.radius**2 - object.radius**2 + distance.length**2) / (2 * distance.length)
					h = sqrt(self.radius**2 - a**2)
					intercenter = Point(
						self.x + a * (object.x - self.x) / distance.length,
						self.y + a * (object.y - self.y) / distance.length,
					)
					return [
						Point(intercenter.x + h*(object.y - self.y) / distance.length, intercenter.y - h*(object.x - self.x) / distance.length),
						Point(intercenter.x - h*(object.y - self.y) / distance.length, intercenter.y + h*(object.x - self.x) / distance.length),
					]

				elif distance.length + self.radius < object.radius and check_inside:
					return self.center
				elif distance.length + object.radius < self.radius and check_inside:
					return object.center

				return []

		else:
			raise ValueError() # недоделал

	def at_pos(self, position: List[[int, int]]):
		if self.center != position:
			return Circle(position, self.radius, name=self.name)

	def to_pos(self, position: List[[int, int]]):
		if isinstance(position, (tuple, list)):
			position = Point(*object)

		if self.center != position:
			self.center = position
			self.center_of_mass = position

	def distane_to(object: Union[Primitive, Shape, Point, tuple, list]) -> Segment:
		if isinstance(object, (tuple, list)):
			object = Point(*object)

		if isinstance(object, Point):
			return Segment(self.center, object)
		elif isinstance(object, Line):
			return self.center.height_to(object)
		elif isinstance(object, Ray):
			res = self.center.height_to(object)
			return res if res in object else []
		elif isinstance(object, (Segment, Vector, Circle)):
			return Segment(self.center, object.center)
		elif isinstance(object, Polygon):
			return Segment(self.center, object.center_of_mass)

		return []

	def scale(factor: float):
		return Circle(self.pos, self.radius*factor, name=self.name)

	def view(self):
		root = Tk()
		root.title(f"{self.name} view")
		root.geometry(f"{self.radius*2+20}x{self.radius*2+20}")

		canvas = Canvas(root, width=self.radius_x*2+20, height=self.radius_y*2+20, bg='gray')
		canvas.pack()

		canvas.create_oval(10, 10, self.radius_x*2, self.radius_y*2, outline='white', fill='red')

		root.mainloop()

	def plot(self):
		fig, ax = plt.subplots()
		ax.add_patch(Ellipse(xy=(self.center.x, self.center.y), width=self.diameter, height=self.diameter, edgecolor='r', fc='None'))
		ax.axis('equal')
		plt.show()

	def y_from_x(self, x: float) -> float:
		if self.radius**2 - (x - self.x)**2 == 0:
			return self.y
		elif self.radius**2 - (x - self.x)**2 > 0:
			return [
				-sqrt(self.radius**2 - (x - self.x)**2) + self.y,
				+sqrt(self.radius**2 - (x - self.x)**2) + self.y,
			]
	def x_from_y(self, y: float) -> float:
		if self.radius**2 - (y - self.y)**2 == 0:
			return self.x
		elif self.radius**2 - (y - self.y)**2 > 0:
			return [
				-sqrt(self.radius**2 - (y - self.y)**2) + self.x,
				+sqrt(self.radius**2 - (y - self.y)**2) + self.x,
			]

	@property
	def x(self):
		return self.center.x
	@property
	def y(self):
		return self.center.y

	@property
	def area(self):
		return pi * self.radius**2
	@property
	def perimeter(self):
		return 2 * pi * self.radius

	@property
	def as_geometry(self):
		return f'(x - {self.x})**2 + (y - {self.y})**2 = {self.radius}**2'
	
	def __contains__(self, object):
		return self.intersects(object, check_inside=True)

	def __add__(self, point: Union[Point, list, tuple]):
		if isinstance(point, (list,tuple)):
			point = Point(*point)
		return Circle(self.center + point, self.radius, name=self.name)
	def __sub__(self, point: Point):
		if isinstance(point, (list,tuple)):
			point = Point(*point)
		return Circle(self.center - point, self.radius, name=self.name)
	def __mul__(self, point: Point):
		if isinstance(point, (list,tuple)):
			point = Point(*point)
		return Circle(self.center * point, self.radius, name=self.name)
	def __truediv__(self, point: Point):
		if isinstance(point, (list,tuple)):
			point = Point(*point)
		return Circle(self.center / point, self.radius, name=self.name)
	def __floordiv__(self, point: Point):
		if isinstance(point, (list,tuple)):
			point = Point(*point)
		return Circle(self.center // point, self.radius, name=self.name)
	def __pow__(self, point: Point):
		if isinstance(point, (list,tuple)):
			point = Point(*point)
		return Circle(self.center ** point, self.radius, name=self.name)

	def __call__(self, x: float) -> float:
		return self.y_from_x(x)
	
	def __str__(self):
		return f"{self.name}({self.center}, radius={self.radius})"
	def __repr__(self):
		return f"{self.__class__.__name__}({self.center}, {self.radius})"


class Composite(Shape):
	def __init__(self, shapes: List[Shape], name: str = 'Composite', pos: Point = [0,0]):
		self.pos = Point(*pos) if isinstance(pos, (tuple, list)) else (pos if isinstance(pos, Point) else Point(0,0))
		self.name = name
		self.shapes = []

		for i, shape in enumerate(shapes):
			if isinstance(shape, Composite):
				for sh in shape.shapes:
					self.shapes.append(sh)
			elif isinstance(shape, Shape):
				self.shapes.append(shape)
			else:
				raise ValueError(f'Composite constructor arg shapes[{i}] is not a Shape: {shape}')

	def intersects(self, object: Union[Primitive, Shape], check_inside: bool = True) -> List[Point]:
		if not isinstance(object, Composite):
			return [ shape.intersects(object, check_inside=check_inside) for shape in self.shapes ]
		elif isinstance(object, PrimitiveGroup):
			return object.intersects(self)
		else:
			points = []
			for sh1 in self.shapes:
				for sh2 in object.shapes:
					if sh1 in sh2:
						res = sh1.intersects(sh2, check_inside=check_inside)
						if isinstance(res, list):
							for i in res:
								points.append(i)
						else:
							points.append(res)

			return points

	def plot(self):
		Scene(*self.shapes, self.center_of_mass).show()

	@property
	def intersections(self) -> List[Point]:
		points = []
		for shape1 in self.shapes:
			for shape2 in self.shapes:
				if shape1 in shape2 and shape1 != shape2:
					res = shape1.intersects(shape2)
					for point in res:
						points.append(point)

		return list(set(points))

	@property
	def center_of_mass(self) -> float:
		y = [ vertice.y for vertice in self.shapes.center_of_mass ]
		x = [ vertice.x for vertice in self.shapes.center_of_mass ]
		return Point(sum(x)/len(x), sum(y)/len(y), name=f'{self.name} center')

	@property
	def area(self) -> float:
		return sum([shape.area for shape in self.shapes])
	@property
	def perimeter(self) -> float:
		return sum([shape.perimeter for shape in self.shapes])
	
	def __contains__(self, object):
		return self.intersects(object, check_inside=True)

	def __len__(self):
		return int(self.perimeter)
	def __str__(self):
		return f'{self.name}([{self.pos}], {len(self.shapes)} shapes)'
	def __repr__(self):
		return f'{self.__class__.__name__}({self.shapes}, name="{self.name}", pos={self.pos})'