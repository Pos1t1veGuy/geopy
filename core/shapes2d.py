from typing import *
from tkinter import Tk, Canvas
from math import cos, sin, pi

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse as MPLEllipse

from .primitives import *
from .math import *
from .exceptions import *
from .base_shapes import *


class Polygon(Shape2D):
	def __init__(self, *args: List[[ Point, Point, ... ]],
		name: str = 'Polygon', pos: Point = None, segment_object = Segment, color: str = 'cyan',
		segments_color: str = 'r', alpha: float = 0.5, space_check: bool = True):

		if len(args) < 3:
			raise ConstructError(f'Number of points list at constructor must be >2, not {len(args)}')

		self.vertices = []
		for i, point in enumerate(args):
			if isinstance(point, (tuple, list, np.ndarray)):
				self.vertices.append(Point(point))
			elif isinstance(point, Point):
				self.vertices.append(point)
			else:
				raise ConstructError(f'Point {i} at points list at constructor must be Point object or list/tuple with 2 numbers, not {point}, {type(point)}')


		self.segments = []
		self.angles = []
		self.name = name

		self.color = color
		self.segments_color = segments_color
		self.alpha = alpha
		self.pos = pos
		self.segment_object = segment_object
		self.space_check = space_check

		self._setup(segment_object)

		if isinstance(pos, (tuple, np.ndarray, list)):
			pos = Point(*pos)
		self.pos =  pos if isinstance(pos, Point) else self.center_of_mass

	def _setup(self, segment_object: 'Primitive'):
		for i, vertice in enumerate(self.vertices):
			if isinstance(vertice, (list,tuple,np.ndarray)):
				vertice = Point(*vertice)
				self.vertices[i] = vertice

			if isinstance(self.vertices[i-1], (list,tuple,np.ndarray)):
				self.vertices[i-1] = Point(*self.vertices[i-1])

			vertices = eq_len_axeslists(self.vertices[i-1].axes, vertice.axes)
			dimension = max([ Point[i].dimension for i in vertices ])
			self.segments.append(
				segment_object(
					vertices[0],
					vertices[1],
					name=polygon_segments_name.format(name=self.name, i=i) if vertice.name == 'Point' else vertice.name
				)
			)

		for i, vertice in enumerate(self.vertices):
			self.angles.append(
				Angle(
					self.vertices[i-2],
					self.vertices[i-1],
					vertice, 
					name=polygon_angles_name.format(name=self.name, i=i) if vertice.name == 'Point' else vertice.name
				)
			)

		if self.space_check:
			try:
				self.normal_space = self.get_normal_space()
			except ConstructError:
				raise ConstructError('A multidimensional (3D+, not 2D) polygon must lie on a 2D surface (2D Space with 2 equal size orthogonal vectors)')

			for point in self.vertices:
				if not point in self.normal_space:
					raise ConstructError('A multidimensional (3D+, not 2D) polygon must lie on a 2D surface (2D Space with 2 equal size orthogonal vectors)')

	def intersects(self, object: Union[Primitive, Shape, Point, tuple, list, np.ndarray],
				   check_inside: bool = True) -> List[Point]:
		if isinstance(object, (tuple, list, np.ndarray)):
			object = Point(*object)

		space = self.get_normal_space()
		if isinstance(object, Line):
			if object.pos1 in space and object.pos2 in space:# line lies on space
				return self.intersects_2d(object, check_inside=check_inside)
		elif isinstance(object, Ellipse):
			if object.center in space and self.center_of_mass in object.space: # ellipse lies on space
				return self.intersects_2d(object, check_inside=check_inside)
			# TODO: когда доделаю пересечение аффинных пространств сделать пересечение Эллипс х Полигон
		elif isinstance(object, Polygon):
			if object.center_of_mass in space and self.center_of_mass in object.normal_space: # ellipse lies on space
				return self.intersects_2d(object, check_inside=check_inside)
			else:
				ions = []
				for segment in object.segments:
					ion = self.intersects(segment)
					for i in ion:
						ions.append(i)
				for segment in self.segments:
					ion = object.intersects(segment)
					for i in ion:
						ions.append(i)

				return ions

		ions = space.intersects(object)

		pol2d = space.get_local_polygons(name=self.name)[0]
		local_ions = [space.to_local(ion) for ion in ions]

		return [ion for ion, lion in zip(ions,local_ions) if pol2d.intersects_2d(lion)]

	def intersects_2d(self, object: Union[Primitive, Shape, Point, tuple, list], check_inside: bool = True) -> List[Point]:
		if isinstance(object, (tuple, list, np.ndarray)):
			object = Point(*object)

		if isinstance(object, Point):
			for segment in self.segments:
				if object in segment:
					return [object.copy()]

		elif isinstance(object, (Segment, Vector)):
			points = []
			for segment in self.segments:
				if object in segment:
					for point in object.intersects(segment):
						points.append(point)

			if points:
				return points
		# Segment and Vector may be inside polygon and do not intersect it, so it will be checked in self.inside_2d

		elif isinstance(object, (Ray, Line)):
			points = []
			for segment in self.segments:
				if object in segment:
					for point in object.intersects(segment):
						if not point in points:
							points.append(point)

			return points
		# Ray and Line can not be inside polygon and do not intersect it, so it will not be checked in self.inside_2d

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

		elif hasattr(object, 'intersects') and not isinstance(object, (Primitive, Shape2D)):
			return object.intersects(self)

		else:
			raise IntersectionError(f'"intesects" method takes Union[Primitive, Shape, Point, tuple, list], not {object}')

		if self.inside_2d(object) and check_inside:
			return [object]
		else:
			return []

	def inside_2d(self, object: Union[Primitive, Shape, Point, tuple, list]) -> bool:
		if isinstance(object, (tuple, list, np.ndarray)):
			point = Point(*object)
		elif isinstance(object, Point):
			point = object
		elif isinstance(object, (Segment, Vector)):
			point = object.center
		elif isinstance(object, (Ray, Line)):
			return len(self.intersects(object)) > 1
		elif isinstance(object, Shape):
			point = object.center_of_mass
		else:
			raise IntersectionError(f'"inside" method takes Union[Primitive, Shape, Point, tuple, list], not {object}')

		ios = []
		ray = Ray(point, point + [0, 1])
		for segment in self.segments:
			if ray in segment:
				ios.append(segment.intersects(ray)[0])

		# if ray intersection count % 2 != 0 and != 0 then point inside
		return len(ios) % 2 != 0 and len(ios) != 0

	def intersection_area(self, object: Union[Primitive, Shape2D, Point, tuple, list]) -> 'Polygon':
		intersections = self.intersects(object)

		if intersections:
			for ion in self.vertices:
				if ion in object and not ion in intersections:
					if isinstance(ion, Point):
						intersections.append(ion)
					elif isinstance(ion, (Segment, Line, Ray)):
						intersections.append(ion.pos1)
						intersections.append(ion.pos2)

			for ion in object.vertices:
				if ion in self and not ion in intersections:
					if isinstance(ion, Point):
						intersections.append(ion)
					elif isinstance(ion, (Segment, Line, Ray)):
						intersections.append(ion.pos1)
						intersections.append(ion.pos2)

			return Polygon(*intersections, name=intersection_result_name.format(self.name, object.name))

		elif self.inside_2d(object):
			return object

		elif isinstance(object, Polygon):
			if object.inside_2d(self):
				return self

	def scale(self, factor: float, center: Point = None) -> 'Polygon':
		if not center:
			center = self.center_of_mass
		elif isinstance(center, (tuple, list, np.ndarray)):
			center = Point(*center)

		new_vertices = []
		for vertice in self.vertices:
			new_vertices.append( Point(center.x + factor * (vertice.x - center.x), center.y + factor * (vertice.y - center.y)) )

		pos = Point(center.x + factor * (self.pos.x - center.x), center.y + factor * (self.pos.y - center.y))
		return Polygon(*new_vertices, name=self.name, pos=pos, segment_object = Segment)

	def rotate(self, angle: Union[int, Angle], center: Point = None) -> 'Polygon':
		if center is None:
			angle = angle.radians if isinstance(angle, Angle) else angle
		if not center:
			center = self.center_of_mass
		elif isinstance(center, (tuple, list, np.ndarray)):
			center = Point(*center)

		cos_angle, sin_angle = cos(angle), sin(angle)

		new_vertices = []
		for vertice in self.vertices:
			translated_x = vertice.x - center.x
			translated_y = vertice.y - center.y

			new_vertices.append(Point(
				translated_x * cos_angle - translated_y * sin_angle + center.x,
				translated_x * sin_angle + translated_y * cos_angle + center.y,
				*vertice.axes[2:]
			))

		return Polygon(*new_vertices, name=self.name, pos=self.pos, segment_object = Segment)

	def segments_by_point(self, point: Point) -> List[Segment]:
		if isinstance(point, (list,tuple,np.ndarray)):
			point = Point(*point)
		return [ segment for segment in self.segments if segment.intersects(point) ]
	def segments_by_fromto_point(self, point: Point) -> List[Segment]:
		if isinstance(point, (list,tuple,np.ndarray)):
			point = Point(*point)
		return [ segment for segment in self.segments if point in [segment.pos1, segment.pos2] ]
	def segments_by_from_point(self, point: Point) -> List[Segment]:
		if isinstance(point, (list,tuple,np.ndarray)):
			point = Point(*point)
		return [ segment for segment in self.segments if point == segment.pos1 ]
	def segments_by_to_point(self, point: Point) -> List[Segment]:
		if isinstance(point, (list,tuple,np.ndarray)):
			point = Point(*point)
		return [ segment for segment in self.segments if point == segment.pos2 ]

	def angle_by_pos(self, point: Point) -> Angle:
		if isinstance(point, (list,tuple,np.ndarray)):
			point = Point(*point)
		return [ vertice for vertice in self.vertices if point.pos == vertice.pos ][0]

	def project_to(self, dimension: int) -> 'Polygon':
		if dimension <= 1:
			raise ConstructError(f'{self.__class__.__name__} can be 2D+ object, not {dimension}D')
		elif dimension >= 2:
			if dimension >= self.dimension:
				return self
			else:
				return self.__class__([point.project_to(dimension) for point in self.points], name=self.name,
									  segment_object=Segment, pos=self.pos, color=self.color, alpha=self.alpha,
									  segments_color=self.segments_color)

	def at_pos(self, point: Union[Point, list, tuple, np.ndarray]) -> 'Polygon':
		if isinstance(point, (list, tuple, np.ndarray)):
			point = Point(*point)
		shift = point - self.pos1
		return self.__class__(*[vertex + shift for vertex in self.vertices],
							  name=self.name, pos=self.pos + point, segment_object=Segment, space_check=False)

	def to_pos(self, point: Union[Point, list, tuple, np.ndarray]):
		if isinstance(point, (list,tuple,np.ndarray)):
			point = Point(*point)

		shift = Point(point.x - self.pos.x, point.y - self.pos.y)
		self.pos = point
		self.vertices = [vertex + shift for vertex in self.vertices]
		self._setup(self.segments[0].__class__)

	def distane_to(object: Union[Primitive, Shape, Point, tuple, list, np.ndarray]) -> Segment:
		if isinstance(object, (tuple, list, np.ndarray)):
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

	def get_normal_space(self) -> 'Space':
		origin = self.vertices[0]
		segments = self.segments_by_fromto_point(origin)
		perps = gram_schmidt([segments[0].to_vector.to_zero, segments[1].to_vector.to_zero])
		sp = Space(origin, perps, name=space_of_object_name.format(self.name)).at_pos(self.center_of_mass)
		sp.add_object(self.copy(space_check=False))
		return sp

	def copy(self, space_check: bool = True) -> 'Polygon':
		return self.__class__(*self.vertices,
			  name=self.name, pos=self.pos, segment_object=self.segment_object, color=self.color, alpha=self.alpha,
			  segments_color=self.segments_color, space_check=space_check)

	@property
	def projected_polygon(self) -> 'Polygon':
		if self.dimension > 2:
			return self.normal_space.get_local_polygons(name=self.name)[0]
		else:
			return self

	@property
	def dimension(self) -> int:
		return max([ vertice.dimension for vertice in self.vertices ])

	@property
	def center_of_mass(self) -> Point:
		return Point((sum(self.vertices) / len(self.vertices)).axes, name=center_name.format(self.name))

	@property
	def area(self) -> float:
		pol = self.projected_polygon
		return to_fraction(abs(sum([
			segment.pos1[0] * segment.pos2[1] - segment.pos1[1] * segment.pos2[0] for segment in pol.segments
		])), 2)
	@property
	def perimeter(self) -> float:
		return sum([segment.length for segment in self.segments])

	@property
	def convex(self) -> bool:
		for segment in self.segments:
			s_inters = len([ segment.intersects(s) for s in self.segments if s != segment ])
			if s_inters <= 2:
				line = segment.to_line
				l_inters = len([
					line.intersects(s) for s in self.segments if s != segment and not (line.intersects(s.pos1) and line.intersects(s.pos2))
				])
				if l_inters > s_inters:
					return False
			else:
				return False

		return True

	@property
	def as_tk_polygon(self) -> List[float]:
		res = []
		for pos in self.vertices:
			for xy in pos:
				res.append(xy)
		return res

	@property
	def width(self) -> float:
		return abs(self.max_pos.x - self.min_pos.x)
	@property
	def height(self) -> float:
		return abs(self.max_pos.y - self.min_pos.y)

	@property
	def min_pos(self) -> float:
		return Point([
			min([vertice[i] for vertice in self.vertices]) for i in range(self.dimension)
		], name=f'{self.name}_min')
	@property
	def max_pos(self) -> float:
		return Point([
			max([vertice[i] for vertice in self.vertices]) for i in range(self.dimension)
		], name=f'{self.name}_max')

	@property
	def box(self):
		if isinstance(self, Rectangle):
			if self.min_pos in self.vertices or self.max_pos in self.vertices:
				return self
			else:
				return Rectangle(self.min_pos, self.max_pos)
		else:
			return Rectangle(self.min_pos, self.max_pos)
	
	def __contains__(self, object):
		return self.intersects(object, check_inside=True)

	def __add__(self, point: Union[Point, list, tuple, np.ndarray]) -> 'Polygon':
		if isinstance(point, (list, tuple, np.ndarray)):
			point = Point(*point)
		return self.__class__(*[ vertice + point for vertice in self.vertices ],
							  name=self.name, pos=self.pos + point, segment_object = Segment)
	def __sub__(self, point: Union[Point, list, tuple, np.ndarray]) -> 'Polygon':
		if isinstance(point, (list, tuple, np.ndarray)):
			point = Point(*point)
		return self.__class__(*[ vertice - point for vertice in self.vertices ],
							  name=self.name, pos=self.pos - point, segment_object = Segment)
	def __mul__(self, point: Union[Point, list, tuple, np.ndarray]) -> 'Polygon':
		if isinstance(point, (list, tuple, np.ndarray)):
			point = Point(*point)
		return self.__class__(*[ vertice * point for vertice in self.vertices ],
							  name=self.name, pos=self.pos * point, segment_object = Segment)
	def __truediv__(self, point: Union[Point, list, tuple, np.ndarray]) -> 'Polygon':
		if isinstance(point, (list, tuple, np.ndarray)):
			point = Point(*point)
		return self.__class__(*[ vertice / point for vertice in self.vertices ],
							  name=self.name, pos=self.pos / point, segment_object = Segment)
	def __floordiv__(self, point: Union[Point, list, tuple, np.ndarray]) -> 'Polygon':
		if isinstance(point, (list, tuple, np.ndarray)):
			point = Point(*point)
		return self.__class__(*[ vertice // point for vertice in self.vertices ],
							  name=self.name, pos=self.pos // point, segment_object = Segment)
	def __pow__(self, point: Union[Point, list, tuple, np.ndarray]) -> 'Polygon':
		if isinstance(point, (list, tuple, np.ndarray)):
			point = Point(*point)
		return self.__class__(*[ vertice ** point for vertice in self.vertices ],
							  name=self.name, pos=self.pos ** point, segment_object = Segment)

	def __len__(self):
		return int(self.perimeter)
	def __str__(self):
		return f'{self.name}({self.pos}, {len(self.vertices)} vertices)'
	def __repr__(self):
		return f'{self.__class__.__name__}({self.vertices}, name="{self.name}", pos={self.pos})'

class Rectangle(Polygon): # only 2D
	def __init__(self, pos1: Point, pos2: Point, name: str = 'Rect', pos: Point = None, segment_object = Segment,
				 color: str = 'cyan', segments_color: str = 'r', alpha: str = 0.5, space_check: bool = True):
		if isinstance(pos1, (tuple, list, np.ndarray)):
			pos1 = Point(pos1)
		if isinstance(pos2, (tuple, list, np.ndarray)):
			pos2 = Point(pos2)

		self.pos1 = pos1
		self.pos2 = pos2

		super().__init__(pos1, Point(pos1.x, pos2.y, pos1.z), pos2, Point(pos2.x, pos1.y, pos2.z),
						 pos=pos, name=name, segment_object=segment_object, color=color, segments_color=segments_color,
						 alpha=alpha, space_check=space_check)

	def copy(self, space_check: bool = True) -> 'Polygon':
		return self.__class__(self.pos1, self.pos2, name=self.name, pos=self.pos, segment_object=self.segment_object,
				color=self.color, alpha=self.alpha, segments_color=self.segments_color, space_check=space_check)

	@property
	def circum_circle(self) -> 'Circle':
		return Circle(self.center_of_mass, Segment(self.center_of_mass, self.vertices[0]).length)

class Triangle(Polygon):
	def __init__(self, pos1: Point, pos2: Point, pos3: Point, name: str = 'Triangle', pos: Point = None,
				 segment_object = Segment, color: str = 'cyan', segments_color: str = 'r', alpha: str = 0.5,
				 space_check: bool = True):

		self.pos1, self.pos2, self.pos3 = pos1, pos2, pos3

		super().__init__(pos1, pos2, pos3, name=name, pos=pos, segment_object=segment_object, space_check=space_check,
						 color=color, segments_color=segments_color, alpha=alpha)

		self.side1, self.side2, self.side3 = self.segments

	def copy(self, space_check: bool = True) -> 'Polygon':
		return self.__class__(self.pos1, self.pos2, self.pos3, name=self.name, pos=self.pos, space_check=space_check,
		segment_object=self.segment_object, color=self.color, alpha=self.alpha, segments_color=self.segments_color)

	@property
	def circum_circle(self) -> 'Circle':
		return Circle(self.orthocenter, Segment(self.orthocenter, self.vertices[0]).length)

	@property
	def hypotenuse(self) -> Segment:
		if self.angle_type == 'right':
			return max(self.segments, key=lambda segment: segment.length)
	@property
	def legs(self) -> List[Segment]:
		if self.angle_type == 'right':
			return [ segment for segment in self.segments if segment != self.hypotenuse ]

	@property
	def orthocenter(self) -> Point:
		return self.segments[0].get_perpendicular().intersects(self.segments[1].get_perpendicular())[0]

	@property
	def angle_type(self) -> str:
		if 90 in [ angle.degrees for angle in self.angles ]:
			return 'right'
		elif len([ angle for angle in self.angles if angle.degrees > 90 ]) > 0:
			return 'obtuse'
		elif len([ angle for angle in self.angles if angle.degrees < 90 ]) == 3:
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

	@staticmethod
	def by_lengths(len1: Union[int, float], len2: Union[int, float], len3: Union[int, float], option: int = 0,
		name: str = 'Triangle', pos: Point = None, segment_object = Segment, space_check: bool = True) -> 'Triangle':
		'''
		triangle can be of two types with setted lengths. Third point is intersection of two Circles from points first and second.
		So, if lengths are correct Circles must have 2 interstcion points and 2 options of third point.
		OPTION kwarg is an integer index of intersection from Circles intersections. It can be 0 or 1, because here is two intersections.
		'''
		p1 = Point[0, 0]
		p2 = Point[len1, 0]
		c1 = Circle(p1, len2)
		c2 = Circle(p2, len3)

		points3 = c1.intersects(c2)
		if len(points3) == 2:
			return Triangle(p1, p2, points3[option], name=name, pos=pos, segment_object=segment_object, space_check=space_check)
		else:
			raise ValueError(f'Lengths are incorrect: {len1}, {len2}, {len3}')

	@staticmethod
	def by_angles_and_length(ang1: Union[int, float, Angle], ang2: Union[int, float, Angle], length: Union[int, float],
		name: str = 'Triangle', pos: Point = None, segment_object = Segment, space_check: bool = True) -> 'Triangle':
		if isinstance(ang1, Angle):
			ang1 = ang1.degrees
		if isinstance(ang2, Angle):
			ang2 = ang2.degrees

		p1 = Point[0, 0]
		p2 = Point[0, length]

		l1 = Line(p1, p2)
		l2 = Line.by_angle(ang1, pos1=p1)
		l3 = Line.by_angle(abs(180 - ang1), pos1=p2)

		p3 = l2.intersects(l3)[0]
		return Triangle(p1, p2, p3, name=name, pos=pos, segment_object=segment_object, space_check=space_check)

	@staticmethod
	def by_lengths_and_angle(len1: Union[int, float], len2: Union[int, float], angle: Union[int, float, Angle],
		name: str = 'Triangle', pos: Point = None, segment_object = Segment, space_check: bool = True) -> 'Triangle':
		if isinstance(angle, Angle):
			angle = angle.degrees

		p1 = Point[0, 0]
		p2 = Point[len1, 0]

		segment = Segment.by_angle(angle, len2, pos1=p1)
		return Triangle(p1, p2, segment.pos2, name=name, pos=pos, segment_object=segment_object, space_check=space_check)

class Rhombus(Polygon):
	def __init__(self, center: Point, diagonal_x: int, diagonal_y: int, name: str = 'Rhombus', pos: Point = None, segment_object = Segment, space_check: bool = True):
		if isinstance(center, (tuple, list)):
			center = Point(center)

		self.center = center
		self.diagonal_x = diagonal_x
		self.diagonal_y = diagonal_y

		self.segment_x = Segment(self.center - [to_fraction(diagonal_x,2), 0], self.center + [to_fraction(diagonal_x,2), 0])
		self.segment_y = Segment(self.center + [0, to_fraction(diagonal_y,2)], self.center + [to_fraction(diagonal_x,2), 0])

		super().__init__(
			self.center + [0, to_fraction(diagonal_y, 2)],
			self.center - [to_fraction(diagonal_x, 2), 0],
			self.center - [0, to_fraction(diagonal_y, 2)],
			self.center + [to_fraction(diagonal_x, 2), 0],
		name=name, pos=pos, segment_object=segment_object, space_check=space_check)


class Ellipse(Shape2D):
	def __init__(self, center: Point, vec1: Vector, vec2: Vector, name: str = 'Ellipse', color: str = 'r',
				 alpha: Union[int, float] = 1):

		if isinstance(center, (tuple, list, np.ndarray)):
			center = Point(*center)
		if isinstance(vec1, (tuple, list, np.ndarray)):
			vec1 = Vector[vec1]
		if isinstance(vec2, (tuple, list, np.ndarray)):
			vec2 = Vector[vec2]

		self.center = center

		self.space = Space(self.center, [vec1.normalize, vec2.normalize])
		self.normal_vector = self.space.normal
		self.vec1, self.vec2 = vec1, vec2
		self.radius_x = vec1.length
		self.radius_y = vec2.length
		self.diameter_x = self.radius_x*2
		self.diameter_y = self.radius_y*2

		self.name = name
		self.color = color
		self.alpha = alpha

	def intersects(self, object: Union[Primitive, Shape, 'Point', tuple, list], check_inside: bool = True) -> List['Point']:
		if isinstance(object, (tuple, list, np.ndarray)):
			object = Point(*object)

		space = self.space.copy()
		global_ions = lambda o: [
			space.to_global(lion) for lion in self.intersects_2d(o, check_inside=check_inside)
		]
		if isinstance(object, Line):
			if object.pos1 in space and object.pos2 in space: # line lies on space
				return global_ions(space.to_local(object))
		elif isinstance(object, Ellipse):
			if object.center in space and self.center in object.space: # ellipse lies on space
				return global_ions(object)
		elif isinstance(object, Polygon):
			if object.center_of_mass in space and self.center in object.space: # polygon lies on space
				global_ions(space.to_local(object))

		res = []
		for ion in space.intersects(object):
			for lion in global_ions(ion):
				res.append(lion)

		return res

	def intersects_2d(self, object: Union[Primitive, Shape, 'Point', tuple, list], check_inside: bool = True) -> List['Point']:
		if isinstance(object, (tuple, list, np.ndarray)):
			object = Point(*object)

		if isinstance(object, Point):
			dx = to_fraction((object.x - self.center.x)**2, self.radius_x**2)
			dy = to_fraction((object.y - self.center.y)**2, self.radius_y**2)
			return [object] if dx+dy <= 1 else []
		elif isinstance(object, (Segment, Ray, Line, Vector)):
			x1, y1 = object.pos1.x - self.center.x, object.pos1.y - self.center.y
			x2, y2 = object.pos2.x - self.center.x, object.pos2.y - self.center.y

			dx, dy = x2 - x1, y2 - y1
			eq = QuadraticEq(
				a=to_fraction(dx**2, self.radius_x**2) + to_fraction(dy**2, self.radius_y**2),
				b=to_fraction(2 * x1 * dx, self.radius_x**2) + to_fraction(2 * y1 * dy, self.radius_y**2),
				c=to_fraction(x1**2, self.radius_x**2) + to_fraction(y1**2, self.radius_y**2) - 1,
			)

			if eq.D < 0:
				return []
			elif eq.D == 0:
				t = eq.solve()[0]
				point = Point(x1 + t * dx + self.center.x, y1 + t * dy + self.center.y)
				return [point] if point in object else []
			else:
				t1, t2 = eq.solve()
				return [ pos for pos in [
					Point(x1 + t1 * dx + self.center.x, y1 + t1 * dy + self.center.y),
					Point(x1 + t2 * dx + self.center.x, y1 + t2 * dy + self.center.y)
				] if pos in object ]

		elif isinstance(object, Polygon):
			object = self.space.to_local(object)
			res = []
			for segment in object.segments:
				if segment in self:
					for point in self.intersects(segment):
						res.append(point)
			return res

		elif isinstance(object, Ellipse):
			...

			'''
			(x1 - a)/r1 + (y1 - b)/r2 = (x2 - c)/r3 + (y2 - d)/r4
			'''

		else:
			return []

	def project_to(self, dimension: int) -> Union['Circle', Line]:
		if dimension <= 1:
			raise ConstructError(f'{self.__class__.__name__} can be 2D+ object, not {dimension}D')
		elif dimension >= self.dimension:
			return self
		else:
			result_vectors_i = [
				i for i, vec in enumerate(self.space.zero_vectors) if vec.pos2[:dimension] != [0] * dimension
			]

			match len(result_vectors_i):
				case 2: # TODO: добавить превращение круга в овал при двумерной проекции
					return self.__class__(self.center.project_to(dimension), self.radius,
										  *[self.space.zero_vectors[i] for i in result_vectors_i],
										  name=self.name, color=self.color, alpha=self.alpha)
				case 1:
					vec = self.space.vectors[result_vectors_i[0]].at_pos(self.center)
					return Segment(vec.pos2, (vec*-1).pos2, name=self.name, color=self.color, alpha=self.alpha)
				case 0:
					return self.center

	def scale(self, factor: float) -> 'Ellipse':
		return self.__class__(self.pos, self.vec1*factor, self.vec2*factor, name=self.name, color=self.color,
							  alpha=self.alpha)

	def at_pos(self, position: List[[int, int]]) -> 'Ellipse':
		if self.center != position:
			return self.__class__(position, self.vec1, self.self.vec2, name=self.name, color=self.color,
								  alpha=self.alpha)
		return self.copy()

	def to_pos(self, position: List[[int, int]]):
		if isinstance(position, (tuple, list, np.ndarray)):
			position = Point(*object)
		self.center = position

	def y_from_x(self, x: float) -> float:
		'''                (x - x0)^2
y = y0 +-sqrt( r2^2 * (1 - ----------)
                              r1^2
		'''
		sqrt_expr = self.radius_y**2 * (1 - ((x - self.x)**2) / self.radius_x**2)

		if sqrt_expr < 0:
			return []
		elif sqrt_expr == 0:
			return [self.y]
		else:
			sqrt_val = sqrt(sqrt_expr)
			return [self.y - sqrt_val, self.y + sqrt_val]

	def x_from_y(self, y: float) -> float:
		'''                (y - y0)^2
x = x0 +-sqrt( r1^2 * (1 - ----------)
                              r2^2
		'''
		sqrt_expr = self.radius_x**2 * (1 - ((y - self.y)**2) / self.radius_y**2)

		if sqrt_expr < 0:
			return []
		elif sqrt_expr == 0:
			return [self.x]
		else:
			sqrt_val = sqrt(sqrt_expr)
			return [self.x - sqrt_val, self.x + sqrt_val]

	@property
	def area(self):
		return pi * self.radius_x * self.radius_y
	@property
	def perimeter(self) -> float:
		return 2 * pi * sqrt( (self.radius_x**2 * self.radius_y**2)/2 )

	@property
	def as_geometry(self):
		return f'({self.x} - x)**2/{self.radius_x}**2 + ({self.y} - y)**2/{self.radius_y}**2 = 1'

	@property
	def random_point(self):
		return self(r.random())

	@property
	def center_of_mass(self) -> 'Point':
		return self.center

	@property
	def x(self) -> float:
		return self.center.x
	@property
	def y(self) -> float:
		return self.center.y
	@property
	def z(self) -> float:
		return self.center.z
	@property
	def w(self) -> float:
		return self.center.w

	@property
	def dimension(self) -> int:
		return self.space.dimension
	@property
	def projection(self) -> 'Circle':
		if self.dimension > 2:
			return Ellipse2D(self.center.project_to(2), self.vec1, self.vec2, name=self.name, color=self.color,
							 alpha=self.alpha)
		else:
			return self
	
	def __contains__(self, object):
		return self.intersects(object)

	def __add__(self, point: Union[Point, list, tuple]) -> 'Ellipse':
		if isinstance(point, (list,tuple,np.ndarray)):
			point = Point(*point)
		return self.__class__(self.center + point, self.vec1, self.vec2, name=self.name, color=self.color, alpha=self.alpha)
	def __sub__(self, point: Union[Point, list, tuple]) -> 'Ellipse':
		if isinstance(point, (list,tuple,np.ndarray)):
			point = Point(*point)
		return self.__class__(self.center - point, self.vec1, self.vec2, name=self.name, color=self.color, alpha=self.alpha)
	def __mul__(self, point: Union[Point, list, tuple]) -> 'Ellipse':
		if isinstance(point, (list,tuple,np.ndarray)):
			point = Point(*point)
		return self.__class__(self.center * point, self.vec1, self.vec2, name=self.name, color=self.color, alpha=self.alpha)
	def __truediv__(self, point: Union[Point, list, tuple]) -> 'Ellipse':
		if isinstance(point, (list,tuple,np.ndarray)):
			point = Point(*point)
		return self.__class__(self.center / point, self.vec1, self.vec2, name=self.name, color=self.color, alpha=self.alpha)
	def __floordiv__(self, point: Union[Point, list, tuple]) -> 'Ellipse':
		if isinstance(point, (list,tuple,np.ndarray)):
			point = Point(*point)
		return self.__class__(self.center // point, self.vec1, self.vec2, name=self.name, color=self.color, alpha=self.alpha)
	def __pow__(self, point: Union[Point, list, tuple]) -> 'Ellipse':
		if isinstance(point, (list,tuple,np.ndarray)):
			point = Point(*point)
		return self.__class__(self.center ** point, self.vec1, self.vec2, name=self.name, color=self.color, alpha=self.alpha)

	# 'num' is a number from 0 to 1
	# Returns point at a Circle that makes with second point sector of a circle like angle (0 -> 0 deg; 1 -> 360 deg)
	def __call__(self, num: Union[float, int]) -> Point:
		local_ray = Ray.by_angle(num * 360, pos1=self.center)
		global_ray = self.space.to_global(local_ray)
		return self.projection.intersects(global_ray, check_inside=False)[0]

	def __str__(self):
		return f"{self.name}([{self.center}], radius_x={self.radius_x}, radius_y={self.radius_y})"
	def __repr__(self):
		return f"{self.__class__.__name__}({self.center}, {self.radius_x}, {self.radius_y}, name={self.name})"


class Circle(Ellipse):
	def __init__(self, center: 'Point', radius: Union[int, float], vec1: Vector, vec2: Vector,
				 name: str = 'Circle', color: str = 'purple', alpha: Union[int, float] = 1):

		if not isinstance(radius, (float, int)):
			raise ConstructError(
				f'Incorrect radius for circle: it must be positive number, not {radius}, {type(radius)}')
		if radius <= 0:
			raise ConstructError(
				f'Incorrect radius for circle: it must be positive number, not {radius}, {type(radius)}')
		if isinstance(vec1, (tuple, list, np.ndarray)):
			vec1 = Vector[vec1]
		if isinstance(vec2, (tuple, list, np.ndarray)):
			vec2 = Vector[vec2]

		self.radius = radius
		self.diameter = radius * 2
		super().__init__(center, vec1*radius, vec2*radius, name=name, color=color, alpha=alpha)

	def at_pos(self, position: List[[int, int]]) -> 'Ellipse':
		if self.center != position:
			return self.__class__(position, self.radius, *self.space.vectors, name=self.name, color=self.color,
								  alpha=self.alpha)
		return self.copy()

	def scale(self, factor: float) -> 'Circle':
		return self.__class__(self.pos, self.radius * factor, *self.space.vectors, name=self.name, color=self.color,
							  alpha=self.alpha)

	def y_from_x(self, x: float) -> float:
		d = self.radius ** 2 - (x - self.x) ** 2
		if d == 0:
			return [self.y]
		elif d > 0:
			sqrtd = sqrt(d)
			return [
				-sqrtd + self.y,
				+sqrtd + self.y,
			]
	def x_from_y(self, y: float) -> float:
		d = self.radius ** 2 - (y - self.y) ** 2
		if d == 0:
			return [self.x]
		elif d > 0:
			sqrtd = sqrt(d)
			return [
				-sqrtd + self.x,
				+sqrtd + self.x,
			]

	@property
	def area(self) -> float:
		return pi * self.radius ** 2
	@property
	def perimeter(self) -> float:
		return 2 * pi * self.radius

	@property
	def projection(self) -> 'Circle':
		if self.dimension > 2:
			return Circle2D(self.center.project_to(2), self.radius, name=self.name, color=self.color, alpha=self.alpha)
		else:
			return self

	@property
	def as_geometry(self) -> str:
		return f'(x - {self.x})**2 + (y - {self.y})**2 = {self.radius}**2'

	def __str__(self):
		return f"{self.name}({self.center}, radius={self.radius})"
	def __repr__(self):
		return f"{self.__class__.__name__}({self.center}, {self.radius}, name='{self.name}')"


def Circle2D(center: Point, radius: Union[int, float], name: str = 'Circle', color: str = 'purple',
			 alpha: Union[int, float] = 1):
	return Circle(center, radius, [1, 0], [0, 1], name=name, color=color, alpha=alpha)

def Ellipse2D(center: Point, rx: Union[int, float], ry: Union[int, float], name: str = 'Circle', color: str = 'red',
			 alpha: Union[int, float] = 1):
	return Ellipse(center, Vector[1, 0] * rx, Vector[0, 1] * ry, name=name, color=color, alpha=alpha)