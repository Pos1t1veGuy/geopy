from typing import *
from itertools import product

from .shapes2d import *
from .primitives import *


class Shape3D(Shape):
	dimension = 3


class Parallelepiped(Shape3D):
	def __init__(self, pos1: Union['Point', list, tuple], pos2: Union['Point', list, tuple], name: str = 'Parallelepiped'):
		if isinstance(pos1, (list, tuple)):
			pos1 = Point(*pos1)
		if isinstance(pos2, (list, tuple)):
			pos2 = Point(*pos2)

		self.name = name
		self.pos1, self.pos2 = pos1, pos2

		if self.dimension <= 2:
			raise ValueError(f'Vertices must to be not on the 2D plane')

		self.vertices = self._vertices(pos1, pos2)
		self.vector_x = Vector(pos1, pos2) + Vector(pos2, Point(pos1.x, pos1.y, pos2.z))
		self.vector_y = Vector(pos1, pos2) + Vector(pos2, Point(pos2.x, pos1.y, pos1.z))
		self.vector_z = Vector(pos1, pos2) + Vector(pos2, Point(pos1.x, pos2.y, pos1.z))

	def _vertices(self, pos1: 'Point', pos2: 'Point') -> List['Point']:
		return [Point(x, y, z) for x, y, z in product([pos1.x, pos2.x], [pos1.y, pos2.y], [pos1.z, pos2.z])]

	def intersects(self, object: Union[Primitive, Shape, Point, tuple, list], check_inside: bool = True):
		if isinstance(object, (tuple, list)):
			object = Point(*object)

		if isinstance(object, Point):
			for edge in self.edge:
				if object in edge:
					return [object]

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
		elif isinstance(object, Shape):
			point = object.center_of_mass
		else:
			raise ValueError(f'"inside" method takes Union[Primitive, Shape, Point, tuple, list], not {object}')

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

	@property
	def edges(self) -> List['Polygon']:
		return [
			Polygon(self.vertices[0], self.vertices[1], self.vertices[3], self.vertices[2], name='Polygon0'),  # down
			Polygon(self.vertices[4], self.vertices[5], self.vertices[7], self.vertices[6], name='Polygon1'),  # up
			Polygon(self.vertices[0], self.vertices[1], self.vertices[5], self.vertices[4], name='Polygon2'),  # front
			Polygon(self.vertices[2], self.vertices[3], self.vertices[7], self.vertices[6], name='Polygon3'),  # back
			Polygon(self.vertices[0], self.vertices[2], self.vertices[6], self.vertices[4], name='Polygon4'),  # left
			Polygon(self.vertices[1], self.vertices[3], self.vertices[7], self.vertices[5], name='Polygon5'),  # right
		]

	@property
	def segments(self) -> List['Segment']:
		segments = []
		for i, edge in enumerate(self.edges):
			for j, segment in enumerate(edge.segments):
				if not segment.center in segments:
					seg = segment.copy()
					seg.name = f'Segment{len(segments)}'
					segments.append(seg)
		return segments

	@property
	def segments_centers(self) -> List['Point']:
		return [ segment.center for segment in self.segments ]

	@property
	def dimension(self) -> int:
		return max(self.pos1.dimension, self.pos2.dimension)

	@property
	def center_of_mass(self) -> 'Point':
		return Point([
			sum([ vertice[i] for vertice in self.vertices ])/len(self.vertices) for i in range(self.dimension)
		])

	@property
	def min_pos(self) -> float:
		return Point([ min([vertice[i] for vertice in self.vertices]) for i in range(self.dimension) ], name=f'{self.name}_min')
	@property
	def max_pos(self) -> float:
		return Point([ max([vertice[i] for vertice in self.vertices]) for i in range(self.dimension) ], name=f'{self.name}_max')

	@property
	def volume(self) -> float:
		return self.vector_x.length * self.vector_y.length * self.vector_z.length
	@property
	def area(self) -> float:
		return sum([ edge.area for edge in self.edges ])
	@property
	def perimeter(self) -> float:
		return sum([ segment.length for segment in self.segments ])

	def __len__(self):
		return int(self.perimeter)
	def __str__(self):
		return f'{self.name}({self.center_of_mass}, volume={self.volume}, area={self.area})'
	def __repr__(self):
		return f'{self.__class__.__name__}({self.pos1}, {self.pos2} name="{self.name}")'
Box = Parallelepiped


class Tetrahedron(Shape3D):
	def __init__(self, *args: Union['Point', list, tuple], name: str = 'Tetrahedron'):
		self.vertices = []
		for vertice in args:
			if isinstance(vertice, (list, tuple)):
				vertice = Point(*vertice)

			self.vertices.append(vertice)

		self.name = name

		if self.dimension <= 2:
			raise ValueError(f'Vertices must to be not on the 2D plane')

	def intersects(self, object: Union[Primitive, Shape, Point, tuple, list], check_inside: bool = True):
		if isinstance(object, (tuple, list)):
			object = Point(*object)

		if isinstance(object, Point):
			for edge in self.edge:
				if object in edge:
					return [object]

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
		elif isinstance(object, Shape):
			point = object.center_of_mass
		else:
			raise ValueError(f'"inside" method takes Union[Primitive, Shape, Point, tuple, list], not {object}')

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

	@property
	def edges(self) -> List['Polygon']:
		edges = []
		for i, vertice in enumerate(self.vertices):
			poly_vertices = self.vertices.copy()
			vertices.remove(vertice)
			edges.append(Polygon(poly_vertices, name=f'Polygon{i}'))
		return edges

	@property
	def segments(self) -> List['Segment']:
		segments = []
		for vert1 in self.vertices:
			for vert2 in self.vertices:
				segment = Segment(vert1, vert2, name=f'Segment{len(segments)}')
				if not segment in segments:
					segments.append(segment)
		return segments

	@property
	def segments_centers(self) -> List['Point']:
		return [ segment.center for segment in self.segments ]

	@property
	def dimension(self) -> int:
		return max([ vertice.dimension for vertice in self.vertices ])

	@property
	def center_of_mass(self) -> 'Point':
		return Point([
			sum([ vertice[i] for vertice in self.vertices ])/len(self.vertices) for i in range(self.dimension)
		])

	@property
	def min_pos(self) -> float:
		return Point([ min([vertice[i] for vertice in self.vertices]) for i in range(self.dimension) ], name=f'{self.name}_min')
	@property
	def max_pos(self) -> float:
		return Point([ max([vertice[i] for vertice in self.vertices]) for i in range(self.dimension) ], name=f'{self.name}_max')

	@property
	def volume(self) -> float:
		return self.width * self.height * self.length
	@property
	def area(self) -> float:
		return sum([ edge.area for edge in self.edges ])
	@property
	def perimeter(self) -> float:
		return sum([ segment.length for segment in self.segments ])

	def __len__(self):
		return int(self.perimeter)
	def __str__(self):
		return f'{self.name}({self.center_of_mass}, volume={self.volume}, area={self.area})'
	def __repr__(self):
		return f'{self.__class__.__name__}({self.pos1}, {self.pos2} name="{self.name}")'