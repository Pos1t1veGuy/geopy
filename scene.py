import matplotlib.pyplot as plt

from typing import *
from matplotlib.patches import Polygon as MPLPolygon, Ellipse as MPLEllipse
from matplotlib.lines import Line2D

from .primitives import *
from .shapes import *


class Scene2D:
	def __init__(self, *args):
		self.fig, self.ax = plt.subplots()

		for object in args:
			self.add(object)

	def add(self, object: Union[Shape, Primitive, Point]):
		if object.dimension == 2:
			if isinstance(object, Point):
				self.add_point(object)
			elif isinstance(object, Vector):
				self.add_vector(object)
			elif isinstance(object, Segment):
				self.add_segment(object)
			elif isinstance(object, Ray):
				self.add_ray(object)
			elif isinstance(object, Line):
				self.add_line(object)
			elif isinstance(object, Polygon):
				self.add_polygon(object)
			elif isinstance(object, Circle):
				self.add_circle(object)
			elif isinstance(object, Composite):
				for shape in object.shapes:
					self.add(shape)
			elif isinstance(object, PrimitiveGroup):
				for pr in object.primitives:
					self.add(pr)
			else:
				raise ValueError(f'Unexpected type {type(object)}. Object must be Shape/Primitive/Point')
		else:
			raise ValueError(f'Unexpected dimension {object.dimension}. Object must be 2D')

	def add_circle(self, circle: Circle):
		if circle.dimension == 2:
			self.ax.add_patch( MPLEllipse(xy=circle.center, width=circle.diameter, height=circle.diameter, edgecolor='r', fc='None') )
		else:
			raise ValueError(f'Unexpected dimension {object.dimension}. Object must be 2D')

	def add_polygon(self, polygon: Polygon):
		if polygon.dimension == 2:
			vertices = [(point.x, point.y) for point in polygon.vertices]
			polygon = MPLPolygon(vertices, edgecolor='b', fc='None')
			self.ax.add_patch(polygon)
		else:
			raise ValueError(f'Unexpected dimension {object.dimension}. Object must be 2D')

	def add_segment(self, segment: Segment):
		if segment.dimension == 2:
			self.ax.add_line(Line2D([segment.pos1.x, segment.pos2.x], [segment.pos1.y, segment.pos2.y], linestyle='--', color='g', marker='.'))
		else:
			raise ValueError(f'Unexpected dimension {object.dimension}. Object must be 2D')

	def add_ray(self, ray: Ray):
		if ray.dimension == 2:
			self.ax.add_line(Line2D([ray.pos1.x, ray.pos2.x], [ray.pos1.y, ray.pos2.y], color='r', marker='.', markevery=[0], linewidth=2))
			self.ax.annotate('', xy=(ray.pos2.x, ray.pos2.y), xytext=(ray.pos1.x, ray.pos1.y), arrowprops=dict(arrowstyle='->', color='r'))
		else:
			raise ValueError(f'Unexpected dimension {object.dimension}. Object must be 2D')

	def add_vector(self, vector: Vector):
		if vector.dimension == 2:
			self.ax.add_line(Line2D([vector.pos1.x, vector.pos2.x], [vector.pos1.y, vector.pos2.y], color='c', marker='.', linewidth=2))
			self.ax.annotate('', xy=(vector.pos2.x, vector.pos2.y), xytext=(vector.pos1.x, vector.pos1.y), arrowprops=dict(arrowstyle='->', color='c'))
		else:
			raise ValueError(f'Unexpected dimension {object.dimension}. Object must be 2D')

	def add_line(self, line: Line):
		if line.dimension == 2:
			self.ax.add_line(Line2D([line.pos1.x, line.pos2.x], [line.pos1.y, line.pos2.y], color='y'))
		else:
			raise ValueError(f'Unexpected dimension {object.dimension}. Object must be 2D')

	def add_point(self, point: Point):
		if point.dimension == 2:
			self.ax.scatter(point.x, point.y, color='m', marker='o')
		else:
			raise ValueError(f'Unexpected dimension {object.dimension}. Object must be 2D')

	def show(self):
		self.ax.axis('equal')
		plt.show()


class Composite(Shape):
	def __init__(self, *args: List[Shape], name: str = 'Composite', pos: Point = [0,0]):
		self.pos = Point(*pos) if isinstance(pos, (tuple, list)) else (pos if isinstance(pos, Point) else Point(0,0))
		self.name = name
		self.shapes = []

		if len(args) == 0:
			raise ValueError(f'Composite must contain at least one Shape object')

		for i, shape in enumerate(args):
			if isinstance(shape, Composite):
				for sh in shape.shapes:
					self.shapes.append(sh)
			elif isinstance(shape, Shape):
				self.shapes.append(shape)
			else:
				raise ValueError(f'Composite constructor arg shapes[{i}] is not a Shape: {shape}')

	def intersects(self, object: Union[Primitive, Shape], check_inside: bool = True) -> List[Point]:
		if not isinstance(object, Composite):
			res = []
			for shape in self.shapes:
				for point in shape.intersects(object, check_inside=check_inside):
					res.append(point)
					
			return res
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
		y = [ shape.center_of_mass.y for shape in self.shapes ]
		x = [ shape.center_of_mass.x for shape in self.shapes ]
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
		return int(self.shapes)
	def __str__(self):
		return f'{self.name}([{self.pos}], {len(self.shapes)} shapes)'
	def __repr__(self):
		return f'{self.__class__.__name__}({self.shapes}, name="{self.name}", pos={self.pos})'

class PrimitiveGroup:
	def __init__(self, *args: List[Primitive], name: str = 'PrimitiveGroup'):
		self.primitives = []
		self.name = name

		if len(args) < 1:
			raise ValueError(f'PrimitiveGroup must contain at least one Primitive object')

		for i, pr in enumerate(args):
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

	def __len__(self):
		return len(self.primitives)
	def __str__(self):
		return f'{self.name}({len(self.primitives)} primitives)'
	def __repr__(self):
		return f'{self.__class__.__name__}({self.primitives}, name="{self.name}")'