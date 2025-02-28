import matplotlib.pyplot as plt

from typing import *
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon as MPLPolygon, Ellipse as MPLEllipse
from matplotlib.lines import Line2D

from .primitives import *
from .shapes2d import *
from .shapes3d import *
from .exceptions import *


# Matplotlib interface


class Scene:
	def save(self, filepath: str):
		plt.savefig(filepath)


class Scene2D(Scene):
	def __init__(self, *args):
		self.fig, self.ax = plt.subplots()
		self.lines = []
		self.rays = []
		self.points = []
		self.objects = args

		for object in args:
			self.add(object)

	def add(self, object: Union[Shape, Primitive, Point]):
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
		elif isinstance(object, Oval):
			self.add_oval(object)
		elif isinstance(object, Composite):
			self.add_composite(object)
		elif isinstance(object, PrimitiveGroup):
			for pr in object.primitives:
				self.add(pr)
		else:
			raise SceneError(f'Unexpected type {type(object)}. Object must be Shape/Primitive/Point')

	def add_composite(self, composite: 'Composite'):
		# for shape in object.shapes:
		# 	self.add(shape)
		pol = composite.polygons[0]
		c = composite.circles[0]
		

	def add_circle(self, circle: Circle):
		self.points.append(circle.center)

		self.points.append(circle.center - [circle.radius, 0])
		self.points.append(circle.center + [circle.radius, 0])
		self.points.append(circle.center - [0, circle.radius])
		self.points.append(circle.center + [0, circle.radius])

		self.ax.add_patch(
			MPLEllipse(xy=(circle.center.x, circle.center.y), width=circle.diameter, height=circle.diameter, edgecolor=circle.color, fc='None', alpha=circle.alpha)
		)
	
	def add_oval(self, oval: Oval):
		self.points.append(oval.center)

		self.points.append(oval.center - [oval.radius_x, 0])
		self.points.append(oval.center + [oval.radius_x, 0])
		self.points.append(oval.center - [0, oval.radius_y])
		self.points.append(oval.center + [0, oval.radius_y])
		
		self.ax.add_patch(
			MPLEllipse(xy=(oval.center.x, oval.center.y), width=oval.diameter_x, height=oval.diameter_y, edgecolor=oval.color, fc='None', alpha=oval.alpha)
		)

	def add_polygon(self, polygon: Polygon):
		for point in polygon.vertices:
			self.points.append(point)

		polygon = MPLPolygon([(point.x, point.y) for point in polygon.vertices], edgecolor=polygon.segments_color, facecolor=polygon.bg_color, alpha=polygon.alpha)
		self.ax.add_patch(polygon)

	def add_segment(self, segment: Segment):
		self.ax.add_line(
			Line2D([segment.pos1.x, segment.pos2.x], [segment.pos1.y, segment.pos2.y], linestyle='--', color=segment.color, marker='.', alpha=segment.alpha)
		)
		self.points.append(segment.pos1)
		self.points.append(segment.pos2)

	def add_ray(self, ray: Ray):
		self.rays.append(ray)
		self.points.append(ray.pos1)
		circle = Circle(ray.pos1, 6)
		self.points.append(circle.intersects(ray)[0])

	def add_vector(self, vector: Vector):
		self.ax.add_line(Line2D([vector.pos1.x, vector.pos2.x], [vector.pos1.y, vector.pos2.y], color=vector.color, marker='.', linewidth=2, alpha=vector.alpha))
		self.ax.annotate('', xy=(vector.pos2.x, vector.pos2.y), xytext=(vector.pos1.x, vector.pos1.y), arrowprops=dict(arrowstyle='->', color=vector.color))
		self.points.append(vector.pos1)
		self.points.append(vector.pos2)

	def add_line(self, line: Line):
		self.lines.append(line)

	def add_point(self, point: Point):
		self.ax.scatter(point.x, point.y, color=point.color, marker='o', zorder=5, s=point.size, alpha=point.alpha)
		self.points.append(point)

	def draw_line(self, line: Line):
		self.ax.add_line(Line2D([line.pos1.x, line.pos2.x], [line.pos1.y, line.pos2.y], color=line.color, alpha=line.alpha))

	def show(self):
		if self.points:
			min_x = min([ point.x for point in self.points ])-1
			min_y = min([ point.y for point in self.points ])-1
			max_x = max([ point.x for point in self.points ])+1
			max_y = max([ point.y for point in self.points ])+1

			min_point = Point[min_x, min_y]
			max_point = Point[max_x, max_y]

			if max_point != min_point:
				if len(self.points) > 1:
					if min_point.x == max_point.x:
						y_distance = max_y - min_y
						min_point -= Point[y_distance/2, 0]
						max_point += Point[y_distance/2, 0]
					if min_point.y == max_point.y:
						x_distance = max_x - min_x
						min_point -= Point[0, x_distance/2]
						max_point += Point[0, x_distance/2]

				scene_rect = Rectangle(max_point, min_point)

			for ray in self.rays:
				if max_point != min_point:
					ions = scene_rect.intersects(ray)
					self.ax.add_line(
						Line2D([ray.pos1.x, ions[0].x], [ray.pos1.y, ions[0].y], color=ray.color, marker='.', markevery=[0], linewidth=2, alpha=ray.alpha)
					)
					self.ax.annotate('', xy=tuple(ions[0].axes), xytext=(ray.pos1.x, ray.pos1.y), arrowprops=dict(arrowstyle='->', color=ray.color))
				else:
					self.ax.add_line(
						Line2D([ray.pos1.x, ray.pos2.x], [ray.pos1.y, ray.pos2.y], color=ray.color, marker='.', markevery=[0], linewidth=2, alpha=ray.alpha)
					)
					self.ax.annotate('', xy=(ray.pos2.x, ray.pos2.y), xytext=(ray.pos1.x, ray.pos1.y), arrowprops=dict(arrowstyle='->', color=ray.color))

			for line in self.lines:
				ions = scene_rect.intersects(line)

				if ions:
					if all([ isinstance(point, Point) for point in ions ]):
						self.ax.add_line(Line2D([ions[0].x, ions[1].x], [ions[0].y, ions[1].y], color=line.color, alpha=line.alpha))
					else:
						for ion in ions:
							if isinstance(ion, (Point)):
								self.add(ion)
							elif isinstance(ion, (Line, Segment, Ray)):
								self.draw_line(ion)
							else:
								raise ConstructError(f'Intersection of Scene and Line gets invalid result {ions}, it must be Point, Line, Ray or Segment list')
				else:
					self.points.append(scene_rect.center_of_mass.height_to(line).pos2)
					self.show()
					return
				
			# Scene draws lines and rays to the end by the min and max points that makes box. Intersection with box is a finish point of line and ray
		else:
			for line in self.lines:
				self.ax.add_line(Line2D([line.pos1.x, line.pos2.x], [line.pos1.y, line.pos2.y], color=line.color, alpha=line.alpha))

		self.ax.axis('equal')
		plt.show()

class Scene3D(Scene):
	def __init__(self, *args):
		self.fig = plt.figure()
		self.ax = self.fig.add_subplot(111, projection='3d')
		self.lines = []
		self.rays = []
		self.points = []
		self.objects = args

		for object in args:
			self.add(object)

	def add(self, object: Union[Shape, Primitive, Point]):
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
		elif isinstance(object, Shape3D):
			self.add_shape3d(object)
		elif isinstance(object, Composite):
			for shape in object.shapes:
				self.add(shape)
		elif isinstance(object, PrimitiveGroup):
			for pr in object.primitives:
				self.add(pr)
		else:
			raise SceneError(f'Unexpected type {type(object)}. Object must be Shape/Primitive/Point')

	def add_point(self, point: Point):
		self.ax.scatter(float(point.x), float(point.y), float(point.z), color=point.color, marker='o', alpha=point.alpha)
		self.points.append(point)

	def add_vector(self, vector: Vector):
		self.ax.plot([vector.pos1.x, vector.pos2.x], [vector.pos1.y, vector.pos2.y], [vector.pos1.z, vector.pos2.z], color=vector.color, marker='.', alpha=vector.alpha)
		self.points.append(vector.pos1)
		self.points.append(vector.pos2)

	def add_segment(self, segment: Segment):
		self.ax.plot(
			[segment.pos1.x, segment.pos2.x], [segment.pos1.y, segment.pos2.y], [segment.pos1.z, segment.pos2.z],
			linestyle='--', color=segment.color, marker='.', alpha=segment.alpha,
		)
		self.points.append(segment.pos1)
		self.points.append(segment.pos2)
		
	def add_ray(self, ray: Ray):
		self.rays.append(ray)
		self.points.append(ray.pos1)

	def add_line(self, line: Line):
		self.lines.append(line)
		self.points.append(line.pos1)
		self.points.append(line.pos2)

	def add_circle(self, circle: Circle):
		cx, cy, cz = circle.center_of_mass.x, circle.center_of_mass.y, circle.center_of_mass.z
		r = circle.radius

		theta = np.linspace(0, 2 * np.pi, 100)
		x = r * np.cos(theta) + cx
		y = r * np.sin(theta) + cy
		z = np.full_like(x, cz)

		self.ax.plot(x, y, z, color=circle.color, alpha=circle.alpha)

		self.points.append(circle.center)

		self.points.append(circle.center - [circle.radius, 0])
		self.points.append(circle.center + [circle.radius, 0])
		self.points.append(circle.center - [0, circle.radius])
		self.points.append(circle.center + [0, circle.radius])

	def add_shape3d(self, shape: Shape3D):
		for polygon in shape.edges:
			self.add_polygon(polygon)

	def add_polygon(self, polygon: Polygon):
		for point in polygon.vertices:
			self.points.append(point)

		verts = [[ tuple([float(axis) for axis in point]) for point in eq_len_axeslists( *map(lambda point: point.axes, polygon.vertices), dimension=3 ) ]]
		polygon = Poly3DCollection(verts, alpha=polygon.alpha, facecolor=polygon.bg_color, edgecolor=polygon.segments_color, color=polygon.color)
		self.ax.add_collection3d(polygon)

	def draw_line(self, line: Line):
		self.ax.plot([line.pos1.x, line.pos2.x], [line.pos1.y, line.pos2.y], [line.pos1.z, line.pos2.z], color=line.color, alpha=line.alpha)

	def show(self):
		if self.points:
			min_x = min([ point.x for point in self.points ])
			min_y = min([ point.y for point in self.points ])

			max_x = max([ point.x for point in self.points ])
			max_y = max([ point.y for point in self.points ])

			min_z = min([ point.z for point in self.points ])
			max_z = max([ point.z for point in self.points ])

			min_point = Point[min_x, min_y, min_z]
			max_point = Point[max_x, max_y, max_z]

			if max_point.axes != min_point.axes:
				if max_point.x != min_point.x and max_point.y != min_point.y and max_point.z != min_point.z:
					scene_rect = Box(max_point, min_point)
				else:
					scene_rect = Rectangle(max_point, min_point)
			else:
				scene_rect = Box(max_point, [0,0,0])

			for ray in self.rays:
				if max_point != min_point:
					ions = scene_rect.intersects(ray)
					self.ax.add_line(
						Line2D([ray.pos1.x, ions[0].x], [ray.pos1.y, ions[0].y], color=ray.color, marker='.', markevery=[0], linewidth=2, alpha=ray.alpha)
					)
					self.ax.annotate('', xy=tuple(ions[0].axes), xytext=(ray.pos1.x, ray.pos1.y), arrowprops=dict(arrowstyle='->', color=ray.color))
				else:
					self.ax.add_line(
						Line2D([ray.pos1.x, ray.pos2.x], [ray.pos1.y, ray.pos2.y], color=ray.color, marker='.', markevery=[0], linewidth=2, alpha=ray.alpha)
					)
					self.ax.annotate('', xy=(ray.pos2.x, ray.pos2.y), xytext=(ray.pos1.x, ray.pos1.y), arrowprops=dict(arrowstyle='->', color=ray.color))

			for line in self.lines:
				ions = scene_rect.intersects(line)
				self.draw_line(line) #TODO: поправить
				#self.ax.plot([ions[0].x, ions[1].x], [ions[0].y, ions[1].y], [ions[0].z, ions[1].z], color=line.color)

			# Scene draws lines and rays to the end by the min and max points that makes box. Intersection with box is a finish point of line and ray
		else:
			for line in self.lines:
				self.ax.plot([line.pos1.x, line.pos2.x], [line.pos1.y, line.pos2.y], [line.pos1.z, line.pos2.z], color=line.color)

		plt.show()


class Composite(Shape):
	def __init__(self, *args: List[Shape], name: str = 'Composite', pos: Point = [0,0], mode='intersection'):
		self.pos = Point(*pos) if isinstance(pos, (tuple, list)) else (pos if isinstance(pos, Point) else Point(0,0))
		self.name = name
		self.shapes = []
		self.mode = mode

		if len(args) == 0:
			raise ConstructError(f'Composite must contain at least one Shape object')

		for i, shape in enumerate(args):
			if isinstance(shape, Composite):
				for sh in shape.shapes:
					self.shapes.append(sh)
			elif isinstance(shape, Shape):
				self.shapes.append(shape)
			else:
				raise ConstructError(f'Composite constructor arg shapes[{i}] is not a Shape: {shape}')

	def area(self):
		if self.mode == 'intersection':
			return self.intersect_area()
		elif self.mode == 'union':
			return self.union_area()
		elif self.mode == 'difference':
			return self.difference_area()
		elif self.mode == 'xor':
			return self.xor_area()
		else:
			raise ValueError(f"Unknown mode: {self.mode}")

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

	# def circles_intersections(self) -> dict:
	# 	ions = {}
	# 	if len(self.circles) > 1:
	# 		for circle in self.circles:
	# 			ions[circle] = [c.intersects(circle) for c in self.circles if c != circle]
		
	# 	return ions

	def plot(self):
		if self.dimension <= 2:
			Scene2D(*self.shapes, self.center_of_mass).show()
		else:
			Scene3D(*self.shapes, self.center_of_mass).show()

	@property
	def circles(self) -> List[Circle]:
		return [shape for shape in self.shapes if isinstance(shape, Circle)]
	@property
	def polygons(self) -> List[Circle]:
		return [shape for shape in self.shapes if isinstance(shape, Polygon)]

	@property
	def intersections(self) -> List[Point]:
		points = []
		for shape1 in self.shapes:
			for shape2 in self.shapes:
				if shape1 in shape2 and shape1 != shape2:
					res = shape1.intersects(shape2)
					for point in res:
						points.append(point)

		res = []
		for point in points:
			if not point in res:
				res.append(point)

		return res

	@property
	def center_of_mass(self) -> float:
		y = [ shape.center_of_mass.y for shape in self.shapes ]
		x = [ shape.center_of_mass.x for shape in self.shapes ]
		return Point(sum(x)/len(x), sum(y)/len(y), name=f'{self.name} center')

	@property
	def union_area(self) -> float:
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
			raise ConstructError(f'PrimitiveGroup must contain at least one Primitive object')

		for i, pr in enumerate(args):
			if isinstance(pr, Primitive):
				self.primitives.append(pr)
			else:
				raise ConstructError(f'Composite constructor arg primitives[{i}] is not a Primitive: {pr}')

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
		if self.dimension <= 2:
			Scene2D(*self.primitives).show()
		else:
			Scene3D(*self.primitives).show()

	@property
	def dimension(self) -> int:
		return max([ shape.dimension for shape in self.shapes])

	@property
	def intersections(self) -> List[Point]:
		points = []
		for pr1 in self.primitives:
			for pr2 in self.primitives:
				if pr1 in pr2 and pr1 != pr2:
					res = pr1.intersects(pr2)
					for point in res:
						points.append(point)

		res = []
		for point in points:
			if not point in res:
				res.append(point)

		return res
	
	def __contains__(self, object):
		return self.intersects(object)

	def __len__(self):
		return len(self.primitives)
	def __str__(self):
		return f'{self.name}({len(self.primitives)} primitives)'
	def __repr__(self):
		return f'{self.__class__.__name__}({self.primitives}, name="{self.name}")'