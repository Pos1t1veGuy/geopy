import matplotlib.pyplot as plt

from typing import *
from matplotlib.patches import Polygon as MPLPolygon, Ellipse as MPLEllipse
from matplotlib.lines import Line2D

from .primitives import *
from .shapes import *


class Scene:
	def __init__(self, *args):
		self.fig, self.ax = plt.subplots()

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
		elif isinstance(object, Composite):
			for shape in object.shapes:
				self.add(shape)
		elif isinstance(object, PrimitiveGroup):
			for pr in object.primitives:
				self.add(pr)
		else:
			raise ValueError(f'Unexpected type {type(object)}. Object must be Shape/Primitive/Point')

	def add_circle(self, circle: Circle):
		self.ax.add_patch( MPLEllipse(xy=circle.center, width=circle.diameter, height=circle.diameter, edgecolor='r', fc='None') )

	def add_polygon(self, polygon: Polygon):
		vertices = [(point.x, point.y) for point in polygon.vertices]
		polygon = MPLPolygon(vertices, edgecolor='b', fc='None')
		self.ax.add_patch(polygon)

	def add_segment(self, segment: Segment):
		self.ax.add_line(Line2D([segment.pos1.x, segment.pos2.x], [segment.pos1.y, segment.pos2.y], linestyle='--', color='g', marker='.'))

	def add_ray(self, ray: Ray):
		self.ax.add_line(Line2D([ray.pos1.x, ray.pos2.x], [ray.pos1.y, ray.pos2.y], color='r', marker='.', markevery=[0], linewidth=2))
		self.ax.annotate('', xy=(ray.pos2.x, ray.pos2.y), xytext=(ray.pos1.x, ray.pos1.y), arrowprops=dict(arrowstyle='->', color='r'))

	def add_vector(self, vector: Vector):
		self.ax.add_line(Line2D([vector.pos1.x, vector.pos2.x], [vector.pos1.y, vector.pos2.y], color='c', marker='.', linewidth=2))
		self.ax.annotate('', xy=(vector.pos2.x, vector.pos2.y), xytext=(vector.pos1.x, vector.pos1.y), arrowprops=dict(arrowstyle='->', color='c'))

	def add_line(self, line: Line):
		self.ax.add_line(Line2D([line.pos1.x, line.pos2.x], [line.pos1.y, line.pos2.y], color='y'))

	def add_point(self, point: Point):
		self.ax.scatter(point.x, point.y, color='m', marker='o')

	def show(self):
		self.ax.axis('equal')
		plt.show()