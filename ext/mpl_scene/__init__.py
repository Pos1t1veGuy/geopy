import matplotlib.pyplot as plt

from typing import *
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Polygon as MPLPolygon, Ellipse as MPLEllipse
from matplotlib.lines import Line2D as mpl_Line2D

from geopy.core import *

# Matplotlib interface

__all__ = ['Scene', 'Scene2D', 'Scene3D']

class Scene:
    def add(self, object: Union['Shape', 'Primitive', 'Point']):
        object = object.project_to(self.dimension)
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
        elif isinstance(object, Ellipse): # TODO: поменять с кругом местами
            self.add_ellipse(object)
        elif isinstance(object, Circle):
            self.add_circle(object)
        else:
            raise SceneError(f'Unexpected type {type(object)}. Object must be Shape/Primitive/Point')

    def save(self, filepath: str):
        self.make_scene()
        plt.savefig(filepath)

    def show(self):
        self.make_scene()
        plt.show()


class Scene2D(Scene):
    def __init__(self, *args):
        self.fig, self.ax = plt.subplots()
        self.lines = []
        self.rays = []
        self.points = []
        self.objects = args
        self.dimension = 2

        for object in args:
            self.add(object)

    def add_circle(self, circle: 'Circle'): # TODO: поправить превращение круга в овал под нужным углом при проекции
        center_proj = circle.center.project_to(self.dimension)
        self.points.append(center_proj)

        self.points.append(center_proj - [circle.radius, 0])
        self.points.append(center_proj + [circle.radius, 0])
        self.points.append(center_proj - [0, circle.radius])
        self.points.append(center_proj + [0, circle.radius])

        self.ax.add_patch(
            MPLEllipse(xy=(circle.center.x, circle.center.y), width=circle.diameter, height=circle.diameter,
                       edgecolor=circle.color, fc='None', alpha=circle.alpha)
        )

    def add_ellipse(self, ellipse: 'Ellipse'):
        self.points.append(ellipse.center)

        self.points.append(ellipse.center - [ellipse.radius_x, 0])
        self.points.append(ellipse.center + [ellipse.radius_x, 0])
        self.points.append(ellipse.center - [0, ellipse.radius_y])
        self.points.append(ellipse.center + [0, ellipse.radius_y])

        self.ax.add_patch(
            MPLEllipse(xy=(ellipse.center.x, ellipse.center.y), width=ellipse.diameter_x, height=ellipse.diameter_y,
                       edgecolor=ellipse.color, fc='None', alpha=ellipse.alpha)
        )

    def add_polygon(self, polygon: 'Polygon'):
        for point in polygon.vertices:
            self.points.append(point)

        polygon = MPLPolygon([(point.x, point.y) for point in polygon.vertices], edgecolor=polygon.segments_color,
                             facecolor=polygon.color, alpha=polygon.alpha)
        self.ax.add_patch(polygon)

    def add_segment(self, segment: 'Segment'):
        self.ax.add_line(
            mpl_Line2D([segment.pos1.x, segment.pos2.x], [segment.pos1.y, segment.pos2.y], linestyle='--',
                   color=segment.color, marker='.', alpha=segment.alpha)
        )
        self.points.append(segment.pos1)
        self.points.append(segment.pos2)

    def add_ray(self, ray: 'Ray'):
        if ray.pos1.project_to(self.dimension) == ray.pos2.project_to(self.dimension):
            self.add(ray.pos1)
            return

        ray = ray.project_to(self.dimension)
        self.rays.append(ray)
        self.points.append(ray.pos1)
        circle = Circle2D(ray.pos1, 6)
        self.points.append(circle.intersects(ray)[0])

    def add_vector(self, vector: 'Vector'):
        if vector.pos1.project_to(self.dimension) == vector.pos2.project_to(self.dimension):
            self.add(vector.pos1.project_to(self.dimension))
            return

        self.ax.add_line(
            mpl_Line2D([vector.pos1.x, vector.pos2.x], [vector.pos1.y, vector.pos2.y], color=vector.color, marker='.',
                   linewidth=2, alpha=vector.alpha))
        self.ax.annotate('', xy=(vector.pos2.x, vector.pos2.y), xytext=(vector.pos1.x, vector.pos1.y),
                         arrowprops=dict(arrowstyle='->', color=vector.color))
        self.points.append(vector.pos1)
        self.points.append(vector.pos2)

    def add_line(self, line: 'Line'):
        if line.pos1.project_to(self.dimension) == line.pos2.project_to(self.dimension):
            self.add(line.pos1)
            return

        self.lines.append(line.project_to(self.dimension))
        self.points.append(line.pos1)
        self.points.append(line.pos2)

    def add_point(self, point: 'Point'):
        self.ax.scatter(point.x, point.y, color=point.color, marker='o', zorder=5, s=point.size, alpha=point.alpha)
        self.points.append(point)

    def draw_line(self, line: 'Line'):
        self.ax.add_line(
            mpl_Line2D([line.pos1.x, line.pos2.x], [line.pos1.y, line.pos2.y], color=line.color, alpha=line.alpha))

    def make_scene(self):
        if self.points:
            min_x = min([point.x for point in self.points]) - 1
            min_y = min([point.y for point in self.points]) - 1
            max_x = max([point.x for point in self.points]) + 1
            max_y = max([point.y for point in self.points]) + 1

            min_point = Point[min_x, min_y]
            max_point = Point[max_x, max_y]

            if max_point != min_point:
                if len(self.points) > 1:
                    if min_point.x == max_point.x:
                        y_distance = max_y - min_y
                        min_point -= Point[y_distance / 2, 0]
                        max_point += Point[y_distance / 2, 0]
                    if min_point.y == max_point.y:
                        x_distance = max_x - min_x
                        min_point -= Point[0, x_distance / 2]
                        max_point += Point[0, x_distance / 2]

                scene_rect = Rectangle(max_point, min_point)

            for ray in self.rays:
                if max_point != min_point:
                    ions = scene_rect.intersects(ray)
                    self.ax.add_line(
                        mpl_Line2D([ray.pos1.x, ions[0].x], [ray.pos1.y, ions[0].y], color=ray.color, marker='.',
                               markevery=[0], linewidth=2, alpha=ray.alpha)
                    )
                    self.ax.annotate('', xy=tuple(ions[0].axes), xytext=(ray.pos1.x, ray.pos1.y),
                                     arrowprops=dict(arrowstyle='->', color=ray.color))
                else:
                    self.ax.add_line(
                        mpl_Line2D([ray.pos1.x, ray.pos2.x], [ray.pos1.y, ray.pos2.y], color=ray.color, marker='.',
                               markevery=[0], linewidth=2, alpha=ray.alpha)
                    )
                    self.ax.annotate('', xy=(ray.pos2.x, ray.pos2.y), xytext=(ray.pos1.x, ray.pos1.y),
                                     arrowprops=dict(arrowstyle='->', color=ray.color))

            for line in self.lines:
                ions = scene_rect.intersects(line)

                if ions:
                    if all([isinstance(point, Point) for point in ions]):
                        self.ax.add_line(
                            mpl_Line2D([ions[0].x, ions[1].x], [ions[0].y, ions[1].y], color=line.color, alpha=line.alpha))
                    else:
                        for ion in ions:
                            if isinstance(ion, (Point)):
                                self.add(ion)
                            elif isinstance(ion, (Line, Segment, Ray)):
                                self.draw_line(ion)
                            else:
                                raise ConstructError(
                                    f'Intersection of Scene and Line gets invalid result {ions}, it must be Point, Line, Ray or Segment list')
                else:
                    self.ax.axis('equal')
                    self.points.append(scene_rect.center_of_mass.height_to(line).pos2)
                    return

        # Scene draws lines and rays to the end by the min and max points that makes box. Intersection with box is a finish point of line and ray
        else:
            for line in self.lines:
                self.ax.add_line(
                    mpl_Line2D([line.pos1.x, line.pos2.x], [line.pos1.y, line.pos2.y], color=line.color, alpha=line.alpha))

        self.ax.axis('equal')


class Scene3D(Scene):
    def __init__(self, *args):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.lines = []
        self.rays = []
        self.points = []
        self.vectors = []
        self.objects = args
        self.dimension = 3

        for object in args:
            self.add(object)

    def add(self, object: Union['Shape', 'Primitive', 'Point']):
        if isinstance(object, Shape3D):
            self.add_shape3d(object)
        else:
            return super().add(object)

    def add_point(self, point: 'Point'):
        self.ax.scatter(float(point.x), float(point.y), float(point.z), color=point.color, marker='o',
                        alpha=point.alpha)
        self.points.append(point)

    def add_vector(self, vector: 'Vector'):
        self.vectors.append(vector)
        self.points.append(vector.pos1)
        self.points.append(vector.pos2)

    def add_segment(self, segment: 'Segment'):
        self.ax.plot(
            [segment.pos1.x, segment.pos2.x], [segment.pos1.y, segment.pos2.y], [segment.pos1.z, segment.pos2.z],
            linestyle='--', color=segment.color, marker='.', alpha=segment.alpha,
        )
        self.points.append(segment.pos1)
        self.points.append(segment.pos2)

    def add_ray(self, ray: 'Ray'):
        self.rays.append(ray)
        self.points.append(ray.pos1)
        self.points.append(ray.pos2)

    def add_line(self, line: 'Line'):
        self.lines.append(line)
        self.points.append(line.pos1)
        self.points.append(line.pos2)

    def add_circle(self, circle: 'Circle'):
        r = circle.radius
        theta = np.linspace(0, 2 * np.pi, 100)

        local_points = [Point(r * np.cos(t), r * np.sin(t)) for t in theta]
        global_points = [circle.space.point_to_global(p) for p in local_points]

        self.ax.plot(
            [p.x for p in global_points],
            [p.y for p in global_points],
            [p.z for p in global_points],
            color=circle.color, alpha=circle.alpha)

        self.points.append(circle.center)
        self.points.append(circle.space.point_to_global(Point(-r, 0)))
        self.points.append(circle.space.point_to_global(Point(r, 0)))
        self.points.append(circle.space.point_to_global(Point(0, -r)))
        self.points.append(circle.space.point_to_global(Point(0, r)))

    def add_ellipse(self, ellipse: 'Ellipse'):
        theta = np.linspace(0, 2 * np.pi, 100)
        points = []
        for t in theta:
            vec1 = ellipse.space.zero_vectors[0] * np.cos(t) * ellipse.radius_x
            vec2 = ellipse.space.zero_vectors[1] * np.sin(t) * ellipse.radius_y
            points.append(
                ellipse.center + (vec1 if type(vec1)==Point else vec1.pos2) + (vec2 if type(vec2)==Point else vec2.pos2)
            )

        self.points.append(ellipse.center)
        self.points.append((ellipse.space.vectors[0] * ellipse.radius_x).pos2)
        self.points.append((ellipse.space.vectors[0] * -ellipse.radius_x).pos2)
        self.points.append((ellipse.space.vectors[1] * ellipse.radius_y).pos2)
        self.points.append((ellipse.space.vectors[1] * -ellipse.radius_y).pos2)

        self.ax.plot([p.x for p in points], [p.y for p in points], [p.z for p in points], color=ellipse.color)

    def add_shape3d(self, shape: 'Shape3D'):
        for polygon in shape.edges:
            self.add_polygon(polygon)

    def add_polygon(self, polygon: 'Polygon'):
        for point in polygon.vertices:
            self.points.append(point)

        verts = [[tuple([float(axis) for axis in point]) for point in
                  eq_len_axeslists(*map(lambda point: point.axes, polygon.vertices), dimension=3)]]
        polygon = Poly3DCollection(verts, alpha=polygon.alpha, facecolor=polygon.color,
                                   edgecolor=polygon.segments_color, color=polygon.color)
        self.ax.add_collection3d(polygon)

    def draw_line(self, line: 'Line'):
        self.ax.plot([line.pos1.x, line.pos2.x], [line.pos1.y, line.pos2.y], [line.pos1.z, line.pos2.z],
                     color=line.color, alpha=line.alpha)

    def make_scene(self):
        if self.points:
            min_x = min([point.x for point in self.points])
            min_y = min([point.y for point in self.points])

            max_x = max([point.x for point in self.points])
            max_y = max([point.y for point in self.points])

            min_z = min([point.z for point in self.points])
            max_z = max([point.z for point in self.points])

            min_point = Point[min_x, min_y, min_z]
            max_point = Point[max_x, max_y, max_z]

            if max_point.axes != min_point.axes:
                x, y, z = 0, 0, 0
                if max_point.x == min_point.x:
                    x = 1
                if max_point.y == min_point.y:
                    y = 1
                if max_point.z == min_point.z:
                    z = 1
                scene_rect = Box(max_point + Point[x, y, z], min_point - Point[x, y, z])
            else:
                scene_rect = Box(max_point, [0, 0, 0])

            for vector in self.vectors:
                direction = vector.pos2 - vector.pos1

                self.ax.quiver(
                    vector.pos1.x, vector.pos1.y, vector.pos1.z,
                    direction.x, direction.y, direction.z,
                    color=vector.color,
                    alpha=vector.alpha,
                    arrow_length_ratio=0.1
                )
                self.ax.plot(
                    [vector.pos1.x, vector.pos2.x],
                    [vector.pos1.y, vector.pos2.y],
                    [vector.pos1.z, vector.pos2.z],
                    color=vector.color, marker='.', alpha=vector.alpha
                )

            for ray in self.rays:
                if max_point != min_point:
                    direction = scene_rect.intersects(ray)[0] - ray.pos1
                else:
                    direction = ray.pos2

                self.ax.quiver(
                    ray.pos1.x, ray.pos1.y, ray.pos1.z,
                    direction.x, direction.y, direction.z,
                    color=ray.color,
                    alpha=ray.alpha,
                    arrow_length_ratio=0.02
                )
                start_pos = ray.pos1
                start_pos.color, start_pos.alpha = ray.color, ray.alpha
                self.add_point(start_pos)

            for line in self.lines:
                ions = scene_rect.intersects(line)
                # self.draw_line(line)
                self.ax.plot([ions[0].x, ions[1].x], [ions[0].y, ions[1].y], [ions[0].z, ions[1].z], color=line.color)

        # Scene draws lines and rays to the end by the min and max points that makes box.
        # Intersection with box is a finish point of line and ray.
        else:
            for line in self.lines:
                self.ax.plot([line.pos1.x, line.pos2.x], [line.pos1.y, line.pos2.y], [line.pos1.z, line.pos2.z],
                             color=line.color)


def make_aspace_scene(space) -> 'Scene':
    if not hasattr(space, '_scene'):
        if space.dimension in [1,2]:
            space._scene = Scene2D(*space.local_objects)
        else:
            space._scene = Scene3D(*space.local_objects)
    return space._scene

AffineSpace.scene = property(make_aspace_scene)
AffineSpace.show = lambda space: space.scene.show