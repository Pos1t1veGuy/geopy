# Geopy - python geometry library

# Primitives
Polygons are made from primitives. Firstly they have a common parent empty class Primitive, so you can do:
```py
>>> from geopy import Primitive
>>> a = "123"
>>> isinstance(a, Primitive)
False
```

## Point primitive
Point costructed from X and Y, also you can name it:
```py
>>> from geopy import Point
>>> Point(1,1, name="Maybe it is a point")
Point(1.0, 1.0, name="Maybe it is a point")
>>>
>>> Point([777,228], name="It is point too")
Point(777.0, 228.0, name="It is point too")
>>> Point[0,0]
Point(0.0, 0.0, name="Point")
```
You can get X and Y from Point:
```py
>>> a = Point(1,1, name="a")
>>> a.x; a.y
1.0
1.0
>>> a.pos
[1.0, 1.0]
```
You can calculate points mathematically:
```py
>>> a = Point[0,0]
>>> b = Point[10,5]
>>>
>>> a + b
Point(10.0, 5.0, name="Point")
>>> a - b
Point(-10.0, -5.0, name="Point")
>>> a * b
Point(0.0, 0.0, name="Point")
>>> a / b
Point(0.0, 0.0, name="Point")
>>> b / a
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "...", line 105, in __truediv__
    return Point(self.x / object.x, self.y / object.y)
                 ~~~~~~~^~~~~~~~~~
ZeroDivisionError: float division by zero
>>> b ** a
Point(1.0, 1.0, name="Point")
```
You can get random point in rect from pos1 to pos2:
```py
>>> Point.random([0,0], [10,10])
Point(8.085890440593959, 4.454630028598453, name="Point")
>>> Point.random([0,0], [10,10], uniform=False)
Point(4.0, 5.0, name="Point")
```
Also you can get height to primitive object, like Line or Segment:
```py
>>> from geopy import Point, Segment
>>>
>>> seg = Segment([1,0], [-1,0])
>>> p = Point[0,1]
>>> p.height_to(seg)
Line(Point[0.0, 1.0], Point[0.0, 2.0], name="Line")
```

## linear primitives
They are a Line, Segment, Ray and Vector, Line is a Ray and Segment parent, Segment is a Vector parent.
Linear primitives is defined by 2 points ( or list/tuple, then it will be converted to Point ):
```py
>>> from geopy import Line, Segment, Ray, Vector
>>> Line([1,0], [0,0], name='abrakadabra')
Line(Point[1.0, 0.0], Point[0.0, 0.0], name="abrakadabra")
>>> Segment([1,0], [0,0], name='123')
Segment(Point[1.0, 0.0], Point[0.0, 0.0], name="123")
```
Line and its children has a linear function, you can get it as string or call methods to get X from Y and Y from X:
```py
>>> line = Line([1,0], [0,0], name='abrakadabra')
>>> line.as_geometry
'y = 0.0'
>>> line = Line([1,1], [0,0])
>>> line.as_geometry
'y = x'
>>> line = Line([1,1], [2,3], name='123')
>>> line.as_geometry
'y = 2.0x - 1.0'
>>> line.x_from_y(1)
1.0
>>> line.y_from_x(1)
1.0
>>> line.y_from_x(2)
3.0
>>> line(2)
3.0
>>> line(2) == line.y_from_x(2) # Line and its children __call__ method returns line.y_from_x result
True
```
Line and its children also save start pos and end pos from constructor:
```py
>>> line.pos1
Point(1.0, 1.0, name="Point")
>>> line.pos2
Point(2.0, 3.0, name="Point")
>>> line.pos2.x
2.0
```
You can get their angle:
```py
Line(Point[1.0, -1.0], Point[2.0, -2.0], name="Line")
>>> Line([1,1], [0,0]).angle
Angle(Point[0.0, 0.0] -> Point[1.0, 1.0] -> Point[2.0, 1.0], name="Angle")
>>> Line([1,1], [0,0]).angle.degrees
135.0
```
Linear primitives can get each other`s intersection points:
```py
>>> a = Line([1,1], [0,0])
>>> b = Vector([0,1], [1,0])
>>> a.intersects(b)
[Point(0.5, 0.5, name="Point")]
>>> b.intersects(a)
[Point(0.5, 0.5, name="Point")]
>>> a in b; b in a # returns True if a intersects b
True
True
```
You can move it to any position:
```py
>>> Segment([1,0], [2,0]).at_pos([0,1])
Segment(Point[0.0, 1.0], Point[1.0, 1.0], name="Segment")
```
And can calculate linear primitives mathematically:
```py
>>> Segment([1,0], [2,1]) + [2,2]
Segment(Point[3.0, 2.0], Point[4.0, 3.0], name="Segment")
>>> Segment([1,0], [2,1]) - [2,2]
Segment(Point[-1.0, -2.0], Point[0.0, -1.0], name="Segment")
```
### Line primitive
Line and Segment has the same direction options, you can get its directions:
```py
>>> Line([1,0], [0,0]).direction
'horizontal'
>>> Line([1,1], [0,0]).direction
'normal'
>>> Line([0,1], [0,0]).direction
'vertical'
```
You can get perpendicular:
```py
>>> Line([1,1], [0,0]).perpendicular
Line(Point[1.0, -1.0], Point[2.0, -2.0], name="Line")
```
### Line primitive
Line and Segment has the same direction options, you can get its directions:
```py
>>> Line([1,0], [0,0]).direction
'horizontal'
>>> Line([1,1], [0,0]).direction
'normal'
>>> Line([0,1], [0,0]).direction
'vertical'
```
You can get perpendicular:
```py
>>> Line([1,1], [0,0]).perpendicular
Line(Point[1.0, -1.0], Point[2.0, -2.0], name="Line")
```
Also you can get line by point and degree and by linear function:
```py
>>> from geopy import line_by_angle, line_by_function
>>> line_by_angle([0,0], 45)
Line(Point[1.0, 0.9999999999999999], Point[2.0, 1.9999999999999998], name="Line")
>>>
>>> line_by_function( lambda x: 2 * x + 10 )
Line(Point[1.0, 12.0], Point[2.0, 14.0], name="Line")
>>> line_by_function( lambda x: 2 * x + 10 ).as_geometry
'y = 2.0x + 10.0'
```
### Segment primitive
It is a Line child.
Line and Segment has the same direction options, you can get its directions:
```py
>>> Segment([1,0], [0,0]).direction
'horizontal'
>>> Segment([1,1], [0,0]).direction
'normal'
>>> Segment([0,1], [0,0]).direction
'vertical'
```
You can get perpendicular:
```py
>>> Segment([1,1], [0,0]).perpendicular
Line(Point[1.0, 0.0], Point[2.0, -1.0], name="Line")
```
Segment has a length:
```py
>>> Segment([1,0], [2,3]).length
3.1622776601683795
```
And center point:
```py
>>> Segment([1,0], [2,3]).center
Point(1.5, 1.5, name="Point")
```
### Vector primitive
It is a Segment child.
It may be created by one point:
```py
>>> Vector[1,1]
Vector(Point[0.0, 0.0], Point[1.0, 1.0], name="Vector")
```
Vector has a detailed direction:
```py
>>> Vector[1,1].direction
'right-up'
>>> Vector[1,0].direction
'right'
>>> Vector[1,-1].direction
'right-down'
>>> Vector[0,-1].direction
'down'
>>> Vector[-1,-1].direction
'left-down'
>>> Vector[-1,0].direction
'left'
>>> Vector[-1,1].direction
'left-up'
>>> Vector[0,1].direction
'up'
```
You can normalize and move Vector to (0,0) point:
```py
>>> Vector([1,1], [2,2]).to_zero
Vector(Point[0.0, 0.0], Point[1.0, 1.0], name="Vector")
>>> Vector([1,1], [2,2]).normalize
Vector(Point[1.0, 1.0], Point[1.7071067811865475, 1.7071067811865475], name="Vector")
```
Also you can calculate vectors mathematically:
```py
>>> Vector[1,1] + Vector[0,1]
Vector(Point[0.0, 0.0], Point[1.0, 2.0], name="Vector")
>>> Vector[1,1] * Vector[0,1]
Vector(Point[0.0, 0.0], Point[0.0, 1.0], name="Vector")
>>> Vector[1,1] - Vector[0,1]
Vector(Point[0.0, 0.0], Point[0.0, 1.0], name="Vector")
```
### Ray primitive
Ray has a detailed direction like a Vector:
```py
>>> Vector[1,1].direction
'right-up'
>>> Vector[1,0].direction
'right'
>>> Vector[1,-1].direction
'right-down'
>>> Vector[0,-1].direction
'down'
>>> Vector[-1,-1].direction
'left-down'
>>> Vector[-1,0].direction
'left'
>>> Vector[-1,1].direction
'left-up'
>>> Vector[0,1].direction
'up'
```
But it is all.


# Angle
Angle is not a Primitive. It defined by 3 points, second point is a midpoint. You can type tuple or list to Angle constructor, then it will be converted to the Point objects:
```py
>>> Angle([1,0], [0,0], [0,1], name="angle ABC")
Angle(Point[1.0, 0.0] -> Point[0.0, 0.0] -> Point[0.0, 1.0], name="angle ABC")
```
You can get its type:
```py
>>> Angle([1,0], [0,0], [0,1]).type
'right'
>>> Angle([1,0], [0,0], [1,1]).type
'acute'
>>> Angle([1,0], [0,0], [-1,1]).type
'obtuse'
```
And its bisector ray:
```py
>>> Angle([1,0], [0,0], [0,1]).bisector.as_geometry
'y = x'
```
Angle has a 3 points, so there is 2 vectors between them:
```py
>>> a = Angle([1,0], [0,0], [0,1])
>>> a.vec1; a.vec2
Vector(Point[0.0, 0.0], Point[1.0, 0.0], name="Vector")
Vector(Point[0.0, 0.0], Point[0.0, 1.0], name="Vector")
```
Also you can calculate cosinus, sinus and tangens:
```py
>>> a = Angle([1,0], [0,0], [0,1])
>>> a.cos; a.sin; a.tan
0.0
1.0
0.0
```
And get angle degrees, minutes and radians:
```py
>>> a = Angle([1,0], [0,0], [0,1])
>>> a.degrees; a.minutes; a.radians
90.0
5400.0
1.5707963267948966
```

# Shapes
They are Circle and Polygon, they have a common parent empty class Shape, so you can do:
```py
>>> from geopy import Shape
>>> a = "123"
>>> isinstance(a, Shape)
False
```
Also they have an intersects method that returns list of Point objects:
```py
>>> from geopy import Polygon, Circle, Vector
>>>
>>> polygon = Polygon([0,0], [1,1], [1,-1], [0,-1])
>>> vec = Vector[10,10]
>>> polygon.intersects(vec)
[Point(0.0, 0.0, name="Point"), Point(1.0, 1.0, name="Point")]
>>> vec.intersects(polygon)
[Point(0.0, 0.0, name="Point"), Point(1.0, 1.0, name="Point")]
>>> vec in polygon; polygon in vec
True
True
```
You can move created Shape to position or get a new Shape at position:
```py
>>> polygon = Polygon([0,0], [1,1], [1,-1], [0,-1])
>>> polygon.at_pos([1,0])
Polygon([Point(0.5, 0.25, name="Point"), Point(1.5, 1.25, name="Point"), Point(1.5, -0.75, name="Point"), Point(0.5, -0.75, name="Point")], name="Polygon", pos=Point[1.5, -0.25])
>>> polygon.to_pos([1,0])
>>> polygon
Polygon([Point(0.5, 0.25, name="Point"), Point(1.5, 1.25, name="Point"), Point(1.5, -0.75, name="Point"), Point(0.5, -0.75, name="Point")], name="Polygon", pos=Point[1.0, 0.0])
```
You can show your Shape in matplotlib:
```py
Polygon([0,0], [1,1], [1,-1], [0,-1]).plot()
```
And in simply tkinter:
```py
Circle([0,0], 1).view()
```
Also you can scale it:
```py
>>> Polygon([0,0], [1,1], [1,-1], [0,-1]).scale(2)
Polygon([Point(-0.5, 0.25, name="Point"), Point(1.5, 2.25, name="Point"), Point(1.5, -1.75, name="Point"), Point(-0.5, -1.75, name="Point")], name="Polygon", pos=Point[0.5, -0.25])
```
And calculate center point mathematically
```py
>>> Polygon([0,0], [1,1], [1,-1], [0,-1]) + [1,0]
Polygon([Point(1.0, 0.0, name="Point"), Point(2.0, 1.0, name="Point"), Point(2.0, -1.0, name="Point"), Point(1.0, -1.0, name="Point")], name="Polygon", pos=Point[1.5, -0.25])
```

## Polygon
It is defined by Point list:
```py
>>> from geopy import Polygon
>>>
>>> Polygon([0,0], [1,1], [1,-1], [0,-1])
Polygon([Point(0.0, 0.0, name="Point"), Point(1.0, 1.0, name="Point"), Point(1.0, -1.0, name="Point"), Point(0.0, -1.0, name="Point")], name="Polygon", pos=Polygon center[0.5, -0.25])
```
Polygon automaticly initialize segments and angles between the points:
```py
>>> polygon = Polygon([0,0], [1,1], [1,-1], [0,-1])
>>> polygon.vertices
[Point(0.0, 0.0, name="Point"), Point(1.0, 1.0, name="Point"), Point(1.0, -1.0, name="Point"), Point(0.0, -1.0, name="Point")]
>>> polygon.segments
[Segment(Point[0.0, -1.0], Point[0.0, 0.0], name="sPolygon0"), Segment(Point[0.0, 0.0], Point[1.0, 1.0], name="sPolygon1"), Segment(Point[1.0, 1.0], Point[1.0, -1.0], name="sPolygon2"), Segment(Point[1.0, -1.0], Point[0.0, -1.0], name="sPolygon3")]
>>> polygon.angles
[Angle(Point[1.0, -1.0] <- Point[0.0, -1.0] -> Point[0.0, 0.0], name="aPolygon0"), Angle(Point[0.0, -1.0] <- Point[0.0, 0.0] -> Point[1.0, 1.0], name="aPolygon1"), Angle(Point[0.0, 0.0] <- Point[1.0, 1.0] -> Point[1.0, -1.0], name="aPolygon2"), Angle(Point[1.0, 1.0] <- Point[1.0, -1.0] -> Point[0.0, -1.0], name="aPolygon3")]
```
You can change type of segments to Vector or your custom:
```py
>>> from geopy import Polygon, Vector
>>> polygon = Polygon([0,0], [1,1], [1,-1], [0,-1], segment_object=Vector)
>>> polygon.segments
[Vector(Point[0.0, -1.0], Point[0.0, 0.0], name="sPolygon0"), Vector(Point[0.0, 0.0], Point[1.0, 1.0], name="sPolygon1"), Vector(Point[1.0, 1.0], Point[1.0, -1.0], name="sPolygon2"), Vector(Point[1.0, -1.0], Point[0.0, -1.0], name="sPolygon3")]
```
And rotate it by degree:
```py
>>> Polygon([0,0], [1,1], [1,-1], [0,-1]).rotate(45)
Polygon([Point(0.02461312445760555, -0.5441212650626268, name="Point"), Point(-0.30096841125878315, 0.8321042482892214, name="Point"), Point(1.4008386378094537, -0.21853972934623805, name="Point"), Point(0.8755166489917239, -1.0694432538803564, name="Point")], name="Polygon", pos=Polygon center[0.5, -0.25])
```
Polygon has a property to get random point inside and center of mass:
```py
>>> polygon = Polygon([0,0], [1,1], [1,-1], [0,-1])
>>> polygon.random_point
Point(0.4057370412021215, 0.07220382319695706, name="Point")
>>> polygon.center_of_mass
Point(0.5, -0.25, name="Polygon center")
```
And property to get perimeter and area:
```py
>>> polygon.perimeter
5.414213562373095
>>> polygon.area
1.5
```
You can get polygon objects by position:
```py
polygon.segments_by_point(self, point: Point) -> List[Segment]
polygon.segments_by_fromto_point(self, point: Point) -> List[Segment]
polygon.segments_by_from_point(self, point: Point) -> List[Segment]
polygon.segments_by_to_point(self, point: Point) -> List[Segment]
polygon.angle_by_pos(self, point: Point) -> Angle
```
And get a Box on top of the Polygon:
```py
>>> Polygon([0,0], [1,1], [1,-1], [0,-1]).box
Box([Point(-1.0, 0.0, name="Polygon_min"), Point(-1.0, 1.0, name="Point"), Point(1.0, 1.0, name="Polygon_min"), Point(1.0, 0.0, name="Point")], name="Box", pos=Box center[0.0, 0.5])
```
### Box
It is a Polygon child and defined by 2 points:
```py
>>> from geopy import Box
>>> Box([0,0], [1,1])
Box([Point(0.0, 0.0, name="Point"), Point(0.0, 1.0, name="Point"), Point(1.0, 1.0, name="Point"), Point(1.0, 0.0, name="Point")], name="Box", pos=Box center[0.5, 0.5])
```
### Rhombus
It is a Polygon child and defined by center point and 2 diagonals:
```py
>>> from geopy import Rhombus
>>> Rhombus([0,0], 1, 2)
Rhombus([Point(0.0, 1.0, name="Point"), Point(-0.5, 0.0, name="Point"), Point(0.0, -1.0, name="Point"), Point(0.5, 0.0, name="Point")], name="Rhombus", pos=Rhombus center[0.0, 0.0])
```
### Triangle
It is a Polygon child and defined by 3 points:
```py
>>> Triangle([1,0], [0,0], [0,1])
Triangle([Point(1.0, 0.0, name="Point"), Point(0.0, 0.0, name="Point"), Point(0.0, 1.0, name="Point")], name="Triangle", pos=Triangle center[0.3333333333333333, 0.3333333333333333])
```
It has type by angle and by side:
```py
>>> tr = Triangle([1,0], [0,0], [0,1])
>>> tr.angle_type
'right'
>>> tr.side_type
'isosceles'
>>> tr.type
'right isosceles'
```
You can get legs and hypotenuse if it is right triangle:
```py
>>> tr = Triangle([1,0], [0,0], [0,1])
>>> tr.hypotenuse
Segment(Point[0.0, 1.0], Point[1.0, 0.0], name="sTriangle0")
>>> tr.legs
[Segment(Point[1.0, 0.0], Point[0.0, 0.0], name="sTriangle1"), Segment(Point[0.0, 0.0], Point[0.0, 1.0], name="sTriangle2")]
```
And orthocenter, the center by sides perpendiculars:
```py
>>> tr.orthocenter
Point(0.5, 0.5, name="Point")
```

## Circle
It is defined by center point and radius
```py
>>> from geopy import Circle
>>> Circle([0,0], 2)
Circle(Point[0.0, 0.0], 2)
```
Or by 2 points:
```py
>>> from geopy import circle_by_points
>>> circle_by_points([0,0], [1,1])
Circle(Point[0.0, 0.0], 1.4142135623730951)
>>> circle_by_points([0,0], [1,0])
Circle(Point[0.0, 0.0], 1.0)
```
