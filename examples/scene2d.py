from geopy import *

Scene2D(

Line([1,1], [0,0]),
Point[3,-3],
Circle([9,9], 4),
Polygon([6,5], [4,5], [0,1], [3,2], [6,0]),
Ray([-8,2], [6,7]),
Segment([5,-5], [-10,5]).at_pos([20,0]),
Vector([5,0], [-10,0])

).show()