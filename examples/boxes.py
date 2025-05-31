from geopy import *

b1 = Box([1,1,1], [3,3,3], color='red', alpha=0.1)
b2 = Box([2,2,2], [4,4,4], alpha=0.1)

ions = []
for edge in b1.edges:
    for segment in b2.segments:
        ion = segment.intersects(edge)
        if ion:
            for i in ion:
                ions.append(i)
for edge in b2.edges:
    for segment in b1.segments:
        ion = segment.intersects(edge)
        if ion:
            for i in ion:
                ions.append(i)

for v in b1.vertices:
    if v in b2:
        ions.append(v)
for v in b2.vertices:
    if v in b1:
        ions.append(v)

Scene3D(b1,b2,*ions).show()