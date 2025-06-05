from geopy import *
from geopy.ext.mpl_scene import Scene3D
import random


output_image_path = 'AssertImage.png'
save_scene = True
show_scene = True

def make_scene(*objects):
	print(' OBJECTS:')
	for i in objects:
		print(' ', i)
	scene = Scene3D(*objects)
	if save_scene:
		scene.save(output_image_path)
	if show_scene:
		scene.show()

def symmetric_intersection(a, b):
	res1, res2 = a.intersects(b), b.intersects(a)
	# print(res1,res2)
	assert res1 == res2, f'''{a} and {b} intersections are not symmetrical:
{a} x {b} = {res1}
{b} x {a} = {res2}
'''
	return res1


def test_lines_intersection():
	ions = []
	try:
		l1 = Line([0], [1,2,3])
		l2 = Line([0], [1,1,1])
		ions = symmetric_intersection(l1,l2)
		assert ions == [Point[0]], f'Lines must intersect in [0,0,0], but it is {ions}'
		l1 = Line([0], [1,2,3])
		l2 = Line([1,1,1], [2,4,6])
		ions = symmetric_intersection(l1,l2)
		assert ions == [Point[2,4,6]], f'Lines must intersect in [2,4,6], but it is {ions}'

		l2 = l1.copy()
		ions = symmetric_intersection(l1,l2)
		assert ions == [l1] == [l2], f'Lines intersection must be a Line, but it is {ions}'

		l1 = Line(Point[0,0,0], Point[1,1,1])
		l2 = Line(Point[1,0,0], Point[1,1,1])
		ions = symmetric_intersection(l1,l2)
		assert ions == [Point[1,1,1]], f'Lines must intersect in [1,1,1], but it is {ions}'

		l1 = Line([0, 0, 0], [1, 1, 1])
		l2 = Line([0, 1, 0], [1, -1, 0])
		ions = symmetric_intersection(l1,l2)
		assert ions == [], f'Lines must not intersect, but it is {ions}'

	except AssertionError as e:
		make_scene(l1, l2, *ions)
		raise e

def test_line_space_intersection():
	ions = []
	try:
		...

	except AssertionError as e:
		make_scene(l1, l2, *ions)
		raise e


if __name__ == '__main__':
	tests = { key: value for key, value in globals().items() if key.startswith('test_') and callable(value) }
	for test in tests.values():
		test()