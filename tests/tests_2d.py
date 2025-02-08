from geopy import *
import random


output_image_path = 'AssertImage.png'


def test_line_circle_intersection():
	step = 0.1
	circle = Circle([1,1], 10)
	horizontal = Line([0,0], [1,0]).at_pos([0,0])
	vertical = Line([0,1], [0,0]).at_pos([0,0])

	try:
		line = horizontal.at_pos([0,0]) + Point[circle.x, circle.y + circle.radius + 0.1]
		ions = line.intersects(circle)
		assert len(ions) == 0, f'First horizontal line intersects circle but must to do not'

		line = horizontal.at_pos([0,0]) + Point[circle.x, circle.y + circle.radius]
		ions = line.intersects(circle)
		assert len(ions) == 1, f'Second horizontal line intersects circle in {len(ions)} points but must to intersects in one point'

		for s in range(1, int(circle.diameter/step)):
			line = horizontal + Point[circle.x, circle.y - circle.radius + s * step]
			ions = line.intersects(circle)
			assert (len(ions) == 2 and
				ions[0] != ions[1]), f'Line intersects circle in {len(ions)} points but must to intersects in 2 different points'

		line = horizontal.at_pos([0,0]) + Point[circle.x, circle.y - circle.radius]
		ions = line.intersects(circle)
		assert len(ions) == 1, f'Penultimate horizontal line intersects circle in {len(ions)} points but must to intersects in one point'

		line = horizontal.at_pos([0,0]) + Point[circle.x, circle.y - circle.radius - 0.1]
		ions = line.intersects(circle)
		assert len(ions) == 0, f'Last horizontal line intersects circle but must to do not'


		line = vertical.at_pos([0,0]) + Point[circle.x + circle.radius + 0.1, circle.y]
		ions = line.intersects(circle)
		assert len(ions) == 0, f'First vertical line intersects circle but must to do not'

		line = vertical.at_pos([0,0]) + Point[circle.x + circle.radius, circle.y]
		ions = line.intersects(circle)
		assert len(ions) == 1, f'Second vertical line intersects circle in {len(ions)} points but must to intersects in one point'

		for s in range(1, int(circle.diameter/step)):
			line = vertical + Point[circle.x - circle.radius + s * step, circle.y]
			ions = line.intersects(circle)
			assert (len(ions) == 2 and
				ions[0] != ions[1]), f'Line intersects circle in {len(ions)} points but must to intersects in 2 different points'

		line = vertical.at_pos([0,0]) + Point[circle.x - circle.radius, circle.y]
		ions = line.intersects(circle)
		assert len(ions) == 1, f'Penultimate vertical line intersects circle in {len(ions)} points but must to intersects in one point'

		line = vertical.at_pos([0,0]) + Point[circle.x - circle.radius - 0.1, circle.y]
		ions = line.intersects(circle)
		assert len(ions) == 0, f'Last vertical line intersects circle but must to do not'

		for degree in range(180):
			line = Line.by_angle(degree)
			ions = line.intersects(circle)
			assert len(ions) == 2 and ions[0] != ions[1], f'Line intersects circle in {len(ions)} points but must to intersects in 2 different points'

	except AssertionError as e:
		for i in [ray, line, ions]:
			print(i)
		Scene2D(circle, line, *ions).save(output_image_path)
		raise e

def test_ray_segment_intersects_circle():
	step = 0.1
	circle = Circle([1,1], 5)

	try:
		for degree in range(-360, 360):
			ray = Ray.by_angle(degree, pos1=circle.center)
			ions = circle.intersects(ray)
			assert len(ions) == 1, f'Ray intersects circle in {len(ions)} points but must to intersects in one point'

		RAY = Ray.by_angle(random.randint(-360, 360))
		for s in range(int(circle.radius/step), 0, -1):
			ray = RAY + Point[circle.x, circle.y + circle.radius - s * step]
			ions = ray.intersects(circle)
			assert len(ions) == 1, f'Ray intersects circle in {len(ions)} points but must to intersects in one point'

		ray = RAY + Point[circle.x, circle.y + circle.radius + 1]
		ions = ray.intersects(circle)
		assert len(ions) != 1, f'Ray intersects circle in {len(ions)} points but must to intersects in 0 or 2 points'

	except AssertionError as e:
		for i in [ray, circle, ions]:
			print(i)
		Scene2D(circle, ray, *ions).save(output_image_path)
		raise e

	try:
		for degree in range(-360, 360):
			segm = Segment.by_angle(degree, pos1=circle.center, length=circle.radius)
			ions = circle.intersects(segm)
			assert len(ions) == 1, f'Segment intersects circle in {len(ions)} points but must to intersects in one point'

		segment = Segment([0,0], [1,0])
		for s in range(int(circle.radius/step), 0, -1):
			segm = segment + Point[circle.x, circle.y + circle.radius - s * step]
			ions = segm.intersects(circle)
			assert len(ions) == 1, f'Segment intersects circle in {len(ions)} points but must to intersects in one point'

		segm = segment + Point[circle.x, circle.y + circle.radius + 1]
		ions = segm.intersects(circle)
		assert len(ions) != 1, f'Segment intersects circle in {len(ions)} points but must to intersects in 0 or 2 points'

	except AssertionError as e:
		for i in [segm, circle, ions]:
			print(i)
		Scene2D(circle, segm, *ions).save(output_image_path)
		raise e


# TODO:
# Сделать тест на Полигон (прямоугольник и непонятная хренатень) и прямую
# Сделать тест на Прямые (параллельные осям и нет) и луч на 360
# Сделать тест на две прямые и два отрезка
# Сделать тест на высоту от точки до прямой
# Сделать тест на пренадлежность точки прямой, отрезку и лучу
# Сделать тест на окружности