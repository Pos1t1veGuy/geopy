from geopy import *
import random


output_image_path = 'AssertImage.png'
save_scene = True
show_scene = True

def make_scene(*objects):
	print('OBJECTS:')
	for i in objects:
		print(i)
	scene = Scene2D(*objects)
	if save_scene:
		scene.save(output_image_path)
	if show_scene:
		scene.show()


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
		make_scene(circle, line, *ions)
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
		make_scene(circle, ray, *ions)
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

def test_circles_intersection():
	try:
		c1 = Circle([1,1], 5)
		c2 = Circle([0,0], 4)
		path = Circle(c1.center, c1.radius+2)

		for degree in range(89, 360):
			new_circle_center = path(Fraction(degree, 360))
			c2.to_pos(new_circle_center)
			ions = c1.intersects(c2, check_inside=False)
			assert len(ions) == 2 and ions[0] != ions[1], f'Circles intersect in {len(ions)} points but must to intersect in 2 different points'
		
		c2.to_pos(c1.center)
		ions = c2.intersects(c1, check_inside=False)
		assert len(ions) == 0, f'Circles intersect in {len(ions)} points but must to intersect in 0 points'
		ions = c2.intersects(c1, check_inside=True)
		assert len(ions) == 1 and ions[0] == c1.center == c2.center, f'Circles intersect in {len(ions)} points but must to intersect in 1 point - center of Circles'

	except AssertionError as e:
		make_scene(c1, c2, *ions)
		raise e

def test_line_height():
	try:
		point = Point[1,1]

		for degree in range(180):
			line = Line.by_angle(degree)
			if degree != 45: # Line not intersects [0,0]
				height = point.height_to(line)
				l = line
			else: # Line intersects [0,0]
				try:
					height = point.height_to(line)
				except Exception as ex:
					assert isinstance(ex, ConstructError), f'There should be an ConstructError when turning 45 degrees'
				l = line+[1,0]
				height = point.height_to(l)

			ions = l.intersects(height)
			assert (height.pos2 in l and height.pos1 == point and l.is_perpendicular(height) and
				len(ions) == 1), f'Height must to intersect in one point and be perpendicular'

	except AssertionError as e:
		make_scene(l, height, *ions)
		raise e

def test_space_point_intersection():
	def check_intersection(point: Point, origin: Point, vectors: List[Vector], result: bool) -> bool:
		try:
			space = Space(origin, vectors)
			assert (space in point) == result, f"Space {'intersects' if not result else 'not intersects'} point"
		except AssertionError as e:
			make_scene(*space.vectors, point)
			print(space)
			raise e

	check_intersection(Point[0,0,0], Point[0,0,0], [Vector[1,0], Vector[0,1]], True)
	check_intersection(Point[5,5,0], Point[0,0,0], [Vector[1,0], Vector[0,1]], True)
	check_intersection(Point[5,5,1], Point[0,0,0], [Vector[1,0], Vector[0,1]], False)

	check_intersection(Point[2,2,1], Point[0,0,0], [Vector[1,1,0].normalize, Vector[0,0,1]], True)
	check_intersection(Point[3,3,3], Point[0,0,0], [Vector[1,1,0].normalize, Vector[0,0,1]], True)
	check_intersection(Point[3,2,3], Point[0,0,0], [Vector[1,1,0].normalize, Vector[0,0,1]], False)

	check_intersection(Point[2,2,2,2,1], Point[1,1,1,1,1], [Vector[1,0,0,0,0],Vector[0,1,0,0,0],Vector[0,0,1,0,0],Vector[0,0,0,1,0]], True)
	check_intersection(Point[2,2,2,2,3], Point[1,1,1,1,1], [Vector[1,0,0,0,0],Vector[0,1,0,0,0],Vector[0,0,1,0,0],Vector[0,0,0,1,0]], False)

def test_lines_intersection():
		def check_intersection_at(pos: Point):
			try:
				if not isinstance(pos, Point):
					pos = Point(pos)

				ions = []
				l1 = Line([0,0], [1,1], color='red').at_pos(pos)

				for degree in range(180):
					l2 = Line.by_angle(degree).at_pos(pos)
					if degree == 45:
						# Lines are equal, intersection is a Line
						assert l1.is_parallel(l2), 'Lines must to be a parallel but not'

						ions = l1.intersects(l2)
						assert len(ions) == 1, f'Lines are equal and must have ONE intersection Line: {ions}'
						assert isinstance(ions[0], Line), f'Lines are equal and must have ONE intersection Line: {ions}'
						assert (
							ions[0].pos1 in l1 and ions[0].pos1 in l2 and ions[0].pos2 in l1 and ions[0].pos2 in l2
						), f'Intersection Line must be equal to both given paralel lines: {ions}'

						assert l1.is_parallel(l2), 'Lines must to be a parallel but not'

						# Lines are parallel, intersection does not exists
						l2 = l2.at_pos(pos + [2,0])
						ions = l1.intersects(l2)
						assert ions == [], f'Lines are parallel but have intersection: {ions}'
					else:
						ions = l1.intersects(l2)
						assert len(ions) > 0, 'Lines are not parallel but have not intersection'
			except AssertionError as e:
				make_scene(l1, l2, *ions)
				raise e

		def check_intersection_by_deg(deg1: int, deg2: int):
			try:
				l1 = Line.by_angle(deg1).at_pos([99,-99])
				l2 = Line.by_angle(deg2).at_pos([1,1])
				ions = l1.intersects(l2)
				assert len(ions) > 0, f'Perpendicular lines must intersect at ONE point: {ions}'
				assert isinstance(ions[0], Point), f'Perpendicular lines must intersect at one POINT: {ions}'
			except AssertionError as e:
				make_scene(l1, l2, *ions)
				raise e

		check_intersection_at([0,0])
		check_intersection_at([2,0])
		check_intersection_at([0,-3])
		check_intersection_at([.5,.5])

		check_intersection_by_deg(0, 90)
		check_intersection_by_deg(45, -45)

def test_segments_intersection():
	try:
		s1, s2 = Segment([0,0],[1,0]), Segment([0,0], [0,-1])
		ions = s1.intersects(s2)
		assert ions == [Point[0]], f"Expected one intersection point [0,0], but it is: {ions}"
		s1, s2 = Segment([0,0],[-1,0]), Segment([0,2], [0,0])
		ions = s1.intersects(s2)
		assert ions == [Point[0]], f"Expected one intersection point [0,0], but it is: {ions}"
		s1 = s1.at_pos([2,2])
		ions = s1.intersects(s2)
		assert ions == [], f"Expected no intersections, but it is: {ions}"
		s1, s2 = Segment([0,0],[1,1]), Segment([1,0], [0,1])
		ions = s1.intersects(s2)
		assert ions == [Point[.5,.5]], f"Expected one intersection point [.5,.5], but it is: {ions}"
		s1, s2 = Segment([-1,1],[1,1]), Segment([0,0], [0,-1])
		ions = s1.intersects(s2)
		assert ions == [], f"Expected no intersections, but it is: {ions}"

		s1, s2 = Segment([1, 0], [-2, 0]), Segment([1, 0], [2, 0])
		ions = s1.intersects(s2)
		assert ions == [Point[1,0]], f"Expected intersection in [1,0], got: {ions}"
		s1, s2 = Segment([1, 0], [-2, 0]), Segment([1.0001, 0], [2, 0])
		ions = s1.intersects(s2)
		assert ions == [], f"Expected no intersections, got: {ions}"
		s1, s2 = Segment([1, 0], [-2, 0]), Segment([0.5, 0], [2, 0])
		ions = s1.intersects(s2)
		assert len(ions) == 1, f"Expected ONE intersection Segment, got: {ions}"
		assert isinstance(ions[0], Segment), f"Expected one intersection SEGMENT, got: {ions}"
		assert ions[0].length == .5, f"Expected 0.5 lengths Segment, but it is: {ions[0].length}"
		points = [s1.pos1, s2.pos1]
		assert (
				ions[0].pos1 in points and ions[0].pos2 in points
		), f"Intersection segment must be from [1,0] to [0.5], but it is {ions[0].pos1} and {ions[0].pos2}"

		s1, s2 = Segment([0, 1], [0, -2]), Segment([0, 1], [0, 2])
		ions = s1.intersects(s2)
		assert ions == [Point[0,1]], f"Expected intersection in [0,1], got: {ions}"
		s1, s2 = Segment([0, 1], [0, -2]), Segment([0, 1.0001], [0, 2])
		ions = s1.intersects(s2)
		assert ions == [], f"Expected no intersections, got: {ions}"
		s1, s2 = Segment([0, 1], [0, -2]), Segment([0, 0.5], [0, 2])
		ions = s1.intersects(s2)
		assert len(ions) == 1, f"Expected ONE intersection Segment, got: {ions}"
		assert isinstance(ions[0], Segment), f"Expected one intersection SEGMENT, got: {ions}"
		assert ions[0].length == .5, f"Expected 0.5 lengths Segment, but it is: {ions[0].length}"
		points = [s1.pos1, s2.pos1]
		assert (
				ions[0].pos1 in points and ions[0].pos2 in points
		), f"Intersection segment must be from [1,0] to [0.5], but it is {ions[0].pos1} and {ions[0].pos2}"

		s1, s2 = Segment([1, 1], [2, 2]), Segment([2, 2], [3, 3])
		ions = s1.intersects(s2)
		assert ions == [Point[2,2]], f"Expected intersection in [2,2], got: {ions}"
		s1, s2 = Segment([1, 1], [2, 2]), Segment([2.0001, 2.1], [3, 3])
		ions = s1.intersects(s2)
		assert ions == [], f"Expected no intersections, got: {ions}"
		s1, s2 = Segment([1, 1], [2, 2]), Segment([1.5,1.5], [3, 3])
		ions = s1.intersects(s2)
		assert len(ions) == 1, f"Expected ONE intersection Segment, got: {ions}"
		assert isinstance(ions[0], Segment), f"Expected one intersection SEGMENT, got: {ions}"
		assert 0.7 < ions[0].length < 0.8, f"Expected ~0.71 lengths Segment, but it is: {ions[0].length}"
		points = [s1.pos2, s2.pos1]
		assert (
				ions[0].pos1 in points and ions[0].pos2 in points
		), f"Intersection segment must be from [2, 2] and [1.5, 1.5], but it is {ions[0].pos1} and {ions[0].pos2}"

		s1, s2 = Segment([-1, 1], [1, -1]), Segment([-1, 1], [-2, 2])
		ions = s1.intersects(s2)
		assert ions == [Point[-1,1]], f"Expected intersection in [-1,1], got: {ions}"
		s1, s2 = Segment([-1, 1], [1, -1]), Segment([-1.1, 1.0001], [-2, 2])
		ions = s1.intersects(s2)
		assert ions == [], f"Expected no intersections, got: {ions}"
		s1, s2 = Segment([-1, 1], [1, -1]), Segment([-.5, .5], [-2, 2])
		ions = s1.intersects(s2)
		assert len(ions) == 1, f"Expected ONE intersection Segment, got: {ions}"
		assert isinstance(ions[0], Segment), f"Expected one intersection SEGMENT, got: {ions}"
		assert 0.7 < ions[0].length < 0.8, f"Expected ~0.71 lengths Segment, but it is: {ions[0].length}"
		points = [s1.pos1, s2.pos1]
		assert (
				ions[0].pos1 in points and ions[0].pos2 in points
		), f"Intersection segment must be from [-1, 1] and [-0.5, 0.5], but it is {ions[0].pos1} and {ions[0].pos2}"

	except AssertionError as e:
		make_scene(s1, s2, *ions)
		raise e

def test_rays_intersection():
	try:
		s1, s2 = Segment([0,0],[1,0]), Segment([0,0], [0,-1])
		ions = s1.intersects(s2)
		assert ions == [Point[0]], f"There is must to be one intersection point [0,0], but it is: {ions}"

	except AssertionError as e:
		make_scene(s1, s2, *ions)
		raise e


if __name__ == '__main__':
	tests = { key: value for key, value in globals().items() if key.startswith('test_') and callable(value) }
	for test in tests.values():
		test()