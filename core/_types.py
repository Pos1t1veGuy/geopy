from typing import *
from fractions import Fraction


intersection_result_name = '|{}-{}|_ion'
height_name = '{}_height'
projection_name = '{}_proj'
perpendicular_name = '{}_perp'
vector_of_object_name = '{}_vec'
center_name = '{}_center'
normalize_result_name = '{}_norm'

angle_result_name = '{}_ang'
angle_between_name = '|{}-{}|_angle'
angle_bisector_name = '{}_bis'

affine_space_vectors_name = '({name})_vector{i}'
affine_space_local_object_name = '{}_local'
affine_space_global_object_name = '{}_global'
affine_space_of_object_name = '{}_space'
space_of_object_name = '{}_space'

polygon_segments_name = '({name})_segment{i}'
polygon_angles_name = '({name})_angle{i}'


class AxesList(list):
	# Infinity list. If use __getitem__ with i > length it returns 0, that means 0 coordinate at axis.
	# AxesList([1,2,3])[i] with 0 <= i < length returns 1, 2 or 3; with i > length returns 0
	def __getitem__(self, i):
		if isinstance(i, slice):
			try:
				return list(self)[i]
			except IndexError:
				pass
		if isinstance(i, int):
			if i >= 0:
				try:
					return list(self)[i]
				except IndexError:
					pass
		else:
			raise IndexError('Axis index must be integer >=0')
		return Fraction(0)

	def __setitem__(self, index, value):
		while len(self) < index+1:
			self.append(Fraction(0))
		super().__setitem__(index, to_fraction(value))

	def as_list(self, length: int = -1) -> list:
		# Convert to default list
		if length == -1 or len(self) == length:
			return list(self)
		elif len(self) < length:
			return list(self) + [0] * (length - len(self))
		else:
			return list(self)[:length]

def eq_len_axeslists(*args: 'AxesList', dimension: int = -1) -> List[list]:
	# takes axeslists with different lengths and returns default lists with equal lengths by adding zeroes to end of lists
	max_dimension = max([ len(l) for l in args ]) if dimension == -1 else dimension
	return [ l.as_list(length=max_dimension) for l in args ]

def reduse_axeslists(*args: 'AxesList') -> List['AxesList']:
	# slices all extra zeros at the beginning of the axeslists. For example [0,0,1] and [0,1,1,2] => [0,1] [1,1,2]
	max_dimension = max([ len(l) for l in args ])
	for i in range(max_dimension)[::-1]:
		if not all([ l[i] == 0 for l in args ]):
			return [ AxesList(l[:i+1]) for l in args ]
	return args



def to_fraction(num1: Union[int, float, Fraction], num2: Union[int, float, Fraction] = 1) -> Fraction:
	# Conrerts any number to Fraction. If takes floats it will multiply until it becomes integer
	if isinstance(num1, Fraction) and isinstance(num2, Fraction):
		return num1 / num2
	if isinstance(num1, Fraction):
		return num1 / Fraction(num2)
	if isinstance(num2, Fraction):
		return Fraction(num1) / num2

	if isinstance(num1, float):
		num1 = Fraction(*num1.as_integer_ratio())
	else:
		num1 = Fraction(num1)
	if isinstance(num2, float):
		num2 = Fraction(*num2.as_integer_ratio())
	else:
		num2 = Fraction(num2)

	fraction = Fraction(num1) / Fraction(num2)
	return fraction.limit_denominator(10**20)