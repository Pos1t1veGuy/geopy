from typing import *

class Space:
	def __init__(self, vectors: List['Vector']):
		self.athenian = (
			any([ round(vector.length[i]/vector.length[i-1], 5) != 0 for i in range(len(vectors)) ]) or
			
		)