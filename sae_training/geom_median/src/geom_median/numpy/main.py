import numpy as np

from .weiszfeld_array import geometric_median_array, geometric_median_per_component
from .weiszfeld_list_of_array import geometric_median_list_of_array
from . import utils

def compute_geometric_median(
	points, weights=None, per_component=False, skip_typechecks=False,
	eps=1e-6, maxiter=100, ftol=1e-20
):
	""" Compute the geometric median of points `points` with weights given by `weights`. 
	"""
	if weights is None:
		n = len(points)
		weights = np.ones(n)
	if type(points) == np.ndarray:
		# `points` are given as an array of shape (n, d)
		points = [p for p in points]  # translate to list of arrays format
	if type(points) not in [list, tuple]:
		raise ValueError(
			f"We expect `points` as a list of arrays or a list of tuples of arrays. Got {type(points)}"
		)
	if type(points[0]) == np.ndarray: # `points` are given in list of arrays format
		if not skip_typechecks:
			utils.check_list_of_array_format(points)
		to_return = geometric_median_array(points, weights, eps, maxiter, ftol)
	elif type(points[0]) in [list, tuple]: # `points` are in list of list of arrays format
		if not skip_typechecks:
			utils.check_list_of_list_of_array_format(points)
		if per_component:
			to_return = geometric_median_per_component(points, weights, eps, maxiter, ftol)
		else:
			to_return = geometric_median_list_of_array(points, weights, eps, maxiter, ftol)
	else:
		raise ValueError(f"Unexpected format {type(points[0])} for list of list format.")
	return to_return
		