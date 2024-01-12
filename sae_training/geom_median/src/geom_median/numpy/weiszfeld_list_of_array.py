import numpy as np
from types import SimpleNamespace

def geometric_median_list_of_array(points, weights, eps=1e-6, maxiter=100, ftol=1e-20):
    """
    :param points: list of length :math:``n``, where each element is itself a list of ``numpy.ndarray``.
        Each inner list has the same "shape".
    :param weights: ``numpy.ndarray`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero. 
    	Equivalently, this is a smoothing parameter. Default 1e-6. 
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a list of ``numpy.ndarray`` of the same "shape" as the input.
        - `termination`: string explaining how the algorithm terminated.
        - `logs`: function values encountered through the course of the algorithm in a list.
    """
    # initialize median estimate at mean
    median = weighted_average(points, weights)
    objective_value = geometric_median_objective(median, points, weights)
    logs = [objective_value]

    # Weiszfeld iterations
    early_termination = False
    for _ in range(maxiter):
        prev_obj_value = objective_value
        new_weights = weights / np.maximum(eps, np.asarray([l2distance(p, median) for p in points]))
        median = weighted_average(points, new_weights)

        objective_value = geometric_median_objective(median, points, weights)
        logs.append(objective_value)
        if abs(prev_obj_value - objective_value) <= ftol * objective_value:
            early_termination = True
            break

    return SimpleNamespace(
        median=median,
        termination="function value converged within tolerance" if early_termination else "maximum iterations reached",
        logs=logs,
    )

def weighted_average(points, weights):
    return [np.average(component, weights=weights, axis=0) for component in zip(*points)]

def geometric_median_objective(median, points, weights):
    return np.average([l2distance(p, median) for p in points], weights=weights)

# Simple operators for list-of-array format
def l2distance(p1, p2):
    return np.linalg.norm([np.linalg.norm(x1 - x2) for (x1, x2) in zip(p1, p2)])

def subtract(p1, p2):
    return [x1 - x2 for (x1, x2) in zip(p1, p2)]