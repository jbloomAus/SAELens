import numpy as np
from types import SimpleNamespace

def geometric_median_array(points, weights, eps=1e-6, maxiter=100, ftol=1e-20):
    """
    :param points: list of length :math:`n`, whose elements are each a ``numpy.array`` of shape ``(d,)``
    :param weights: ``numpy.array`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero. 
    	Equivalently, this is a smoothing parameter. Default 1e-6. 
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a ``numpy.array`` object of shape :math:``(d,)``
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
        norms = [np.linalg.norm((p - median).reshape(-1)) for p in points]
        new_weights = weights / np.maximum(eps, norms)
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

def geometric_median_per_component(points, weights, eps=1e-6, maxiter=100, ftol=1e-20):
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
        - `termination`: string explaining how the algorithm terminated, one for each component. 
        - `logs`: function values encountered through the course of the algorithm.
    """
    components = list(zip(*points))
    median = []
    termination = []
    logs = []
    for component in components:
        ret = geometric_median_array(component, weights, eps, maxiter, ftol)
        median.append(ret.median)
        termination.append(ret.termination)
        logs.append(ret.logs)
    return SimpleNamespace(median=median, termination=termination, logs=logs)

def weighted_average(points, weights):
    """
    Compute a weighted average of rows of `points`, with each row weighted by the corresponding entry in `weights`
    :param points: ``np.ndarray`` of shape (n, d, ...)
    :param weights: ``np.ndarray`` of shape (n,)
    :return: weighted average, np.ndarray of shape (d, ...)
    """
    return np.average(points, weights=weights, axis=0)


def geometric_median_objective(median, points, weights):
    return np.average([np.linalg.norm((p - median).reshape(-1)) for p in points], weights=weights)