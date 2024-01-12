import numpy as np
import torch
from types import SimpleNamespace

def geometric_median_list_of_array(points, weights, eps=1e-6, maxiter=100, ftol=1e-20):
    """
    :param points: list of length :math:``n``, where each element is itself a list of ``torch.Tensor``.
        Each inner list has the same "shape".
    :param weights: ``torch.Tensor`` of shape :math:``(n,)``.
    :param eps: Smallest allowed value of denominator, to avoid divide by zero. 
    	Equivalently, this is a smoothing parameter. Default 1e-6. 
    :param maxiter: Maximum number of Weiszfeld iterations. Default 100
    :param ftol: If objective value does not improve by at least this `ftol` fraction, terminate the algorithm. Default 1e-20.
    :return: SimpleNamespace object with fields
        - `median`: estimate of the geometric median, which is a list of ``torch.Tensor`` of the same "shape" as the input.
        - `termination`: string explaining how the algorithm terminated.
        - `logs`: function values encountered through the course of the algorithm in a list.
    """
    with torch.no_grad():
        # initialize median estimate at mean
        median = weighted_average(points, weights)
        new_weights = weights
        objective_value = geometric_median_objective(median, points, weights)
        logs = [objective_value]

        # Weiszfeld iterations
        early_termination = False
        for _ in range(maxiter):
            prev_obj_value = objective_value
            denom = torch.stack([l2distance(p, median) for p in points])
            new_weights = weights / torch.clamp(denom, min=eps) 
            median = weighted_average(points, new_weights)

            objective_value = geometric_median_objective(median, points, weights)
            logs.append(objective_value)
            if abs(prev_obj_value - objective_value) <= ftol * objective_value:
                early_termination = True
                break
        
    median = weighted_average(points, new_weights)  # for autodiff

    return SimpleNamespace(
        median=median,
        new_weights=new_weights,
        termination="function value converged within tolerance" if early_termination else "maximum iterations reached",
        logs=logs,
    )

def weighted_average_component(points, weights):
    ret = points[0] * weights[0]
    for i in range(1, len(points)):
        ret += points[i] * weights[i]
    return ret

def weighted_average(points, weights):
    weights = weights / weights.sum()
    return [weighted_average_component(component, weights=weights) for component in zip(*points)]

@torch.no_grad()
def geometric_median_objective(median, points, weights):
    return np.average([l2distance(p, median).item() for p in points], weights=weights.cpu())

@torch.no_grad()
def l2distance(p1, p2):
    return torch.linalg.norm(torch.stack([torch.linalg.norm(x1 - x2) for (x1, x2) in zip(p1, p2)]))
