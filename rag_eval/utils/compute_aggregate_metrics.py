from typing import List

import numpy as np

def compute_average(scores: List[float]) -> float:
    """
    Computes the mean of a list of float scores.
    """
    return float(np.mean(scores))

def compute_standard_deviation(scores: List[float]) -> float:
    """
    Computes the sample standard deviation of a list of float scores.

    Uses ddof=1 to produce an unbiased estimate of the
    population standard deviation from a sample. This is appropriate when the
    scores represent a sample of runs rather than the full population.
    """
    return float(np.std(scores, ddof=1))
