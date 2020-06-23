"""
Contains utility functions for sampling.
"""
from numba import njit
import numpy as np


@njit(fastmath=True)
def sample_from_cumulative_weights(cumulative_weights, random_number):
    """
    Sample from a multinomial using linear search method.

    Parameters
    ----------
    cumulative_weights : np.ndarray[float]
        The cumulative weights for the multinomial.
    random_number : float
        The random number to be used.

    Returns
    -------
    sampled_value : int
        The sampled value.
    """
    number_of_weights = cumulative_weights.shape[0]
    scaled_random_number = random_number * cumulative_weights[-1]

    counter = 0
    maximum_counter_value = number_of_weights - 1

    while counter < maximum_counter_value:
        mid = counter + ((maximum_counter_value - counter) // 2)
        if scaled_random_number > cumulative_weights[mid]:
            counter = mid + 1
        else:
            maximum_counter_value = mid

    sampled_value = counter
    return sampled_value


@njit(fastmath=True)
def sample_many_from_cumulative_weights(cumulative_weights, random_numbers):
    """
    Sample from a multinomial using linear search method.

    Parameters
    ----------
    cumulative_weights : np.ndarray[float]
        The cumulative weights for the multinomial.
    random_numbers : np.ndarray[float]
        The random numbers to be used.

    Returns
    -------
    sampled_values : np.array[int]
        The sampled values.
    """
    number_of_weights = cumulative_weights.shape[0]
    scaled_random_numbers = random_numbers * cumulative_weights[-1]

    counters = np.zeros_like(random_numbers, dtype=np.int32)
    maximum_counter_value = number_of_weights - 1
    upper_bounds = number_of_weights - np.ones_like(random_numbers, dtype=np.int32)

    for _ in range(maximum_counter_value):
        mid_values = (counters + ((upper_bounds - counters) // 2)).astype(np.int32)
        indicators = scaled_random_numbers > cumulative_weights[mid_values]
        counters = (indicators * mid_values) + ((np.int32(1) - indicators) * counters) + indicators
        upper_bounds = (indicators * upper_bounds) + ((np.int32(1) - indicators) * mid_values)

    sampled_values = counters
    return sampled_values
