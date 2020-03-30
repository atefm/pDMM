"""
Contains utility functions for sampling.
"""
from numba import njit


@njit(fastmath=True)
def sample_from_cumulative_weights(cumulative_weights, random_number):
    """
    Sample from a multinomial using linear search method.

    Parameters
    ----------
    cumulative_weights : np.ndarray[float]
        The cumulative weights for the multinomial.
    random_number : float
        The random number passed into the array.

    Returns
    -------
    sampled_value : int
        The samples value.
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
