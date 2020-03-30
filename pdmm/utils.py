"""
Contains utility functions for sampling.
"""
from numba import njit


@njit(fastmath=True)
def sample_from_multinomial_and_mutate_weights(weights, random_number):
    """
    Sample from a multinomial using linear search method.

    Parameters
    ----------
    weights : np.ndarray[float]
        The weights for the multinomial.
    random_number : float
        The random number passed into the array.

    Returns
    -------
    sampled_value : int
        The samples value.

    Notes
    -----
    - Sampling like this will mutate the input array.
    - Weights do NOT need to sum to 1.
    """
    weights = weights.copy()
    number_of_weights = weights.shape[0]
    for i in range(1, number_of_weights):
        weights[i] += weights[i - 1]

    scaled_random_number = random_number * weights[-1]

    counter = 0
    maximum_counter_value = number_of_weights - 1

    while counter < maximum_counter_value:
        mid = counter + ((maximum_counter_value - counter) // 2)
        if scaled_random_number > weights[mid]:
            counter = mid + 1
        else:
            maximum_counter_value = mid

    sampled_value = counter
    return sampled_value
