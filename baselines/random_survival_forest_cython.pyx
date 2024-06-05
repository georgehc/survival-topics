#cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False
#distutils: language=c++
"""
Random survival forest helper cython code

Author: George H. Chen (georgechen [at symbol] cmu.edu)
"""
import numpy as np
cimport numpy as np
from libcpp.map cimport map
from numpy.math cimport INFINITY
from libc.math cimport fabs


def logrank(np.ndarray[np.float64_t, ndim=1] feature_values,
            np.ndarray[np.float64_t, ndim=2] y,
            double split_threshold,
            double min_samples_split,
            double min_samples_leaf):
    """
    Computes the log-rank splitting score provided that both sides of the split
    have more than a user-specified minimum number of samples; otherwise
    outputs 0.

    Parameters
    ----------
    feature_values : 1D numpy array, shape = [n_samples]
        Feature values for a single feature that we split on.

    y : 2D numpy array, shape = [n_samples, 2]
        Survival labels (first column is for observed times, second column is
        for event indicators). The i-th row corresponds to the i-th entry in
        `feature_values`.

    split_threshold : float
        Split `feature_values` by the `split_threshold` into the two pieces:
        `feature_values <= split_threshold`, `feature_values > split_threshold`

    min_samples_split : int
        If the number of samples at the current node is less than this
        quantity, then return 0.

    min_samples_leaf : int, float
        Integer case: if the split results in either side having fewer than
        `min_samples_leaf` samples, then return 0. The float case is similar:
        if either side of the split has fewer than `min_samples_leaf` fraction
        of the total number of samples (`n_samples`), then return 0.

    Returns
    -------
    score : float
      log-rank splitting score (float) or 0 (if one of the sides of the split
      has too few samples).
    """
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] left_mask \
        = (feature_values <= split_threshold)
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] right_mask = ~left_mask
    cdef np.ndarray[np.float64_t, ndim=2] y_left = y[left_mask]
    cdef np.ndarray[np.float64_t, ndim=2] y_right = y[right_mask]
    cdef int n_left = np.sum(left_mask)
    cdef int n_right = np.sum(right_mask)
    cdef int n_samples = y.shape[0]
    cdef double min_samples, n_alive_left, n_alive_right, score
    cdef double num = 0.
    cdef double den = 0.
    cdef int idx, idx2
    cdef map[double, int] time_to_idx
    cdef np.ndarray[np.float64_t, ndim=2] alive_counts = np.zeros((n_samples, 3))
    cdef np.ndarray[np.float64_t, ndim=2] death_counts = np.zeros((n_samples, 3))
    cdef np.ndarray[np.float64_t, ndim=1] count_left = np.zeros(n_samples)
    cdef np.ndarray[np.float64_t, ndim=1] count_right = np.zeros(n_samples)

    if n_samples < min_samples_split:
        return 0.

    if min_samples_leaf >= 1:
        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            return 0.
    else:
        min_samples = min_samples_leaf * n_samples
        if n_left < min_samples or n_right < min_samples:
            return 0.

    for idx in range(n_samples):
        time_to_idx[y[idx, 0]] = idx

    for idx2 in range(n_left):
        idx = time_to_idx[y_left[idx2, 0]]
        count_left[idx] += 1
        if y_left[idx2, 1]:
            death_counts[idx, 1] += 1

    for idx2 in range(n_right):
        idx = time_to_idx[y_right[idx2, 0]]
        count_right[idx] += 1
        if y_right[idx2, 1]:
            death_counts[idx, 2] += 1

    n_alive_left = <double>n_left
    n_alive_right = <double>n_right
    for idx in range(n_samples):
        alive_counts[idx, 1] = n_alive_left
        alive_counts[idx, 2] = n_alive_right
        alive_counts[idx, 0] = n_alive_left + n_alive_right
        n_alive_left -= count_left[idx]
        n_alive_right -= count_right[idx]
        death_counts[idx, 0] = death_counts[idx, 1] + death_counts[idx, 2]

    for idx in range(n_samples):
        if alive_counts[idx, 0] > 0:
            num += death_counts[idx, 1] - \
                alive_counts[idx, 1] * death_counts[idx, 0] \
                / alive_counts[idx, 0]
        if alive_counts[idx, 0] > 1:
            den += (alive_counts[idx, 1] * alive_counts[idx, 2]
                    / (alive_counts[idx, 0]**2)) \
                * ((alive_counts[idx, 0] - death_counts[idx, 0])
                   / (alive_counts[idx, 0] - 1)) * death_counts[idx, 0]

    if num < 0:
        num = -num

    if den > 0:
        score = num / np.sqrt(den)
    else:
        score = 0.

    return score


def predict_leaf_helper(np.ndarray[np.float64_t, ndim=1] leaf_times,
                        np.ndarray[np.float64_t, ndim=1] leaf_func,
                        np.ndarray[np.float64_t, ndim=1] times,
                        double init_func_value):
    cdef int n_leaf_times, n_times, leaf_time_idx, time_idx
    cdef double last_seen_value
    cdef np.ndarray[np.float64_t, ndim=1] output

    n_leaf_times = leaf_times.shape[0]
    n_times = times.shape[0]
    output = np.zeros(n_times)

    leaf_time_idx = 0
    last_seen_value = init_func_value
    for time_idx in range(n_times):
        time = times[time_idx]
        while leaf_time_idx < n_leaf_times:
            if leaf_times[leaf_time_idx] <= time:
                last_seen_value = leaf_func[leaf_time_idx]
                leaf_time_idx += 1
            else:
                break
        output[time_idx] = last_seen_value

    return output


def predict_leaf_helper_limit_from_left(
        np.ndarray[np.float64_t, ndim=1] leaf_times,
        np.ndarray[np.float64_t, ndim=1] leaf_func,
        np.ndarray[np.float64_t, ndim=1] times,
        double init_func_value):
    cdef int n_leaf_times, n_times, leaf_time_idx, time_idx
    cdef double last_seen_value
    cdef np.ndarray[np.float64_t, ndim=1] output

    n_leaf_times = leaf_times.shape[0]
    n_times = times.shape[0]
    output = np.zeros(n_times)

    leaf_time_idx = 0
    last_seen_value = init_func_value
    for time_idx in range(n_times):
        time = times[time_idx]
        while leaf_time_idx < n_leaf_times:
            if leaf_times[leaf_time_idx] < time:
                last_seen_value = leaf_func[leaf_time_idx]
                leaf_time_idx += 1
            else:
                break
        output[time_idx] = last_seen_value

    return output
