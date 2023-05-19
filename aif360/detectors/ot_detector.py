from typing import Union

import pandas as pd

import numpy as np

import ot

def _normalize(distribution1, distribution2):
    """
    Transform distributions to pleasure form, that is their sums are equal with precise at least 0.0000001
    and in case if there is negative values, increase all values with absolute value of smallest number.

    Args:
        distribution1 (numpy array): nontreated distribution
        distribution2 (numpy array): nontreated distribution
    """
    switch = False
    if sum(distribution2) < sum(distribution1):
        switch = True
        distribution2, distribution1 = distribution1, distribution2

    sum1 = np.sum(distribution1)
    sum2 = 0
    
    distribution = distribution2
    for i in range(len(distribution2)):
        distribution[i] = float(format(distribution2[i], '.10f'))
        sum2 += distribution[i]
    
    # Trying to make two distributions make equal, changing the precision of values
    for val in range(9, 0, -1):
        if abs(sum1 - sum2) < 0.0000001:
            break
        if sum2 <= sum1:
            break
        for i in range(len(distribution2)):
            if distribution[i] < float(format(distribution2[i], f'.{val}f')): 
                continue
            sum2 -= distribution[i]
            sum2 += float(format(distribution2[i], f'.{val}f'))
            if sum2 < sum1 and abs(sum1 - sum2) > 0.0000001:
                sum2 += distribution[i]
                sum2 -= float(format(distribution2[i], f'.{val}f'))
                continue

            distribution[i] = float(format(distribution2[i], f'.{val}f'))
            if sum2 <= sum1:
                break
    if sum2 < sum1 and abs(sum1 - sum2) > 0.0000001:
        need = sum1 - sum2
        val = np.max(distribution)
        for i in range(0, np.size(distribution)):
            if distribution[i] != val:
                continue
            distribution[i] += need
            break
    
    # If we encounter with negative values, we get rid of them, adding their absolute value to all elements
    if min(min(distribution), min(distribution1)) < 0:
        extra = -min(min(distribution), min(distribution1))
        distribution += extra
        distribution1 += extra
    distribution2 = distribution

    if switch:
        distribution2, distribution1 = distribution1, distribution2


def _transform(observations, ideal_distribution, data):
    """
    Transoform given distributions from pandas type to numpy arrays, and _normalize them.

    In case, if the datas are different even after multiple operations, 
    it rearanges distributions, with totall data allocated of one.

    Args:
        observations (series): ground truth (correct) target values
        ideal_distribution (series,  dataframe, optional): pandas series estimated targets
            as returned by a model for binary, continuous and ordinal modes.
        data (dataframe): the dataset (containing the features) the model was trained on

    Returns:
        initial_distribution, which is an processed observations (numpy array)
        required_distribution, which is an processed ideal_distribution (numpy array)
        matrix_distance, which stores the distances between the cells of distributions (2d numpy array)
    
    Raises:
        AssertionError: An error occur, when two distributions have totally different sizes,
                        their difference is greater than 0.0000001 after all manipulations.
    """
    initial_distribution = (pd.Series.to_numpy(observations)).astype(float)
    required_distribution = (pd.Series.to_numpy(ideal_distribution)).astype(float)

    _normalize(initial_distribution, required_distribution)
    if abs(sum(initial_distribution) - sum(required_distribution)) > 0.0000001:
        total_of_distribution = np.sum(initial_distribution)
        initial_distribution /= total_of_distribution
        
        total_of_distribution = np.sum(required_distribution)
        required_distribution /= total_of_distribution
        print(required_distribution[0])

    assert abs(sum(initial_distribution) - sum(required_distribution)) <= 0.0000001, \
        f"Datas are different, must have the same sum value! {abs(sum(initial_distribution[:]))} != {sum(required_distribution[:])}"

    # Creating the distance matrix for future obtaining optimal transport matrix
    d1_ = np.tile(range(len(initial_distribution)), len(initial_distribution))
    d2_ = np.repeat(range(len(required_distribution)), len(required_distribution))
    matrix_distance = np.reshape(np.abs(d1_ - d2_),
                                 newshape=(len(initial_distribution), len(required_distribution))
                                 ).astype(float)

    return initial_distribution, required_distribution, matrix_distance

def ot_bias_scan(
    observations: pd.Series,
    ideal_distribution: Union[pd.Series, pd.DataFrame],
    data: pd.DataFrame = None,
    favorable_value: Union[str, float] = None,
    overpredicted: bool = True,
    scoring: str = "Optimal Transport",
    num_iters: int = 15,
    penalty: float = 1e-17,
    mode: str = "ordinal",
    **kwargs,
):
    """Calculated the Wasserstein distance for two given distributions.

    Transforms pandas Series into numpy arrays, transofrms and normalize them.
    After all, solves the optimal transport problem.

    Args:
        observations (series): ground truth (correct) target values
        ideal_distribution (series,  dataframe, optional): pandas series estimated targets
            as returned by a model for binary, continuous and ordinal modes.
            If mode is nominal, this is a dataframe with columns containing ideal_distribution for each nominal class.
            If None, model is assumed to be a dumb model that predicts the mean of the targets
                    or 1/(num of categories) for nominal mode.
        data (dataframe): the dataset (containing the features) the model was trained on
        favorable_value(str, float, optional): Should be high or low or float if the mode in [binary, ordinal, or continuous].
                If float, value has to be minimum or maximum in the observations column. Defaults to high if None for these modes.
                Support for float left in to keep the intuition clear in binary classification tasks.
                If mode is nominal, favorable values should be one of the unique categories in the observations.
                Defaults to a one-vs-all scan if None for nominal mode.
        overpredicted (bool, optional): flag for group to scan for.
            True means we scan for a group whose ideal_distribution/predictions are systematically higher than observed.
            In other words, True means we scan for a group whose observeed is systematically lower than the ideal_distribution.
            False means we scan for a group whose ideal_distribution/predictions are systematically lower than observed.
            In other words, False means we scan for a group whose observed is systematically higher than the ideal_distribution.
        scoring (str or class): Only 'Optimal Transport'
        num_iters (int, optional): number of iterations (random restarts). Should be positive.
        penalty (float, optional): penalty term. Should be positive. The penalty term as with any regularization parameter may need to be
            tuned for ones use case. The higher the penalty, the less complex (number of features and feature values) the
            highest scoring subset that gets returned is.
        mode: one of ['binary', 'continuous', 'nominal', 'ordinal']. Defaults to binary.
                In nominal mode, up to 10 categories are supported by default.
                To increase this, pass in keyword argument max_nominal = integer value.

    Returns:
        ot.emd (float): Earth mover's distance

    Raises:
        AssertionError: If scoring variable is not "Optimal Transport"
        AssertionError: If type mode does not belong to any, of the possible options 
                        ["binary", "continuous", "nominal", "ordinal"].
        AssertionError: If favorable_value does not belong to any, of the possible options 
                        [min_val, max_val, "flag-all", *uniques].
    """
    # Check whether scoring correspond to "Optimal Transport"
    assert scoring == "Optimal Transport", \
        f"Scoring mode can only be \"Optimal Transport\", got {scoring}."

    # Ensure correct mode is passed in.
    assert mode in ['binary', 'continuous', 'nominal', 'ordinal'], \
        f"Expected one of {['binary', 'continuous', 'nominal', 'ordinal']}, got {mode}."
    
    # Set ideal_distribution to mean targets for non-nominal modes
    if ideal_distribution is None and mode != "nominal":
        ideal_distribution = pd.Series(observations.mean(), index=observations.index)

    # Set correct favorable value (this tells us if higher or lower is better)
    min_val, max_val = observations.min(), observations.max()
    uniques = list(observations.unique())

    if favorable_value == 'high':
        favorable_value = max_val
    elif favorable_value == 'low':
        favorable_value = min_val
    elif favorable_value is None:
        if mode in ["binary", "ordinal", "continuous"]:
            favorable_value = max_val # Default to higher is better
        elif mode == "nominal":
            favorable_value = "flag-all" # Default to scan through all categories

    assert favorable_value in [min_val, max_val, "flag-all", *uniques,], \
        f"Favorable_value should be high, low, or one of categories {uniques}, got {favorable_value}."

    if mode == "binary": # Flip observations if favorable_value is 0 in binary mode.
        observations = pd.Series(observations == favorable_value, dtype=int)
    elif mode == "nominal":
        unique_outs = set(sorted(observations.unique()))
        size_unique_outs = len(unique_outs)
        if ideal_distribution is None: # Set ideal_distribution to 1/(num of categories) for nominal mode
            ideal_distribution = pd.Series(1 / observations.nunique(), index=observations.index)

        if favorable_value != "flag-all": # If favorable flag is set, use one-vs-others strategy to scan, else use one-vs-all strategy
            observations = observations.map({favorable_value: 1})
            observations = observations.fillna(0)
            if isinstance(ideal_distribution, pd.DataFrame):
                ideal_distribution = ideal_distribution[favorable_value]
        else:
            results = {}
            orig_observations = observations.copy()
            orig_ideal_distribution = ideal_distribution.copy()
            for unique in uniques:
                observations = orig_observations.map({unique: 1})
                observations = observations.fillna(0)

                if isinstance(ideal_distribution, pd.DataFrame):
                    ideal_distribution = orig_ideal_distribution[unique]

                initial_distribution, required_distribution, matrix_distance = _transform(observations, ideal_distribution, data)
                result = ot.emd(initial_distribution, required_distribution, matrix_distance, num_iters, True)[1]["cost"]
                results[unique] = result
            return results
    initial_distribution, required_distribution, matrix_distance = _transform(observations, ideal_distribution, data)
    
    return ot.emd(initial_distribution, required_distribution, matrix_distance, num_iters, True)[1]["cost"]