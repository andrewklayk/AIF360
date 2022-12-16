from typing import Union

from aif360.detectors.mdss.ScoringFunctions import *

from aif360.detectors.mdss.MDSS import MDSS

import pandas as pd
import numpy as np

import ot

def _normalize(distribution):
    """
    transform distribution into distribution with total data allocation of one
    """
    total_of_distribution = np.sum(distribution)
    for i in range(np.size(distribution)):
        distribution[i] /= total_of_distribution

def _transform(observations, ideal_distribution, data):
    """
    transoforms given distributions from pandas type to numpy arrays, and prepare them
    """
    initial_distribution = (pd.Series.to_numpy(observations)).astype(np.float)
    required_distribution = (pd.Series.to_numpy(ideal_distribution)).astype(np.float)

    _normalize(initial_distribution)
    _normalize(required_distribution)
    
    matrix_distance = np.empty(shape = (np.size(initial_distribution), np.size(required_distribution)))
    for u in range(len(initial_distribution)):
        for v in range(len(required_distribution)):
            matrix_distance[u, v] = abs(u - v)

    return initial_distribution, required_distribution, matrix_distance

def ot_bias_scan(
    data: pd.DataFrame,
    observations: pd.Series,
    ideal_distribution: Union[pd.Series, pd.DataFrame] = None,
    favorable_value: Union[str, float] = None,
    overpredicted: bool = True,
    scoring: Union[str, ScoringFunction] = "Optimal Transport",
    num_iters: int = 10,
    penalty: float = 1e-17,
    mode: str = "binary",
    **kwargs,
):
    """
    scan to find the highest scoring subset of records

    :param data (dataframe): the dataset (containing the features) the model was trained on
    :param observations (series): ground truth (correct) target values
    :param ideal_distribution (series,  dataframe, optional): pandas series estimated targets
        as returned by a model for binary, continuous and ordinal modes.
        If mode is nominal, this is a dataframe with columns containing ideal_distribution for each nominal class.
        If None, model is assumed to be a dumb model that predicts the mean of the targets
                or 1/(num of categories) for nominal mode.
    :param favorable_value(str, float, optional): Should be high or low or float if the mode in [binary, ordinal, or continuous].
            If float, value has to be minimum or maximum in the observations column. Defaults to high if None for these modes.
            Support for float left in to keep the intuition clear in binary classification tasks.
            If mode is nominal, favorable values should be one of the unique categories in the observations.
            Defaults to a one-vs-all scan if None for nominal mode.
    :param overpredicted (bool, optional): flag for group to scan for.
        True means we scan for a group whose ideal_distribution/predictions are systematically higher than observed.
        In other words, True means we scan for a group whose observeed is systematically lower than the ideal_distribution.
        False means we scan for a group whose ideal_distribution/predictions are systematically lower than observed.
        In other words, False means we scan for a group whose observed is systematically higher than the ideal_distribution.
    :param scoring (str or class): Only 'Optimal Transport'
            In order to use others params for scoring, it is essential to use from mdss_detector
    :param num_iters (int, optional): number of iterations (random restarts). Should be positive.
    :param penalty (float,optional): penalty term. Should be positive. The penalty term as with any regularization parameter may need to be
        tuned for ones use case. The higher the penalty, the less complex (number of features and feature values) the
        highest scoring subset that gets returned is.
    :param mode: one of ['binary', 'continuous', 'nominal', 'ordinal']. Defaults to binary.
            In nominal mode, up to 10 categories are supported by default.
            To increase this, pass in keyword argument max_nominal = integer value.

    :returns: the highest scoring subset and the score or dict of the highest scoring subset and the score for each category in nominal mode
    """
    # Ensure correct mode is passed in.
    modes = ["binary", "continuous", "nominal", "ordinal"]
    assert mode in modes, f"Expected one of {modes}, got {mode}."

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
            assert favorable_value in [
                "flag-all",
                *uniques,
            ], f"Expected one of {uniques}, got {favorable_value}."

    assert favorable_value in [
        min_val,
        max_val,
        "flag-all",
        *uniques,
    ], f"Favorable_value should be high, low, or one of categories {uniques}, got {favorable_value}."

    # Set appropriate direction for scanner depending on mode and overppredicted flag
    if mode in ["ordinal", "continuous"]:
        if favorable_value == max_val:
            kwargs["direction"] = "negative" if overpredicted else "positive"
        else:
            kwargs["direction"] = "positive" if overpredicted else "negative"
    else:
        kwargs["direction"] = "negative" if overpredicted else "positive"

    # Set ideal_distribution to mean targets for non-nominal modes
    if ideal_distribution is None and mode != "nominal":
        ideal_distribution = pd.Series(observations.mean(), index=observations.index)

    # Check whether scoring correspond to "Optimal Transport"
    assert scoring == "Optimal Transport", f"Scoring mode can only be \"Optimal Transport\", got {scoring}."

    if mode == "binary": # Flip observations if favorable_value is 0 in binary mode.
        observations = pd.Series(observations == favorable_value, dtype=int)
    elif mode == "nominal":
        unique_outs = set(sorted(observations.unique()))
        size_unique_outs = len(unique_outs)
        if ideal_distribution is not None: # Set ideal_distribution to 1/(num of categories) for nominal mode
            ideal_distribution_cols = set(sorted(ideal_distribution.columns))
            assert (
                unique_outs == ideal_distribution_cols
            ), f"Expected {unique_outs} in expectation columns, got {ideal_distribution_cols}"
        else:
            ideal_distribution = pd.Series(
                1 / observations.nunique(), index=observations.index
            )
        max_nominal = kwargs.get("max_nominal", 10)

        assert (
            size_unique_outs <= max_nominal
        ), f"Nominal mode only support up to {max_nominal} labels, got {size_unique_outs}. Use keyword argument max_nominal to increase the limit."

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
                result = ot.emd(initial_distribution, required_distribution, matrix_distance, num_iters, False)
                results[unique] = result
            return results
    
    initial_distribution, required_distribution, matrix_distance = _transform(observations, ideal_distribution, data)
    return ot.emd(initial_distribution, required_distribution, matrix_distance, num_iters, False)