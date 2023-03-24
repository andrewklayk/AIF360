import numpy as np
import pandas as pd

from aif360.detectors.ot_detector import ot_bias_scan

def check(initial, final, expected_value, mode="ordinal", favorable_value=None, overpredicted=True):
    assert abs(sum(initial[:]) - sum(final[:])) <= 0.0001, \
        f"Datas are different, must have the same sum value! {sum(initial[:])} != {sum(final[:])}"
    res = ot_bias_scan(observations=initial, ideal_distribution=final, 
                       favorable_value=favorable_value, overpredicted=overpredicted, mode=mode)
    assert abs(res - expected_value) < 0.0000001, \
        f"The values do not match! Expected {expected_value}, but got {res}"
    print("TEST PASSED")

def test0():
    init_data = {"Empty": 0}
    end_data = {"Empty": 0}
    initial = pd.Series(data=init_data)
    final = pd.Series(data=end_data)
    expected_value = 0
    check(initial, final, expected_value)

def test1():
    init_data = {"Cat": 14, "Dog": 23, "Fish": 8}
    end_data = {"Cat": 15, "Dog": 16, "Fish": 14}
    initial = pd.Series(data=init_data)
    final = pd.Series(data=end_data)
    expected_value = 7
    check(initial, final, expected_value)

def test2():
    init_data = {"Bottle 1": 12.4, "Bottle 2": 4.3, "Bottle 3": 38.26, "Bottle 4": 21.14, "Bottle 5": 8.9}
    end_data = {"Bottle 1": 17, "Bottle 2": 17, "Bottle 3": 17, "Bottle 4": 17, "Bottle 5": 17}
    initial = pd.Series(data=init_data)
    final = pd.Series(data=end_data)
    expected_value = 33.96
    check(initial, final, expected_value)

def test3():
    init_data = {"January": 95.89, "Febuary": -26.28, "March": -9.67, "April": 25.74, "May": 56.40, "June": -20.9, 
                "July": -59.17, "August": 6.15, "September": -54.26, "October": 38.77, "November": 38.96, "December": -0.74}
    end_data = {"January": 11.14, "Febuary": 16.3, "March": -12.25, "April": 48.72, "May": -51.48, "June": -10.32, 
                "July": -0.23, "August": 17.6, "September": -23.33, "October": 13.38, "November": 51.67, "December": 29.69}
    initial = pd.Series(data=init_data)
    final = pd.Series(data=end_data)
    expected_value = 642.29
    check(initial, final, expected_value)

def test4():
    init_data = {0: 105.41751589143566, 1: -279.90626557631117, 2: 635.0683774379818, 3: 251.1221088327333, 4: 448.77098271495805, 5: 517.9042047261698, 6: 398.5836128520691, 7: 196.20387854448828, 8: 131.149425661516, 9: -49.313841085040615}
    end_data = {0: 237.66968240668567, 1: -544.778111408537, 2: 869.28182441595, 3: -1112.1404735810463, 4: 1390.0744460186522, 5: -501.3887201540594, 6: 243.22781074611106, 7: -172.62185585793216, 8: 1057.714355185456, 9: 887.9610422287196}
    initial = pd.Series(data=init_data)
    final = pd.Series(data=end_data)
    expected_value = 7584.286830610824
    check(initial, final, expected_value)

def test5():
    init_data = {0: 0, 1: 1, 2: 1, 3: 0, 4: 0}
    end_data = {0: 0.4793187959759956, 1: 0.4022045809764221, 2: 0.5190645039240965, 3: 0.568800185245431, 4: 0.030611933878054703}
    initial = pd.Series(data=init_data)
    final = pd.Series(data=end_data)
    expected_value = 1.2278194720251183
    check(initial, final, expected_value, "binary")

def test6():
    init_data = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5}
    end_data = {0: 0.5, 1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
    initial = pd.Series(data=init_data)
    final = pd.Series(data=end_data)
    expected_value = 2.0
    check(initial, final, expected_value, "continuous", "high")

def test7():
    init_data = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5}
    end_data = {0: -0.19, 1: 3.0, 2: -1.31}
    initial = pd.Series(data=init_data)
    final = pd.Series(data=end_data)
    expected_value = 4.2
    check(initial, final, expected_value, "continuous", "low", False)

def test8():
    init_data = {0: 0.4, 1: 0.6}
    end_data = {0: 0.3, 1: 0.7}
    initial = pd.Series(data=init_data)
    final = pd.Series(data=end_data)
    expected_value = 0.7
    check(initial, final, expected_value, "nominal", "low", False)


test0()
test1()
test2()
test3()
test4()
test5()
test6()
test7()
test8()
