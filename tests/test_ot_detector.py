import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from aif360.datasets import AdultDataset

from aif360.detectors.ot_detector import ot_bias_scan, _normalize, _transform

# create a subset of the AdultDataset to test on and a set of predictions

ad = AdultDataset(protected_attribute_names=['race', 'sex', 'native-country'],
                  privileged_classes=[['White'], ['Male'], ['United-States']],
                  categorical_features=['workclass', 'education',
                          'marital-status', 'occupation', 'relationship'],
                  custom_preprocessing=lambda df: df.fillna('Unknown'))
adult_test, adult_train = ad.split([16281], shuffle=False)
adult_train = adult_train.subset(range(1000))
adult_test = adult_test.subset(range(500))
scaler = StandardScaler()
X = scaler.fit_transform(adult_train.features)
test_X = scaler.transform(adult_test.features)
clf = LogisticRegression(C=1.0, random_state=0, solver='liblinear')
adult_pred = adult_test.copy()
adult_pred.labels = clf.fit(X, adult_train.labels.ravel()).predict(test_X)

rng = np.random.default_rng(seed = 42)
s = 100
d1 = rng.normal(loc=0, size=s)
d2 = rng.normal(loc=1, size=s)


def test_normalization():
    # test normalization: must make every value non-negative
    _normalize(d1, d2)
    print(np.sum(d1), np.sum(d2))
    assert isinstance(d1, np.ndarray)
    assert isinstance(d2, np.ndarray)
    assert np.all(d1 >= 0), "ot_detector._normalize: negatives present"
    assert np.all(d2 >= 0), "ot_detector._normalize: negatives present"

def test_transform():
    # check if transform returns np.ndarrays, makes sums equal
    s = 100
    d1 = pd.Series(rng.normal(loc=0, size=s))
    d2 = pd.Series(rng.normal(loc=10, size=s))
    d1_, d2_, dist_ = _transform(d1, d2, None)

    assert isinstance(d1_, np.ndarray)
    assert isinstance(d2_, np.ndarray)
    assert abs(np.sum(d1_) - np.sum(d2_)) < 1e-6, "ot_detector._transform: sums differ"

def test_diff_distance():
    # check wasserstein distance against POT implementation
    actual = ot_bias_scan(observations=pd.Series(adult_test.labels.flatten()),
                 ideal_distribution=pd.Series(adult_pred.labels.flatten()),
                 favorable_value=adult_test.favorable_label,
                 mode='binary', num_iters=10000)
    expected = 10.564426877470353
    assert abs(actual - expected) <= 1e-6, f"WD must be {expected}, got {actual}"
    
def test_same_distance():
    # check returns 0 for same data
    actual = ot_bias_scan(observations=pd.Series(adult_pred.labels.flatten()),
                 ideal_distribution=pd.Series(adult_pred.labels.flatten()),
                 favorable_value=adult_test.favorable_label,
                 mode='binary', num_iters=100000)
    expected = 0
    assert abs(actual - expected) <= 1e-6, f"WD must be {expected}, got {actual}"