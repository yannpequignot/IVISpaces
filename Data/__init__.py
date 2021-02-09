
from abc import ABC
import importlib.util
import torch


from sklearn.model_selection import train_test_split
from Tools import fitStandardScalerNormalization, normalize

test_ratio=0.1 #split ratio : 90% train 10% test

def switch_setup(setup):
    return {
        'foong':  importlib.util.spec_from_file_location("foong", "Data/foong/__init__.py") ,
        'boston': importlib.util.spec_from_file_location("boston", "Data/boston/__init__.py"),
        'concrete': importlib.util.spec_from_file_location("concrete", "Data/concrete/__init__.py"),
        'energy':  importlib.util.spec_from_file_location("energy", "Data/energy/__init__.py") ,
        'wine': importlib.util.spec_from_file_location("wine", "Data/winequality/__init__.py"),
        'yacht': importlib.util.spec_from_file_location("yacht", "Data/yacht/__init__.py"),

        'powerplant': importlib.util.spec_from_file_location("powerplant", "Data/ccpowerplant/__init__.py"),
        'navalC': importlib.util.spec_from_file_location("navalC", "Data/naval/__init__.py"),
        'protein': importlib.util.spec_from_file_location("protein", "Data/protein/__init__.py"),

    }[setup]


def get_setup(setup):
    spec=switch_setup(setup)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    return setup



class AbstractRegressionSetup(ABC):
    def __init__(self):
        self.device=None
        self.sigma_noise=None
        self.seed=None
        self.test_ratio=test_ratio


    def _split_holdout_data(self):
        X_tv, self._X_test, y_tv, self._y_test = train_test_split(self._X, self._y, test_size=test_ratio, random_state=self.seed)
        self._X_train, self._y_train = X_tv, y_tv


    def _normalize_data(self):        
        self._scaler_X, self.scaler_y = fitStandardScalerNormalization(self._X_train, self._y_train)
        self._X_train, self._y_train = normalize(self._X_train, self._y_train, self._scaler_X, self.scaler_y)
        self._X_test, self._y_test = normalize(self._X_test, self._y_test, self._scaler_X, self.scaler_y)

    def _flip_data_to_torch(self):
        self._X_train = torch.tensor(self._X_train, device=self.device).float()
        self._y_train = torch.tensor(self._y_train, device=self.device).float()
        self._X_test = torch.tensor(self._X_test, device=self.device).float()
        self._y_test = torch.tensor(self._y_test, device=self.device).float()
        self.n_train_samples=self._X_train.shape[0]

    def train_data(self):
        return self._X_train, self._y_train
    
    def test_data(self):
        return self._X_test, self._y_test
        
