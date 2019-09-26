
from sklearn.base import BaseEstimator
from .panelctmc_func import panelctmc
from ctmc import simulate
import datetime
import numpy as np


class PanelCtmc(BaseEstimator):
    """Continous Time Markov Chain for Panel Data, sklearn API class"""

    def __init__(self, mapping: list = None,
                 lastdate: datetime.datetime = None,
                 transintv: float = 1.0,
                 toltime: float = 1e-8,
                 debug: bool = False):
        self.mapping = mapping
        self.lastdate = lastdate
        self.transintv = transintv
        self.toltime = toltime
        self.debug = debug
        self.transmat = None
        self.genmat = None
        self.transcount = None
        self.statetime = None
        self.datalist = None

    def fit(self, X: np.ndarray, y=None):
        (
            self.transmat,
            self.genmat,
            self.transcount,
            self.statetime,
            self.datalist
        ) = panelctmc(
            X, self.mapping,
            lastdate=self.lastdate,
            transintv=self.transintv,
            toltime=self.toltime,
            debug=self.debug)
        return self

    def predict(self, X: np.ndarray, steps: int = 1) -> np.ndarray:
        return simulate(X, self.transmat, steps)
