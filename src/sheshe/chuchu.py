from __future__ import annotations

_EPS = 1e-12

def _safe_div(a: np.ndarray, b: np.ndarray, eps: float=_EPS) -> np.ndarray:
    pass
def _whitener_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pass
def _apply_whiten(X: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    pass
def _scott(n: int, d: int) -> float:
    pass
def _bounding_box(X: np.ndarray, q_low: float=0.01, q_high: float=0.99) -> Tuple[np.ndarray, np.ndarray]:
    pass
def _train_calib_split(X: np.ndarray, y: Optional[np.ndarray], calib_frac: float, rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray, None]:
    pass
class BaseDensityModel:
    def fit(self, X: np.ndarray) -> 'BaseDensityModel':
        pass
    def score(self, X: np.ndarray) -> np.ndarray:
        pass
    def sample(self, n: int, rng: Optional[np.random.RandomState]=None) -> np.ndarray:
        pass
    def get_params(self, deep: bool=True) -> Dict[str, Any]:
        pass
    def set_params(self, **params) -> 'BaseDensityModel':
        pass
class KDEDensityModel(BaseDensityModel):
    def __init__(self, bandwidth_factor: float=1.0, whiten: bool=True):
        pass
    def fit(self, X: np.ndarray) -> 'KDEDensityModel':
        pass
    def score(self, X: np.ndarray) -> np.ndarray:
        pass
    def sample(self, n: int, rng: Optional[np.random.RandomState]=None) -> np.ndarray:
        pass
class GMMDensityModel(BaseDensityModel):
    def __init__(self, n_components: int=3, covariance_type: str='full', reg_covar: float=1e-06, whiten: bool=True, random_state: int=0):
        pass
    def fit(self, X: np.ndarray) -> 'GMMDensityModel':
        pass
    def score(self, X: np.ndarray) -> np.ndarray:
        pass
    def sample(self, n: int, rng: Optional[np.random.RandomState]=None) -> np.ndarray:
        pass
def _softmax(z, axis=1):
    pass
def _sigmoid(z):
    pass
class ScoreModelAdapterConfig:
    def get_params(self, deep: bool=True) -> Dict[str, Any]:
        pass
    def set_params(self, **params) -> 'ScoreModelAdapterConfig':
        pass
class ScoreDensityModel(BaseDensityModel):
    def __init__(self, cfg: ScoreModelAdapterConfig):
        pass
    def get_params(self, deep: bool=True) -> Dict[str, Any]:
        pass
    def set_params(self, **params) -> 'ScoreDensityModel':
        pass
    def fit(self, X: np.ndarray) -> 'ScoreDensityModel':
        pass
    def _scores_all(self, X: np.ndarray) -> np.ndarray:
        pass
    def score(self, X: np.ndarray) -> np.ndarray:
        pass
    def sample(self, n: int, rng: Optional[np.random.RandomState]=None) -> np.ndarray:
        pass
class RegionSpec:
    def get_params(self, deep: bool=True) -> Dict[str, Any]:
        pass
    def set_params(self, **params) -> 'RegionSpec':
        pass
def _conformal_threshold(scores_calib: np.ndarray, mass: float) -> float:
    pass
def _hdr_region(X_train: np.ndarray, X_calib: np.ndarray, mass: float, model: BaseDensityModel) -> Tuple[float, BaseDensityModel]:
    pass
def _estimate_volume_MC(indicator: Callable[[np.ndarray], np.ndarray], lo: np.ndarray, hi: np.ndarray, n_samples: int=20000, rng: Optional[np.random.RandomState]=None) -> float:
    pass
def _mvs_region(X_train: np.ndarray, X_calib: np.ndarray, mass: float, model: BaseDensityModel, lo: np.ndarray, hi: np.ndarray, rng: Optional[np.random.RandomState]=None) -> Tuple[float, BaseDensityModel]:
    pass
def f1_binary_from_masks(y_true: np.ndarray, y_pred: np.ndarray, positive_label) -> float:
    pass
def macro_f1_ignore_rejects(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pass
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pass
def _build_density_model(spec: DensitySpec) -> BaseDensityModel:
    pass
class ChuchuConfig:
    def get_params(self, deep: bool=True) -> Dict[str, Any]:
        pass
    def set_params(self, **params) -> 'ChuchuConfig':
        pass
class ChuchuClassifier:
    def __init__(self, config: ChuchuConfig):
        pass
    def get_params(self, deep: bool=True) -> Dict[str, Any]:
        pass
    def set_params(self, **params) -> 'ChuchuClassifier':
        pass
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ChuchuClassifier':
        pass
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray]=None, **fit_kwargs) -> np.ndarray:
        pass
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        pass
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray]=None, **fit_kwargs) -> np.ndarray:
        pass
    def save(self, filepath: Union[str, Path]) -> None:
        pass
    def load(cls, filepath: Union[str, Path]) -> 'ChuchuClassifier':
        pass
    def _scores_by_class(self, X: np.ndarray) -> Dict[Any, np.ndarray]:
        pass
    def membership(self, X: np.ndarray) -> Dict[Any, np.ndarray]:
        pass
    def predict_proba(self, X: np.ndarray) -> Dict[Any, np.ndarray]:
        pass
    def predict(self, X: np.ndarray, reject_if_outside: bool=True) -> np.ndarray:
        pass
    def predict_regions(self, X: ArrayLike) -> pd.DataFrame:
        pass
    def plot_classes(self, X: ArrayLike, y: ArrayLike, *, feature_names: Optional[Sequence[str]]=None):
        pass
    def plot_pairs(self, X: ArrayLike, *, feature_names: Optional[Sequence[str]]=None):
        pass
    def plot_pair_3d(self, *args, **kwargs):
        pass
    def optimize(self, X: np.ndarray, y: np.ndarray, density_grid: List[DensitySpec], coverage_grid: Iterable[float]=(0.8, 0.85, 0.9, 0.95), method: str='hdr', n_splits: int=5, random_state: int=0, reject_if_outside: bool=True) -> Tuple[ChuchuConfig, float]:
        pass
class ChuchuRegressor:
    def __init__(self, config: ChuchuConfig):
        pass
    def get_params(self, deep: bool=True) -> Dict[str, Any]:
        pass
    def set_params(self, **params) -> 'ChuchuRegressor':
        pass
    def _nw_predict(self, Xq: np.ndarray, density: BaseDensityModel) -> np.ndarray:
        pass
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ChuchuRegressor':
        pass
    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray]=None, **fit_kwargs) -> np.ndarray:
        pass
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        pass
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray]=None, **fit_kwargs) -> np.ndarray:
        pass
    def save(self, filepath: Union[str, Path]) -> None:
        pass
    def load(cls, filepath: Union[str, Path]) -> 'ChuchuRegressor':
        pass
    def region_mask(self, X: np.ndarray) -> np.ndarray:
        pass
    def predict(self, X: np.ndarray, reject_if_outside: bool=False) -> np.ndarray:
        pass
    def predict_regions(self, X: ArrayLike) -> pd.DataFrame:
        pass
    def plot_classes(self, X: ArrayLike, y: ArrayLike, *, feature_names: Optional[Sequence[str]]=None):
        pass
    def plot_pairs(self, X: ArrayLike, *, feature_names: Optional[Sequence[str]]=None):
        pass
    def plot_pair_3d(self, *args, **kwargs):
        pass
    def optimize(self, X: np.ndarray, y: np.ndarray, density_grid: List[DensitySpec], coverage_grid: Iterable[float]=(0.75, 0.8, 0.85, 0.9), method: str='hdr', n_splits: int=5, random_state: int=0, min_coverage: float=0.6) -> Tuple[ChuchuConfig, float, float]:
        pass
def hdr_polygons_2d(X: np.ndarray, region: RegionSpec, pair: Tuple[int, int], grid_res: int=600, pad: float=0.25) -> List[np.ndarray]:
    pass
