# %%
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from typing_extensions import Self, Unpack

Parameters: TypeAlias = tuple[float, ...]
ModelCallable: TypeAlias = Callable[[NDArray[Any], Unpack[Parameters]], NDArray[Any]]

# Parameters: TypeAlias = Sequence[float]
# ModelCallable: TypeAlias = Callable[[NDArray[Any], Parameters], NDArray[Any]]


def linear_model(X: NDArray[Any], intercept: float, *coef: float) -> NDArray[Any]:
    """Linear model of the form:
    y = a1 * x1 + a2 * x2 ... + an * xn + b"""
    return intercept + np.sum(X * coef, axis=1)


def exponential_model_glm(X: NDArray[Any], A0: float, *coef: float) -> NDArray[Any]:
    """Exponential model of the form:
    y = A0 * (np.exp(b1 * x1) + np.exp(b2 * x2) + ... + np.exp(bn * xn))"""
    return A0 * np.sum(np.exp(X * coef), axis=1)


def exponential_model_with_offset(
    X: NDArray[Any], A0: float, C: float, *coef: float
) -> NDArray[Any]:
    """Exponential model with an offset term.
    y = A0 * (np.exp(b1 * x1) + np.exp(b2 * x2) + ... + np.exp(bn * xn)) + C"""
    return A0 * np.sum(np.exp(X * coef), axis=1) + C


@dataclass
class Model(ABC):
    _model: ModelCallable
    _params: Parameters | None = None
    _feature_names: list[str] | None = None
    _target_name: str | None = None
    loss: str = "linear"

    @abstractmethod
    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> Self:
        if isinstance(X, pd.DataFrame):
            self._feature_names = list(X.columns)
        if isinstance(y, pd.Series):
            self._target_name = y.name

    @property
    @abstractmethod
    def y_intercept(self) -> float:
        pass

    @property
    def params(self) -> Parameters:
        if self._params is None:
            raise RuntimeError(
                "Can't access `self.params` befor the model has been fitted!"
            )
        return self._params

    def predict(self, X: NDArray[Any]) -> NDArray[Any]:
        return self._model(X, *self.params)

    def _correction_factor(self, X: NDArray[Any]) -> NDArray[Any]:
        return 1 / self.predict(X) * self.y_intercept

    def correct(self, X: NDArray[Any], y: NDArray[Any]) -> NDArray[Any]:
        return y * self._correction_factor(X)

    def save(
        self, file_name: str | None = None, directory: str | Path = Path(".")
    ) -> None:
        if file_name is None:
            file_name = f"{self.__class__.__name__}"

        full_out_file = Path(directory) / f"{file_name}.pkl"
        print(f"Saving model to {full_out_file.resolve()}")
        full_out_file.parent.mkdir(exist_ok=True)
        with open(full_out_file, "xb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, full_file_path: str | Path):
        with open(full_file_path, "rb") as f:
            model = pickle.load(f)
        return model

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        if self._params is None:
            return f"{class_name}: not fitted."
        return f"{class_name}: {self.params}"


@dataclass
class LinearModel(Model):
    _model: ModelCallable = linear_model

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> Self:
        super().fit(X, y)
        self._params, _ = curve_fit(
            self._model, X, y, loss=self.loss, method="trf", p0=np.ones(X.shape[1] + 1)
        )
        return self

    @property
    def y_intercept(self) -> float:
        return self.params[0]


@dataclass
class ExponentialModelFitLinear(Model):
    _model: ModelCallable = exponential_model_glm

    def fit(self, X: NDArray[Any], y: NDArray[Any]):
        super().fit(X, y)
        # Fit a linear model on log-transformed data.
        lin_params, _ = curve_fit(
            linear_model,
            X,
            np.log(np.clip(y, 1e-6, None)),
            loss=self.loss,
            method="trf",
            p0=np.ones(X.shape[1] + 1),
        )
        # Transform the first coefficient by applying `np.exp`
        self._params = tuple([np.exp(lin_params[0]), *lin_params[1:]])
        return self

    @property
    def y_intercept(self) -> float:
        return self.params[0]


@dataclass
class ExponentialModelNoOffset(Model):
    _model: ModelCallable = exponential_model_glm

    def fit(self, X: NDArray[Any], y: NDArray[Any]):
        super().fit(X, y)
        # Fit an `ExponentialModelFitLinear` to find initialization parameters -> better convergence.
        mdl = ExponentialModelFitLinear().fit(X, y)
        # Then fit the actual model.
        self._params, _ = curve_fit(
            self._model, X, y, loss=self.loss, method="trf", p0=mdl.params
        )
        return self

    @property
    def y_intercept(self) -> float:
        return self.params[0]


@dataclass
class ExponentialModel(Model):
    _model: ModelCallable = exponential_model_with_offset
    offset_init: int = 0
    enforce_positive_offset: bool = False

    def fit(self, X: NDArray[Any], y: NDArray[Any]) -> Self:
        super().fit(X, y)
        # Estimate initial parameters using an `ExponentialModelNoOffset`.
        mdl = ExponentialModelNoOffset().fit(X, y)
        p0 = np.array([mdl.params[0], self.offset_init, *mdl.params[1:]])
        lower_bounds = np.array([-np.inf for _ in range(len(p0))])
        if self.enforce_positive_offset:
            lower_bounds[1] = 0.0
        upper_bounds = np.array([np.inf for _ in range(len(p0))])
        bounds = (lower_bounds, upper_bounds)
        # Then fit the actual model.
        self._params, _ = curve_fit(
            self._model, X, y, loss=self.loss, method="trf", p0=p0, bounds=bounds
        )
        return self

    @property
    def y_intercept(self) -> float:
        return self.params[0] * (len(self.params) - 2) + self.params[1]


class NoiseModel(ABC):
    @abstractmethod
    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        pass


class NoNoise(NoiseModel):
    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        return X


@dataclass
class GaussianNoise(NoiseModel):
    apply: Callable
    sigma: float = 1

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        noise = np.random.randn(X.shape) * self.sigma
        return self.apply(X, noise)


@dataclass
class AdditiveGaussianNoise(GaussianNoise):
    apply: Callable = np.add


@dataclass
class MultiplicativeGaussianNoise(GaussianNoise):
    apply: Callable = np.multiply


@dataclass
class DataGenerator:
    model: Model
    noise_model: NoiseModel


# %%
if __name__ == "__main__":
    # %%

    mdl = LinearModel()
    out_file = mdl.save(file_name="test")
# print(Path('.').resolve())


# def stratified_sample(df, attr, n="max", s=5):
#     strat = pd.cut(
#         df[attr],
#         bins=np.linspace(df[attr].min() - 1, df[attr].max() + 1, s + 1),
#         labels=np.arange(s),
#     )
#     if n == "max":
#         n = strat.value_counts().min() * s
#     sample = df.groupby(strat, group_keys=False).apply(lambda x: x.sample(n // s))
#     return sample


# def stratified_sample_nd(df, attrs, n="max", s=3):
#     strats = []
#     for attr in attrs:
#         strats.append(
#             pd.cut(
#                 df[attr],
#                 bins=np.linspace(df[attr].min() - 1, df[attr].max() + 1, s + 1),
#                 labels=np.arange(s),
#             )
#         )
#     return strats


# def illumination_correction(img, fg, fg_emb):
#     medium_path = np.cumsum(fg_emb, axis=0)
#     embryo_path = np.cumsum(np.logical_not(fg_emb), axis=0)
#     data = pd.DataFrame(
#         np.vstack(
#             [medium_path.flatten(), embryo_path.flatten(), fg.flatten(), img.flatten()]
#         ).T,
#         columns=("medium_path", "embryo_path", "foreground", "intensity"),
#     )
#     data = data[data["foreground"] == 1]
#     sample = data
#     regressor = LinearRegression()
#     regressor.fit(
#         sample[["medium_path", "embryo_path"]],
#         np.log(sample[["intensity"]].replace({0: 1})),
#     )
#     normalizer = np.exp(regressor.predict([[0, 0]])) / np.exp(
#         regressor.predict(np.stack([medium_path.flatten(), embryo_path.flatten()]).T)
#     ).reshape(medium_path.shape)
#     return img * normalizer

# %%
