from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import chronological_split
from .regime import CausalHMMRegimeModel


REGIME_CACHE_VERSION = 1


class CausalHMMRegimePipeline:
    def __init__(
        self,
        n_states=2,
        volatility_window=20,
        random_state=42,
        n_iter=500,
        train_ratio=0.60,
        validation_ratio=0.20,
        processed_dir="data/processed/regimes",
    ):
        if train_ratio <= 0 or validation_ratio <= 0:
            raise ValueError("Split ratios must be positive")

        if train_ratio + validation_ratio >= 1:
            raise ValueError("Train and validation ratios must sum to less than 1")

        self.n_states = n_states
        self.volatility_window = volatility_window
        self.random_state = random_state
        self.n_iter = n_iter
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.processed_dir = Path(processed_dir)
        self.model_ = None
        self.state_summary_ = None
        self.last_posterior_ = None
        self.loaded_from_cache_ = False

    def get_regimes(
        self,
        ticker,
        start_date,
        end_date,
        price_data,
        refresh=False,
    ):
        ticker = _normalize_ticker(ticker)
        start, end = _normalize_date_range(start_date, end_date)
        cache_path = self.cache_path(ticker, start, end)

        if cache_path.exists() and not refresh:
            print(f"[cache] Loading {ticker} regimes from {cache_path}")
            result = self._load_cache(cache_path)
            self.model_ = None
            self.state_summary_ = self._summarize_training_states(result)
            self.last_posterior_ = self._last_probability_vector(result)
            self.loaded_from_cache_ = True
            return result

        result = self._compute_regimes(price_data, start, end)
        self._save_cache(cache_path, result)
        self.loaded_from_cache_ = False
        print(f"[cache] Saved regimes to {cache_path}")
        return result

    def cache_path(self, ticker, start_date, end_date):
        ticker = _normalize_ticker(ticker)
        start, end = _normalize_date_range(start_date, end_date)
        safe_ticker = "".join(
            character if character.isalnum() else "_"
            for character in ticker
        )
        filename = "_".join(
            [
                safe_ticker,
                start.strftime("%Y%m%d"),
                end.strftime("%Y%m%d"),
                f"s{self.n_states}",
                f"w{self.volatility_window}",
                f"rs{self.random_state}",
                f"i{self.n_iter}",
                f"tr{_float_tag(self.train_ratio)}",
                f"va{_float_tag(self.validation_ratio)}",
                f"cv{REGIME_CACHE_VERSION}",
            ]
        )
        return self.processed_dir / f"{filename}.parquet"

    def get_state_summary(self):
        if self.state_summary_ is None:
            raise RuntimeError("Generate or load regimes before requesting the summary")

        return self.state_summary_.copy()

    def _compute_regimes(self, price_data, start, end):
        model = CausalHMMRegimeModel(
            n_states=self.n_states,
            volatility_window=self.volatility_window,
            random_state=self.random_state,
            n_iter=self.n_iter,
        )
        features = model.prepare_features(price_data)
        features = _normalize_feature_index(features)
        features = features.loc[
            (features.index >= start) & (features.index < end)
        ].copy()

        train, validation, test = chronological_split(
            features,
            train_ratio=self.train_ratio,
            validation_ratio=self.validation_ratio,
        )

        model.fit(train)
        train_result, train_posterior = model.infer(train)
        validation_result, validation_posterior = model.infer(
            validation,
            initial_probs=train_posterior,
        )
        test_result, test_posterior = model.infer(
            test,
            initial_probs=validation_posterior,
        )

        train_result["dataset_split"] = "train"
        validation_result["dataset_split"] = "validation"
        test_result["dataset_split"] = "test"

        result = pd.concat(
            [train_result, validation_result, test_result],
            axis=0,
        ).sort_index()
        result.index.name = "date"
        result = result.reset_index()
        result = result[self._output_columns()]

        self.model_ = model
        self.state_summary_ = model.get_state_summary()
        self.last_posterior_ = test_posterior.copy()
        return result

    def _load_cache(self, cache_path):
        result = pd.read_parquet(cache_path)
        missing = set(self._output_columns()).difference(result.columns)

        if missing:
            raise ValueError(
                "Regime cache is missing required columns: "
                + ", ".join(sorted(missing))
            )

        result = result[self._output_columns()].copy()
        result["date"] = pd.to_datetime(result["date"]).dt.normalize()
        result = result.sort_values("date").reset_index(drop=True)
        self._validate_result(result)
        return result

    def _save_cache(self, cache_path, result):
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(".parquet.tmp")
        result.to_parquet(temporary_path, index=False)
        temporary_path.replace(cache_path)

    def _validate_result(self, result):
        if result.empty:
            raise ValueError("Regime cache is empty")

        if result["date"].duplicated().any():
            raise ValueError("Regime cache contains duplicate dates")

        if not result["date"].is_monotonic_increasing:
            raise ValueError("Regime cache is not chronologically ordered")

        expected_splits = {"train", "validation", "test"}
        actual_splits = set(result["dataset_split"].astype(str))

        if actual_splits != expected_splits:
            raise ValueError("Regime cache has invalid dataset split labels")

        probability_columns = self._probability_columns()
        probabilities = result[probability_columns].to_numpy(dtype=float)

        if not np.isfinite(probabilities).all():
            raise ValueError("Regime cache contains non-finite probabilities")

        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
            raise ValueError("Regime cache probabilities do not sum to 1")

        states = result["state"].to_numpy(dtype=int)

        if ((states < 0) | (states >= self.n_states)).any():
            raise ValueError("Regime cache contains invalid state values")

    def _summarize_training_states(self, result):
        train = result.loc[result["dataset_split"] == "train"]
        summary = (
            train.groupby("state")
            .agg(
                mean_return=("return", "mean"),
                mean_volatility=("volatility", "mean"),
                observations=("return", "size"),
                regime=("regime", "first"),
                volatility_regime=("volatility_regime", "first"),
                regime_label=("regime_label", "first"),
            )
            .reindex(range(self.n_states))
        )
        summary.index.name = "state"
        return summary

    def _last_probability_vector(self, result):
        return result.iloc[-1][self._probability_columns()].to_numpy(dtype=float)

    def _probability_columns(self):
        return [
            f"state_{state}_probability"
            for state in range(self.n_states)
        ]

    def _output_columns(self):
        return [
            "date",
            "return",
            "volatility",
            "state",
            "state_probability",
            *self._probability_columns(),
            "regime",
            "volatility_regime",
            "regime_label",
            "dataset_split",
        ]


def _normalize_feature_index(features):
    result = features.copy()
    index = pd.DatetimeIndex(pd.to_datetime(result.index))

    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)

    result.index = index.normalize()
    result = result.sort_index()

    if result.index.duplicated().any():
        raise ValueError("Price features contain duplicate dates")

    return result


def _normalize_ticker(ticker):
    normalized = str(ticker).strip().upper()

    if not normalized:
        raise ValueError("ticker cannot be empty")

    return normalized


def _normalize_date_range(start_date, end_date):
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start.tzinfo is not None:
        start = start.tz_convert("UTC").tz_localize(None)

    if end.tzinfo is not None:
        end = end.tz_convert("UTC").tz_localize(None)

    start = start.normalize()
    end = end.normalize()

    if end <= start:
        raise ValueError("end_date must be later than start_date")

    return start, end


def _float_tag(value):
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")
