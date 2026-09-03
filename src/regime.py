import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler


class CausalHMMRegimeModel:
    def __init__(
        self,
        n_states=2,
        volatility_window=20,
        random_state=42,
        n_iter=500,
    ):
        if n_states < 2:
            raise ValueError(
                "n_states must be at least 2"
            )

        if volatility_window < 2:
            raise ValueError(
                "volatility_window must be at least 2"
            )

        self.n_states = n_states
        self.volatility_window = volatility_window
        self.random_state = random_state
        self.n_iter = n_iter

        self.feature_columns = [
            "return",
            "volatility",
        ]

        self.scaler = StandardScaler()

        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
        )

        self.state_summary_ = None
        self.state_labels_ = None
        self.train_last_posterior_ = None
        self.training_index_ = None
        self.training_observation_count_ = 0
        self.is_fitted_ = False

    def prepare_features(self, price_data):
        close = self._extract_close(
            price_data
        )

        returns = close.pct_change()

        volatility = (
            returns.rolling(
                window=self.volatility_window,
                min_periods=self.volatility_window,
            ).std()
            * np.sqrt(252)
        )

        features = pd.DataFrame(
            {
                "return": returns,
                "volatility": volatility,
            },
            index=close.index,
        )

        return (
            features
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )

    def fit(self, train_features):
        self._validate_features(
            train_features
        )

        x_train = (
            train_features[
                self.feature_columns
            ]
            .to_numpy(dtype=float)
        )

        x_train_scaled = (
            self.scaler.fit_transform(
                x_train
            )
        )

        self.model.fit(
            x_train_scaled
        )

        self.is_fitted_ = True

        (
            train_probabilities,
            self.train_last_posterior_,
        ) = self._filter_probabilities(
            train_features,
            initial_probs=None,
        )

        train_states = train_probabilities.argmax(
            axis=1
        )

        (
            self.state_summary_,
            self.state_labels_,
        ) = self._build_state_labels(
            train_features,
            train_states,
        )

        self.training_index_ = train_features.index.copy()
        self.training_observation_count_ = len(train_features)

        return self

    def infer(
        self,
        features,
        initial_probs=None,
    ):
        if not self.is_fitted_:
            raise RuntimeError(
                "Fit the HMM before calling infer"
            )

        (
            probabilities,
            previous_posterior,
        ) = self._filter_probabilities(
            features,
            initial_probs,
        )

        states = probabilities.argmax(
            axis=1
        )

        result = features.copy()

        result["state"] = states

        result[
            "state_probability"
        ] = probabilities.max(
            axis=1
        )

        for state in range(
            self.n_states
        ):
            result[
                f"state_{state}_probability"
            ] = probabilities[:, state]

        result["regime"] = [
            self.state_labels_[
                state
            ]["regime"]
            for state in states
        ]

        result[
            "volatility_regime"
        ] = [
            self.state_labels_[
                state
            ]["volatility_regime"]
            for state in states
        ]

        result[
            "regime_label"
        ] = [
            self.state_labels_[
                state
            ]["regime_label"]
            for state in states
        ]

        return (
            result,
            previous_posterior,
        )

    def infer_after_training(
        self,
        features,
    ):
        return self.infer(
            features,
            initial_probs=(
                self.train_last_posterior_
            ),
        )

    def get_state_summary(self):
        if self.state_summary_ is None:
            raise RuntimeError(
                "Fit the HMM before requesting the state summary"
            )

        return (
            self.state_summary_.copy()
        )

    def _filter_probabilities(
        self,
        features,
        initial_probs,
    ):
        self._validate_features(
            features
        )

        x = (
            features[
                self.feature_columns
            ]
            .to_numpy(dtype=float)
        )

        x_scaled = self.scaler.transform(
            x
        )

        if initial_probs is not None:
            initial_probs = np.asarray(
                initial_probs,
                dtype=float,
            )

            if initial_probs.shape != (
                self.n_states,
            ):
                raise ValueError(
                    "initial_probs has an invalid shape"
                )

            if (
                not np.isfinite(initial_probs).all()
                or (initial_probs < 0).any()
            ):
                raise ValueError(
                    "initial_probs must contain finite non-negative values"
                )

            total = initial_probs.sum()

            if total <= 0:
                raise ValueError(
                    "initial_probs must sum to a positive value"
                )

            initial_probs = (
                initial_probs / total
            )

        probabilities = np.zeros(
            (
                len(x_scaled),
                self.n_states,
            ),
            dtype=float,
        )

        previous_posterior = (
            initial_probs
        )

        for row_index, observation in enumerate(
            x_scaled
        ):
            if previous_posterior is None:
                prior = (
                    self.model.startprob_
                )
            else:
                prior = (
                    previous_posterior
                    @ self.model.transmat_
                )

            prior = np.clip(
                prior,
                1e-300,
                None,
            )

            log_prior = np.log(
                prior
            )

            log_emission = np.array(
                [
                    multivariate_normal.logpdf(
                        observation,
                        mean=self.model.means_[
                            state
                        ],
                        cov=self.model.covars_[
                            state
                        ],
                        allow_singular=True,
                    )
                    for state in range(
                        self.n_states
                    )
                ]
            )

            log_posterior = (
                log_prior
                + log_emission
            )

            log_posterior -= logsumexp(
                log_posterior
            )

            posterior = np.exp(
                log_posterior
            )

            probabilities[
                row_index
            ] = posterior

            previous_posterior = (
                posterior
            )

        return (
            probabilities,
            previous_posterior,
        )

    def _build_state_labels(
        self,
        train_features,
        train_states,
    ):
        working = (
            train_features[
                self.feature_columns
            ].copy()
        )

        working["state"] = (
            train_states
        )

        summary = (
            working.groupby("state")
            .agg(
                mean_return=(
                    "return",
                    "mean",
                ),
                mean_volatility=(
                    "volatility",
                    "mean",
                ),
                observations=(
                    "return",
                    "size",
                ),
            )
            .reindex(
                range(
                    self.n_states
                )
            )
        )

        return_order = (
            summary["mean_return"]
            .fillna(-np.inf)
            .sort_values()
            .index
            .tolist()
        )

        regimes = {
            state: "neutral"
            for state in range(
                self.n_states
            )
        }

        regimes[
            return_order[0]
        ] = "bear"

        regimes[
            return_order[-1]
        ] = "bull"

        volatility_median = float(
            summary[
                "mean_volatility"
            ]
            .dropna()
            .median()
        )

        labels = {}

        for state in range(
            self.n_states
        ):
            state_volatility = (
                summary.loc[
                    state,
                    "mean_volatility",
                ]
            )

            if pd.isna(
                state_volatility
            ):
                volatility_regime = (
                    "unknown"
                )
            elif (
                state_volatility
                >= volatility_median
            ):
                volatility_regime = (
                    "high_volatility"
                )
            else:
                volatility_regime = (
                    "low_volatility"
                )

            regime = regimes[
                state
            ]

            labels[state] = {
                "regime": regime,
                "volatility_regime": (
                    volatility_regime
                ),
                "regime_label": (
                    f"{regime}_{volatility_regime}"
                ),
            }

        summary["regime"] = [
            labels[state]["regime"]
            for state in summary.index
        ]

        summary[
            "volatility_regime"
        ] = [
            labels[state][
                "volatility_regime"
            ]
            for state in summary.index
        ]

        summary[
            "regime_label"
        ] = [
            labels[state][
                "regime_label"
            ]
            for state in summary.index
        ]

        return (
            summary,
            labels,
        )

    def _validate_features(
        self,
        features,
    ):
        missing = [
            column
            for column in (
                self.feature_columns
            )
            if column
            not in features.columns
        ]

        if missing:
            raise ValueError(
                "Missing required feature columns: "
                + ", ".join(missing)
            )

        if features.empty:
            raise ValueError(
                "Feature data is empty"
            )

        values = (
            features[
                self.feature_columns
            ]
            .to_numpy(dtype=float)
        )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                "Feature data contains NaN or infinite values"
            )

    @staticmethod
    def _extract_close(
        price_data,
    ):
        if "Close" in price_data.columns:
            close = (
                price_data["Close"]
            )

        elif isinstance(
            price_data.columns,
            pd.MultiIndex,
        ):
            if (
                "Close"
                in price_data.columns.get_level_values(
                    0
                )
            ):
                close = price_data.xs(
                    "Close",
                    axis=1,
                    level=0,
                )

            elif (
                "Close"
                in price_data.columns.get_level_values(
                    1
                )
            ):
                close = price_data.xs(
                    "Close",
                    axis=1,
                    level=1,
                )

            else:
                raise ValueError(
                    "Price data does not contain a Close column"
                )

        else:
            raise ValueError(
                "Price data does not contain a Close column"
            )

        if isinstance(
            close,
            pd.DataFrame,
        ):
            if close.shape[1] != 1:
                raise ValueError(
                    "Expected price data for one ticker at a time"
                )

            close = close.iloc[:, 0]

        return (
            close
            .astype(float)
            .sort_index()
        )
