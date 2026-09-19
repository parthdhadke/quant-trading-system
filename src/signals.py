import numpy as np

from config import MAX_POSITION_SIZE, MIN_POSITION_SIZE, TARGET_VOLATILITY


def sentiment_signal(
    score,
    buy_threshold,
    sell_threshold,
):
    if sell_threshold >= buy_threshold:
        raise ValueError(
            "sell_threshold must be lower than buy_threshold"
        )

    if score >= buy_threshold:
        return 1

    if score <= sell_threshold:
        return -1

    return 0


def regime_signal(regime):
    normalized = (
        str(regime)
        .strip()
        .lower()
    )

    if normalized.startswith(
        "bull"
    ):
        return 1

    if normalized.startswith(
        "bear"
    ):
        return -1

    return 0


def combined_signal(
    sentiment_score,
    regime,
    buy_threshold,
    sell_threshold,
    sentiment_weight,
    regime_weight,
    decision_threshold,
):
    if (
        sentiment_weight < 0
        or regime_weight < 0
    ):
        raise ValueError(
            "Signal weights cannot be negative"
        )

    weight_total = (
        sentiment_weight
        + regime_weight
    )

    if weight_total <= 0:
        raise ValueError(
            "At least one signal weight must be positive"
        )

    if (
        decision_threshold <= 0
        or decision_threshold > 1
    ):
        raise ValueError(
            "decision_threshold must be greater than 0 and at most 1"
        )

    sentiment_component = (
        sentiment_signal(
            sentiment_score,
            buy_threshold,
            sell_threshold,
        )
    )

    regime_component = (
        regime_signal(
            regime
        )
    )

    combined_score = (
        sentiment_weight
        * sentiment_component
        + regime_weight
        * regime_component
    ) / weight_total

    if (
        combined_score
        >= decision_threshold
    ):
        return 1

    if (
        combined_score
        <= -decision_threshold
    ):
        return -1

    return 0


def volatility_position_size(
    current_volatility,
    target_volatility=TARGET_VOLATILITY,
    max_position_size=MAX_POSITION_SIZE,
    min_position_size=MIN_POSITION_SIZE,
):
    if not np.isfinite([target_volatility, max_position_size, min_position_size]).all():
        raise ValueError("Sizing settings must be finite")
    if target_volatility <= 0:
        raise ValueError(
            "target_volatility must be positive"
        )

    if max_position_size <= 0:
        raise ValueError(
            "max_position_size must be positive"
        )

    if max_position_size > 1.0:
        raise ValueError("max_position_size cannot exceed 1.0; leverage is disabled")

    if min_position_size < 0:
        raise ValueError(
            "min_position_size cannot be negative"
        )

    if (
        min_position_size
        > max_position_size
    ):
        raise ValueError(
            "min_position_size cannot exceed max_position_size"
        )

    if (
        current_volatility is None
        or not np.isfinite(
            current_volatility
        )
    ):
        return min_position_size

    if current_volatility <= 0:
        return max_position_size

    if current_volatility <= target_volatility / max_position_size:
        return float(max_position_size)

    raw_size = (
        target_volatility
        / current_volatility
    )

    return float(
        np.clip(
            raw_size,
            min_position_size,
            max_position_size,
        )
    )


def add_strategy_signals(
    data,
    buy_threshold,
    sell_threshold,
    sentiment_weight,
    regime_weight,
    decision_threshold,
):
    required = {
        "sentiment_score",
        "regime",
    }

    missing = (
        required.difference(
            data.columns
        )
    )

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    result = data.copy()

    result[
        "sentiment_signal"
    ] = result[
        "sentiment_score"
    ].apply(
        lambda score: sentiment_signal(
            score,
            buy_threshold,
            sell_threshold,
        )
    )

    result[
        "regime_signal"
    ] = result[
        "regime"
    ].apply(
        regime_signal
    )

    result[
        "combined_signal"
    ] = result.apply(
        lambda row: combined_signal(
            row[
                "sentiment_score"
            ],
            row["regime"],
            buy_threshold,
            sell_threshold,
            sentiment_weight,
            regime_weight,
            decision_threshold,
        ),
        axis=1,
    )

    return result


def apply_volatility_sizing(
    data,
    signal_column,
    volatility_column,
    target_volatility=TARGET_VOLATILITY,
    max_position_size=MAX_POSITION_SIZE,
    min_position_size=MIN_POSITION_SIZE,
    output_column="position",
    size_column="position_size",
):
    if signal_column not in data.columns:
        raise ValueError(
            f"Missing required column: {signal_column}"
        )

    if (
        volatility_column
        not in data.columns
    ):
        raise ValueError(
            f"Missing required column: {volatility_column}"
        )

    if not data.columns.is_unique:
        raise ValueError("Sizing requires unique input columns")
    if (output_column in (signal_column, volatility_column)
            or size_column in (signal_column, volatility_column, output_column)):
        raise ValueError("Sizing output columns must not overwrite signals or volatility")
    if not data[signal_column].isin([-1, 0, 1]).all():
        raise ValueError("Directional signals must contain only -1, 0, +1")
    volatility_position_size(None, target_volatility, max_position_size, min_position_size)
    result = data.copy()

    sizes = result[
        volatility_column
    ].apply(
        lambda volatility: volatility_position_size(
            volatility,
            target_volatility,
            max_position_size,
            min_position_size,
        )
    )

    result[size_column] = sizes.astype(float)

    result[
        output_column
    ] = (
        result[
            signal_column
        ].astype(float)
        * sizes
    )

    return result
