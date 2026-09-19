import numpy as np
import pandas as pd

from config import BACKTEST_INITIAL_CAPITAL, COMMISSION_BPS, SLIPPAGE_BPS


def run_backtest(
    data,
    position_column,
    return_column="return",
    commission_bps=COMMISSION_BPS,
    slippage_bps=SLIPPAGE_BPS,
    initial_capital=BACKTEST_INITIAL_CAPITAL,
):
    if data.empty or not data.columns.is_unique:
        raise ValueError("Backtest requires nonempty data with unique columns")

    settings = np.asarray([commission_bps, slippage_bps, initial_capital], dtype=float)
    if not np.isfinite(settings).all():
        raise ValueError("Backtest settings must be finite")
    if (
        position_column
        not in data.columns
    ):
        raise ValueError(
            f"Missing required column: {position_column}"
        )

    if (
        return_column
        not in data.columns
    ):
        raise ValueError(
            f"Missing required column: {return_column}"
        )

    if (
        commission_bps < 0
        or slippage_bps < 0
    ):
        raise ValueError(
            "Transaction costs cannot be negative"
        )

    if initial_capital <= 0:
        raise ValueError(
            "initial_capital must be positive"
        )

    result = data.copy()
    if "date" in result.columns:
        dates = pd.DatetimeIndex(pd.to_datetime(result["date"], errors="raise"))
        if dates.isna().any() or dates.has_duplicates:
            raise ValueError("Backtest dates must be non-missing and unique")
        result["date"] = dates
        result = result.sort_values("date")
    else:
        if result.index.has_duplicates or result.index.isna().any():
            raise ValueError("Backtest index must be non-missing and unique")
        result = result.sort_index()

    result[
        return_column
    ] = pd.to_numeric(
        result[
            return_column
        ],
        errors="raise",
    )

    result[
        position_column
    ] = pd.to_numeric(
        result[
            position_column
        ],
        errors="raise",
    )

    if not np.isfinite(result[[return_column, position_column]].to_numpy(dtype=float)).all():
        raise ValueError("Returns and positions must be finite; rows cannot be silently dropped")
    if (result[return_column] < -1.0).any():
        raise ValueError("Simple market returns cannot be less than -100%")

    result[
        "target_position"
    ] = result[
        position_column
    ].astype(float)

    result[
        "executed_position"
    ] = (
        result[
            "target_position"
        ]
        .shift(1)
        .fillna(0.0)
    )

    result[
        "turnover"
    ] = (
        result[
            "executed_position"
        ]
        .diff()
        .abs()
        .fillna(
            result[
                "executed_position"
            ].abs()
        )
    )

    total_cost_rate = (
        float(
            commission_bps
        )
        + float(
            slippage_bps
        )
    ) / 10000.0

    result[
        "gross_return"
    ] = (
        result[
            "executed_position"
        ]
        * result[
            return_column
        ]
    )

    result[
        "transaction_cost"
    ] = (
        result[
            "turnover"
        ]
        * total_cost_rate
    )

    result[
        "net_return"
    ] = (
        result[
            "gross_return"
        ]
        - result[
            "transaction_cost"
        ]
    )

    if (result[["gross_return", "net_return"]] <= -1.0).any().any():
        raise ValueError("Strategy loses all capital; insolvency handling is not implemented")

    result[
        "gross_equity"
    ] = (
        float(
            initial_capital
        )
        * (
            1.0
            + result[
                "gross_return"
            ]
        ).cumprod()
    )

    result[
        "equity"
    ] = (
        float(
            initial_capital
        )
        * (
            1.0
            + result[
                "net_return"
            ]
        ).cumprod()
    )

    running_peak = (
        result[
            "equity"
        ].cummax().clip(lower=float(initial_capital))
    )

    result[
        "drawdown"
    ] = (
        result[
            "equity"
        ]
        / running_peak
    ) - 1.0

    result[
        "trade_event"
    ] = (
        result[
            "turnover"
        ] > 0
    ).astype(int)

    values = result[
        [
            "target_position",
            "executed_position",
            "turnover",
            "gross_return",
            "transaction_cost",
            "net_return",
            "gross_equity",
            "equity",
            "drawdown",
        ]
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "Backtest produced non-finite values"
        )

    return result
