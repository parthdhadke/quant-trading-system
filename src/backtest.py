import numpy as np
import pandas as pd


def run_backtest(
    data,
    position_column,
    return_column="return",
    commission_bps=5.0,
    slippage_bps=5.0,
    initial_capital=100000.0,
):
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

    result = (
        data.copy()
        .sort_index()
    )

    result[
        return_column
    ] = pd.to_numeric(
        result[
            return_column
        ],
        errors="coerce",
    )

    result[
        position_column
    ] = pd.to_numeric(
        result[
            position_column
        ],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            return_column,
            position_column,
        ]
    ).copy()

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
        ].cummax()
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
            "gross_return",
            "transaction_cost",
            "net_return",
            "equity",
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