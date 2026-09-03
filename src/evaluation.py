import numpy as np
import pandas as pd


def chronological_split(
    data,
    train_ratio=0.60,
    validation_ratio=0.20,
):
    if data.empty:
        raise ValueError(
            "Cannot split an empty dataset"
        )

    if (
        train_ratio <= 0
        or validation_ratio <= 0
    ):
        raise ValueError(
            "train_ratio and validation_ratio must be positive"
        )

    if (
        train_ratio
        + validation_ratio
        >= 1
    ):
        raise ValueError(
            "train_ratio + validation_ratio must be less than 1"
        )

    ordered = (
        data.copy()
        .sort_index()
    )

    total_rows = len(
        ordered
    )

    train_end = int(
        total_rows
        * train_ratio
    )

    validation_end = int(
        total_rows
        * (
            train_ratio
            + validation_ratio
        )
    )

    if train_end < 1:
        raise ValueError(
            "Training split is empty"
        )

    if (
        validation_end
        <= train_end
    ):
        raise ValueError(
            "Validation split is empty"
        )

    if (
        validation_end
        >= total_rows
    ):
        raise ValueError(
            "Test split is empty"
        )

    train = ordered.iloc[
        :train_end
    ].copy()

    validation = ordered.iloc[
        train_end:validation_end
    ].copy()

    test = ordered.iloc[
        validation_end:
    ].copy()

    return (
        train,
        validation,
        test,
    )


def performance_metrics(
    backtest,
    return_column="net_return",
    annualization_factor=252,
):
    if (
        return_column
        not in backtest.columns
    ):
        raise ValueError(
            f"Missing required column: {return_column}"
        )

    returns = pd.to_numeric(
        backtest[
            return_column
        ],
        errors="coerce",
    ).dropna()

    if returns.empty:
        raise ValueError(
            "No valid returns available"
        )

    total_return = float(
        (
            1.0 + returns
        ).prod()
        - 1.0
    )

    periods = len(
        returns
    )

    if total_return <= -1.0:
        annualized_return = -1.0
    else:
        annualized_return = float(
            (
                1.0
                + total_return
            )
            ** (
                annualization_factor
                / periods
            )
            - 1.0
        )

    daily_std = float(
        returns.std(
            ddof=1
        )
    )

    if np.isfinite(
        daily_std
    ):
        annualized_volatility = (
            daily_std
            * np.sqrt(
                annualization_factor
            )
        )
    else:
        annualized_volatility = (
            0.0
        )

    if (
        daily_std > 0
        and np.isfinite(
            daily_std
        )
    ):
        sharpe_ratio = float(
            returns.mean()
            / daily_std
            * np.sqrt(
                annualization_factor
            )
        )
    else:
        sharpe_ratio = 0.0

    equity_curve = (
        1.0 + returns
    ).cumprod()

    running_peak = (
        equity_curve.cummax()
    )

    drawdown = (
        equity_curve
        / running_peak
        - 1.0
    )

    max_drawdown = float(
        drawdown.min()
    )

    if (
        "executed_position"
        in backtest.columns
    ):
        active_mask = (
            backtest.loc[
                returns.index,
                "executed_position",
            ].abs()
            > 0
        )

        active_returns = (
            returns[
                active_mask
            ]
        )

    else:
        active_returns = (
            returns[
                returns != 0
            ]
        )

    if active_returns.empty:
        win_rate = 0.0
    else:
        win_rate = float(
            (
                active_returns
                > 0
            ).mean()
        )

    if (
        "trade_event"
        in backtest.columns
    ):
        number_of_trades = int(
            backtest[
                "trade_event"
            ].sum()
        )

    elif (
        "turnover"
        in backtest.columns
    ):
        number_of_trades = int(
            (
                backtest[
                    "turnover"
                ]
                > 0
            ).sum()
        )

    else:
        number_of_trades = 0

    if (
        "turnover"
        in backtest.columns
    ):
        total_turnover = float(
            backtest[
                "turnover"
            ].sum()
        )
    else:
        total_turnover = 0.0

    if (
        "transaction_cost"
        in backtest.columns
    ):
        total_transaction_cost = float(
            backtest[
                "transaction_cost"
            ].sum()
        )
    else:
        total_transaction_cost = 0.0

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "number_of_trades": number_of_trades,
        "total_turnover": total_turnover,
        "total_transaction_cost": total_transaction_cost,
    }


def compare_strategies(
    backtests,
):
    if not backtests:
        raise ValueError(
            "No backtests were provided"
        )

    rows = []

    for (
        strategy_name,
        backtest,
    ) in backtests.items():
        metrics = (
            performance_metrics(
                backtest
            )
        )

        metrics[
            "strategy"
        ] = strategy_name

        rows.append(
            metrics
        )

    columns = [
        "strategy",
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "win_rate",
        "number_of_trades",
        "total_turnover",
        "total_transaction_cost",
    ]

    return pd.DataFrame(
        rows
    )[columns]