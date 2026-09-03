from .backtest import run_backtest

from .evaluation import (
    chronological_split,
    compare_strategies,
    performance_metrics,
)

from .news import (
    AlphaVantageNewsClient,
    NewsSentimentPipeline,
    aggregate_daily_sentiment,
    align_news_to_trading_days,
    normalize_alpha_vantage_news,
)

from .regime import (
    CausalHMMRegimeModel,
)

from .regime_pipeline import (
    CausalHMMRegimePipeline,
)

from .sentiment import (
    FinBERTSentiment,
)

from .signals import (
    add_strategy_signals,
    apply_volatility_sizing,
    combined_signal,
    regime_signal,
    sentiment_signal,
    volatility_position_size,
)

__all__ = [
    "FinBERTSentiment",
    "CausalHMMRegimeModel",
    "CausalHMMRegimePipeline",
    "AlphaVantageNewsClient",
    "NewsSentimentPipeline",
    "normalize_alpha_vantage_news",
    "align_news_to_trading_days",
    "aggregate_daily_sentiment",
    "sentiment_signal",
    "regime_signal",
    "combined_signal",
    "add_strategy_signals",
    "volatility_position_size",
    "apply_volatility_sizing",
    "run_backtest",
    "chronological_split",
    "performance_metrics",
    "compare_strategies",
]
