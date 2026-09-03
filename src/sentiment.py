import pandas as pd
import torch
from transformers import pipeline


class FinBERTSentiment:
    def __init__(self, model_name="ProsusAI/finbert"):
        device = 0 if torch.cuda.is_available() else -1

        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            tokenizer=model_name,
            device=device,
        )

    def score_texts(self, texts, batch_size=16):
        clean_texts = [
            str(text).strip()
            for text in texts
            if str(text).strip()
        ]

        if not clean_texts:
            return pd.DataFrame(
                columns=[
                    "text",
                    "positive",
                    "neutral",
                    "negative",
                    "sentiment_score",
                    "sentiment_label",
                ]
            )

        outputs = self.classifier(
            clean_texts,
            top_k=None,
            truncation=True,
            batch_size=batch_size,
        )

        if outputs and isinstance(outputs[0], dict):
            outputs = [outputs]

        rows = []

        for text, result in zip(clean_texts, outputs):
            scores = {
                item["label"].lower(): float(item["score"])
                for item in result
            }

            positive = scores.get("positive", 0.0)
            neutral = scores.get("neutral", 0.0)
            negative = scores.get("negative", 0.0)

            sentiment_score = positive - negative

            label_scores = {
                "positive": positive,
                "neutral": neutral,
                "negative": negative,
            }

            sentiment_label = max(
                label_scores,
                key=label_scores.get,
            )

            rows.append(
                {
                    "text": text,
                    "positive": positive,
                    "neutral": neutral,
                    "negative": negative,
                    "sentiment_score": sentiment_score,
                    "sentiment_label": sentiment_label,
                }
            )

        return pd.DataFrame(rows)

    def aggregate_score(self, texts, batch_size=16):
        scored = self.score_texts(
            texts,
            batch_size=batch_size,
        )

        if scored.empty:
            return 0.0

        return float(
            scored["sentiment_score"].mean()
        )

    def score_dataframe(
        self,
        news,
        text_column="headline",
        date_column="date",
        batch_size=16,
    ):
        if text_column not in news.columns:
            raise ValueError(
                f"Missing required column: {text_column}"
            )

        if date_column not in news.columns:
            raise ValueError(
                f"Missing required column: {date_column}"
            )

        working = news.copy()

        working = working.dropna(
            subset=[
                text_column,
                date_column,
            ]
        )

        working[text_column] = (
            working[text_column]
            .astype(str)
            .str.strip()
        )

        working = working[
            working[text_column] != ""
        ].copy()

        if working.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "sentiment_score",
                    "positive",
                    "neutral",
                    "negative",
                    "headline_count",
                ]
            )

        scored = self.score_texts(
            working[text_column].tolist(),
            batch_size=batch_size,
        )

        working = working.reset_index(drop=True)
        scored = scored.reset_index(drop=True)

        working["sentiment_score"] = (
            scored["sentiment_score"]
        )

        working["positive"] = scored["positive"]
        working["neutral"] = scored["neutral"]
        working["negative"] = scored["negative"]

        working[date_column] = pd.to_datetime(
            working[date_column]
        ).dt.normalize()

        daily = (
            working.groupby(date_column)
            .agg(
                sentiment_score=(
                    "sentiment_score",
                    "mean",
                ),
                positive=(
                    "positive",
                    "mean",
                ),
                neutral=(
                    "neutral",
                    "mean",
                ),
                negative=(
                    "negative",
                    "mean",
                ),
                headline_count=(
                    text_column,
                    "count",
                ),
            )
            .reset_index()
            .rename(
                columns={
                    date_column: "date"
                }
            )
        )

        return daily