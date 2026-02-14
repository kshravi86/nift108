#!/usr/bin/env python3
"""Train astrology-only NIFTY model on a TPU VM."""

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import callbacks, layers

ASTROLOGY_FEATURES = [
    "Sun_Nakshatra",
    "Moon_Nakshatra",
    "Mars_Nakshatra",
    "Mercury_Nakshatra",
    "Jupiter_Nakshatra",
    "Venus_Nakshatra",
    "Saturn_Nakshatra",
    "Rahu_Nakshatra",
    "Ketu_Nakshatra",
    "Sun_Pada",
    "Moon_Pada",
    "Mars_Pada",
    "Mercury_Pada",
    "Jupiter_Pada",
    "Venus_Pada",
    "Saturn_Pada",
    "Rahu_Pada",
    "Ketu_Pada",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train astrology-only model on TPU.")
    parser.add_argument(
        "--data-uri",
        default="gs://nift108-bucket/nifty50_full_with_nakshatra_10am_mumbai.csv",
        help="Input CSV path (local or gs://).",
    )
    parser.add_argument(
        "--output-dir",
        default="tpu_outputs/latest",
        help="Local output directory for model/metrics/predictions.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=120,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--gcs-output-prefix",
        default="",
        help="Optional gs:// prefix to upload outputs.",
    )
    return parser.parse_args()


def configure_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data(data_uri: str) -> pd.DataFrame:
    with tf.io.gfile.GFile(data_uri, "r") as f:
        df = pd.read_csv(f)
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" not in out.columns:
        raise ValueError("Input data must include a Date column.")
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")

    if "open_close_diff" not in out.columns:
        if "Open" in out.columns and "Close" in out.columns:
            out["open_close_diff"] = out["Open"] - out["Close"]
        else:
            raise ValueError("Need open_close_diff or Open/Close columns.")

    needed = ["Date", "open_close_diff"] + ASTROLOGY_FEATURES
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = out[needed].dropna(subset=needed).copy()
    if out.empty:
        raise ValueError("No rows after null filtering.")

    for col in ASTROLOGY_FEATURES:
        out[col] = out[col].astype(str)
    out["open_close_diff"] = out["open_close_diff"].astype("float32")
    return out


def split_train_valid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_date = df["Date"].max()
    cutoff_date = max_date - pd.DateOffset(years=1)
    train_df = df[df["Date"] < cutoff_date].copy()
    valid_df = df[df["Date"] >= cutoff_date].copy()
    if train_df.empty or valid_df.empty:
        raise ValueError("Train/validation split is empty.")
    return train_df, valid_df


def make_tf_dataset(
    frame: pd.DataFrame,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    x = {c: frame[c].to_numpy() for c in ASTROLOGY_FEATURES}
    y = frame["open_close_diff"].to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(min(len(frame), 20000), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(train_df: pd.DataFrame, learning_rate: float) -> tf.keras.Model:
    inputs = {}
    encoded = []
    for col in ASTROLOGY_FEATURES:
        inp = tf.keras.Input(shape=(1,), name=col, dtype=tf.string)
        lookup = layers.StringLookup(output_mode="int", num_oov_indices=1)
        lookup.adapt(train_df[col].to_numpy())
        vocab_size = max(lookup.vocabulary_size(), 2)
        emb_dim = min(16, max(4, vocab_size // 2))
        x = lookup(inp)
        x = layers.Embedding(vocab_size + 1, emb_dim, name=f"{col}_emb")(x)
        x = layers.Flatten()(x)
        inputs[col] = inp
        encoded.append(x)

    x = layers.Concatenate()(encoded)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    out = layers.Dense(1, name="open_close_diff")(x)
    model = tf.keras.Model(inputs=inputs, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


def get_strategy() -> tf.distribute.Strategy:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu="local")
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    return tf.distribute.TPUStrategy(resolver)


def evaluate_predictions(valid_df: pd.DataFrame, pred: np.ndarray) -> dict:
    actual = valid_df["open_close_diff"].to_numpy()
    error = actual - pred
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    sse = float(np.sum(np.square(error)))
    sst = float(np.sum(np.square(actual - np.mean(actual))))
    r2 = float(1.0 - (sse / sst)) if sst > 0 else float("nan")
    direction_acc = float(np.mean(np.sign(actual) == np.sign(pred)))
    return {
        "validation_rows": int(len(valid_df)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "direction_accuracy": direction_acc,
    }


def save_outputs(
    output_dir: Path,
    model: tf.keras.Model,
    metrics: dict,
    valid_df: pd.DataFrame,
    pred: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "model.keras")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pred_df = pd.DataFrame(
        {
            "Date": valid_df["Date"].dt.strftime("%Y-%m-%d"),
            "actual_open_close_diff": valid_df["open_close_diff"].to_numpy(),
            "predicted_open_close_diff": pred,
            "error": valid_df["open_close_diff"].to_numpy() - pred,
        }
    )
    pred_df.to_csv(output_dir / "predictions.csv", index=False)


def upload_outputs_to_gcs(local_output_dir: Path, gcs_prefix: str) -> None:
    gcs_prefix = gcs_prefix.rstrip("/")
    for file_path in local_output_dir.iterdir():
        if not file_path.is_file():
            continue
        target = f"{gcs_prefix}/{file_path.name}"
        tf.io.gfile.copy(str(file_path), target, overwrite=True)


def main() -> None:
    args = parse_args()
    configure_seed(args.seed)

    print("Loading data:", args.data_uri, flush=True)
    df = prepare_dataframe(load_data(args.data_uri))
    train_df, valid_df = split_train_valid(df)
    print(
        f"Rows train={len(train_df)} valid={len(valid_df)} "
        f"date_min={df['Date'].min().date()} date_max={df['Date'].max().date()}",
        flush=True,
    )

    strategy = get_strategy()
    print("TPU strategy replicas:", strategy.num_replicas_in_sync, flush=True)

    train_ds = make_tf_dataset(train_df, args.batch_size, shuffle=True)
    valid_ds = make_tf_dataset(valid_df, args.batch_size, shuffle=False)

    with strategy.scope():
        model = build_model(train_df, args.learning_rate)

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
    )

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        callbacks=[early_stop],
        verbose=2,
    )

    pred = model.predict(valid_ds, verbose=0).reshape(-1)
    metrics = evaluate_predictions(valid_df, pred)
    metrics["train_rows"] = int(len(train_df))
    metrics["epochs_trained"] = int(len(history.history.get("loss", [])))
    metrics = {
        k: (float(v) if isinstance(v, (float, np.floating)) and math.isfinite(v) else v)
        for k, v in metrics.items()
    }

    output_dir = Path(args.output_dir)
    save_outputs(output_dir, model, metrics, valid_df, pred)

    if args.gcs_output_prefix.strip():
        upload_outputs_to_gcs(output_dir, args.gcs_output_prefix.strip())
        print("Uploaded outputs to:", args.gcs_output_prefix, flush=True)

    print("Metrics:", json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()

