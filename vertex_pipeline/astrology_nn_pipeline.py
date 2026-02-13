"""Vertex AI Pipeline definition for astrology-only NIFTY regression."""

from kfp import dsl
from kfp.dsl import Dataset, Metrics, Model, Output

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


@dsl.component(
    base_image="python:3.10",
    packages_to_install=[
        "db-dtypes==1.2.0",
        "google-cloud-bigquery==3.25.0",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "pyarrow==17.0.0",
        "tensorflow-cpu==2.15.1",
    ],
)
def train_astrology_nn_from_bq(
    project_id: str,
    bq_table: str,
    target_column: str,
    feature_columns_csv: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    early_stopping_patience: int,
    seed: int,
    model_artifact: Output[Model],
    metrics_artifact: Output[Metrics],
    predictions_artifact: Output[Dataset],
) -> None:
    import json
    import math
    import os
    import random

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from google.cloud import bigquery
    from tensorflow.keras import callbacks, layers

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    feature_columns = [c.strip() for c in feature_columns_csv.split(",") if c.strip()]
    if not feature_columns:
        raise ValueError("feature_columns_csv resolved to an empty feature list.")

    select_cols = ["Date", *feature_columns, target_column]
    select_sql = ", ".join(select_cols)
    query = f"""
        SELECT {select_sql}
        FROM `{bq_table}`
        WHERE {target_column} IS NOT NULL
    """

    bq_client = bigquery.Client(project=project_id)
    df = bq_client.query(query).to_dataframe(create_bqstorage_client=False)
    if df.empty:
        raise RuntimeError("No rows returned from BigQuery.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", target_column] + feature_columns).copy()
    if df.empty:
        raise RuntimeError("Dataset became empty after dropping nulls.")

    max_date = df["Date"].max()
    cutoff_date = max_date - pd.DateOffset(years=1)

    train_df = df[df["Date"] < cutoff_date].copy()
    valid_df = df[df["Date"] >= cutoff_date].copy()

    if train_df.empty or valid_df.empty:
        raise RuntimeError(
            "Train/validation split is empty. Ensure data spans at least one year."
        )

    for col in feature_columns:
        train_df[col] = train_df[col].astype(str)
        valid_df[col] = valid_df[col].astype(str)

    train_y = train_df[target_column].astype("float32").to_numpy()
    valid_y = valid_df[target_column].astype("float32").to_numpy()

    inputs = {}
    encoded_features = []
    for col in feature_columns:
        inp = tf.keras.Input(shape=(1,), name=col, dtype=tf.string)
        lookup = layers.StringLookup(output_mode="int", num_oov_indices=1)
        lookup.adapt(train_df[col].to_numpy())
        vocab_size = max(lookup.vocabulary_size(), 2)
        emb_dim = min(16, max(4, vocab_size // 2))

        x = lookup(inp)
        x = layers.Embedding(
            input_dim=vocab_size + 1,
            output_dim=emb_dim,
            name=f"{col}_embedding",
        )(x)
        x = layers.Flatten()(x)
        inputs[col] = inp
        encoded_features.append(x)

    if len(encoded_features) == 1:
        x = encoded_features[0]
    else:
        x = layers.Concatenate()(encoded_features)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    output = layers.Dense(1, name=target_column)(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.Huber(),
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )

    train_x = {col: train_df[col].to_numpy() for col in feature_columns}
    valid_x = {col: valid_df[col].to_numpy() for col in feature_columns}

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
    )
    history = model.fit(
        train_x,
        train_y,
        validation_data=(valid_x, valid_y),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )

    pred_y = model.predict(valid_x, verbose=0).reshape(-1)
    error = valid_y - pred_y
    abs_error = np.abs(error)

    mae = float(np.mean(abs_error))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    sse = float(np.sum(np.square(error)))
    sst = float(np.sum(np.square(valid_y - np.mean(valid_y))))
    r2 = float(1.0 - (sse / sst)) if sst > 0 else float("nan")
    direction_accuracy = float(np.mean(np.sign(valid_y) == np.sign(pred_y)))

    os.makedirs(model_artifact.path, exist_ok=True)
    model_path = os.path.join(model_artifact.path, "model.keras")
    model.save(model_path)
    model_artifact.metadata["framework"] = "tensorflow"
    model_artifact.metadata["target_column"] = target_column
    model_artifact.metadata["feature_columns"] = feature_columns

    pred_df = pd.DataFrame(
        {
            "Date": valid_df["Date"].dt.strftime("%Y-%m-%d"),
            "actual_open_close_diff": valid_y,
            "predicted_open_close_diff": pred_y,
            "error": error,
            "abs_error": abs_error,
        }
    ).sort_values("Date")
    pred_df.to_csv(predictions_artifact.path, index=False)

    metrics = {
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "epochs_trained": int(len(history.history.get("loss", []))),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "direction_accuracy": direction_accuracy,
    }
    with open(metrics_artifact.path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    for key, value in metrics.items():
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            metrics_artifact.log_metric(key, float(value))


@dsl.pipeline(
    name="nifty-astrology-nn-pipeline",
    description="Train astrology-only neural net to predict NIFTY open-close difference.",
)
def astrology_nn_pipeline(
    project_id: str = "nift108",
    bq_table: str = "nift108.nift108_ds.nifty50_full_with_nakshatra_10am_mumbai",
    target_column: str = "open_close_diff",
    feature_columns_csv: str = ",".join(ASTROLOGY_FEATURES),
    epochs: int = 120,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 15,
    seed: int = 42,
) -> None:
    train_astrology_nn_from_bq(
        project_id=project_id,
        bq_table=bq_table,
        target_column=target_column,
        feature_columns_csv=feature_columns_csv,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
    ).set_display_name("train-astrology-nn")
