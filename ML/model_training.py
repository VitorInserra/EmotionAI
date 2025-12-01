import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import tensorflow as tf
from typing import List, Tuple, Optional
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.optimizers import Adam


def train_random_forest(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    estimators: int = 100,
    max_depth: int = 8,
    n_jobs: int = -1,
):
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, verbose=True
    )
    rf.fit(X_train, y_train)

    return rf, X_test, y_test


def train_random_forest_regressor(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    estimators: int = 100,
    max_depth: int = 8,
    n_jobs: int = -1,
):
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1, verbose=True
    )
    rf.fit(X_train, y_train)

    return rf, X_test, y_test


def build_decod_lstm_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "arousal",
    group_by: str = "session_id",
    thresh: float = 0.5,
    fixed_T: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (num_trials, timesteps, n_features) and (num_trials,) from EEGo
    window-level data.
    """
    groups = df.groupby([group_by], sort=False)

    X_seqs: list[np.ndarray] = []
    y_labels: list[float] = []

    for (_), g in groups:

        X_seq = g[feature_cols].to_numpy(dtype=np.float32)

        y_vals = g[target_col].values
        assert np.allclose(y_vals, y_vals[0]), "Label not constant within group!"

        y_val = float(y_vals[0])
        y_bin = 1.0 if y_val > thresh else 0.0

        X_seqs.append(X_seq)
        y_labels.append(y_bin)

    if fixed_T is None:
        fixed_T = max(seq.shape[0] for seq in X_seqs)

    n_features = X_seqs[0].shape[1]
    X_padded = np.zeros((len(X_seqs), fixed_T, n_features), dtype=np.float32)

    for i, seq in enumerate(X_seqs):
        T = seq.shape[0]
        if T >= fixed_T:
            X_padded[i, :, :] = seq[:fixed_T, :]
        else:
            X_padded[i, :T, :] = seq

    y_arr = np.asarray(y_labels, dtype=np.float32)
    return X_padded, y_arr


def train_lstm(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | pd.DataFrame | np.ndarray,
    y_test: pd.Series | pd.DataFrame | np.ndarray,
    *,
    units: int = 64,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.0,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    bidirectional: bool = True,
    patience: int = 8,
    verbose: int = 0,
    random_seed: int = 42,
):
    if random_seed is not None:
        tf.keras.utils.set_random_seed(random_seed)

    X_train_arr = np.asarray(X_train).astype(np.float32, copy=False)
    X_test_arr = np.asarray(X_test).astype(np.float32, copy=False)
    y_train_arr = np.asarray(y_train)

    if y_train_arr.dtype.kind in "ifu": 
        y_train_arr = y_train_arr.astype(np.float32)
    else:
        y_train_arr = np.array(
            [1.0 if str(v).lower() == "high" else 0.0 for v in y_train_arr],
            dtype=np.float32,
        )

    if y_test is not None:
        y_test_arr = np.asarray(y_test)
        if y_test_arr.dtype.kind in "ifu":
            y_test_arr = y_test_arr.astype(np.float32)
        else:
            y_test_arr = np.array(
                [1.0 if str(v).lower() == "high" else 0.0 for v in y_test_arr],
                dtype=np.float32,
            )
    else:
        y_test_arr = None

    if X_train_arr.ndim != 3:
        raise ValueError(
            f"X_train must be 3D (trials, timesteps, features), got {X_train_arr.shape}"
        )

    timesteps = X_train_arr.shape[1]
    n_features = X_train_arr.shape[2]

    inp = layers.Input(shape=(timesteps, n_features))
    lstm_block = layers.LSTM(
        units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        return_sequences=False,
    )
    x = layers.Bidirectional(lstm_block)(inp) if bidirectional else lstm_block(inp)
    # lstm_block2 = layers.LSTM(
    #     units,
    #     dropout=dropout,
    #     recurrent_dropout=recurrent_dropout,
    #     return_sequences=False,
    # )
    # x = layers.Bidirectional(lstm_block2)(x) if bidirectional else lstm_block2(x)
    x = layers.Dense(units // 2, activation="leaky_relu")(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            mode="max",
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=max(2, patience // 2),
            min_lr=1e-6,
            mode="max",
        ),
    ]

    model.fit(
        X_train_arr,
        y_train_arr,
        epochs=epochs,
        validation_split=0.15,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=cbs,
        shuffle=False,
    )

    return model, X_test_arr, y_test_arr


def train_lstm_regressor(
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_train: pd.Series | pd.DataFrame | np.ndarray,
    y_test: pd.Series | pd.DataFrame | np.ndarray,
    *,
    units: int = 64,
    dropout: float = 0.2,
    recurrent_dropout: float = 0.0,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,
    bidirectional: bool = True,
    patience: int = 8,
    verbose: int = 0,
    random_seed: int = 42,
):
    if random_seed is not None:
        tf.keras.utils.set_random_seed(random_seed)

    X_train_arr = np.asarray(X_train).astype(np.float32, copy=False)
    X_test_arr = np.asarray(X_test).astype(np.float32, copy=False)

    y_train_arr = np.asarray(y_train, dtype=np.float32)
    y_test_arr = np.asarray(y_test, dtype=np.float32)

    X_test_arr

    timesteps = X_train_arr.shape[1]
    n_features = X_train_arr.shape[2]

    inp = layers.Input(shape=(timesteps, n_features))
    lstm_block = layers.LSTM(
        units,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        return_sequences=False,
    )
    x = layers.Bidirectional(lstm_block)(inp) if bidirectional else lstm_block(inp)
    x = layers.Dense(units // 2, activation="leaky_relu")(x)
    x = layers.Dropout(dropout)(x)

    out = layers.Dense(1, activation="linear")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="mse",
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
        ],
    )

    cbs = [
        callbacks.EarlyStopping(
            monitor="loss", patience=patience, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-6
        ),
    ]

    model.fit(
        X_train_arr,
        y_train_arr,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=cbs,
        shuffle=False,
    )

    return model, X_test_arr, y_test_arr
