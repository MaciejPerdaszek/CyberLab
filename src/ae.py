from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE
from src.data import validate_fraction


@dataclass
class AutoencoderTrainingResult:
    autoencoder: Any
    encoder: Any
    keras: Any
    history: Any
    normal_validation: pd.DataFrame
    training_time_seconds: float
    fit_samples: int
    validation_samples: int


def build_autoencoder(
    input_dimension: int,
    latent_dimension: int,
    learning_rate: float = 0.001,
) -> tuple[Any, Any, Any]:
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as error:
        raise ImportError(
            "TensorFlow nie jest zainstalowany. Zainstaluj tensorflow, "
            "aby uruchomić model ae_rf."
        ) from error

    tf.random.set_seed(RANDOM_STATE)

    inputs = keras.Input(shape=(input_dimension,), name="network_features")
    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    latent = layers.Dense(
        latent_dimension,
        activation="relu",
        name="latent_vector",
    )(x)
    x = layers.Dense(32, activation="relu")(latent)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(
        input_dimension,
        activation="linear",
        name="reconstruction",
    )(x)

    autoencoder = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name=f"anomaly_autoencoder_latent_{latent_dimension}",
    )
    encoder = keras.Model(
        inputs=inputs,
        outputs=latent,
        name=f"encoder_latent_{latent_dimension}",
    )
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return autoencoder, encoder, keras


def make_callbacks(
    keras: Any,
    *,
    early_stopping_patience: int = 5,
    reduce_lr_patience: int = 2,
) -> list[Any]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
        ),
    ]


def fit_autoencoder_on_normal(
    X_normal: pd.DataFrame,
    *,
    latent_dimension: int,
    epochs: int,
    batch_size: int,
    validation_size: float,
    verbose: int = 1,
    learning_rate: float = 0.001,
) -> AutoencoderTrainingResult:
    validate_fraction(validation_size, "normal_validation_size")
    if latent_dimension <= 0:
        raise ValueError("latent_dimension musi być większy od 0.")
    if len(X_normal) < 2:
        raise ValueError("Za mało próbek Normal do treningu autoenkodera.")

    X_fit, X_validation = train_test_split(
        X_normal,
        test_size=validation_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    if len(X_fit) == 0 or len(X_validation) == 0:
        raise ValueError("Podział próbek Normal utworzył pusty zbiór.")

    autoencoder, encoder, keras = build_autoencoder(
        input_dimension=X_normal.shape[1],
        latent_dimension=latent_dimension,
        learning_rate=learning_rate,
    )
    callbacks = make_callbacks(keras)
    X_fit_array = X_fit.to_numpy(dtype=np.float32, copy=False)
    X_validation_array = X_validation.to_numpy(dtype=np.float32, copy=False)

    started = time.perf_counter()
    history = autoencoder.fit(
        X_fit_array,
        X_fit_array,
        validation_data=(X_validation_array, X_validation_array),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - started

    return AutoencoderTrainingResult(
        autoencoder=autoencoder,
        encoder=encoder,
        keras=keras,
        history=history,
        normal_validation=X_validation.reset_index(drop=True),
        training_time_seconds=elapsed,
        fit_samples=len(X_fit),
        validation_samples=len(X_validation),
    )


def reconstruct_and_encode(
    autoencoder: Any,
    encoder: Any,
    X: pd.DataFrame | np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_array = (
        X.to_numpy(dtype=np.float32, copy=False)
        if hasattr(X, "to_numpy")
        else np.asarray(X, dtype=np.float32)
    )
    reconstructed = autoencoder.predict(
        X_array,
        batch_size=batch_size,
        verbose=0,
    )
    errors = np.mean(np.square(X_array - reconstructed), axis=1).astype(np.float32)
    latent = encoder.predict(
        X_array,
        batch_size=batch_size,
        verbose=0,
    ).astype(np.float32)
    return errors, latent


def augment_with_ae_features(
    autoencoder: Any,
    encoder: Any,
    X: pd.DataFrame | np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_array = (
        X.to_numpy(dtype=np.float32, copy=False)
        if hasattr(X, "to_numpy")
        else np.asarray(X, dtype=np.float32)
    )
    errors, latent = reconstruct_and_encode(
        autoencoder,
        encoder,
        X_array,
        batch_size,
    )
    augmented = np.column_stack(
        [X_array, errors.reshape(-1, 1), latent]
    ).astype(np.float32, copy=False)
    return augmented, errors, latent


def anomaly_threshold(normal_errors: np.ndarray, percentile: float) -> float:
    if not 0.0 < percentile <= 100.0:
        raise ValueError("Percentyl progu musi należeć do (0, 100].")
    return float(np.percentile(normal_errors, percentile))
