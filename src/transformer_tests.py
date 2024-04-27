import tensorflow as tf
from utils import (
    get_datasets,
    waveform_to_spectrograms,
    waveform_to_log_mel_spectrogram,
    eval_and_save,
    get_callbacks,
)
from Transformer import Transformer


def main():
    # make_dataset("../data/train/audio")
    sample_rate = 16000
    frame_length = 255
    frame_step = 128
    num_mel_bins = 129
    ds_train_raw, ds_val_raw, ds_test_raw = get_datasets()
    model_type = "Transformer"

    # Model 1
    path = "Transformer1.keras"
    config = {
        "Spectrogram": "Normal",
        "Regularization": "Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
        "d_model": 256,
        "num_layers": 2,
        "num_heads": 2,
        "dropout_rate": 0.2,
    }

    ds_train = (
        ds_train_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_spectrograms(
                    x, frame_length=frame_length, frame_step=frame_step
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_val = (
        ds_val_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_spectrograms(
                    x, frame_length=frame_length, frame_step=frame_step
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_test = (
        ds_test_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_spectrograms(
                    x, frame_length=frame_length, frame_step=frame_step
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    model = Transformer(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        dff=4 * config["d_model"],
        block_size=62,
        dropout_rate=config["dropout_rate"],
        num_classes=30,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=100,
        callbacks=get_callbacks("../models/" + path),
    )

    eval_and_save(model_type, ds_test, config, history, path)

    # Model 2
    path = "Transformer2.keras"
    config = {
        "Spectrogram": "Normal",
        "Regularization": "Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 2,
        "dropout_rate": 0.2,
    }

    ds_train = (
        ds_train_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_spectrograms(
                    x, frame_length=frame_length, frame_step=frame_step
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_val = (
        ds_val_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_spectrograms(
                    x, frame_length=frame_length, frame_step=frame_step
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_test = (
        ds_test_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_spectrograms(
                    x, frame_length=frame_length, frame_step=frame_step
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    model = Transformer(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        dff=4 * config["d_model"],
        block_size=62,
        dropout_rate=config["dropout_rate"],
        num_classes=30,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=100,
        callbacks=get_callbacks("../models/" + path),
    )

    eval_and_save(model_type, ds_test, config, history, path)

    # Model 3
    path = "Transformer3.keras"
    config = {
        "Spectrogram": "Normal",
        "Regularization": "Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.0001,
        "Batch Size": 128,
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 4,
        "dropout_rate": 0.2,
    }

    ds_train = (
        ds_train_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_spectrograms(
                    x, frame_length=frame_length, frame_step=frame_step
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_val = (
        ds_val_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_spectrograms(
                    x, frame_length=frame_length, frame_step=frame_step
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_test = (
        ds_test_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_spectrograms(
                    x, frame_length=frame_length, frame_step=frame_step
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    model = Transformer(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        dff=4 * config["d_model"],
        block_size=62,
        dropout_rate=config["dropout_rate"],
        num_classes=30,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=100,
        callbacks=get_callbacks("../models/" + path),
    )

    eval_and_save(model_type, ds_test, config, history, path)

    # Model 4
    path = "Transformer4.keras"
    config = {
        "Spectrogram": "Normal",
        "Regularization": "Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.0001,
        "Batch Size": 128,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 4,
        "dropout_rate": 0.2,
    }

    ds_train = (
        ds_train_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_spectrograms(
                    x, frame_length=frame_length, frame_step=frame_step
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_val = (
        ds_val_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_spectrograms(
                    x, frame_length=frame_length, frame_step=frame_step
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_test = (
        ds_test_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_spectrograms(
                    x, frame_length=frame_length, frame_step=frame_step
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    model = Transformer(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        dff=4 * config["d_model"],
        block_size=62,
        dropout_rate=config["dropout_rate"],
        num_classes=30,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=100,
        callbacks=get_callbacks("../models/" + path),
    )

    eval_and_save(model_type, ds_test, config, history, path)

    # Model 5
    path = "Transformer5.keras"
    config = {
        "Spectrogram": "Log-Mel",
        "Regularization": "Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
        "d_model": 256,
        "num_layers": 2,
        "num_heads": 2,
        "dropout_rate": 0.2,
    }

    ds_train = (
        ds_train_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_log_mel_spectrogram(
                    x,
                    sample_rate=sample_rate,
                    frame_length=frame_length,
                    fram_step=frame_step,
                    num_mel_bins=num_mel_bins,
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_val = (
        ds_val_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_log_mel_spectrogram(
                    x,
                    sample_rate=sample_rate,
                    frame_length=frame_length,
                    fram_step=frame_step,
                    num_mel_bins=num_mel_bins,
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_test = (
        ds_test_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_log_mel_spectrogram(
                    x,
                    sample_rate=sample_rate,
                    frame_length=frame_length,
                    fram_step=frame_step,
                    num_mel_bins=num_mel_bins,
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    model = Transformer(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        dff=4 * config["d_model"],
        block_size=62,
        dropout_rate=config["dropout_rate"],
        num_classes=30,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=100,
        callbacks=get_callbacks("../models/" + path),
    )

    eval_and_save(model_type, ds_test, config, history, path)

    # Model 6
    path = "Transformer6.keras"
    config = {
        "Spectrogram": "Log-Mel",
        "Regularization": "Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.001,
        "Batch Size": 256,
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 2,
        "dropout_rate": 0.2,
    }

    ds_train = (
        ds_train_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_log_mel_spectrogram(
                    x,
                    sample_rate=sample_rate,
                    frame_length=frame_length,
                    fram_step=frame_step,
                    num_mel_bins=num_mel_bins,
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_val = (
        ds_val_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_log_mel_spectrogram(
                    x,
                    sample_rate=sample_rate,
                    frame_length=frame_length,
                    fram_step=frame_step,
                    num_mel_bins=num_mel_bins,
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_test = (
        ds_test_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_log_mel_spectrogram(
                    x,
                    sample_rate=sample_rate,
                    frame_length=frame_length,
                    fram_step=frame_step,
                    num_mel_bins=num_mel_bins,
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    model = Transformer(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        dff=4 * config["d_model"],
        block_size=62,
        dropout_rate=config["dropout_rate"],
        num_classes=30,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=100,
        callbacks=get_callbacks("../models/" + path),
    )

    eval_and_save(model_type, ds_test, config, history, path)

    # Model 7
    path = "Transformer7.keras"
    config = {
        "Spectrogram": "Log-Mel",
        "Regularization": "Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.0001,
        "Batch Size": 128,
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 4,
        "dropout_rate": 0.2,
    }

    ds_train = (
        ds_train_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_log_mel_spectrogram(
                    x,
                    sample_rate=sample_rate,
                    frame_length=frame_length,
                    fram_step=frame_step,
                    num_mel_bins=num_mel_bins,
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_val = (
        ds_val_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_log_mel_spectrogram(
                    x,
                    sample_rate=sample_rate,
                    frame_length=frame_length,
                    fram_step=frame_step,
                    num_mel_bins=num_mel_bins,
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_test = (
        ds_test_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_log_mel_spectrogram(
                    x,
                    sample_rate=sample_rate,
                    frame_length=frame_length,
                    fram_step=frame_step,
                    num_mel_bins=num_mel_bins,
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    model = Transformer(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        dff=4 * config["d_model"],
        block_size=62,
        dropout_rate=config["dropout_rate"],
        num_classes=30,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=100,
        callbacks=get_callbacks("../models/" + path),
    )

    eval_and_save(model_type, ds_test, config, history, path)

    # Model 8
    path = "Transformer8.keras"
    config = {
        "Spectrogram": "Log-Mel",
        "Regularization": "Dropout",
        "Optimizer": "Adam",
        "Learning Rate": 0.0001,
        "Batch Size": 128,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 4,
        "dropout_rate": 0.2,
    }

    ds_train = (
        ds_train_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_log_mel_spectrogram(
                    x,
                    sample_rate=sample_rate,
                    frame_length=frame_length,
                    fram_step=frame_step,
                    num_mel_bins=num_mel_bins,
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_val = (
        ds_val_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_log_mel_spectrogram(
                    x,
                    sample_rate=sample_rate,
                    frame_length=frame_length,
                    fram_step=frame_step,
                    num_mel_bins=num_mel_bins,
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    ds_test = (
        ds_test_raw.batch(config["Batch Size"])
        .map(
            lambda x, y: (
                waveform_to_log_mel_spectrogram(
                    x,
                    sample_rate=sample_rate,
                    frame_length=frame_length,
                    fram_step=frame_step,
                    num_mel_bins=num_mel_bins,
                ),
                y,
            )
        )
        .cache()
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    model = Transformer(
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        dff=4 * config["d_model"],
        block_size=62,
        dropout_rate=config["dropout_rate"],
        num_classes=30,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["Learning Rate"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=100,
        callbacks=get_callbacks("../models/" + path),
    )

    eval_and_save(model_type, ds_test, config, history, path)


if __name__ == "__main__":
    main()
