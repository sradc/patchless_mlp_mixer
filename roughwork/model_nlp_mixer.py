"""This is still a work in progress.

It's a novel, simplified language model, based on MLP-Mixer [1].

I'm surprised at how well it seems to be doing,
despite the fact that it's working with character unigrams,
and the simplicity of the architecture.

So far I only have the training set up, not inference.

TODO:
- inference
- improve logging
- model versioning
- parameter search
- try it on larger dataset
- try it on nlp tasks, e.g. filling in missing words

[1] https://arxiv.org/abs/2105.01601

Run of best performing model configuration:
Epoch 1/10
2883/2883 [==============================] - 154s 53ms/step - loss: 0.1274 - accuracy: 0.1305 - precision: 0.7510 - recall: 0.4292 - f1: 0.5395 - val_loss: 0.0441 - val_accuracy: 0.2054 - val_precision: 0.8112 - val_recall: 0.6409 - val_f1: 0.7161
Epoch 2/10
2883/2883 [==============================] - 160s 55ms/step - loss: 0.0501 - accuracy: 0.2164 - precision: 0.8982 - recall: 0.7900 - f1: 0.8406 - val_loss: 0.0473 - val_accuracy: 0.2208 - val_precision: 0.7818 - val_recall: 0.6816 - val_f1: 0.7283
Epoch 3/10
2883/2883 [==============================] - 162s 56ms/step - loss: 0.0362 - accuracy: 0.2295 - precision: 0.9236 - recall: 0.8535 - f1: 0.8872 - val_loss: 0.0476 - val_accuracy: 0.1858 - val_precision: 0.7640 - val_recall: 0.7128 - val_f1: 0.7376
Epoch 4/10
2883/2883 [==============================] - 163s 57ms/step - loss: 0.0289 - accuracy: 0.2387 - precision: 0.9373 - recall: 0.8845 - f1: 0.9102 - val_loss: 0.0481 - val_accuracy: 0.1966 - val_precision: 0.7981 - val_recall: 0.7037 - val_f1: 0.7479
Epoch 5/10
2883/2883 [==============================] - 170s 59ms/step - loss: 0.0237 - accuracy: 0.2439 - precision: 0.9478 - recall: 0.9074 - f1: 0.9271 - val_loss: 0.0539 - val_accuracy: 0.2071 - val_precision: 0.8440 - val_recall: 0.6697 - val_f1: 0.7468
Epoch 6/10
2883/2883 [==============================] - 165s 57ms/step - loss: 0.0197 - accuracy: 0.2510 - precision: 0.9550 - recall: 0.9237 - f1: 0.9391 - val_loss: 0.0561 - val_accuracy: 0.2253 - val_precision: 0.7673 - val_recall: 0.7087 - val_f1: 0.7368
Epoch 7/10
2883/2883 [==============================] - 166s 57ms/step - loss: 0.0172 - accuracy: 0.2525 - precision: 0.9604 - recall: 0.9333 - f1: 0.9467 - val_loss: 0.0592 - val_accuracy: 0.2033 - val_precision: 0.8433 - val_recall: 0.6643 - val_f1: 0.7431


val_precision: 0.8433 - val_recall: 0.6643 - val_f1: 0.7431

Does better than baseline model. Is more precise, but lower f1 than en_core_web_sm.

Can potentially be improved, have not done a thorough hyperparameter exploration,
or tried pre-training.

"""
from datetime import datetime
import pathlib
from sys import path
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from prep_data import load_CoNLL_2003, load_newline_split_text
from keras_metrics import Precision, Recall, F1Score

MODEL_DIR = pathlib.Path("model")
MODEL_DIR.mkdir(exist_ok=True)

N_WORDS = 32  # number of words to consider at a time


def process_data(X, y, no_repeat=False):
    "Such that each sample contains multiple words."
    X = sliding_window_view(X, (N_WORDS, 1), writeable=False)
    X = X.squeeze(-1).transpose(0, 2, 1)  # [batch, word, char]
    y = sliding_window_view(y, (N_WORDS), writeable=False)
    if no_repeat:
        X = X[::N_WORDS, ...]
        y = y[::N_WORDS, ...]
    return X, y


def train_model(train_data_path: str, val_data_path: str):
    "Train the model, using text files as data source."

    # Prepare the data:
    train_data = load_CoNLL_2003(train_data_path)#.iloc[:1000, :]
    y_true = (train_data["entity"] == "I-PER").astype(np.float32)

    vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(1, 1))
    ngram_matrix = vectorizer.fit_transform(train_data["word"]).todense()

    X_train, y_train = process_data(ngram_matrix, y_true)
    X_train = X_train * (np.random.random(X_train.shape) < 0.8)  # add noise to input

    val_data = load_CoNLL_2003(val_data_path)  # .iloc[:1000, :]  # smaller for testing
    y_true_val = (val_data["entity"] == "I-PER").astype(np.float32)
    X_val, y_val = process_data(
        vectorizer.transform(val_data["word"]).todense(), y_true_val
    )

    # Train the model:
    model = make_model(N_WORDS, ngram_matrix.shape[1])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            "accuracy",
            Precision(from_logits=True),
            Recall(from_logits=True),
            F1Score(from_logits=True),
        ],
    )
    timestamp = datetime.now().strftime("%d-%m-%Y_%H%M%S")
    history = model.fit(
        X_train,  # full output
        y_train,
        batch_size=64,
        epochs=10,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_f1",
                mode="max",
                patience=3,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(
                    MODEL_DIR / (f"{timestamp}_" + "model.{epoch:02d}-{val_f1:.2f}.h5")
                )
            ),
            tf.keras.callbacks.CSVLogger(
                f"model/log_{timestamp}.csv", append=True, separator=";"
            ),
        ],
    )
    plt.plot(history.history["precision"], label="precision")
    plt.plot(history.history["val_precision"], label="val_precision")
    plt.plot(history.history["f1"], label="f1")
    plt.plot(history.history["val_f1"], label="val_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.savefig(f"model/{timestamp}_training_metrics.png")


def make_model(n_words, n_ngrams):
    "Create MLP-Mixer (but simpler) style model."

    x_in = tf.keras.layers.Input([n_words, n_ngrams])
    h = x_in

    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(512, tf.nn.gelu)(h)
    h = tf.keras.layers.Dense(1024, tf.nn.gelu)(h)
    h = tf.keras.layers.Dense(n_ngrams)(h)

    h = tf.keras.layers.Permute([2, 1])(h)

    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.Dense(512, tf.nn.gelu)(h)
    h = tf.keras.layers.Dense(1024, tf.nn.gelu)(h)
    h = tf.keras.layers.Dense(N_WORDS)(h)

    h = tf.keras.layers.Permute([2, 1])(h)

    # To classify:
    h = tf.keras.layers.Dense(1)(h)
    h = tf.keras.layers.Reshape([N_WORDS])(h)

    model = tf.keras.Model(x_in, h)
    print(model.summary())
    return model


if __name__ == '__main__':
    train_model("data/train.txt", "data/validate.txt")
