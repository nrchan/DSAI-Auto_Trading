import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

WINDOW = 30
EPOCH = 100
TESTING = True

def make_data(data, window):
    x = []
    y = []
    for i in range(len(data) - window):
        row = [r for r in data[i:i+window]]
        x.append(row)
        y.append(data[i+window][0]) #take only opening price
    return np.array(x), np.array(y)

def take_input(data, window):
    x = []
    start = len(data)-window
    for i in range(window):
        x.append(data[start+i])
    return np.array([x])

def make_model():
    model = Sequential()
    model.add(InputLayer((WINDOW, 4)))
    model.add(LSTM(128))
    model.add(Dense(64, activation="leaky_relu"))
    model.add(Dense(16, activation="leaky_relu"))
    model.add(Dense(8, activation="leaky_relu"))
    model.add(Dense(1))
    return model

if __name__ == "__main__":
    # You should not modify this part.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training", default="training_data.csv", help="input training data file name")
    parser.add_argument("--testing", default="testing_data.csv", help="input testing data file name")
    parser.add_argument("--output", default="output.csv", help="output file name")
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    training_data = pd.read_csv(args.training, header = None).to_numpy()
    train_x, train_y = make_data(training_data, WINDOW)

    trader = make_model()

    trader.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam(),
        metrics = ["mse"]
    )

    cb = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=15, restore_best_weights=True)

    trader.fit(train_x, train_y, epochs = EPOCH, validation_split=0.2, callbacks=cb)

    testing_data = pd.read_csv(args.testing, header = None).to_numpy()

    if TESTING:
        prediction = []
        truth = []

        for row in testing_data:
            test_x = take_input(training_data, WINDOW)
            pred = trader.predict(test_x).flatten().item()
            training_data = np.concatenate([training_data, [row]], axis = 0)
            prediction.append(pred)
            truth.append(row[0])

        plt.plot(prediction, label = "Predition")
        plt.plot(truth, label = "Truth")
        plt.legend(loc = "best")
        plt.show()
        print(mean_squared_error(prediction, truth))
    else:
        """
        with open(args.output, "w") as output_file:
            for row in testing_data:
                # We will perform your action as the open price in the next day.
                action = trader.predict_action(row)
                output_file.write(action)

                # this is your option, you can leave it empty.
                trader.re_training()
        """