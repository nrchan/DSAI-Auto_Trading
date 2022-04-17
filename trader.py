import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

WINDOW = 30
EPOCH = 100
PREDICT_DAYS = 19
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

def predict_action(pred, CurrentStock, CurrentPrice):
    if pred > CurrentPrice:
        if CurrentStock == 1:
            return 0, 1
        elif CurrentStock == 0:
            return 1, 1
        else:
            return 1, 0
    else:
        if CurrentStock == 1:
            return -1, 0
        elif CurrentStock == 0:
            return -1, -1
        else:
            return 0, -1

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

    CurrentPrice = training_data[-1][0]
    CurrentStock = 0

    trader = make_model()

    trader.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam(),
        metrics = ["mse"]
    )

    cb = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=15, restore_best_weights=True)

    trader.fit(train_x, train_y, epochs = EPOCH, validation_split=0.2, callbacks=cb)
"""
    testing_data = pd.read_csv(args.testing, header = None).to_numpy()

    count = 0
    prediction = []
    truth = []

    with open(args.output, "w") as output_file:
        for row in testing_data:
            if count == PREDICT_DAYS:
                break
            # We will perform your action as the open price in the next day.
            test_x = take_input(training_data, WINDOW)
            pred = trader.predict(test_x).flatten().item()
            training_data = np.concatenate([training_data, [row]], axis = 0)
            action, CurrentStock = predict_action(pred, CurrentStock, CurrentPrice)
            output_file.write("{}{}".format(action, "\n" if count < PREDICT_DAYS-1 else ""))
            CurrentPrice = row[0]
            prediction.append(pred)
            truth.append(row[0])

            # this is your option, you can leave it empty.
            #trader.re_training()

            count += 1

    if TESTING:
        plt.plot(prediction, label = "Predition")
        plt.plot(truth, label = "Truth")
        plt.legend(loc = "best")
        plt.show()
        print(mean_squared_error(prediction, truth))
"""