#python trader.py --training mytraining.csv --testing mytesting.csv
#python profit_calculator.py mytesting.csv output.csv

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

WINDOW = 2
EPOCH = 100
PREDICT_DAYS = 19
TESTING = False
HOLDING_PRICE = 0

def make_data(data, window, step=1, rshift=0):
    x = []
    y = []
    for i in range(len(data) - step*window - rshift):
        row = [np.concatenate([r, [data[i+idx+1][0]/data[i+idx][0]]]) for idx, r in enumerate(data[i:i+step*window:step])]
        x.append(row)
        y.append([data[i+step*window+rshift][0]])
    return np.array(x), np.array(y)

def take_input(data, window):
    x = []
    start = len(data)-window
    for i in range(window):
        x.append(np.concatenate([data[start+i], [data[min(len(data)-1,start+i+1)][0]/data[start+i][0]]]))
    return np.array([x])

def predict_action(pred, CurrentStock, YesterPrice):
    if pred > YesterPrice:
        if CurrentStock == 1:
            if pred > HOLDING_PRICE: return -1, False
            else: return 0, False
        elif CurrentStock == 0:
            return 1, True
        else:
            if pred < HOLDING_PRICE: return 1, False
            else: return 0, False
    else:
        if CurrentStock == 1:
            if pred > HOLDING_PRICE: return -1, False
            else: return 0, False
        elif CurrentStock == 0:
            return -1, True
        else:
            if pred < HOLDING_PRICE: return 1, False
            else: return 0, False

def make_model():
    model = Sequential()
    model.add(InputLayer((WINDOW, 5)))
    model.add(LSTM(256))
    model.add(Dense(128, activation="leaky_relu"))
    model.add(Dense(64, activation="leaky_relu"))
    model.add(Dense(16, activation="leaky_relu"))
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

    YesterPrice = training_data[-1][0]
    CurrentStock = 0

    trader = make_model()

    trader.compile(
        loss = tf.keras.losses.MeanSquaredError(),
        optimizer = tf.keras.optimizers.Adam(),
        metrics = ["mse"]
    )

    cb = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=15, restore_best_weights=True)

    trader.fit(train_x, train_y, epochs = EPOCH, validation_split=0.2, callbacks=cb)

    testing_data = pd.read_csv(args.testing, header = None).to_numpy()

    count = 0
    prediction = []
    truth = []

    with open(args.output, "w") as output_file:
        for row in testing_data:
            if count == PREDICT_DAYS:
                break
            # Extract data to predict from training data
            test_x = take_input(training_data, WINDOW)

            # Predict
            pred = trader.predict(test_x).flatten()

            # Put new into train data
            training_data = np.concatenate([training_data, [row]], axis = 0)

            # Decide action and write
            if count < WINDOW: 
                action = 0
                updateHP = False
            else:
                action, updateHP = predict_action(pred, CurrentStock, YesterPrice)
            CurrentStock += action
            if updateHP:
                HOLDING_PRICE = row[0]
            output_file.write("{}{}".format(action, "\n" if count < PREDICT_DAYS-1 else ""))

            # Record
            YesterPrice = row[0]
            prediction.append(pred)
            truth.append(row[0])

            count += 1

    if TESTING:
        plt.plot(prediction, label = "Predition")
        plt.plot(truth, label = "Truth")
        plt.legend(loc = "best")
        plt.show()
        print(mean_squared_error(prediction, truth))