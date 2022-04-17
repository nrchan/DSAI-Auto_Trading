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

def make_data(data, window, step=1, rshift=0):
    x = []
    y = []
    for i in range(len(data) - step*window - rshift):
        row = [r for r in data[i:i+step*window:step]]
        x.append(row)
        y.append(data[i+step*window+rshift])
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

    testing_data = pd.read_csv(args.testing, header = None).to_numpy()

    count = 0
    prediction = []
    prediction2 = []
    truth = []

    with open(args.output, "w") as output_file:
        for row in testing_data:
            if count == PREDICT_DAYS:
                break
            # Extract data to predict from training data
            test_x = take_input(training_data, WINDOW)

            # Predict
            pred = trader.predict(test_x).flatten()

            """
            # Predict d+2
            training_data = np.concatenate([training_data, [pred]], axis = 0)
            test_x2 = take_input(training_data, WINDOW)
            pred2 = trader.predict(test_x2).flatten()
            training_data = training_data[:-1]
            """

            # Put new into train data
            training_data = np.concatenate([training_data, [row]], axis = 0)

            # Decide action and write
            action, CurrentStock = predict_action(pred[0], CurrentStock, CurrentPrice)
            output_file.write("{}{}".format(action, "\n" if count < PREDICT_DAYS-1 else ""))

            # Record
            CurrentPrice = row[0]
            prediction.append(pred)
            #prediction2.append(pred2)
            truth.append(row[0])

            #trader.re_training()

            count += 1

    if TESTING:
        opening = [p[0] for p in prediction]
        opening2 = [p[0] for p in prediction2]
        plt.plot(opening, label = "Predition")
        #plt.plot(list(range(1, PREDICT_DAYS)),opening2[:-1], label = "Predition D+2")
        plt.plot(truth, label = "Truth")
        plt.legend(loc = "best")
        plt.show()
        print(mean_squared_error(opening, truth))