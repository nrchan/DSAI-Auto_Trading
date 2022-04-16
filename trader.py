import pandas as pd
import tensorflow as tf
import numpy as np

WINDOW = 30

def make_data(data, window):
    x = []
    y = []
    for i in range(len(data) - window):
        row = [[r] for r in data[i:i+window]]
        x.append(row)
        y.append(data[i+window][0]) #take only opening price
    return np.array(x), np.array(y)

def take_input(data, window):
    x = []
    start = len(data)-window
    for i in range(window):
        x.append([data[start+i]])
    return np.array(x)


def make_model():
    model = tf.keras.layers.Sequential()
    model.add(tf.keras.layers.InputLayer((4,1)))
    model.add(tf.keras.layers.LSTM(64))
    model.add(tf.keras.layers.Dense(1))
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
    print(take_input(training_data, 3))

    """
    with open(args.output, "w") as output_file:
        for row in testing_data:
            # We will perform your action as the open price in the next day.
            action = trader.predict_action(row)
            output_file.write(action)

            # this is your option, you can leave it empty.
            trader.re_training()
    """
