import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from exporter import BitMexExporter


class MPPModel:
    """Market Price Prediction Model"""

    def __init__(self, sym1, sym2, frec="1d", batch_size=1, epochs=1):
        self.batch_size = batch_size
        self.epochs = 1
        self.sym1 = sym1
        self.sym2 = sym2
        self.frec = frec if frec in ("1m", "5m", "1h", "1d") else "1d"
        self.model = None
        self.scaler = None
        self.training_data_len = None
        self.data = None
        self.train_ds = None
        self.test_ds = None
        self.predictions = None

    def get_dataset(self, test=True):
        data = BitMexExporter().export(self.sym1.upper() + self.sym2.upper(), self.frec, save=False)
        self.data = data.filter(["close"])
        dataset = self.data.values
        self.training_data_len = math.ceil(len(dataset) * .8)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(dataset)
        train_data = scaled_data[0:self.training_data_len, :]
        x_train = []
        y_train = []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60: i, 0])
            y_train.append(train_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], 60, 1))
        self.train_ds = (x_train, y_train)
        if test:
            test_data = scaled_data[self.training_data_len - 60:, :]
            x_test = []
            y_test = dataset[self.training_data_len:, :]
            for i in range(60, len(test_data)):
                x_test.append(test_data[i - 60: i, 0])
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], 60, 1))
            self.test_ds = (x_test, y_test)
            return x_train, y_train, x_test, y_test
        return x_train, y_train

    def evaluate(self):
        if not self.predictions:
            predictions = self.model.predict(self.test_ds[0])
            self.predictions = self.scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(self.predictions - self.test_ds[1]) ** 2)
        print(rmse)

    def build_and_compile(self):
        self.model = Sequential(
            [
                LSTM(50, return_sequences=True, input_shape=(self.train_ds[0].shape[1], 1)),
                LSTM(50, return_sequences=False),
                Dense(25),
                Dense(1)
            ])
        self.model.compile(optimizer="adam", loss="mean_squared_error")
        self.model.summary()

    def train(self):
        x_train, y_train, x_test, y_test = self.get_dataset()
        self.build_and_compile()
        self.model.fit(x_train, y_train, self.batch_size, self.epochs)

    def plot(self):
        if not self.predictions:
            predictions = self.model.predict(self.test_ds[0])
            self.predictions = self.scaler.inverse_transform(predictions)
        train = self.data[:self.training_data_len]
        valid = self.data[self.training_data_len:]
        valid["predictions"] = self.predictions
        plt.style.use("fivethirtyeight")
        plt.figure(figsize=(16, 8))
        plt.title("Model")
        plt.xlabel("Date", fontsize=18)
        plt.ylabel("Close Price %s (%s)" % (self.sym1, self.sym2))
        plt.plot(train["close"])
        plt.plot(valid[["close", "predictions"]])
        plt.legend(["Train", "Val", "Predictions"], loc="lower right")
        plt.show()
