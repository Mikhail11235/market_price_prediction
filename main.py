from model import MPPModel


if __name__ == "__main__":
    model = MPPModel("ETH", "USD", frec="1d", batch_size=1, epochs=3)
    model.train()
    model.plot()
