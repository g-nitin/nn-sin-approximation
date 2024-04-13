import time
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor, rand, sin
from math import pi
from sklearn.model_selection import train_test_split

from sin_network import train_nnet, evaluate_nnet
from utilities import plot_prediction


def main():
    n: int = 100_000
    print(f"Creating data of size {n}\n")
    x: Tensor = (2 * pi - 0) * rand(n)
    y: Tensor = sin(x)

    # Split data and labels into training and testing sets
    test_size: float = 0.2
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=13)

    # Create TensorDatasets for training and testing data
    train_dataset: TensorDataset = TensorDataset(x_train, y_train)
    test_dataset: TensorDataset = TensorDataset(x_test, y_test)

    # Create DataLoaders for training and testing
    batch_size: int = 1
    train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    start_time = time.time()

    nnet: Module = train_nnet(train_loader, batch_size)

    average_loss: float = evaluate_nnet(nnet, test_loader)

    print('\nAverage Test Loss: {:.4f}'.format(average_loss))
    print('Time', time.time() - start_time)

    plot_prediction(nnet, training_size=n)


if __name__ == "__main__":
    main()
