import torch.nn as nn
from torch.optim import Adam
from torch import Tensor, no_grad, Size
from torchsummary import summary


class sinNet(nn.Module):
    def __init__(self, input_dims: int, hidden_dims: int, output_dims: int):
        super().__init__()

        self.lin1 = nn.Linear(input_dims, hidden_dims)
        self.lin2 = nn.Linear(hidden_dims, hidden_dims * 2)
        self.lin3 = nn.Linear(hidden_dims * 2, output_dims)

    def forward(self, x):
        x = self.lin1(x)
        x = nn.functional.relu(x)

        x = self.lin2(x)
        x = nn.functional.relu(x)

        x = self.lin3(x)

        return x


def train_nnet(train_loader, batch_size: int) -> nn.Module:
    nnet: sinNet = sinNet(batch_size, batch_size * 500, batch_size)
    # summary(nnet, input_size=Size([1]))

    nnet.train()

    lr: float = 0.001
    criterion = nn.MSELoss()
    optimizer = Adam(nnet.parameters(), lr=lr)

    num_iters: int = 1
    for epoch in range(num_iters):
        running_loss: float = 0.0

        for i, data in enumerate(train_loader):
            inputs, labels = data

            optimizer.zero_grad()  # Zero the parameter gradients

            nnet_outputs = nnet(inputs)

            loss = criterion(nnet_outputs, labels)

            loss.backward()  # Back-propagation
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            loss_every = 20
            if i % loss_every == loss_every - 1:  # print after every `loss_every` mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / loss_every:.3f}')
                running_loss = 0.0

    return nnet


def evaluate_nnet(nnet: nn.Module, test_loader) -> float:
    nnet.eval()
    criterion = nn.MSELoss()

    with no_grad():
        # Initialize variables for tracking
        total_loss: float = 0.0
        total_samples: int = 0

        # Loop through test data loader
        for data, target in test_loader:
            # Get predictions from the model
            outputs: Tensor = nnet(data)

            loss = criterion(outputs, target)

            # Update tracking variables
            total_loss += loss.item() * target.size(0)  # Consider batch size
            total_samples += target.size(0)

    # Return average test loss
    return total_loss / total_samples
