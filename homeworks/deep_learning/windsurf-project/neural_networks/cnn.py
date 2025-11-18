import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.head = nn.Linear(16 * 28 * 28, 10)
        self.device = device

    def forward(self, x):
        x = self.base(x)
        x = x.view(x.size(0), -1)
        return self.head(x)

    def train_epoch(self, dataloader, loss_fn, optimizer):
        size = len(dataloader.dataset)
        self.train()
        total_loss = 0.0
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)
            pred = self(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                current = (batch_idx + 1) * len(X)
                print(f"  loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
        return total_loss / len(dataloader)

    def test_epoch(self, dataloader, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.eval()
        test_loss, correct = 0.0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        accuracy = correct / size
        print(f"  test accuracy: {(100*accuracy):>0.2f}%, Avg loss: {test_loss:>8f}")
        return accuracy