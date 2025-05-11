import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import os

from mlp_visualizer import ModelCollector, visualize_mlp


class CNN_large(nn.Module):
    def __init__(self):
        super(CNN_large, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.linear_block = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 7 * 7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.linear_block(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 3*3, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 24)
        self.fc4 = nn.Linear(24, 24)
        self.fc5 = nn.Linear(24, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 64 * 3*3)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, training=self.training)
        x = self.fc5(x)
        return x


def train_and_collect(batch_size=64, epochs=1,
                      lr=0.002, seed=1,
                      data_collection_interval=50, num_collections=-1, path="./collection.json",
                      model=None, neuron_cap=48):
    """
    Train a CNN model on MNIST and collect data at specified intervals.

    Args:
        batch_size: Training batch size
        epochs: Number of training epochs
        lr: Learning rate
        seed: Random seed
        data_collection_interval: Collect data every N batches
        num_collections: Number of data collections to make, -1 to collect until the end
    """
    if os.path.exists(path):
        print(f"File {path} already exists. Please remove it before running the script.")
        return
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training model on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'} with seed {seed}")

    # MNIST dataset transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1310,), (0.3085,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    test_iterator = iter(test_loader)

    # Create the CNN model
    model = CNN().to(device) if model is None else model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Create data directory if it doesn't exist
    os.makedirs('./data/collections', exist_ok=True)

    # Training loop
    collections_made = 0
    batch_count = 0
    collector = ModelCollector(model, neuron_cap)

    acc = Accuracy("multiclass", num_classes=10).to(device)

    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            batch_count += 1

            # Move data and target to the specified device
            data, target = data.to(device), target.to(device)

            # Regular training step
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            acc(output, target)

            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tEpoch Acc: {acc.compute():.6f}')

            # Collect data at specified intervals
            if batch_count % data_collection_interval == 0 and (collections_made < num_collections or num_collections == -1):
                model.eval()

                print(f"\nCollecting data (collection {collections_made + 1}{'/' + str(num_collections) if num_collections > 0 else ''})..."
                      f"at step {batch_idx + 1 + epoch * len(train_loader)}")
                sample_image, target = next(test_iterator)
                sample_image = sample_image.to(device)
                target = torch.tensor(target, dtype=torch.int64, device=device)
                output = collector(sample_image, input=sample_image[0])
                prediction = output.argmax(dim=1).item()
                collector.register_value("prediction", prediction)
                collector.register_value("logits", output[0])
                accuracy = acc.compute()
                collector.register_value("loss", loss.item())
                collector.register_value("accuracy", accuracy)
                print(f"Loss: {loss.item():.6f}, Accuracy: {accuracy:.2f}")

                model.train()
                collections_made += 1
        acc.reset()
    print(f'Epoch: {epochs}/{epochs} [{len(train_loader.dataset)}/{len(train_loader.dataset)} (100%)]')

    # Save the collected data
    json_path = path
    collector.dump_to_json(json_path)


def visualize_collected_data(json_path):
    print(f"Visualizing collected data from {json_path}...")
    visualize_mlp(json_path)


if __name__ == "__main__":
    # Train the model and collect data
    path = './data/collections/mnist_collection.json'
    train_and_collect(path=path, model=CNN(), data_collection_interval=50)
    visualize_collected_data(path)

    path = './data/collections/mnist_collection_largeCNN.json'
    train_and_collect(path=path, model=CNN_large(), data_collection_interval=20, neuron_cap=24)
    visualize_collected_data(path)
