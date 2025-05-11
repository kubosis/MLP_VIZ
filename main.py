import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from mlp_visualizer import ModelCollector, visualize_mlp


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=5)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=5)
        self.conv3 = nn.Conv2d(4, 8, kernel_size=5)
        self.fc1 = nn.Linear(3 * 3 * 8, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, 24)
        self.fc4 = nn.Linear(24, 24)
        self.fc5 = nn.Linear(24, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.dropout(x, p=0.5, training=self.training)
        x = x.view(-1, 3 * 3 * 8)
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


def train_and_collect(batch_size=64, test_batch_size=64, epochs=1,
                      lr=0.001, seed=1,
                      data_collection_interval=50, num_collections=5, path="./collection.json"):
    """
    Train a CNN model on MNIST and collect data at specified intervals.

    Args:
        batch_size: Training batch size
        test_batch_size: Testing batch size
        epochs: Number of training epochs
        lr: Learning rate
        seed: Random seed
        data_collection_interval: Collect data every N batches
        num_collections: Number of data collections to make, -1 to collect until the end
    """
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using seed: {seed}")

    # MNIST dataset transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

    # Create the CNN model
    model = CNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Create data directory if it doesn't exist
    os.makedirs('./data/collections', exist_ok=True)

    # Training loop
    collections_made = 0
    batch_count = 0
    collector = ModelCollector(model)
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

            # Print progress
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                      f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

            # Collect data at specified intervals
            if batch_count % data_collection_interval == 0 and (collections_made < num_collections or num_collections == -1):
                model.eval()

                print(f"\nCollecting data (collection {collections_made + 1}{'/' + str(num_collections) if num_collections > 0 else ''})..."
                      f"at step {batch_idx + 1}")
                sample_image, label = test_dataset[collections_made]
                sample_image = sample_image.to(device)
                test_data, test_target = next(iter(test_loader))
                test_data, test_target = test_data.to(device), test_target.to(device)

                output = collector(test_data, input=sample_image, label=label)
                loss = F.cross_entropy(output, test_target)
                loss.backward()

                model.train()
                collections_made += 1

    # Save the collected data
    json_path = path
    collector.dump_to_json(json_path)


def visualize_collected_data(json_path):
    print(f"Visualizing collected data from {json_path}...")
    visualize_mlp(json_path)


def train(path):
    print("Training model and collecting data...")
    train_and_collect(
        batch_size=64,
        epochs=1,
        data_collection_interval=50,
        num_collections=-1,
        path=path
    )


if __name__ == "__main__":
    # Train the model and collect data
    path = './data/collections/mnist_collection.json'
    train(path)
    visualize_collected_data(path)
