import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import os

from mlp_visualizer import ModelCollector, visualize_mlp
from models import *

import argparse

parser = argparse.ArgumentParser(description="Train and visualize neural network data.")
parser.add_argument(
    "--dataset",
    choices=["mnist", "cifar10"],
    default="mnist",
    help="Choose the dataset to use.",
)
parser.add_argument(
    "--model",
    choices=["cnn", "cnn_large", "cifar10_cnn"],
    default="cnn",
    help="Choose the model architecture to use.",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=1,
    help="Number of training epochs.",
)
parser.add_argument(
    "--interval",
    type=int,
    default=50,
    help="Data collection interval during training.",
)
parser.add_argument(
    "--neuron_cap",
    type=int,
    default=24,
    help="Optional neuron cap for the model.",
)
parser.add_argument(
    "--output_path",
    type=str,
    default=None,
    help="Optional path to save the collected data.",
)

def train_and_collect(train_dataset, test_dataset, batch_size=64, epochs=1,
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
    avg_loss = 0
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

            avg_loss += loss.item()

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
                prediction_label = test_dataset.classes[prediction]
                collector.register_value("prediction", prediction)
                collector.register_value("label", prediction_label)
                collector.register_value("logits", output[0])
                accuracy = acc.compute()
                collector.register_value("loss", avg_loss / batch_idx + 1 + epoch * len(train_loader))
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


def train(path, train, test, model, epochs=1, batch_size=32, data_collection_interval=50, cap=24, num_collections=-1):
    print("Training model and collecting data...")
    train_and_collect(
        train,
        test,
        batch_size=batch_size,
        epochs=epochs,
        data_collection_interval=data_collection_interval,
        num_collections=num_collections,
        path=path,
        model=model,
        cap=cap,
    )


def get_cifar():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # random crop with padding
        transforms.RandomHorizontalFlip(),  # random horizontal flip
        transforms.ToTensor(),  # convert to tensor and scale to [0, 1]
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # mean
                             (0.2023, 0.1994, 0.2010))  # std
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    return train, test


def get_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1310,), (0.3085,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    return train_dataset, test_dataset


if __name__ == "__main__":
    args = parser.parse_args()

    dataset = args.dataset
    model_name = args.model
    epochs = args.epochs
    interval = args.interval
    neuron_cap = args.neuron_cap
    output_path = args.output_path

    print(f"Selected Dataset: {dataset}")
    print(f"Selected Model: {model_name}")
    print(f"Epochs: {epochs}")
    print(f"Data Collection Interval: {interval}")
    print(f"Neuron Cap: {neuron_cap}")

    # Load dataset
    if dataset == "mnist":
        train_dataset, test_dataset = get_mnist()
        default_output_path = './data/collections/mnist_collection.json'
    elif dataset == "cifar10":
        train_dataset, test_dataset = get_cifar()
        default_output_path = './data/collections/cifar10_collection.json'
    else:
        print(f"Error: Unknown dataset '{dataset}'")
        sys.exit(1)

    # Instantiate model
    if model_name == "cnn":
        model = CNN()
        current_output_path = output_path if output_path else default_output_path
    elif model_name == "cnn_large":
        model = CNN_large()
        current_output_path = output_path if output_path else './data/collections/mnist_collection_largeCNN.json'
    elif model_name == "cifar10_cnn":
        model = Cifar10CnnModel()
        current_output_path = output_path if output_path else default_output_path
    else:
        print(f"Error: Unknown model '{model_name}'")
        sys.exit(1)

    # Train and collect data
    if not os.path.exists(current_output_path):
        train_and_collect(path=current_output_path, train_dataset=train_dataset,test_dataset=test_dataset,model=model,
                          epochs=epochs, data_collection_interval=interval, neuron_cap=neuron_cap)

    # Visualize collected data
    visualize_collected_data(current_output_path)
