# MLP Activation & Weight Visualizer

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive tool built with PyTorch and PyQt6 to visualize the internal dynamics (activations, weights) of Multi-Layer Perceptron (MLP) classification heads during training.

Check out the full project documentation here: [PDF](https://github.com/user-attachments/files/20417235/Visualization.of.MLP.-.Kolar.Sukdol.pdf)


![UI](https://github.com/user-attachments/assets/204d5e52-9d18-4953-9274-4c7ccebc72b3)
*Figure 1: The MLP Visualizer interface showing network state, input, predictions, and metrics.*

## Usage
```
usage: main.py [-h] [--dataset {mnist,cifar10,fashion_mnist}] [--model {cnn,cnn_large,cifar10_cnn}] [--epochs EPOCHS] [--interval INTERVAL] [--neuron_cap NEURON_CAP] [--output_path OUTPUT_PATH] [--seed SEED] [--force]

Train and visualize neural network data.

options:
  -h, --help            show this help message and exit
  --dataset {mnist,cifar10,fashion_mnist}
                        Choose the dataset to use.
  --model {cnn,cnn_large,cifar10_cnn}
                        Choose the model architecture to use.
  --epochs EPOCHS       Number of training epochs.
  --interval INTERVAL   Data collection interval during training.
  --neuron_cap NEURON_CAP
                        Optional neuron cap for the model.
  --output_path OUTPUT_PATH
                        Optional path to save the collected data.
  --seed SEED           Random seed for reproducibility.
  --force               Force overwrite of existing output file.

```

## Features

*   **Data Collection Backend:** Integrates with PyTorch models using hooks to capture activations, weights, (and optionally gradients) from Linear layers. (`_collector_module.py`, see its [ReadME here](https://github.com/kubosis/VIZ_project/tree/main/mlp_visualizer/torch_collector))
*   **Data Reduction:** Intelligently caps and reduces data from large layers to keep visualization feasible.
*   **Interactive Frontend:** Visualizes the collected data using PyQt6 and pyqtgraph. (`_visualizer.py`, see its [ReadME here](https://github.com/kubosis/VIZ_project/tree/main/mlp_visualizer/visualization))
*   **Dynamic Network View:**
    *   Neurons colored by activation value (diverging blue-white-red scale).
    *   Connections colored by weight sign (red/blue) and sized by magnitude.
    *   Interactive hover effect highlights neurons and their pathways while dimming sibling connections.
    *   Zoom and Pan functionality.
*   **Pass Navigation:** Step through training intervals manually (slider) or automatically (play/pause/stop animation with adjustable delay).
*   **Contextual Information:**
    *   Displays the input image and true label for the current pass.
    *   Shows a histogram of output probabilities and the predicted class.
    *   Plots training Loss and Accuracy over time.
    *   Summarizes the visualized architecture.
*   **Customizable Training:** `main.py` script allows training different models (CNN, CNN_large, Cifar10CnnModel) on different datasets (MNIST, Fashion-MNIST, CIFAR-10) via command-line arguments.
*   **Dark Mode Theme:** Uses `qdarktheme` for a pleasant UI experience.

## Motivation

Neural networks are often treated as "black boxes". This tool aims to provide insights into the learning process of the MLP classification heads commonly used in deep learning models, helping users understand:
*   How neuron activations evolve.
*   Which connections (weights) become important.
*   How the network state relates to predictions and performance metrics.
*   Potential issues like dead neurons or learning dynamics.

## Project Structure
```
.
├── _collector_module.py # PyTorch data collection logic
├── _visualizer.py # PyQt6 visualization GUI, see its ReadME here
├── main.py # Main script for training and launching visualization
├── models.py # PyTorch model definitions
├── requirements.txt # Python dependencies
├── resources/ # Optional: icons, etc.
├── data/collections/ # Default directory for saved JSON data
└── README.md # This file
```


## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://your-repository-url/mlp-visualizer.git
    cd mlp-visualizer
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Install PyTorch 2.6 based on your installed CUDA version or use CPU version:**
   ```bash
   # OSX
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
   # ROCM 6.1 (Linux only)
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.1
   # ROCM 6.2.4 (Linux only)
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
   # CUDA 11.8
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
   # CUDA 12.4
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
   # CUDA 12.6
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
   # CPU only
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
   ```

## Usage

The primary way to use the tool is via `main.py`.

1.  **See available options:**
    ```bash
    python main.py --help
    ```

2.  **Train a model and visualize:**
    *   Train the default CNN on MNIST for 1 epoch, collecting data every 50 steps, capping visualized layers at 24 neurons, save to default path, then visualize:
        ```bash
        python main.py
        ```
    *   Train the large CNN on Fashion-MNIST for 5 epochs, collect every 100 steps, cap at 48 neurons, specify output path, overwrite if exists:
        ```bash
        python main.py --dataset fashion_mnist --model cnn_large --epochs 5 --interval 50 --neuron_cap 24 --output_path ./data/collections/fmnist_large_run.json --force
        ```
    *   Train the CIFAR-10 model:
        ```bash
        python main.py --dataset cifar10 --model cifar10_cnn --epochs 10 --interval 50 --neuron_cap 32
        ```

## How it Works

1.  **Collection:** `main.py` sets up the chosen model and dataset. It wraps the model with `ModelCollector`. During training, at specified intervals, a forward pass is run through the `collector` on a sample input. Hooks capture activations and weights. Custom values (loss, accuracy, label, prediction, logits, input image) are registered. The data is saved to JSON.
2.  **Visualization:** `_visualizer.py` (or `main.py` launching it) loads the JSON. It builds the UI using PyQt6. For the selected pass, it renders the network graph in a `QGraphicsScene`, plotting neurons and connections based on the data. Interactivity allows exploring different passes and network details. `pyqtgraph` handles the metric plots.

## Limitations

*   **Visualization Scalability:** Very wide MLP layers might still appear cluttered despite data capping.
*   **Gradient Visualization:** Currently visualizes weights; visualizing gradients is planned future work.
*   **Performance:** Loading/parsing huge JSON files (many passes) might take time.

## Future Work

*   Implement gradient visualization mode.
*   Add options for visualizing different parameters (e.g., bias).
*   Explore more advanced graph layout or abstraction techniques for large layers.
*   Potentially add comparative visualization features (comparing two passes or models).

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details (if applicable).
