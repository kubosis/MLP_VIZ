# ModelCollector JSON Documentation

## Overview
The ModelCollector generates a JSON that tracks PyTorch (2.6) neural network architecture and execution data during forward and backward passes.

## Purpose

The primary goal is to extract intermediate activations, parameters (weights, biases), and gradients from specified layers of a neural network during training or inference. This data is structured and saved into a JSON file, which serves as the input for the `MLPVisualizer` frontend.

## Key Features

*   **PyTorch Hook Integration:** Uses `register_forward_hook` and `register_full_backward_hook` to non-intrusively capture data.
*   **Selective Layer Capture:** Focuses data collection on specific layer types (currently `nn.Linear`, `nn.Identity` by default, extensible via `_captured_instances`).
*   **Architecture Parsing:** Automatically extracts basic layer information (type, input/output features) into the JSON output.
*   **Data Reduction/Capping:**
    *   Limits the reported dimensions of large layers in the architecture description based on the `cap_visualized_layer_size` parameter.
    *   Reduces the dimensionality of captured tensors (activations, weights, gradients) using averaging/pooling (`_reduce_to_max_len`) to keep visualization feasible and data files manageable.
*   **Custom Value Registration:** Allows logging arbitrary data (e.g., loss, accuracy, labels, predictions, logits) from the training script into the collected data using `collector.register_value()`.
*   **Structured JSON Output:** Organizes data with a static `"architecture"` section and numbered sections for each collected pass.

## Usage

```python
import torch
import torch.nn as nn
from _collector_module import ModelCollector # Assuming it's importable

# 1. Define your PyTorch model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# 2. Instantiate the collector, wrapping your model
# cap_visualized_layer_size limits visualized neurons per layer (e.g., 48)
collector = ModelCollector(model, cap_visualized_layer_size=48)

# 3. Run data through the collector (instead of the raw model)
#    Pass any extra data to be logged for this step via kwargs.
dummy_input = torch.randn(1, 784)
sample_image_tensor = torch.randn(1, 28, 28) # Example input image data

output = collector(dummy_input, input=sample_image_tensor) # 'input' kwarg is captured

# 4. Register custom values (optional)
collector.register_value("loss", 0.5)
collector.register_value("accuracy", 0.95)
collector.register_value("label", "Digit 7")
collector.register_value("prediction", 7)
collector.register_value("logits", output) # Log the raw output tensor

# 5. Perform backward pass if needed (gradients will be captured if enabled)
# loss = criterion(output, target)
# loss.backward()

# 6. Save the collected data
collector.dump_to_json("./data/collections/my_model_data.json")

# 7. Pretty print to console (optional)
# collector.pretty_print()

## JSON Structure

```json
{
    "architecture": { /* model structure */ },
    "1": { /* first pass data */ },
    "2": { /* second pass data (if applicable) */ },
    ...
}
```

## Architecture Section
Contains static information about each layer, identified by `"{index}_{ModuleType}"`:

```json
"architecture": {
    "1_Linear": {
        "in_features": 5.0,
        "out_features": 5.0,
        "bias": true
    },
    "2_ReLU": {},
    "3_Linear": { /* ... */ }
}
```

## Pass Data Sections
Each numbered section contains runtime data for a forward/backward pass:

```json
"1": {
    "1_Linear": {
        "input": [/* input_features -- mean over the batch */],
        "weight": [/* output_features × input_features */],
        "bias": [/* output_features */],
        "grad_input": [/* input_features -- mean over the batch */],
        "grad_weight": [/* gradients for weights */],
        "grad_bias": [/* gradients for bias */]
    },
    "2_ReLU": {
        "input": [/* ... */],
        "grad_output": [/* ... */]
    }
}
```

### Fields by Module Type
- **All modules**: `input`, `grad_input`
- **Parameterized modules** (e.g., Linear): 
  - Parameters: `weight`, `bias`
  - Gradients: `grad_weight`, `grad_bias`