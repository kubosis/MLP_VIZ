from typing import Any, Tuple, Dict

import torch
import torch.nn as nn


class ModelCollector(nn.Module):
    def __init__(self, model: nn.Module):
        super(ModelCollector, self).__init__()
        self._model = model
        self._state_dict: Dict[str, Any] = {}  # Store weights and gradients
        self._pass_no = 0

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Calls forward pass of the wrapped model,
        collects weights and gradients for later visualization.
        """
        self._pass_no += 1
        return CollectorFunction.apply(self, *args, **kwargs)

    def get_collected_data(self):
        """
        Returns the collected weights and gradients.
        """
        return self._state_dict

class CollectorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model: ModelCollector, *args, **kwargs):
        """
        Custom autograd function to collect forward weights.
        """
        ctx.model = model._model  # Save model reference for backward pass
        ctx.state_dict = model._state_dict
        ctx.pass_no = model._pass_no
        ctx.save_for_backward(*args)  # Save inputs for later

        # Forward pass through the model
        output = model(*args, **kwargs)

        # Collect and store weights of Linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                model._state_dict[name] = {
                    "weights": module.weight.detach().clone(),
                    "bias": module.bias.detach().clone() if module.bias is not None else None,
                }

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Custom backward function to collect gradients.
        """
        model = ctx.model  # Retrieve model reference
        state = ctx.state_dict
        pass_no = ctx.pass_no

        state[pass_no] = {}

        # Collect gradients of Linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.grad is not None:
                state[pass_no][name] = {}
                state[pass_no][name]["grad_weights"] = module.weight.grad.clone()
                if module.bias is not None and module.bias.grad is not None:
                    state[pass_no][name]["grad_bias"] = module.bias.grad.clone()

        return (None, *grad_output)  # First output is None since we don’t need gradients for the model itself


