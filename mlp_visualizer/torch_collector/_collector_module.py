from typing import Any, Type, Literal
from collections import defaultdict
import json
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F


def to_list(tensor: Any) -> list:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy().tolist()
    return tensor


def split_by_all(arr_of_strings: list[str], split_by: str, i: int = 0):
    if i == len(split_by):
        return arr_of_strings
    l = list(chain.from_iterable(list(map(lambda x: x.split(split_by[i]), arr_of_strings))))
    return split_by_all(l, split_by, i + 1)


def try_convert(v: str, arr_type: list[Type]) -> Any:
    for type_ in arr_type:
        try:
            return type_(v)
        except ValueError:
            continue
    return v


def _parse_module_str(module_str: str) -> tuple[str, dict]:
    module_specs = split_by_all([module_str], "(),")
    spec_dict = {}
    for spec in module_specs[1:]:
        if not spec:
            continue
        if '=' in spec:
            k, v = spec.split('=')
            spec_dict[k] = try_convert(v, [float, bool])
    return module_specs[0], spec_dict


class ModelCollector(nn.Module):
    _captured_instances = [
        nn.Linear,
    ]

    def __init__(self, model: nn.Module, cap_visualized_layer_size=48, register: Literal["forward", "backward", "both"] = "forward"):
        super(ModelCollector, self).__init__()
        # add dummy identities to make it easier to collect inputs and gradients with hooks
        self._model = nn.Sequential(nn.Identity(), model, nn.Identity())
        self._state_dict: dict[int | str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
        self._pass_no = 0
        self.register = register
        self.cap_visualized_layer_size = cap_visualized_layer_size
        self._register_hooks()

    def _register_hooks(self):
        reg_mod_count = 1
        for i, module in enumerate(self._model.modules()):
            if isinstance(module, nn.Sequential) or isinstance(module, nn.ModuleList):
                continue
            module_name, module_specs = _parse_module_str(str(module))
            module.tag = f"{reg_mod_count}_{module_name}"
            self._state_dict["architecture"][module.tag] = module_specs
            reg_mod_count += 1
            if any([isinstance(module, cls) for cls in self._captured_instances]):
                if self.register == "both" or self.register == "forward":
                    module.register_forward_hook(self._capture_params())
                if self.register == "backward" or self.register == "both":
                    module.register_full_backward_hook(self._capture_gradients())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _cap_model_specs(self, specs):
        new_specs = {}
        for k, v in specs.items():
            if isinstance(v, int) or isinstance(v, float):
                v = v.__class__(min((self.cap_visualized_layer_size, v)))
            new_specs[k] = v
        return new_specs

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Calls forward pass of the wrapped model,
        collects weights and gradients for later visualization.

        kwargs are additional collected data like 'input'
        """
        self._pass_no += 1
        for k, v in kwargs.items():
            self._state_dict[self._pass_no][k] = to_list(v)
        output = self._model(*args)

        return output

    def _reduce_to_max_len(self, tensor: torch.Tensor) -> torch.Tensor:
        max_size = self.cap_visualized_layer_size
        orig_shape = tensor.shape
        d0 = orig_shape[0]

        if d0 <= max_size:
            return tensor

        # Compute the number of groups for reduction along dim 0
        num_groups = max_size
        group_size = d0 // num_groups
        remainder = d0 % num_groups

        # Create a list of group sizes
        sizes = [group_size + 1 if i < remainder else group_size for i in range(num_groups)]

        # Split the tensor into chunks along dimension 0
        chunks = torch.split(tensor, sizes)

        # Reduce each chunk by taking the mean along dimension 0
        mean_chunks = [chunk.mean(dim=0) for chunk in chunks]

        # Stack the mean chunks back together along dimension 0
        reduced_tensor_d0 = torch.stack(mean_chunks)

        # Now, reduce the other dimensions based on the reduction factor of the first dimension
        if reduced_tensor_d0.ndim > 1:
            scale_factor = d0 / max_size
            new_other_dims = [max(1, int(dim / scale_factor)) for dim in orig_shape[1:]]

            if reduced_tensor_d0.ndim == 2:
                reduced_tensor = F.adaptive_avg_pool1d(reduced_tensor_d0.unsqueeze(0), new_other_dims[-1]).squeeze(0)
            elif reduced_tensor_d0.ndim == 3:
                reduced_tensor = F.adaptive_avg_pool2d(reduced_tensor_d0.unsqueeze(0), new_other_dims[-2:]).squeeze(0)
            elif reduced_tensor_d0.ndim == 4:
                reduced_tensor = F.adaptive_avg_pool3d(reduced_tensor_d0.unsqueeze(0), new_other_dims[-3:]).squeeze(0)
            else:
                raise ValueError("Tensors with >4D adaptive pooling are not supported directly.")
        else:
            reduced_tensor = reduced_tensor_d0

        return reduced_tensor

    def _capture_gradients(self):
        def hook(module, grad_input, grad_output):
            self._state_dict[self._pass_no][module.tag]['grad_input'] = to_list(
                self._reduce_to_max_len(self._reduce_to_max_len(torch.mean(grad_input[0], dim=0, keepdim=False))))
            for pname, param in module.named_parameters():
                self._state_dict[self._pass_no][module.tag][f"grad_{pname}"] = to_list(self._reduce_to_max_len(param))
        return hook

    def _capture_params(self):
        def hook(module, input_, res):
            self._state_dict[self._pass_no][module.tag]['input'] = to_list(
                self._reduce_to_max_len((input_[0].mean(dim=0, keepdim=False))))
            for pname, param in module.named_parameters():
                self._state_dict[self._pass_no][module.tag][pname] = to_list(self._reduce_to_max_len(param))
        return hook

    def get_collected_data(self):
        """
        Returns the collected weights and gradients.
        """
        return self._state_dict

    def pretty_print(self):
        formatted_json = json.dumps(self._state_dict, indent=4, separators=(",", ": "))
        print(formatted_json)

    def dump_to_json(self, json_path: str):
        with open(json_path, "w") as f:
            json.dump(self._state_dict, f, indent=4, separators=(",", ": "))

    def register_value(self, name: str, value: Any, pass_no: int = -1):
        """ registers value to the current pass in the state dict or to pass_no if specified """
        pass_no = pass_no if pass_no > 0 else self._pass_no
        self._state_dict[pass_no][name] = to_list(value) if isinstance(value, torch.Tensor) else value


if __name__ == '__main__':
    # driver code
    model = nn.Sequential(
        nn.Linear(5, 5),
        nn.ReLU(),
        nn.Linear(5, 5),
        nn.ReLU(),
        nn.Linear(5, 2),
    )
    collector = ModelCollector(model)

    for i in range(2):
        y = collector.forward(torch.randn(5, 5, requires_grad=True))
        gold = torch.randn(5, 2, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(y, gold)
        loss.backward()

    collector.pretty_print()
    collector.dump_to_json("./example.json")
