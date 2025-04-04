from typing import Any, Type
from collections import defaultdict
import json
from itertools import chain

import torch
import torch.nn as nn

def to_list(tensor: torch.Tensor) -> list:
    return tensor.detach().cpu().numpy().tolist()

def split_by_all(arr_of_strings: list[str], split_by: str, i: int=0):
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
    def __init__(self, model: nn.Module):
        super(ModelCollector, self).__init__()
        # add dummy identities to make it easier to collect inputs and gradients with hooks
        self._model = nn.Sequential(nn.Identity(), model, nn.Identity())
        self._state_dict: dict[int | str, dict[str, dict]] = defaultdict(lambda: defaultdict(dict))
        self._pass_no = 0
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
            module.register_forward_hook(self._capture_params())
            module.register_full_backward_hook(self._capture_gradients())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Calls forward pass of the wrapped model,
        collects weights and gradients for later visualization.
        """
        self._pass_no += 1
        output = self._model(*args, **kwargs)

        return output

    def _capture_gradients(self):
        def hook(module, grad_input, grad_output):
            self._state_dict[self._pass_no][module.tag]['grad_input'] = to_list(torch.mean(grad_input[0], dim=0, keepdim=False))
            for pname, param in module.named_parameters():
                self._state_dict[self._pass_no][module.tag][f"grad_{pname}"] = to_list(param)
        return hook

    def _capture_params(self):
        def hook(module, input_, res):
            self._state_dict[self._pass_no][module.tag]['input'] = to_list(input_[0].mean(dim=0, keepdim=False))
            for pname, param in module.named_parameters():
                self._state_dict[self._pass_no][module.tag][pname] = to_list(param)
        return hook

    def get_collected_data(self):
        """
        Returns the collected weights and gradients.
        """
        return self._state_dict

    def dump_to_json(self, json_path: str):
        with open(json_path, "w") as f:
            json.dump(self._state_dict, f, indent=4, separators=(",", ": "))


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

    print(collector.get_collected_data())
    collector.dump_to_json("./example.json")

