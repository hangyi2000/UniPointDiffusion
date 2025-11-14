# -*- coding: utf-8 -*-

import torch
import numpy as np


def generate_dense_grid_points(bbox_min: np.ndarray,
                               bbox_max: np.ndarray,
                               octree_dept: int,
                               indexing: str = "xy"):
    length = bbox_max - bbox_min
    num_cells = np.exp2(octree_dept)
    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    xyz = xyz.reshape(-1, 3)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


def print_model_parameter_count(model):
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in {type(model).__name__}: {count}")


def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen if t.requires_grad]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs}

    # Filter out redundant entries.
    if skip_redundant:
        entries = [e for e in entries if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)]

    # Construct table.
    rows = [[type(module).__name__, "Parameters", "Buffers", "Output shape", "Datatype"]]
    rows += [["---"] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = "<top-level>" if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(e.outputs[0].shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split(".")[-1] for t in e.outputs]
        rows += [[
            name + (":0" if len(e.outputs) >= 2 else ""),
            str(param_size) if param_size else "-",
            str(buffer_size) if buffer_size else "-",
            (output_shapes + ["-"])[0],
            (output_dtypes + ["-"])[0],
        ]]
        for idx in range(1, len(e.outputs)):
            rows += [[name + f":{idx}", "-", "-", output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [["---"] * len(rows[0])]
    rows += [["Total", str(param_total), str(buffer_total), "-", "-"]]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print("  ".join(cell + " " * (width - len(cell)) for cell, width in zip(row, widths)))
    print()
    return outputs


def get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


def invalid_tensor(tensor):
    return torch.isinf(tensor).any() or torch.isnan(tensor).any()


class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]
