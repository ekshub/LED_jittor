"""
PyTorch to Jittor compatibility layer.
This module provides compatibility functions and utilities for migrating from PyTorch to Jittor.
"""

import jittor as jt
from jittor import nn
import numpy as np

# Enable CUDA if available
if jt.has_cuda:
    jt.flags.use_cuda = 1

# ====================
# Tensor operations
# ====================

def torch_to_jittor(torch_api_name):
    """
    Map PyTorch API to Jittor equivalent.
    """
    # Most APIs are the same, this is for documentation
    mapping = {
        'torch.tensor': 'jt.array',
        'torch.Tensor': 'jt.Var',
        'torch.cuda.is_available': 'jt.has_cuda',
        'torch.cuda.device_count': 'jt.get_device_count',
        'torch.no_grad': 'jt.no_grad',
        'torch.save': 'jt.save',
        'torch.load': 'jt.load',
    }
    return mapping.get(torch_api_name, torch_api_name)


# ====================
# Module compatibility
# ====================

class DataParallel(nn.Module):
    """
    Jittor doesn't need DataParallel as it handles multi-GPU automatically.
    This is a passthrough wrapper for compatibility.
    """
    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__()
        self.module = module
        
    def execute(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class DistributedDataParallel(nn.Module):
    """
    Jittor handles distributed training differently.
    This is a placeholder for compatibility.
    """
    def __init__(self, module, device_ids=None, output_device=None, 
                 broadcast_buffers=True, find_unused_parameters=False):
        super().__init__()
        self.module = module
        
    def execute(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# ====================
# Utility functions
# ====================

def get_device(device_str='cuda'):
    """
    Jittor doesn't use explicit device placement like PyTorch.
    Returns None for compatibility.
    """
    return None


def to_device(data, device):
    """
    Jittor automatically handles device placement.
    Returns data as-is for compatibility.
    """
    return data


def cuda_available():
    """Check if CUDA is available."""
    return jt.has_cuda


def device_count():
    """Get number of CUDA devices."""
    if jt.has_cuda:
        return jt.compiler.cuda_flags.device_count()
    return 0


def synchronize():
    """Synchronize CUDA operations."""
    if jt.has_cuda:
        jt.sync_all()


def empty_cache():
    """Empty CUDA cache."""
    if jt.has_cuda:
        jt.gc()


# ====================
# Torchvision utilities
# ====================

def make_grid(tensor, nrow=8, padding=2, normalize=False, 
              value_range=None, scale_each=False, pad_value=0):
    """
    Make a grid of images. Simplified version of torchvision.utils.make_grid.
    """
    if not isinstance(tensor, jt.Var):
        tensor = jt.array(tensor)
    
    # Ensure 4D tensor (N, C, H, W)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(np.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    
    grid = jt.zeros((num_channels, height * ymaps + padding, width * xmaps + padding))
    grid.fill_(pad_value)
    
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y + 1) * height, 
                 x * width + padding:(x + 1) * width] = tensor[k]
            k += 1
    
    if normalize:
        if value_range is not None:
            grid = (grid - value_range[0]) / (value_range[1] - value_range[0])
        else:
            grid = grid - grid.min()
            grid = grid / grid.max()
    
    return grid


def save_image(tensor, fp, nrow=8, padding=2, normalize=False,
               value_range=None, scale_each=False, pad_value=0, format=None):
    """
    Save a tensor as an image file.
    """
    from PIL import Image
    
    grid = make_grid(tensor, nrow=nrow, padding=padding, normalize=normalize,
                    value_range=value_range, scale_each=scale_each, pad_value=pad_value)
    
    # Convert to numpy and transpose to (H, W, C)
    ndarr = grid.numpy()
    if ndarr.shape[0] == 1:
        ndarr = ndarr[0]
    else:
        ndarr = ndarr.transpose(1, 2, 0)
    
    # Scale to 0-255
    ndarr = np.clip(ndarr * 255, 0, 255).astype(np.uint8)
    
    im = Image.fromarray(ndarr)
    im.save(fp, format=format)


# ====================
# Learning rate scheduler compatibility
# ====================

class _LRScheduler:
    """Base class for learning rate schedulers in Jittor."""
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self._step_count = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def get_lr(self):
        raise NotImplementedError
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# ====================
# Optimizer wrapper
# ====================

def get_optimizer_param_groups(module):
    """Get parameter groups for optimizer."""
    return [{'params': module.parameters()}]


# ====================
# Mixed precision training
# ====================

class GradScaler:
    """
    Gradient scaler for mixed precision training.
    Jittor doesn't require explicit gradient scaling like PyTorch AMP.
    This is a compatibility wrapper.
    """
    def __init__(self, init_scale=2.**16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, enabled=True):
        self.enabled = enabled
        
    def scale(self, loss):
        return loss
    
    def step(self, optimizer):
        optimizer.step()
    
    def update(self):
        pass
    
    def unscale_(self, optimizer):
        pass


class autocast:
    """
    Autocast context manager for mixed precision.
    Jittor has its own implementation.
    """
    def __init__(self, enabled=True, dtype=None):
        self.enabled = enabled
        self.prev_dtype = None
        
    def __enter__(self):
        if self.enabled:
            # Jittor's float16 support
            pass
        return self
        
    def __exit__(self, *args):
        pass


# ====================
# Re-export Jittor components
# ====================

# Common Jittor imports that replace PyTorch
Var = jt.Var
array = jt.array
zeros = jt.zeros
ones = jt.ones
randn = jt.randn
rand = jt.rand
cat = jt.concat  # Note: Jittor uses concat instead of cat
stack = jt.stack
no_grad = jt.no_grad
save = jt.save
load = jt.load

# Re-export nn module
Module = nn.Module
Sequential = nn.Sequential
ModuleList = nn.ModuleList
Parameter = nn.Parameter

# Jittor doesn't have ModuleDict and ParameterList, implement them
class ModuleDict(nn.Module):
    """A dictionary that holds submodules."""
    def __init__(self, modules=None):
        super().__init__()
        self._modules_dict = {}
        if modules is not None:
            self.update(modules)
    
    def __getitem__(self, key):
        return self._modules_dict[key]
    
    def __setitem__(self, key, module):
        self._modules_dict[key] = module
        setattr(self, str(key), module)
    
    def __delitem__(self, key):
        del self._modules_dict[key]
        delattr(self, str(key))
    
    def __len__(self):
        return len(self._modules_dict)
    
    def __iter__(self):
        return iter(self._modules_dict)
    
    def __contains__(self, key):
        return key in self._modules_dict
    
    def clear(self):
        self._modules_dict.clear()
    
    def pop(self, key):
        v = self._modules_dict.pop(key)
        delattr(self, str(key))
        return v
    
    def keys(self):
        return self._modules_dict.keys()
    
    def items(self):
        return self._modules_dict.items()
    
    def values(self):
        return self._modules_dict.values()
    
    def update(self, modules):
        if isinstance(modules, dict):
            for key, module in modules.items():
                self[key] = module
        else:
            for key, module in modules:
                self[key] = module

class ParameterList(nn.Module):
    """A list that holds parameters."""
    def __init__(self, parameters=None):
        super().__init__()
        self._parameters_list = []
        if parameters is not None:
            self += parameters
    
    def __getitem__(self, idx):
        return self._parameters_list[idx]
    
    def __setitem__(self, idx, param):
        self._parameters_list[idx] = param
        setattr(self, f'param_{idx}', param)
    
    def __len__(self):
        return len(self._parameters_list)
    
    def __iter__(self):
        return iter(self._parameters_list)
    
    def append(self, parameter):
        idx = len(self._parameters_list)
        self._parameters_list.append(parameter)
        setattr(self, f'param_{idx}', parameter)
    
    def extend(self, parameters):
        for param in parameters:
            self.append(param)
    
    def __iadd__(self, parameters):
        self.extend(parameters)
        return self


# Common layers
Conv2d = nn.Conv2d
ConvTranspose2d = nn.ConvTranspose2d
BatchNorm2d = nn.BatchNorm2d
InstanceNorm2d = nn.InstanceNorm2d
GroupNorm = nn.GroupNorm
LayerNorm = nn.LayerNorm
Linear = nn.Linear
Dropout = nn.Dropout
ReLU = nn.ReLU
LeakyReLU = nn.LeakyReLU
PReLU = nn.PReLU
GELU = nn.GELU
Sigmoid = nn.Sigmoid
Tanh = nn.Tanh

# Pooling
MaxPool2d = nn.MaxPool2d
AvgPool2d = nn.AvgPool2d
AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d
AdaptiveMaxPool2d = nn.AdaptiveMaxPool2d

# Loss functions
L1Loss = nn.L1Loss
MSELoss = nn.MSELoss
BCELoss = nn.BCELoss
BCEWithLogitsLoss = nn.BCEWithLogitsLoss
CrossEntropyLoss = nn.CrossEntropyLoss

print("[Jittor Compatibility Layer] Successfully loaded. PyTorch -> Jittor migration active.")
