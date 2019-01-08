"""Utils"""
import torch

def show_gpu_info():
    """Show GPU info."""
    pass

def get_device():
    """Get best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
