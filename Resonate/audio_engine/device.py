"""
Device Management - MPS (Metal Performance Shaders) for Apple Silicon

Handles device selection and memory management for M1/M2 chips.
"""

import torch
import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Supported device types for audio processing."""
    MPS = "mps"  # Apple Silicon Metal
    CPU = "cpu"  # Fallback


class DeviceManager:
    """
    Manages device selection and configuration for audio processing.
    
    Automatically detects and configures MPS for Apple Silicon,
    with graceful fallback to CPU if unavailable.
    """
    
    def __init__(self, memory_fraction: float = 0.75):
        """
        Initialize device manager.
        
        Args:
            memory_fraction: Fraction of GPU memory to use (0.0-1.0)
        """
        self.memory_fraction = memory_fraction
        self._device: Optional[torch.device] = None
        self._device_type: Optional[DeviceType] = None
        
    @property
    def device(self) -> torch.device:
        """Get the current device."""
        if self._device is None:
            self._detect_device()
        return self._device
    
    @property
    def device_type(self) -> DeviceType:
        """Get the device type enum."""
        if self._device_type is None:
            self._detect_device()
        return self._device_type
    
    def _detect_device(self) -> None:
        """
        Detect and configure the best available device.
        
        Priority: MPS (Apple Silicon) > CPU
        """
        # Check MPS availability
        if torch.backends.mps.is_available():
            try:
                # Test MPS with actual operation
                test_tensor = torch.ones(1, device='mps') * 2
                _ = test_tensor.cpu()  # Force computation
                
                self._device = torch.device('mps')
                self._device_type = DeviceType.MPS
                
                # Configure memory management
                torch.mps.set_per_process_memory_fraction(self.memory_fraction)
                
                logger.info("✅ Using MPS (Metal) device: Apple Silicon GPU")
                logger.info(f"   Memory fraction: {self.memory_fraction * 100}%")
                
            except Exception as mps_error:
                logger.warning(f"⚠️ MPS test failed: {mps_error}")
                logger.info("   Falling back to CPU")
                self._use_cpu()
        else:
            logger.info("💻 MPS not available, using CPU")
            self._use_cpu()
    
    def _use_cpu(self) -> None:
        """Configure CPU as the processing device."""
        self._device = torch.device('cpu')
        self._device_type = DeviceType.CPU
        logger.info("   Using CPU for audio processing")
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get device information for logging/debugging.
        
        Returns:
            Dictionary with device details
        """
        info = {
            "device_type": self.device_type.value,
            "device": str(self.device),
        }
        
        if self.device_type == DeviceType.MPS:
            info["gpu_name"] = "Apple Silicon MPS"
            info["memory_fraction"] = self.memory_fraction
        
        return info
    
    def __repr__(self) -> str:
        """String representation of device manager."""
        info = self.get_info()
        return f"DeviceManager({info['device_type']}, {info.get('device', 'N/A')})"


# Global device manager instance
_default_device_manager: Optional[DeviceManager] = None


def get_device_manager(memory_fraction: float = 0.75) -> DeviceManager:
    """
    Get or create the default device manager.
    
    Args:
        memory_fraction: Fraction of GPU memory to use
        
    Returns:
        DeviceManager instance
    """
    global _default_device_manager
    
    if _default_device_manager is None:
        _default_device_manager = DeviceManager(memory_fraction)
    
    return _default_device_manager


def get_device() -> torch.device:
    """
    Get the default device for audio processing.
    
    Returns:
        torch.device (MPS or CPU)
    """
    return get_device_manager().device


def is_mps_available() -> bool:
    """Check if MPS is available and working."""
    return torch.backends.mps.is_available()


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    manager = DeviceManager(memory_fraction=0.75)
    print(f"Device: {manager.device}")
    print(f"Info: {manager.get_info()}")
