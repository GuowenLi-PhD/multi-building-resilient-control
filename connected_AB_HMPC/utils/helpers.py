"""
Utility helper functions for hierarchical control

Author: Guowen Li
Date: 2025-02-06
"""

import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def extract_forecast_window(data_dict: Dict[str, List], 
                           start_idx: int,
                           horizon: int) -> Dict[str, List]:
    """
    Extract forecast window from data dictionary
    
    Parameters:
    -----------
    data_dict : Dict[str, List]
        Dictionary of time-series data
    start_idx : int
        Starting index
    horizon : int
        Number of steps to extract
    
    Returns:
    --------
    Dict[str, List]
        Extracted window
    """
    forecast = {}
    for key, values in data_dict.items():
        end_idx = min(start_idx + horizon, len(values))
        forecast[key] = values[start_idx:end_idx]
        
        # Pad if necessary
        if len(forecast[key]) < horizon:
            last_value = forecast[key][-1] if forecast[key] else 0
            forecast[key].extend([last_value] * (horizon - len(forecast[key])))
    
    return forecast


def compute_time_index(current_time: float, dt: float, base_time: float = 0.0) -> int:
    """
    Compute time index for data array
    
    Parameters:
    -----------
    current_time : float
        Current time [s]
    dt : float
        Time step [s]
    base_time : float
        Base time (default 0)
    
    Returns:
    --------
    int
        Index
    """
    return int((current_time - base_time) / dt)


def interpolate_to_timestep(hourly_data: List[float],
                            dt: float,
                            n_steps: int) -> List[float]:
    """
    Interpolate hourly data to different timestep
    
    Parameters:
    -----------
    hourly_data : List[float]
        Data at 1-hour intervals
    dt : float
        Target timestep [s]
    n_steps : int
        Number of steps to generate
    
    Returns:
    --------
    List[float]
        Interpolated data
    """
    hourly_times = np.arange(0, len(hourly_data)) * 3600.0
    target_times = np.arange(0, n_steps) * dt
    
    interpolated = np.interp(target_times, hourly_times, hourly_data)
    
    return interpolated.tolist()


def validate_forecast_data(forecast: Dict[str, List], 
                          required_keys: List[str],
                          horizon: int) -> bool:
    """
    Validate forecast data dictionary
    
    Parameters:
    -----------
    forecast : Dict[str, List]
        Forecast data
    required_keys : List[str]
        Required variables
    horizon : int
        Expected length
    
    Returns:
    --------
    bool
        Valid or not
    """
    for key in required_keys:
        if key not in forecast:
            logger.error(f"Missing key in forecast: {key}")
            return False
        if len(forecast[key]) < horizon:
            logger.warning(f"Forecast for {key} shorter than horizon ({len(forecast[key])} < {horizon})")
            # Allow this but warn
    
    return True


def compute_comfort_violation(T_zone: float, 
                              T_lower: float = 21.0,
                              T_upper: float = 24.0,
                              dt_hours: float = 1.0) -> float:
    """
    Compute comfort violation in degree-hours
    
    Parameters:
    -----------
    T_zone : float
        Zone temperature [°C]
    T_lower : float
        Lower comfort bound [°C]
    T_upper : float
        Upper comfort bound [°C]
    dt_hours : float
        Time step [hours]
    
    Returns:
    --------
    float
        Comfort violation [°C·h]
    """
    violation = 0.0
    
    if T_zone < T_lower:
        violation = (T_lower - T_zone) * dt_hours
    elif T_zone > T_upper:
        violation = (T_zone - T_upper) * dt_hours
    
    return violation


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range"""
    return max(min_val, min(value, max_val))


def ensure_list(value, length: int = 1):
    """Ensure value is a list of specified length"""
    if isinstance(value, (list, tuple)):
        return list(value)
    else:
        return [value] * length
