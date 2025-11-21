"""
Data Loader - Load weather and price data

Author: Guowen Li, AI Assistant
Date: 2025-01-20
"""

import numpy as np
from pvlib.iotools import read_epw
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLoader:
    """Load and process weather and price data"""
    
    @staticmethod
    def load_weather_data(weather_file: str, dt: float = 900.0) -> Dict[str, List[float]]:
        """
        Load weather data from EPW file
        
        Parameters:
        -----------
        weather_file : str
            Path to EPW file
        dt : float
            Timestep in seconds
        
        Returns:
        --------
        Dict with Toa, RHoa, GHI lists
        """
        logger.info(f"📡 Loading weather data from {weather_file}")
        
        dat = read_epw(weather_file)
        
        # Extract hourly data
        weather_hourly = dat[0][['temp_air', 'relative_humidity', 'ghi']].copy()
        weather_hourly.columns = ['Toa', 'RHoa', 'GHI']
        
        # Convert RH to fraction
        weather_hourly['RHoa'] = weather_hourly['RHoa'] / 100.0
        
        # Create index (hours to seconds)
        index_h = np.arange(3600, 3600 * (len(weather_hourly) + 1), 3600)
        weather_hourly.index = index_h
        
        # Interpolate to timestep
        index_step = np.arange(3600, 3600 * (len(weather_hourly) + 1), dt)
        
        weather_interpolated = {}
        for col in weather_hourly.columns:
            weather_interpolated[col] = list(
                np.interp(index_step, weather_hourly.index, weather_hourly[col])
            )
        
        logger.info(f"✅ Weather data loaded: {len(weather_interpolated['Toa'])} timesteps")
        
        return weather_interpolated
    
    @staticmethod
    def load_price_data(dt: float = 900.0, n_days: int = 2) -> List[float]:
        """
        Load TOU pricing data
        
        Parameters:
        -----------
        dt : float
            Timestep in seconds
        n_days : int
            Number of days
        
        Returns:
        --------
        List of prices ($/kWh)
        """
        # Time-of-Use pricing ($/kWh)
        price_tou_24h = [
            0.0640, 0.0640, 0.0640, 0.0640,  # 00:00-03:59
            0.0640, 0.0640, 0.0640, 0.0640,  # 04:00-07:59
            0.1391, 0.1391, 0.1391, 0.1391,  # 08:00-11:59
            0.3548, 0.3548, 0.3548, 0.3548,  # 12:00-15:59 (PEAK)
            0.3548, 0.3548, 0.1391, 0.1391,  # 16:00-19:59
            0.1391, 0.1391, 0.1391, 0.0640   # 20:00-23:59
        ]
        
        # Repeat for n_days
        price_hourly = price_tou_24h * (n_days + 1)  # Extra day for forecast
        
        # Interpolate to timestep
        nsteps_per_hour = int(3600 / dt)
        price_interpolated = []
        
        for price_hour in price_hourly:
            price_interpolated.extend([price_hour] * nsteps_per_hour)
        
        logger.info(f"✅ Price data loaded: {len(price_interpolated)} timesteps")
        
        return price_interpolated
