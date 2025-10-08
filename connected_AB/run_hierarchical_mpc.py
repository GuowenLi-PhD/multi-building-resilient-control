"""
Main execution script for hierarchical multi-building resilient control

Author: Guowen Li, AI Assistant
Date: 2025-01-07
"""

import sys
import os
import numpy as np
from pvlib.iotools import read_epw
import argparse
import yaml

# Add module paths
sys.path.append(os.path.dirname(__file__))

from simulation.coordinator import HierarchicalCoordinator
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_weather_data(weather_file: str, dt: float = 900.0):
    """Load weather data from EPW file"""
    
    logger.info(f"📡 Loading weather data from {weather_file}")
    
    dat = read_epw(weather_file)
    
    # Extract hourly data
    weather_hourly = dat[0][['temp_air', 'relative_humidity', 'ghi']]
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
    
    logger.info(f"✓ Weather data loaded: {len(weather_interpolated['Toa'])} timesteps")
    
    return weather_interpolated

def load_price_data(dt: float = 900.0, n_days: int = 2):
    """Load TOU pricing data"""
    
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
    price_hourly = price_tou_24h * n_days
    
    # Interpolate to timestep
    nsteps_per_hour = int(3600 / dt)
    price_interpolated = []
    
    for price_hour in price_hourly:
        price_interpolated.extend([price_hour] * nsteps_per_hour)
    
    logger.info(f"✓ Price data loaded: {len(price_interpolated)} timesteps")
    
    return price_interpolated

def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description='Hierarchical Multi-Building Resilient Control')
    parser.add_argument('--config', type=str, default='config/system_config.yaml',
                       help='Path to system configuration file')
    parser.add_argument('--weather', type=str, 
                       default='../buildingA_wo_TES/weather_data/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw',
                       help='Path to weather file (EPW format)')
    parser.add_argument('--start-day', type=int, default=212,
                       help='Simulation start day (day of year)')
    parser.add_argument('--duration-days', type=int, default=2,
                       help='Simulation duration (days)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("🚀 STARTING HIERARCHICAL MULTI-BUILDING RESILIENT CONTROL SIMULATION")
    logger.info("="*80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Weather file: {args.weather}")
    logger.info(f"Start day: {args.start_day}")
    logger.info(f"Duration: {args.duration_days} days")
    logger.info("="*80)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Calculate simulation times
    t_start = args.start_day * 24 * 3600  # Convert day to seconds
    t_end = t_start + args.duration_days * 24 * 3600
    
    # Load weather data
    weather_data = load_weather_data(
        weather_file=args.weather,
        dt=config['timing']['aggregator_timestep']
    )
    
    # Load price data
    price_data = load_price_data(
        dt=config['timing']['aggregator_timestep'],
        n_days=args.duration_days + 1  # Extra day for forecast
    )
    
    # Initialize coordinator
    coordinator = HierarchicalCoordinator(config_path=args.config)
    
    # Run simulation
    try:
        coordinator.run_simulation(
            start_time=t_start,
            end_time=t_end,
            weather_data=weather_data,
            price_data=price_data
        )
        
        logger.info("="*80)
        logger.info("✅ SIMULATION COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info("📊 Check the 'results/' directory for detailed outputs")
        
    except Exception as e:
        logger.error("="*80)
        logger.error(f"❌ SIMULATION FAILED: {str(e)}")
        logger.error("="*80)
        raise
    
    finally:
        # Cleanup
        coordinator.building_a.shutdown()
        coordinator.building_b.shutdown()

if __name__ == "__main__":
    main()