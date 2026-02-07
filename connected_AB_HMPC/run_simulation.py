"""
Main Simulation Script for Hierarchical Multi-Building Control

This demonstrates the new log-utility aggregator with flexibility bands
and soft budget constraints.

Usage:
    python run_simulation.py --days 2

Author: Guowen Li
Date: 2025-02-06
"""

import sys
import os
import argparse
import logging
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coordination import HierarchicalCoordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simulation.log')
    ]
)

logger = logging.getLogger(__name__)


def generate_weather_data(duration_days: int, dt: float = 3600.0) -> dict:
    """
    Generate synthetic weather data
    
    Parameters:
    -----------
    duration_days : int
        Simulation duration in days
    dt : float
        Time step (seconds)
    
    Returns:
    --------
    dict
        Weather data with 'Toa' (outdoor air temperature)
    """
    
    n_steps = int(duration_days * 24 * 3600 / dt) + 100
    
    # Outdoor temperature: sinusoidal daily pattern
    # Toa = 25 + 10*sin(2π*t/24hr) [°C]
    time_hours = np.arange(n_steps) * (dt / 3600.0)
    Toa = 25.0 + 10.0 * np.sin(2 * np.pi * time_hours / 24.0)
    
    return {'Toa': Toa.tolist()}


def generate_price_data(duration_days: int, dt: float = 3600.0) -> list:
    """
    Generate Time-of-Use electricity pricing
    
    Returns:
    --------
    list
        Price [$/kWh] at each timestep
    """
    
    # TOU pricing ($/kWh)
    price_24h = [
        0.064, 0.064, 0.064, 0.064,  # 00:00-03:59 (off-peak)
        0.064, 0.064, 0.064, 0.064,  # 04:00-07:59
        0.139, 0.139, 0.139, 0.139,  # 08:00-11:59 (mid-peak)
        0.355, 0.355, 0.355, 0.355,  # 12:00-15:59 (peak)
        0.355, 0.355, 0.139, 0.139,  # 16:00-19:59
        0.139, 0.139, 0.139, 0.064   # 20:00-23:59
    ]
    
    n_steps = int(duration_days * 24 * 3600 / dt) + 100
    n_steps_per_hour = int(3600 / dt)
    
    price = []
    for day in range(duration_days + 2):
        for hour_price in price_24h:
            price.extend([hour_price] * n_steps_per_hour)
    
    return price[:n_steps]


def main():
    """Main execution"""
    
    parser = argparse.ArgumentParser(
        description='Hierarchical Multi-Building Resilient Control'
    )
    parser.add_argument('--config', default='config/system_config.yaml',
                       help='Path to system configuration')
    parser.add_argument('--days', type=int, default=2,
                       help='Simulation duration (days)')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("🚀 HIERARCHICAL MULTI-BUILDING CONTROL SIMULATION")
    logger.info("="*80)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Duration: {args.days} days")
    logger.info("="*80)
    logger.info("")
    
    # Generate data
    logger.info("📊 Generating simulation data...")
    weather_data = generate_weather_data(args.days, dt=3600.0)
    price_data = generate_price_data(args.days, dt=3600.0)
    logger.info(f"  ✓ Weather: {len(weather_data['Toa'])} hourly points")
    logger.info(f"  ✓ Price: {len(price_data)} hourly points")
    logger.info("")
    
    # Initialize coordinator
    logger.info("🔧 Initializing coordinator...")
    try:
        coordinator = HierarchicalCoordinator(args.config)
    except Exception as e:
        logger.error(f"❌ Failed to initialize coordinator: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    logger.info("")
    
    # Run simulation
    logger.info("▶️  Starting simulation...")
    logger.info("")
    
    try:
        start_time = 0.0
        end_time = args.days * 24 * 3600.0
        
        coordinator.run_simulation(
            start_time=start_time,
            end_time=end_time,
            weather_data=weather_data,
            price_data=price_data
        )
        
        logger.info("")
        logger.info("="*80)
        logger.info("✅ SIMULATION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info("📁 Results saved in results/ directory")
        logger.info("")
        
        return 0
        
    except Exception as e:
        logger.error("")
        logger.error("="*80)
        logger.error(f"❌ SIMULATION FAILED: {e}")
        logger.error("="*80)
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        coordinator.shutdown()


if __name__ == '__main__':
    sys.exit(main())
