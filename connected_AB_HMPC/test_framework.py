"""
Quick Test Script

Tests the framework with a minimal simulation

Usage:
    python test_framework.py

Author: Guowen Li
Date: 2025-02-06
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all modules can be imported"""
    logger.info("Testing imports...")
    
    try:
        from aggregator import LogUtilityAggregator, LogUtilityAggregatorConfig
        from buildings import BuildingASimple, BuildingBSimple
        from communication import FlexibilityBand, PowerBudget
        from coordination import HierarchicalCoordinator
        logger.info("  ✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"  ✗ Import failed: {e}")
        return False

def test_aggregator():
    """Test aggregator with dummy flexibility bands"""
    logger.info("Testing aggregator...")
    
    try:
        from aggregator import LogUtilityAggregator, LogUtilityAggregatorConfig
        from communication import FlexibilityBand
        
        config = LogUtilityAggregatorConfig(
            PH=4,
            dt=3600.0,
            P_feeder_limit_kW=50.0
        )
        
        agg = LogUtilityAggregator(config)
        agg.register_building('Building_A', 1)
        agg.register_building('Building_B', 2)
        
        # Create dummy bands
        band_a = FlexibilityBand(
            building_id='Building_A',
            timestamp=0.0,
            time_horizon=[0, 3600, 7200, 10800],
            P_lower_kW=[5.0, 5.0, 5.0, 5.0],
            P_upper_kW=[15.0, 15.0, 15.0, 15.0],
            baseline_P_kW=[10.0, 10.0, 10.0, 10.0],
            computation_time_s=0.1,
            feasible=True
        )
        
        band_b = FlexibilityBand(
            building_id='Building_B',
            timestamp=0.0,
            time_horizon=[0, 3600, 7200, 10800],
            P_lower_kW=[8.0, 8.0, 8.0, 8.0],
            P_upper_kW=[20.0, 20.0, 20.0, 20.0],
            baseline_P_kW=[12.0, 12.0, 12.0, 12.0],
            computation_time_s=0.1,
            feasible=True
        )
        
        # Solve allocation
        allocation = agg.allocate_power(
            flexibility_bands=[band_a, band_b],
            feeder_limit=[50.0, 50.0, 50.0, 50.0],
            current_time=0.0
        )
        
        if allocation.feasible:
            logger.info("  ✓ Aggregator test passed")
            logger.info(f"    Building A: {allocation.budgets[0].P_ref_kW[0]:.2f} kW")
            logger.info(f"    Building B: {allocation.budgets[1].P_ref_kW[0]:.2f} kW")
            return True
        else:
            logger.error("  ✗ Allocation infeasible")
            return False
            
    except Exception as e:
        logger.error(f"  ✗ Aggregator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_building():
    """Test building MPC"""
    logger.info("Testing building MPC...")
    
    try:
        from buildings import BuildingASimple
        
        config = {'timing': {'building_a_timestep': 900}}
        bldg = BuildingASimple('Building_A', config)
        
        weather = {'Toa': [30.0, 30.0, 30.0, 30.0]}
        price = [0.1, 0.1, 0.1, 0.1]
        
        # Test two-pass MPC
        band = bldg.compute_flexibility_band(0.0, weather, price, 4)
        
        if band.feasible:
            logger.info("  ✓ Building test passed")
            logger.info(f"    Flexibility: [{band.P_lower_kW[0]:.2f}, {band.P_upper_kW[0]:.2f}] kW")
            return True
        else:
            logger.error("  ✗ MPC infeasible")
            return False
            
    except Exception as e:
        logger.error(f"  ✗ Building test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger.info("="*60)
    logger.info("🧪 FRAMEWORK QUICK TEST")
    logger.info("="*60)
    logger.info("")
    
    tests = [
        test_imports,
        test_aggregator,
        test_building
    ]
    
    results = []
    for test in tests:
        results.append(test())
        logger.info("")
    
    logger.info("="*60)
    if all(results):
        logger.info("✅ ALL TESTS PASSED")
    else:
        logger.error("❌ SOME TESTS FAILED")
    logger.info("="*60)
    
    return 0 if all(results) else 1

if __name__ == '__main__':
    sys.exit(main())
