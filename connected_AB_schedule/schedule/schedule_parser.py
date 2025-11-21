"""
Schedule Parser - Parse YAML schedule configurations

Author: Guowen Li, AI Assistant
Date: 2025-01-20
"""

import yaml
import logging
from typing import Dict, List, Optional
from pathlib import Path
from schedule.control_models import (
    ControlAction, DailySchedule, AttackEvent, SimulationConfig,
    BuildingAVariables, BuildingBVariables
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScheduleParser:
    """Parse and validate schedule configurations"""
    
    # Valid variable names and bounds for Building A
    BUILDING_A_VARS = {
        'bcp': (0, 1),           # Binary
        'bahu': (0, 1),          # Binary
        'Tchw': (5, 15),         # °C
        'Tcw': (15, 35),         # °C
        'Tsa': (10, 20),         # °C
        'Vcore': (0, 1),         # Damper position [0-1]
        'Veast': (0, 1),
        'Vnorth': (0, 1),
        'Vsouth': (0, 1),
        'Vwest': (0, 1),
        'epsilon': (0, 100)      # Slack variable
    }
    
    # Valid variable names and bounds for Building B
    BUILDING_B_VARS = {
        'uMod': (-1, 2)  # Integer: -1, 0, 1, 2
    }
    
    @staticmethod
    def parse_schedule_file(filepath: str, building_id: str) -> DailySchedule:
        """
        Parse schedule YAML file for a specific building
        
        Parameters:
        -----------
        filepath : str
            Path to schedule YAML file
        building_id : str
            "Building_A" or "Building_B"
        
        Returns:
        --------
        DailySchedule object
        """
        logger.info(f"📋 Parsing schedule file: {filepath} for {building_id}")
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get building-specific schedule
        building_key = 'building_a' if building_id == 'Building_A' else 'building_b'
        building_config = config.get(building_key, {})
        schedule_data = building_config.get('schedule', [])
        
        # Get control interval (if specified)
        control_interval = building_config.get('control_interval_minutes', 
                                                15 if building_id == 'Building_A' else 60)
        control_interval_seconds = control_interval * 60
        
        # Parse actions
        actions = []
        for item in schedule_data:
            time_str = item['time']
            controls = item['controls']
            
            # Validate controls
            validated_controls = ScheduleParser._validate_controls(
                controls, building_id
            )
            
            action = ControlAction(
                time_of_day=time_str,
                building_id=building_id,
                scheduled_vars=validated_controls
            )
            actions.append(action)
        
        schedule = DailySchedule(
            building_id=building_id,
            control_interval_seconds=control_interval_seconds,
            actions=actions
        )
        
        logger.info(f"✅ Parsed {len(actions)} control actions for {building_id}")
        logger.info(f"   Control interval: {control_interval} minutes")
        
        return schedule
    
    @staticmethod
    def _validate_controls(controls: Dict[str, float], building_id: str) -> Dict[str, float]:
        """Validate control variable names and bounds"""
        
        var_bounds = (ScheduleParser.BUILDING_A_VARS if building_id == 'Building_A' 
                     else ScheduleParser.BUILDING_B_VARS)
        
        validated = {}
        for var_name, value in controls.items():
            # Check if variable exists
            if var_name not in var_bounds:
                logger.warning(f"⚠️ Unknown variable '{var_name}' for {building_id}, skipping")
                continue
            
            # Check bounds
            min_val, max_val = var_bounds[var_name]
            if not (min_val <= value <= max_val):
                logger.warning(
                    f"⚠️ Variable '{var_name}' value {value} out of bounds "
                    f"[{min_val}, {max_val}], clipping"
                )
                value = max(min_val, min(max_val, value))
            
            validated[var_name] = value
        
        return validated
    
    @staticmethod
    def parse_attack_scenarios(filepath: str) -> List[AttackEvent]:
        """Parse attack scenario configuration"""
        
        logger.info(f"🔒 Parsing attack scenarios: {filepath}")
        
        if not Path(filepath).exists():
            logger.warning(f"⚠️ Attack file not found: {filepath}, no attacks will occur")
            return []
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        attacks = []
        for attack_data in config.get('attacks', []):
            attack = AttackEvent(
                name=attack_data['name'],
                target_building=attack_data['target_building'],
                start_day=attack_data['start_day'],
                start_hour=attack_data['start_hour'],
                duration_hours=attack_data['duration_hours'],
                affected_variables=attack_data.get('affected_variables', []),
                attack_type=attack_data.get('type', 'unknown'),
                attack_params=attack_data.get('params', {})
            )
            attacks.append(attack)
        
        logger.info(f"✅ Parsed {len(attacks)} attack scenarios")
        
        return attacks
    
    @staticmethod
    def parse_system_config(filepath: str) -> Dict:
        """Parse system configuration"""
        
        logger.info(f"⚙️ Parsing system config: {filepath}")
        
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✅ System configuration loaded")
        
        return config
    
    @staticmethod
    def create_simulation_config(
        system_config_path: str,
        schedule_path: str,
        attack_path: str,
        start_day: int,
        duration_days: int
    ) -> SimulationConfig:
        """Create complete simulation configuration"""
        
        # Parse system config
        system_config = ScheduleParser.parse_system_config(system_config_path)
        
        # Parse schedules
        building_a_schedule = ScheduleParser.parse_schedule_file(schedule_path, 'Building_A')
        building_b_schedule = ScheduleParser.parse_schedule_file(schedule_path, 'Building_B')
        
        # Parse attacks
        attack_events = ScheduleParser.parse_attack_scenarios(attack_path)
        
        # Get scenario name
        with open(schedule_path, 'r') as f:
            schedule_config = yaml.safe_load(f)
        scenario_name = schedule_config.get('scenario_name', 'Unnamed Scenario')
        
        sim_config = SimulationConfig(
            scenario_name=scenario_name,
            start_day=start_day,
            duration_days=duration_days,
            building_a_schedule=building_a_schedule,
            building_b_schedule=building_b_schedule,
            attack_events=attack_events,
            feeder_capacity_kW=system_config.get('feeder', {}).get('capacity_kW', 50.0),
            feeder_safety_margin=system_config.get('feeder', {}).get('safety_margin', 0.9)
        )
        
        logger.info(f"✅ Complete simulation config created: {scenario_name}")
        
        return sim_config
