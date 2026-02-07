"""
Attack anticipation and prediction module

Author: Guowen Li, AI Assistant
Date: 2025-01-07
"""

import yaml
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AttackPrediction:
    """Attack prediction result"""
    anticipated: bool
    confidence: float
    time_to_attack_hours: float
    expected_duration_hours: float
    target_building: str
    severity: str

class AttackAnticipator:
    """Predicts and schedules cyber-attacks"""
    
    def __init__(self, method: str = "scheduled", config_path: str = "config/attack_scenarios.yaml"):
        """
        Initialize attack anticipator
        
        Parameters:
        -----------
        method : str
            'scheduled' - Use predefined attack schedule
            'predictive' - Use pattern recognition (future enhancement)
        """
        self.method = method
        self.attack_schedule = []
        self.attack_history = []
        
        # Load attack scenarios
        if method == "scheduled":
            self._load_attack_schedule(config_path)
    
    def _load_attack_schedule(self, config_path: str):
        """Load attack schedule from YAML"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            for scenario in config['scenarios']:
                for schedule in scenario['schedule']:
                    attack_event = {
                        'name': scenario['name'],
                        'target': scenario['target'],
                        'attack_type': scenario['attack_type'],
                        'component': scenario['affected_component'],
                        'start_day': schedule['start_day'],
                        'start_hour': schedule['start_hour'],
                        'duration_hours': schedule['duration_hours'],
                        'severity': schedule['severity']
                    }
                    self.attack_schedule.append(attack_event)
            
            logger.info(f"✓ Loaded {len(self.attack_schedule)} attack events from schedule")
        
        except FileNotFoundError:
            logger.warning(f"⚠️ Attack schedule file not found: {config_path}")
            self.attack_schedule = []
    
    def predict_attack(self, current_time: float, simulation_start: float, 
                       anticipation_hours: float = 3.0) -> AttackPrediction:
        """
        Predict if an attack is anticipated within the given horizon
        
        Parameters:
        -----------
        current_time : float
            Current simulation time (seconds)
        simulation_start : float
            Simulation start time (seconds)
        anticipation_hours : float
            How many hours ahead to look for attacks
        
        Returns:
        --------
        AttackPrediction object
        """
        if self.method == "scheduled":
            return self._scheduled_prediction(current_time, simulation_start, anticipation_hours)
        elif self.method == "predictive":
            return self._ml_prediction(current_time, anticipation_hours)
        else:
            return AttackPrediction(
                anticipated=False,
                confidence=0.0,
                time_to_attack_hours=np.inf,
                expected_duration_hours=0.0,
                target_building="None",
                severity="none"
            )
    
    def _scheduled_prediction(self, current_time: float, simulation_start: float, 
                             anticipation_hours: float) -> AttackPrediction:
        """Schedule-based attack prediction"""
        
        # Convert current time to days and hours
        elapsed_seconds = current_time - simulation_start
        current_day = int(elapsed_seconds // 86400) + 1
        current_hour = (elapsed_seconds % 86400) / 3600
        
        # Check each scheduled attack
        for attack in self.attack_schedule:
            attack_day = attack['start_day']
            attack_hour = attack['start_hour']
            
            # Calculate time to attack
            time_to_attack_seconds = (attack_day - current_day) * 86400 + (attack_hour - current_hour) * 3600
            time_to_attack_hours = time_to_attack_seconds / 3600
            
            # Check if attack is within anticipation window
            if 0 <= time_to_attack_hours <= anticipation_hours:
                logger.info(f"🔮 Attack anticipated: {attack['name']} in {time_to_attack_hours:.1f}h")
                return AttackPrediction(
                    anticipated=True,
                    confidence=1.0,  # Perfect confidence for scheduled attacks
                    time_to_attack_hours=time_to_attack_hours,
                    expected_duration_hours=attack['duration_hours'],
                    target_building=attack['target'],
                    severity=attack['severity']
                )
        
        # No attack anticipated
        return AttackPrediction(
            anticipated=False,
            confidence=0.0,
            time_to_attack_hours=np.inf,
            expected_duration_hours=0.0,
            target_building="None",
            severity="none"
        )
    
    def _ml_prediction(self, current_time: float, anticipation_hours: float) -> AttackPrediction:
        """
        Machine learning-based attack prediction (PLACEHOLDER for future work)
        
        This could use:
        - Pattern recognition from historical attacks
        - Anomaly detection in network traffic
        - External threat intelligence feeds
        """
        # TODO: Implement ML-based prediction
        # For now, use conservative assumption: no attack anticipated
        return AttackPrediction(
            anticipated=False,
            confidence=0.0,
            time_to_attack_hours=np.inf,
            expected_duration_hours=0.0,
            target_building="Unknown",
            severity="unknown"
        )
    
    def is_attack_active(self, current_time: float, simulation_start: float) -> Tuple[bool, Optional[Dict]]:
        """
        Check if an attack is currently active
        
        Returns:
        --------
        (is_active, attack_info)
        """
        elapsed_seconds = current_time - simulation_start
        current_day = int(elapsed_seconds // 86400) + 1
        current_hour = (elapsed_seconds % 86400) / 3600
        
        for attack in self.attack_schedule:
            attack_start = (attack['start_day'] - 1) * 24 + attack['start_hour']
            attack_end = attack_start + attack['duration_hours']
            current_absolute_hour = (current_day - 1) * 24 + current_hour
            
            if attack_start <= current_absolute_hour < attack_end:
                return True, attack
        
        return False, None