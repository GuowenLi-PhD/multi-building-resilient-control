"""
Schedule Manager - Map simulation time to daily schedules

Author: Guowen Li, AI Assistant
Date: 2025-01-20
"""

import logging
from typing import Dict, Optional
from schedule.control_models import DailySchedule, ControlAction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScheduleManager:
    """Manage daily repeating schedules"""
    
    def __init__(self, daily_schedule: DailySchedule, simulation_start_time: float):
        """
        Initialize schedule manager
        
        Parameters:
        -----------
        daily_schedule : DailySchedule
            Daily schedule that repeats
        simulation_start_time : float
            Simulation start time (seconds from epoch)
        """
        self.daily_schedule = daily_schedule
        self.simulation_start_time = simulation_start_time
        self.building_id = daily_schedule.building_id
        
        logger.info(f"📅 Schedule manager initialized for {self.building_id}")
        logger.info(f"   Control interval: {daily_schedule.control_interval_seconds/60:.1f} min")
        logger.info(f"   Number of scheduled actions: {len(daily_schedule.actions)}")
    
    def get_control_action(self, current_time: float) -> Optional[Dict[str, float]]:
        """
        Get scheduled control variables for current simulation time
        
        Parameters:
        -----------
        current_time : float
            Current simulation time (seconds from epoch)
        
        Returns:
        --------
        Dict[str, float] or None
            Dictionary of scheduled variables, or None if no schedule
        """
        # Calculate time within day
        elapsed_time = current_time - self.simulation_start_time
        time_of_day_seconds = elapsed_time % 86400  # 86400 sec = 1 day
        
        # If no actions defined, return None (full MPC control)
        if len(self.daily_schedule.actions) == 0:
            return None
        
        # Find the active action for this time
        # Sort actions by time and find the last action before current time
        sorted_actions = sorted(
            self.daily_schedule.actions,
            key=lambda a: self._time_to_seconds(a.time_of_day)
        )
        
        active_action = None
        for action in sorted_actions:
            action_time = self._time_to_seconds(action.time_of_day)
            if action_time <= time_of_day_seconds:
                active_action = action
            else:
                break
        
        # If no action found (before first action), use last action from previous day
        if active_action is None and len(sorted_actions) > 0:
            active_action = sorted_actions[-1]
        
        if active_action:
            return active_action.scheduled_vars
        else:
            return None
    
    def get_control_interval(self) -> float:
        """Get control interval in seconds"""
        return self.daily_schedule.control_interval_seconds
    
    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        """Convert HH:MM to seconds since midnight"""
        parts = time_str.split(':')
        return int(parts[0]) * 3600 + int(parts[1]) * 60
    
    def get_schedule_summary(self) -> str:
        """Get human-readable schedule summary"""
        if len(self.daily_schedule.actions) == 0:
            return f"{self.building_id}: No schedule (full MPC control)"
        
        summary = f"{self.building_id} Daily Schedule:\n"
        for action in sorted(self.daily_schedule.actions, 
                            key=lambda a: self._time_to_seconds(a.time_of_day)):
            vars_str = ", ".join([f"{k}={v}" for k, v in action.scheduled_vars.items()])
            summary += f"  {action.time_of_day}: {vars_str}\n"
        
        return summary
