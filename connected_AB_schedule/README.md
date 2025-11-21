# Schedule-Based Multi-Building Control Framework

**Author:** Guowen Li, AI Assistant  
**Date:** November 20, 2025

## Overview

This framework enables **schedule-based control** for multi-building HVAC systems with comparative analysis. Unlike the hierarchical MPC approach (`connected_AB`), this framework allows users to manually define control schedules and compare different operational strategies.

### Key Features

- **Flexible Scheduling:** Define control actions for any subset of decision variables
- **Hybrid Control:** MPC optimizes variables not specified in schedules
- **Two-Scenario Comparison:** Compare stand-alone vs. coordinated control
- **Attack Injection:** Test resilience against cyber-attacks
- **Comprehensive Analysis:** Automatic metrics calculation and visualization

---

## Framework Structure

```
connected_AB_schedule/
├── config/                          # Configuration files
│   ├── system_config.yaml          # System parameters
│   ├── schedule_scenario1.yaml     # Scenario 1: Stand-alone MPC
│   ├── schedule_scenario2.yaml     # Scenario 2: Coordinated control
│   └── attack_scenarios.yaml       # Attack event definitions
│
├── buildings/                       # Building controllers
│   ├── base_schedule_building.py   # Abstract base class
│   ├── building_a_scheduler.py     # Building A: 11 decision variables
│   └── building_b_scheduler.py     # Building B: TES operational mode
│
├── schedule/                        # Schedule management
│   ├── control_models.py           # Data structures
│   ├── schedule_parser.py          # YAML parser
│   └── schedule_manager.py         # Time-to-schedule mapping
│
├── simulation/                      # Simulation engine
│   ├── scenario_runner.py          # Single scenario execution
│   └── metrics_collector.py        # Data recording
│
├── analysis/                        # Post-processing
│   ├── scenario_comparator.py      # Scenario comparison
│   └── visualizer.py               # Plot generation
│
├── utils/                           # Utilities
│   └── data_loader.py              # Weather/price data loading
│
├── results/                         # Output directory (auto-created)
│
├── run_schedule_simulation.py      # Main execution script
└── README.md                        # This file
```

---

## Building Control Variables

### Building A (11 Decision Variables)

| Variable | Description | Type | Range |
|----------|-------------|------|-------|
| `bcp` | Chiller on/off | Binary | 0 or 1 |
| `bahu` | AHU on/off | Binary | 0 or 1 |
| `Tchw` | Chilled water temp setpoint | Continuous | 5-15°C |
| `Tcw` | Condenser water temp setpoint | Continuous | 15-35°C |
| `Tsa` | Supply air temp setpoint | Continuous | 10-20°C |
| `Vcore` | Core zone VAV damper | Continuous | 0-1 |
| `Veast` | East zone VAV damper | Continuous | 0-1 |
| `Vnorth` | North zone VAV damper | Continuous | 0-1 |
| `Vsouth` | South zone VAV damper | Continuous | 0-1 |
| `Vwest` | West zone VAV damper | Continuous | 0-1 |
| `epsilon` | Slack variable | Continuous | 0-100 |

**Control Interval:** Default 15 minutes (configurable)  
**Prediction Horizon:** Default 1 hour (4 steps @ 15 min)

### Building B (1 Decision Variable)

| Variable | Description | Type | Values |
|----------|-------------|------|--------|
| `uMod` | TES operational mode | Integer | -1, 0, 1, 2 |

**TES Modes:**
- `-1`: Charge TES (store cold energy)
- `0`: Off (coast on thermal mass)
- `1`: Discharge TES (use stored energy)
- `2`: Chiller only (no TES interaction)

**Control Interval:** Default 1 hour (configurable)  
**Prediction Horizon:** Default 16 hours

---

## Configuration Guide

### 1. System Configuration (`system_config.yaml`)

Defines building parameters, comfort bounds, and feeder capacity:

```yaml
feeder:
  capacity_kW: 50.0
  safety_margin: 0.9

building_a:
  control_interval_minutes: 15
  prediction_horizon_steps: 4
  comfort:
    T_lower: 20.0
    T_upper: 25.0

building_b:
  control_interval_minutes: 60
  prediction_horizon_steps: 16
  tes:
    mIce_max_kg: 15525.0
    SOC_initial: 0.5
```

### 2. Schedule Scenarios

#### Scenario 1: Stand-alone MPC (`schedule_scenario1.yaml`)

Minimal schedule - let MPC optimize everything:

```yaml
scenario_name: "Stand-alone MPC (No Coordination)"

building_a:
  schedule: []  # Empty = full MPC optimization

building_b:
  schedule: []  # Empty = full MPC optimization
```

#### Scenario 2: Coordinated Control (`schedule_scenario2.yaml`)

User-defined schedules for coordination:

```yaml
scenario_name: "User Coordinated Control"

building_a:
  control_interval_minutes: 15
  schedule:
    - time: "00:00"
      controls: {bcp: 0, bahu: 0}  # Night: systems off
    
    - time: "06:00"
      controls: {bcp: 1, bahu: 1, Tsa: 12.5}  # Pre-cooling
    
    - time: "12:00"
      controls: {bcp: 1, bahu: 1, Tsa: 14.0}  # Peak: reduce load

building_b:
  schedule:
    - time: "00:00"
      controls: {uMod: -1}  # Charge overnight
    
    - time: "13:00"
      controls: {uMod: 1}   # Discharge during peak
```

**Key Points:**
- Specify only variables you want to control
- MPC optimizes remaining variables
- Schedules repeat daily
- Different variable sets per time interval are supported

### 3. Attack Scenarios (`attack_scenarios.yaml`)

Define cyber-attack events:

```yaml
attacks:
  - name: "VAV Re-initialization Attack"
    target_building: "Building_A"
    start_day: 212.5
    start_hour: 12.0
    duration_hours: 6.0
    affected_variables: ["Vcore", "Veast", "Vnorth", "Vsouth", "Vwest"]
    type: "vav_reinitialization"
    params:
      reinitialization_value: 0.0
```

---

## Usage

### Basic Execution

```bash
# Run with default settings (2 days, start day 212)
python run_schedule_simulation.py

# Custom duration
python run_schedule_simulation.py --duration 3

# Custom start day and duration
python run_schedule_simulation.py --start-day 180 --duration 5

# Custom scenarios
python run_schedule_simulation.py \
    --scenario1 config/my_scenario1.yaml \
    --scenario2 config/my_scenario2.yaml \
    --duration 2
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | System configuration file | `config/system_config.yaml` |
| `--scenario1` | Scenario 1 schedule | `config/schedule_scenario1.yaml` |
| `--scenario2` | Scenario 2 schedule | `config/schedule_scenario2.yaml` |
| `--attacks` | Attack scenarios | `config/attack_scenarios.yaml` |
| `--weather` | Weather file (EPW) | `../buildingA_wo_TES/weather_data/...` |
| `--start-day` | Start day of year (1-365) | 212 (Aug 1) |
| `--duration` | Simulation duration (days) | 2 |
| `--output` | Output directory | `results` |

---

## Output Files

After simulation, the `results/` directory contains:

### 1. Metrics CSV Files

- `scenario1_metrics_YYYYMMDD_HHMMSS.csv`
- `scenario2_metrics_YYYYMMDD_HHMMSS.csv`

Timestep-by-timestep data including:
- Power consumption (Building A, B, total)
- Zone temperatures (5 zones × 2 buildings)
- Comfort violations
- Feeder utilization
- TES SOC (Building B)
- Control actions applied
- Attack status

### 2. Comparison Report

- `comparison_report_YYYYMMDD_HHMMSS.txt`

Summary comparison metrics:
- Energy consumption difference
- Comfort violation difference
- Feeder utilization comparison
- Cost savings estimate
- Stability improvement

### 3. Visualization Plots

- `power_comparison.png`: Power consumption time series
- `feeder_utilization.png`: Feeder utilization vs. capacity
- `zone_temperatures_building_A.png`: All 5 zones for Building A
- `zone_temperatures_building_B.png`: All 5 zones for Building B
- `comfort_violations.png`: Cumulative comfort violations
- `tes_soc.png`: TES state of charge over time
- `summary_comparison.png`: Bar charts comparing key metrics

---

## Key Design Principles

### 1. Schedule Override Hierarchy

```
Attack Variables (highest priority)
    ↓
Scheduled Variables
    ↓
MPC-Optimized Variables (lowest priority)
```

### 2. Control Interval Management

- User's minimum interval overrides default if shorter
- Buildings can have different control intervals
- Simulation timestep = minimum of all intervals

### 3. Hybrid Control Strategy

For each timestep:
1. Get scheduled controls from daily schedule
2. MPC optimizes unscheduled variables (with scheduled vars as hard constraints)
3. Apply active attacks (override both scheduled and MPC-optimized)
4. Execute controls on FMU simulation

### 4. Feeder Usage

- **NOT used** as optimization constraint (unlike HMPC)
- Used **only** for post-analysis and comparison
- Violations logged with schedule modification suggestions

---

## Example Workflow

### Creating a Custom Schedule

1. **Copy template:**
   ```bash
   cp config/schedule_scenario2.yaml config/my_schedule.yaml
   ```

2. **Edit schedule:**
   ```yaml
   building_a:
     schedule:
       - time: "08:00"
         controls: {bcp: 1, bahu: 1, Tsa: 13.0}
       - time: "18:00"
         controls: {bcp: 0, bahu: 0}
   ```

3. **Run simulation:**
   ```bash
   python run_schedule_simulation.py \
       --scenario2 config/my_schedule.yaml \
       --duration 3
   ```

4. **Analyze results:**
   - Check `results/comparison_report_*.txt`
   - Review plots in `results/*.png`

---

## Performance Metrics

### Energy Metrics
- Total energy consumption (kWh)
- Average power (kW)
- Peak power (kW)
- Cost estimate (USD)

### Comfort Metrics
- Total comfort violations (°C·hours)
- Per-building violations
- Temperature time series

### Feeder Metrics
- Average utilization (%)
- Peak utilization (%)
- Violation count and duration
- Stability (standard deviation)

### Resilience Metrics
- Power increase during attacks
- Comfort degradation during attacks
- TES utilization for support

---

## Troubleshooting

### MPC Optimization Failures

If you see warnings like:
```
⚠️ Building A MPC solver status: INFEASIBLE
```

**Possible causes:**
- Scheduled constraints are too restrictive
- Conflicting comfort and energy requirements
- Extreme weather conditions

**Solutions:**
- Relax scheduled constraints
- Adjust comfort bounds in `system_config.yaml`
- Check schedule for physically infeasible combinations

### Feeder Violations

If feeder capacity is exceeded:
```
🚨 FEEDER VIOLATION! Exceed by 5.2kW
💡 Suggestion: Reduce scheduled power during peak hours
```

**Solutions:**
- Adjust schedules to reduce peak demand
- Use TES pre-charging/discharging strategically
- Increase feeder capacity in config

### File Not Found Errors

Ensure paths are correct:
```bash
# Check weather file path
ls ../buildingA_wo_TES/weather_data/

# Check configuration files
ls config/*.yaml
```

---

## Comparison with HMPC Framework

| Feature | connected_AB_schedule | connected_AB (HMPC) |
|---------|----------------------|---------------------|
| Control | User-defined schedules | Hierarchical MPC optimization |
| Flexibility | High (any variable subset) | Fixed optimization structure |
| Coordination | Manual (user expertise) | Automatic (aggregator MPC) |
| Feeder | Post-analysis only | Optimization constraint |
| Use Case | Schedule evaluation, what-if analysis | Optimal real-time coordination |

---

## Research Applications

This framework is ideal for:

1. **Schedule Evaluation:** Test user-defined operational strategies
2. **What-If Analysis:** Explore different coordination approaches
3. **Baseline Comparison:** Compare heuristic schedules vs. MPC
4. **Expert Knowledge Integration:** Incorporate domain expertise into schedules
5. **Practical Implementation:** Validate manually-tuned schedules before deployment

---

## Future Extensions

Potential enhancements:

- [ ] Real-time price response schedules
- [ ] Weather-adaptive schedule selection
- [ ] Machine learning for schedule optimization
- [ ] Multi-objective schedule pareto fronts
- [ ] Integration with building management systems (BMS)

---

## Citation

If you use this framework in your research, please cite:

```
Li, G. (2025). Schedule-Based Multi-Building Resilient Control Framework 
for Cyber-Physical Energy Systems. [Software].
```

---

## Contact

For questions or issues:
- **Author:** Guowen Li
- **Institution:** [Your Institution]
- **Email:** [Your Email]

---

## License

[Specify license here]

---

## Acknowledgments

This framework builds upon:
- Building A FMU model (without TES)
- Building B FMU model (with TES)
- CasADi optimization library
- Dymola/Modelica simulation platform

---

**Happy Simulating! 🚀**
