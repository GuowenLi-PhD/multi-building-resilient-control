# Quick Start Guide - Schedule-Based Multi-Building Control

## 5-Minute Setup and Run

### Step 1: Prerequisites

Ensure you have:
- Python 3.8+
- Dymola FMU files for Building A and Building B
- Weather data (EPW format)

### Step 2: Check File Structure

Your directory should look like:
```
connected_AB_schedule/
├── config/
│   ├── system_config.yaml
│   ├── schedule_scenario1.yaml
│   ├── schedule_scenario2.yaml
│   └── attack_scenarios.yaml
├── buildings/
├── schedule/
├── simulation/
├── analysis/
├── utils/
└── run_schedule_simulation.py
```

### Step 3: Verify Paths

1. **Weather file path** in `config/system_config.yaml`:
   ```yaml
   simulation:
     weather_file: '../buildingA_wo_TES/weather_data/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw'
   ```

2. **FMU paths** (automatically resolved in building schedulers):
   - Building A: `../buildingA_wo_TES/modelica_model/*.fmu`
   - Building B: `../buildingB_w_TES/modelica_model/*.fmu`

3. **Dymola license** (set in code or environment):
   ```python
   os.environ["DYMOLA_RUNTIME_LICENSE"] = "c:/programdata/dassaultsystemes/dymola/dymola.lic"
   ```

### Step 4: Run Default Simulation

```bash
cd connected_AB_schedule
python run_schedule_simulation.py
```

This will:
- Run Scenario 1 (stand-alone MPC)
- Run Scenario 2 (coordinated control)
- Generate comparison report
- Create visualization plots
- Save everything to `results/`

### Step 5: Check Results

Look in `results/` directory:
```bash
ls results/

# You should see:
# - scenario1_metrics_*.csv
# - scenario2_metrics_*.csv
# - comparison_report_*.txt
# - power_comparison.png
# - feeder_utilization.png
# - zone_temperatures_building_*.png
# - comfort_violations.png
# - tes_soc.png
# - summary_comparison.png
```

---

## Customizing Your Simulation

### Change Simulation Duration

```bash
python run_schedule_simulation.py --duration 5
```

### Change Start Date

```bash
# Start on day 180 (June 29) for 3 days
python run_schedule_simulation.py --start-day 180 --duration 3
```

### Create Custom Schedule

1. Copy a template:
   ```bash
   cp config/schedule_scenario2.yaml config/my_schedule.yaml
   ```

2. Edit `config/my_schedule.yaml`:
   ```yaml
   scenario_name: "My Custom Schedule"
   
   building_a:
     schedule:
       - time: "06:00"
         controls: {bcp: 1, bahu: 1}
       - time: "20:00"
         controls: {bcp: 0, bahu: 0}
   
   building_b:
     schedule:
       - time: "00:00"
         controls: {uMod: -1}
       - time: "14:00"
         controls: {uMod: 1}
   ```

3. Run with custom schedule:
   ```bash
   python run_schedule_simulation.py --scenario2 config/my_schedule.yaml
   ```

---

## Understanding the Output

### 1. Comparison Report (`comparison_report_*.txt`)

```
COMPARISON SUMMARY
================================================================================

📊 ENERGY CONSUMPTION:
  Scenario 1: 450.23 kWh
  Scenario 2: 438.15 kWh
  ✅ Scenario 2 saves 12.08 kWh (2.7%)

🌡️  THERMAL COMFORT:
  Scenario 1: 5.42 °C·h
  Scenario 2: 4.18 °C·h
  ✅ Scenario 2 improves comfort by 1.24 °C·h (22.9%)

⚡ FEEDER UTILIZATION:
  Peak utilization:
    Scenario 1: 94.3%
    Scenario 2: 88.7%
  ✅ Scenario 2 reduces peak by 5.6%
```

### 2. Key Plots

- **power_comparison.png**: Shows if your schedule reduces peak demand
- **feeder_utilization.png**: Highlights any capacity violations
- **comfort_violations.png**: Shows thermal comfort performance
- **tes_soc.png**: Shows how TES is being utilized

---

## Common Issues and Solutions

### Issue 1: "Configuration file not found"

**Solution:**
```bash
# Check you're in the right directory
pwd
# Should show: .../connected_AB_schedule

# Check config files exist
ls config/*.yaml
```

### Issue 2: "Weather file not found"

**Solution:**
- Verify the path in `config/system_config.yaml`
- Or specify directly:
  ```bash
  python run_schedule_simulation.py --weather /path/to/weather.epw
  ```

### Issue 3: "FMU file not found"

**Solution:**
- Check relative paths in building schedulers
- Ensure `buildingA_wo_TES` and `buildingB_w_TES` directories are at correct locations

### Issue 4: "MPC optimization infeasible"

**Cause:** Scheduled constraints are too restrictive

**Solution:**
- Check your schedule for conflicting constraints
- Example: Don't schedule `bcp=0` (chiller off) during peak cooling demand
- Relax comfort bounds in `system_config.yaml` if needed

### Issue 5: Feeder violations

**Cause:** Total power exceeds feeder capacity

**Solution:**
- Adjust schedules to stagger peak loads
- Example: Pre-cool Building A earlier, discharge Building B TES during peaks
- Or increase feeder capacity in config (if realistic)

---

## Next Steps

1. **Experiment with schedules**: Try different coordination strategies
2. **Analyze trade-offs**: Energy vs. comfort vs. feeder stability
3. **Test resilience**: Enable attacks to see how schedules handle disruptions
4. **Compare to HMPC**: Run the same scenario with `connected_AB` framework

---

## Tips for Good Schedules

### Building A Scheduling Best Practices

1. **Pre-cooling**: Turn on chiller and set low Tsa (12-13°C) 2-3 hours before occupancy
2. **Peak reduction**: Raise Tsa (14-15°C) during peak electricity price hours
3. **Night setback**: Turn off systems (bcp=0, bahu=0) during unoccupied hours

### Building B TES Scheduling Best Practices

1. **Overnight charging**: Use uMod=-1 during low-price periods (00:00-06:00)
2. **Peak discharge**: Use uMod=1 during high-price periods (12:00-16:00)
3. **Coordination**: Discharge TES when Building A needs support during attacks

### Attack Scenarios

- **VAV attacks**: Most effective during cooling demand (midday)
- **Duration**: 4-6 hour attacks are realistic
- **Frequency**: 1-2 attacks per simulation period

---

## Example: Creating a Load-Shifting Schedule

Goal: Shift Building A's cooling load to off-peak hours

```yaml
building_a:
  schedule:
    # Aggressive pre-cooling (04:00-08:00)
    - time: "04:00"
      controls: {bcp: 1, bahu: 1, Tsa: 11.5}
    
    # Normal operation (08:00-12:00)
    - time: "08:00"
      controls: {bcp: 1, bahu: 1, Tsa: 13.0}
    
    # Reduce cooling during peak (12:00-16:00)
    - time: "12:00"
      controls: {bcp: 1, bahu: 1, Tsa: 15.0}
    
    # Resume normal (16:00-20:00)
    - time: "16:00"
      controls: {bcp: 1, bahu: 1, Tsa: 13.0}
    
    # Off during night (20:00-04:00)
    - time: "20:00"
      controls: {bcp: 0, bahu: 0}

building_b:
  schedule:
    # Charge TES overnight (00:00-06:00)
    - time: "00:00"
      controls: {uMod: -1}
    
    # Coast in morning (06:00-12:00)
    - time: "06:00"
      controls: {uMod: 0}
    
    # Discharge to support Building A during peak (12:00-16:00)
    - time: "12:00"
      controls: {uMod: 1}
    
    # Back to chiller-only (16:00-00:00)
    - time: "16:00"
      controls: {uMod: 2}
```

Test this schedule:
```bash
python run_schedule_simulation.py --scenario2 my_load_shifting_schedule.yaml
```

---

## Getting Help

If you encounter issues:

1. Check the full README.md for detailed documentation
2. Review the log output for specific error messages
3. Verify all paths and configurations
4. Start with default scenarios to ensure basic setup works

---

**Ready to optimize your building schedules! 🏢⚡**
