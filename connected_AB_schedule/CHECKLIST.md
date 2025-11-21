# Implementation Checklist - Schedule-Based Multi-Building Control

**Date:** January 20, 2025  
**Status:** ✅ Framework Complete

---

## ✅ COMPLETED ITEMS

### Core Framework (100% Complete)

- [x] **Data Models** (`schedule/control_models.py`)
  - ControlAction, DailySchedule classes
  - BuildingAVariables, BuildingBVariables
  - AttackEvent, SimulationConfig
  - All data structures implemented

- [x] **Schedule Parser** (`schedule/schedule_parser.py`)
  - YAML configuration parsing
  - Variable validation and bounds checking
  - Attack scenario loading
  - Complete simulation config creation

- [x] **Schedule Manager** (`schedule/schedule_manager.py`)
  - Time-to-schedule mapping
  - Daily repetition logic
  - Control interval management

- [x] **Building Controllers**
  - Base class (`buildings/base_schedule_building.py`)
  - Building A scheduler (`buildings/building_a_scheduler.py`)
  - Building B scheduler (`buildings/building_b_scheduler.py`)
  - Hybrid schedule+MPC control
  - Attack injection handling
  - FMU integration

- [x] **Simulation Engine**
  - Scenario runner (`simulation/scenario_runner.py`)
  - Metrics collector (`simulation/metrics_collector.py`)
  - Multi-rate control coordination
  - Attack event management

- [x] **Analysis Tools**
  - Scenario comparator (`analysis/scenario_comparator.py`)
  - Visualizer with 8 plot types (`analysis/visualizer.py`)
  - Summary metrics calculation
  - Comprehensive comparison reports

- [x] **Utilities**
  - Weather data loader (`utils/data_loader.py`)
  - TOU price data generator

- [x] **Configuration Files**
  - System config (`config/system_config.yaml`)
  - Scenario 1 schedule (`config/schedule_scenario1.yaml`)
  - Scenario 2 schedule (`config/schedule_scenario2.yaml`)
  - Attack scenarios (`config/attack_scenarios.yaml`)

- [x] **Main Script** (`run_schedule_simulation.py`)
  - Command-line interface
  - Two-scenario execution
  - Automatic comparison
  - Results saving

- [x] **Documentation**
  - Comprehensive README (45 KB)
  - Quick-start guide (8 KB)
  - Project summary
  - This checklist

---

## ⏳ USER VALIDATION REQUIRED

### Before First Run

- [ ] **Verify Dymola License Path**
  - Location: `buildings/building_a_scheduler.py` line ~45
  - Location: `buildings/building_b_scheduler.py` line ~60
  - Current: `c:/programdata/dassaultsystemes/dymola/dymola.lic`
  - Action: Update if different on your system

- [ ] **Verify FMU Paths**
  - Building A FMU: `../buildingA_wo_TES/modelica_model/*.fmu`
  - Building B FMU: `../buildingB_w_TES/modelica_model/*.fmu`
  - Action: Confirm relative paths are correct

- [ ] **Verify Weather File Path**
  - Location: `config/system_config.yaml` line ~45
  - Current: `../buildingA_wo_TES/weather_data/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw`
  - Action: Update if file is in different location

- [ ] **Verify Building B Model Paths**
  - Location: `buildings/building_b_scheduler.py` lines ~55-62
  - Neural network models: `../buildingB_w_TES/system_identification/*.h5`
  - Action: Confirm TensorFlow models exist at these paths

### Python Dependencies

- [ ] **Install Required Packages**
  ```bash
  pip install casadi pyfmi tensorflow pandas numpy matplotlib pyyaml pvlib
  ```

- [ ] **Verify Installations**
  ```python
  import casadi
  import pyfmi
  import tensorflow
  import pandas
  import matplotlib
  import yaml
  ```

---

## 🧪 TESTING STEPS

### Step 1: Quick Verification

```bash
cd connected_AB_schedule

# Check all files exist
ls config/*.yaml
ls buildings/*.py
ls schedule/*.py
ls simulation/*.py
ls analysis/*.py
ls utils/*.py
ls *.py
ls *.md
```

Expected: All files listed in PROJECT_SUMMARY.md should be present

### Step 2: Configuration Validation

```bash
# Test YAML parsing (Python check)
python -c "import yaml; yaml.safe_load(open('config/system_config.yaml'))"
python -c "import yaml; yaml.safe_load(open('config/schedule_scenario1.yaml'))"
python -c "import yaml; yaml.safe_load(open('config/schedule_scenario2.yaml'))"
```

Expected: No errors

### Step 3: Module Imports

```bash
# Test all modules can be imported
python -c "from schedule.schedule_parser import ScheduleParser"
python -c "from simulation.scenario_runner import ScenarioRunner"
python -c "from analysis.scenario_comparator import ScenarioComparator"
python -c "from analysis.visualizer import Visualizer"
```

Expected: No import errors

### Step 4: First Simulation Run

```bash
# Run with minimal duration for quick test
python run_schedule_simulation.py --duration 1 --start-day 212

# Check results were created
ls results/
```

Expected:
- Two CSV files (scenario metrics)
- One TXT file (comparison report)
- Multiple PNG files (plots)
- No fatal errors

### Step 5: Validate Outputs

```bash
# Check CSV files
head -n 5 results/scenario1_metrics_*.csv
head -n 5 results/scenario2_metrics_*.csv

# Check comparison report
cat results/comparison_report_*.txt

# View plots
# (Open PNG files in image viewer)
```

Expected:
- CSV files have proper columns
- Comparison report shows metrics
- Plots are generated

---

## 🐛 TROUBLESHOOTING GUIDE

### Common Issues

#### Issue: "ModuleNotFoundError"

**Cause:** Missing Python packages

**Solution:**
```bash
pip install [missing_package]
```

#### Issue: "Configuration file not found"

**Cause:** Running from wrong directory

**Solution:**
```bash
cd /path/to/connected_AB_schedule
pwd  # Should show .../connected_AB_schedule
```

#### Issue: "Weather file not found"

**Cause:** Incorrect path in config

**Solution:**
1. Find actual weather file location: `find .. -name "*.epw"`
2. Update path in `config/system_config.yaml`
3. Or use command line: `--weather /full/path/to/weather.epw`

#### Issue: "FMU file not found"

**Cause:** Incorrect relative paths

**Solution:**
1. Check FMU locations: `find .. -name "*.fmu"`
2. Update paths in:
   - `buildings/building_a_scheduler.py` (~line 50)
   - `buildings/building_b_scheduler.py` (~line 70)

#### Issue: "MPC optimization infeasible"

**Cause:** Conflicting schedule constraints

**Solution:**
1. Review schedule for physical feasibility
2. Don't schedule `bcp=0` during peak cooling
3. Ensure scheduled temps are in bounds
4. Try scenario 1 first (no schedules) to verify MPC works

#### Issue: Slow simulation

**Expected:** 5-10 minutes for 2-day simulation

**If slower:**
- Reduce duration: `--duration 1`
- Check CPU usage
- Ensure not running other heavy processes

---

## 📋 PRE-SUBMISSION CHECKLIST

Before running production simulations:

- [ ] All paths verified and updated
- [ ] Test run completed successfully
- [ ] Results directory populated with outputs
- [ ] Plots generated correctly
- [ ] Comparison metrics look reasonable
- [ ] Custom schedules tested (if applicable)
- [ ] Attack scenarios verified (if enabled)

---

## 🚀 READY TO GO

Once all above items are checked:

```bash
# Run full 2-day simulation
python run_schedule_simulation.py

# Or custom duration
python run_schedule_simulation.py --duration 5

# Or custom scenarios
python run_schedule_simulation.py \
    --scenario2 config/my_custom_schedule.yaml \
    --duration 3
```

---

## 📊 EXPECTED OUTPUTS

After successful run, `results/` should contain:

```
results/
├── scenario1_metrics_20250120_143022.csv      # Timestep data
├── scenario2_metrics_20250120_143022.csv      # Timestep data
├── comparison_report_20250120_143022.txt      # Summary comparison
├── power_comparison.png                        # Power time series
├── feeder_utilization.png                      # Feeder vs capacity
├── zone_temperatures_building_A.png            # 5 zones
├── zone_temperatures_building_B.png            # 5 zones
├── comfort_violations.png                      # Cumulative violations
├── tes_soc.png                                # TES state of charge
└── summary_comparison.png                      # Bar charts
```

**Total:** 2 CSV + 1 TXT + 7 PNG files

---

## 🎯 SUCCESS CRITERIA

Your framework is working correctly if:

1. ✅ No Python errors or exceptions
2. ✅ Both scenarios complete simulation
3. ✅ CSV files contain reasonable data (power > 0, temps 20-30°C)
4. ✅ Comparison report shows metrics differences
5. ✅ Plots are generated and viewable
6. ✅ Feeder utilization < 100% (or expected violations)
7. ✅ TES SOC stays between 0 and 1
8. ✅ Timestamps are sequential

---

## 📞 SUPPORT

If you encounter issues not covered here:

1. Review full documentation:
   - `README.md` - Comprehensive guide
   - `QUICKSTART.md` - Quick setup
   - `PROJECT_SUMMARY.md` - Technical overview

2. Check error messages carefully:
   - File not found → check paths
   - Import error → install packages
   - MPC infeasible → check schedule constraints
   - FMU error → check Dymola license

3. Test incrementally:
   - Start with scenario 1 (no schedules)
   - Try 1-day duration first
   - Add complexity gradually

---

## ✅ FINAL CHECKLIST

Before considering framework ready:

- [ ] All code files created (24 files)
- [ ] All documentation written (3 markdown files)
- [ ] Configuration files complete (4 YAML files)
- [ ] User paths verified
- [ ] Dependencies installed
- [ ] Test run successful
- [ ] Outputs validated

**Once all checked: Framework is READY FOR RESEARCH USE! 🎉**

---

**Implementation Date:** January 20, 2025  
**Framework Version:** 1.0  
**Status:** ✅ Complete and Ready
