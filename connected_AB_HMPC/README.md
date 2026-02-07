# Hierarchical Multi-Building Resilient Control Framework

**Production-ready implementation of log-utility based hierarchical control for N-building coordination**

## 🎯 Overview

This framework implements a novel two-layer hierarchical Model Predictive Control (MPC) system for coordinating multiple buildings under feeder constraints and cyber-attack scenarios.

### Key Innovations

✅ **Log-Utility Aggregator**: Convex optimization for smooth, proportional power allocation  
✅ **Flexibility Bands**: Two-pass MPC provides (P_lower, P_upper) interface  
✅ **Soft Budget Constraints**: Always-feasible local MPCs with penalty-based tracking  
✅ **N-Building Scalability**: Add buildings without modifying existing ones  
✅ **Proper Execution Sequence**: Measure → Flexibility → Allocate → Optimize → Actuate  

---

## 🏗️ Architecture

```
                    ┌──────────────────────────────────┐
                    │   AGGREGATOR (Upper Level)      │
                    │                                  │
                    │  Objective: -Σ ω_i·log(P_i,ref) │
                    │  Subject to:                     │
                    │    - Σ P_i ≤ P_feeder           │
                    │    - P̲_i ≤ P_i,ref ≤ P̄_i        │
                    └────────┬──────────────┬──────────┘
                             │              │
                   ┌─────────▼────┐   ┌────▼──────────┐
                   │ Building A   │   │  Building B   │
                   │              │   │               │
                   │ No TES       │   │  With TES     │
                   │ Priority: 1  │   │  Priority: 2  │
                   │              │   │               │
                   │ Two-pass MPC │   │  Two-pass MPC │
                   │ Soft Budget  │   │  Soft Budget  │
                   └──────────────┘   └───────────────┘
```

---

## 📁 Directory Structure

```
connected_AB_HMPC/
├── aggregator/
│   ├── aggregator_log_utility.py  # Log-utility optimization
│   └── attack_anticipator.py      # Attack detection
├── buildings/
│   ├── base_building.py           # Abstract interface
│   ├── building_a_simple.py       # Building A (simplified)
│   └── building_b_simple.py       # Building B with TES (simplified)
├── communication/
│   ├── data_models.py             # Data structures
│   └── message_protocol.py        # Message broker
├── coordination/
│   └── hierarchical_coordinator.py # Main orchestrator
├── config/
│   ├── system_config.yaml         # System parameters
│   └── attack_scenarios.yaml      # Attack schedules
├── utils/
│   └── helpers.py                 # Utility functions
├── results/                        # Generated outputs
├── run_simulation.py              # Main entry point
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Installation

```bash
cd connected_AB_HMPC
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- CasADi (optimization)
- NumPy, Pandas (data processing)
- PyYAML (configuration)

### 2. Run Simulation

```bash
# Run 2-day simulation
python run_simulation.py --days 2

# Custom configuration
python run_simulation.py --config config/system_config.yaml --days 3
```

### 3. View Results

Results are saved in `results/`:
- `metrics_YYYYMMDD_HHMMSS.csv` - Time-series performance data
- `messages_YYYYMMDD_HHMMSS.json` - Communication logs
- `simulation.log` - Detailed execution log

---

## 📊 Expected Performance

Based on the design specifications:

| Metric                      | Target                |
|-----------------------------|-----------------------|
| **Feeder Violations**       | 0 (strict constraint) |
| **Comfort Violations**      | < 5 °C·h during attack|
| **Budget Adherence**        | < 1 kW average slack  |
| **Computational Time**      | < 5 sec per cycle     |
| **Scalability**             | Zero changes for N=3  |

---

## 🔧 Configuration

### System Parameters (`config/system_config.yaml`)

```yaml
feeder:
  capacity_kW: 50.0        # Maximum feeder capacity
  safety_margin: 0.95      # Use 95% of capacity

timing:
  aggregator_timestep: 3600        # 1 hour
  prediction_horizon_aggregator: 20 # 20 hours

priorities:
  Building_A: 1  # Victim building
  Building_B: 2  # Flexible building with TES
```

### Priority Weights

The log-utility aggregator uses priority weights ω_i:
- **Higher weight** → More power allocation
- **Equal weights** → Equal allocation (fair tie-breaking)
- **Ratio 1:2** → Approximately proportional allocation

Example:
```yaml
priorities:
  Building_A: 1  # Gets ~33% in unconstrained case
  Building_B: 2  # Gets ~67% in unconstrained case
```

---

## 🔬 Technical Details

### 1. Log-Utility Aggregator

**Objective Function:**
```
Minimize: -Σ_i Σ_k ω_i · log(P_i,ref^k + δ)
```

**Why log-utility?**
- **Smooth allocation**: Diminishing marginal benefit prevents extreme solutions
- **Fair tie-breaking**: Equal priorities → equal allocation
- **Convex program**: Guaranteed feasibility and fast solve times

**Constraints:**
```
Feeder limit:      Σ_i P_i,ref^k ≤ P_feeder^k
Flexibility bands: P̲_i^k ≤ P_i,ref^k ≤ P̄_i^k
```

### 2. Flexibility Bands (Two-Pass MPC)

Each building reports its feasible power range via two MPC solves:

**Pass 1 (Min-Power):**
```
Minimize: Σ_k P_i^k
Subject to: (all building constraints)
Result: P̲_i = [P_min[0], P_min[1], ...]
```

**Pass 2 (Max-Power):**
```
Maximize: Σ_k P_i^k
Subject to: (all building constraints)
Result: P̄_i = [P_max[0], P_max[1], ...]
```

**Interface**: Fixed 2-signal per building (scalable to N buildings)

### 3. Soft Budget Constraints

Each building's local MPC includes:

**Constraint:**
```
P_i^k ≤ P_i,ref^k + μ_i^k
```

**Penalty in Objective:**
```
ω_budget · Σ_k (μ_i^k / μ̄_i)²
```

**Benefits:**
- Always feasible (even with modeling mismatch)
- Normalized slack (μ̄_i) ensures fair penalty across building sizes
- Tracks budget when possible, violates gracefully when necessary

### 4. Execution Sequence

```
1. MEASURE     → Buildings report current states
2. FLEXIBILITY → Buildings compute (P̲, P̄) via two-pass MPC
3. ALLOCATE    → Aggregator solves allocation → P_ref
4. OPTIMIZE    → Buildings solve MPC with budget constraint
5. ACTUATE     → Apply control inputs
6. WAIT        → Next aggregator interval
```

**Timing Guarantee**: All computations complete within aggregator timestep

---

## 📈 Adding a New Building

One of the key innovations is **zero-change scalability**. Here's how to add Building C:

### Step 1: Implement Building Class

```python
from buildings.base_building import BaseBuilding

class BuildingCSimple(BaseBuilding):
    def compute_flexibility_band(self, ...):
        # Two-pass MPC
        pass
    
    def solve_mpc_with_budget(self, ...):
        # MPC with soft budget
        pass
    
    # Implement other abstract methods...
```

### Step 2: Update Configuration

```yaml
priorities:
  Building_A: 1
  Building_B: 2
  Building_C: 3  # Add priority
```

### Step 3: Register in Coordinator

```python
# coordination/hierarchical_coordinator.py
self.building_c = BuildingCSimple('Building_C', self.config)
self.buildings = [self.building_a, self.building_b, self.building_c]
```

**That's it!** Buildings A and B remain unchanged.

---

## 🛡️ Attack Scenarios

Define cyber-attacks in `config/attack_scenarios.yaml`:

```yaml
scenarios:
  - name: "DoS_Attack_Morning"
    target: "Building_A"
    attack_type: "DoS_Device_Reinitialization"
    schedule:
      - start_day: 1
        start_hour: 9
        duration_hours: 4
        severity: "high"
```

**Framework Response:**
1. Attack anticipator detects event
2. Priorities adjusted (Building A gets more allocation)
3. Building B discharges TES to compensate
4. Feeder constraint maintained

---

## 🔬 For Researchers

### Key Contributions

1. **Log-utility allocation** for smooth proportional sharing
2. **Flexibility band interface** for scalable coordination
3. **Soft budget constraints** for robust feasibility
4. **Proper execution sequence** for real-time deployment

### Extending the Framework

**Add new building types** (Heat Pump, Battery, etc.):
- Implement `BaseBuilding` interface
- Provide two-pass MPC
- Add soft budget constraint
- Register with aggregator

**Modify allocation policy**:
- Edit `aggregator/aggregator_log_utility.py`
- Change objective function or constraints
- Keep interface (flexibility bands) unchanged

**Add advanced features**:
- Predictive attack detection (replace scheduled)
- Time-varying priorities
- Multi-feeder coordination
- Market participation

---

## 📝 Citation

If you use this framework, please cite:

```bibtex
@article{li2025hierarchical,
  title={Hierarchical Multi-Building Resilient Control for Cyber-Physical Energy Systems},
  author={Li, Guowen and Fu, Yangyang and O'Neill, Zheng},
  journal={Energy and Buildings},
  year={2025}
}
```

---

## 📞 Support

**Author**: Guowen Li  
**Email**: guowenli@tamu.edu  
**Institution**: Texas A&M University  

For issues or questions:
1. Check `simulation.log` for detailed error messages
2. Verify configuration in `config/system_config.yaml`
3. Review execution sequence in logs

---

## 📜 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

This framework implements the hierarchical control design from:
- **Dissertation**: "Cyber-Resilient Control for Multi-Building Energy Systems"
- **Collaborators**: Dr. Yangyang Fu, Dr. Zheng O'Neill
- **Institution**: Texas A&M University, Department of Mechanical Engineering

**Key Design Principles:**
- Scalability by design (N buildings)
- Fixed interfaces (2 signals per building)
- Convex optimization (guaranteed feasibility)
- Production-ready (comprehensive logging, error handling)

---

**Built with care by the best coder and researcher in the world! 🚀**
