# Connected AB — Hierarchical MPC Framework

**Cyber-Resilient Hierarchical Control for a Two-Building Community**

## Overview

This framework implements a hierarchical model predictive control (HMPC) system for coordinating two buildings that share an electrical feeder:

- **Building A** — Commercial HVAC with AHU-VAV-Chiller system (11 decision variables, CasADi/IPOPT solver)
- **Building B** — HVAC with Thermal Energy Storage (1 discrete mode variable, DEAP/GA solver)
- **Aggregator** — Upper-level coordinator that allocates power budgets using hybrid rule-based / log-utility optimisation

The framework maintains feeder capacity constraints and thermal comfort during both normal operation and cyber-attacks (DoS on VAV systems).

## Architecture

```
┌───────────────────────────────────────────┐
│           AGGREGATOR MPC                  │
│  ┌──────────────┐  ┌──────────────────┐   │
│  │  Rule-based   │  │  Log-utility     │   │
│  │  (NORMAL)     │  │  (ATTACK_OPTIM)  │   │
│  └──────┬───────┘  └───────┬──────────┘   │
│         └────────┬─────────┘              │
│          Power budgets + priority θ       │
└──────────┬──────────────────┬─────────────┘
           │                  │
    ┌──────▼──────┐    ┌──────▼──────┐
    │ Building A  │    │ Building B  │
    │ Adaptive    │    │ TES-based   │
    │ MPC Wrapper │    │ MPC Wrapper │
    └─────────────┘    └─────────────┘
```

### Coordination Protocol (Measure → Allocate → Optimize → Actuate)

1. **Measure** — Buildings report status (power, temperatures, flexibility bands, SOC)
2. **Allocate** — Aggregator computes power budgets subject to feeder constraint
3. **Optimize** — Buildings solve local MPC with budget constraints and adaptive weights
4. **Actuate** — Control actions applied, plant advances one timestep

### Multi-Rate Scheduling

| Component   | Timestep | Prediction Horizon |
|-------------|----------|--------------------|
| Building A  | 15 min   | 4 steps (1 hr)     |
| Building B  | 1 hr     | 16 steps (16 hr)   |
| Aggregator  | 15 min   | 4 steps (1 hr)     |

## Folder Structure

```
connected_AB_HMPC/
├── config/
│   └── system_config.yaml       # All parameters in one place
├── communication/
│   ├── __init__.py
│   └── messages.py              # Typed dataclasses for all signals
├── aggregator/
│   ├── __init__.py
│   ├── aggregator_mpc.py        # Hybrid rule-based / log-utility
│   └── attack_manager.py        # Scheduled attack injection/clearance
├── buildings/
│   ├── __init__.py
│   ├── building_a_wrapper.py    # Wraps mpc_a.py (CasADi/IPOPT)
│   └── building_b_wrapper.py    # Wraps mpc_b.py (DEAP/GA)
├── utils/
│   ├── __init__.py
│   └── helpers.py               # Logging, time, comfort, pricing
├── results/                     # Generated output
├── run_hmpc.py                  # Main simulation orchestrator
├── postprocessing.py            # Visualization and metrics
└── README.md
```

## Quick Start (Mock Mode)

```bash
cd connected_AB_HMPC

# Run 1-day simulation with mock plant dynamics
python run_hmpc.py

# Run with overrides
python run_hmpc.py --duration 2 --start-day 200

# Post-process results
python postprocessing.py
```

## Integrating Real MPC Models

Replace `mpc=None, fmu=None` in `run_hmpc.py` with your actual objects:

```python
from buildings.mpc_a import mpc_case as mpc_a_class
from buildings.mpc_b import mpc_case as mpc_b_class

# Initialise your MPC and FMU objects
mpc_a = mpc_a_class(PH=4, CH=4, time=t_start, dt=900, ...)
mpc_b = mpc_b_class(PH=16, ...)

building_a = BuildingAWrapper(cfg, mpc=mpc_a, fmu=fmu_a)
building_b = BuildingBWrapper(cfg, mpc=mpc_b, fmu=fmu_b)
```

### MPC Interface Requirements

**Building A (`mpc_a.py`)**:
- `optimize(fixed_vars=None)` → `(res, solver_status)`
- `get_open_loop_preds(u_seq)` → dict with `"P_pred"` key (optional)
- `w` attribute for weight adjustment

**Building B (`mpc_b.py`)**:
- `optimize(fixed_vars=None)` → `(res, solver_status)`
- `get_open_loop_preds(u_seq)` → dict with `"P_pred"` key (optional)

## Attack Scenarios

Configured in `config/system_config.yaml` under the `attacks` key:

```yaml
attacks:
  - name: "DoS on Core-Zone VAV"
    target: building_a
    type: dos_vav_reinit
    start_time_s: 18360000    # Absolute seconds
    duration_s: 7200          # 2 hours
    affected_zone: core
```

## Key Design Decisions

1. **True hierarchical separation** — Aggregator never touches building-level decision variables
2. **Two-pass scheme** — Buildings report flexibility bands (Pass 1), aggregator allocates, buildings re-optimise (Pass 2)
3. **Adaptive weights** — θ ∈ {0,1,2,3} maps to w_comfort ∈ {1, 10, 100, 1000} and w_cost ∈ {10, 1, 0.1, 0}
4. **Feeder as hard constraint** — Not a soft penalty in objective
5. **Hysteresis** — 4-step delay before returning from ATTACK to NORMAL mode

## Author

Guowen Li — Texas A&M University, 2026
