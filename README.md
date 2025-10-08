# multi-building-resilient-control
Investigation of the Multi-Building Hierarchical MPC Control

## Repository Structure
```bash
multi-building-resilient-control/
├── buildingA_wo_TES/ # Building A standalone MPC
├── buildingB_w_TES/ # Building B standalone MPC
├── connected_AB/ # HIERARCHICAL CONTROL FRAMEWORK
│ ├── aggregator/
│ │ ├── aggregator_mpc.py # Upper-level coordinator MPC
│ │ ├── attack_anticipator.py # Attack prediction/scheduling
│ │ └── power_allocator.py # Dynamic power budget allocation
│ ├── communication/
│ │ ├── message_protocol.py # Communication interface
│ │ ├── data_models.py # Structured data classes
│ │ └── status_monitor.py # Real-time status tracking
│ ├── buildings/
│ │ ├── building_a_interface.py # Wrapper for Building A MPC
│ │ ├── building_b_interface.py # Wrapper for Building B MPC
│ │ └── base_building.py # Abstract base class
│ ├── simulation/
│ │ ├── coordinator.py # Main orchestrator
│ │ ├── scenario_manager.py # Attack scenario definitions
│ │ └── metrics_collector.py # Performance evaluation
│ ├── config/
│ │ ├── system_config.yaml # System-wide parameters
│ │ ├── attack_scenarios.yaml # Predefined attack schedules
│ │ └── feeder_limits.yaml # Grid constraints
│ └── results/ # Generated during simulation
├── run_hierarchical_mpc.py # Main execution script
├── analyze_results.py # Post-processing & visualization
└── README.md # Documentation
```
