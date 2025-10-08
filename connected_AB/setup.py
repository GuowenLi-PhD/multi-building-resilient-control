"""
Cross-platform setup script for Hierarchical Multi-Building Control Framework
Works on Windows, Linux, and macOS

Date: 2025-10-07

Usage:
    python setup.py
"""

import os
import sys
import subprocess
import platform

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)

def print_step(text, status="INFO"):
    """Print step with status"""
    symbols = {
        "OK": "✓",
        "ERROR": "✗",
        "WARNING": "⚠",
        "INFO": "ℹ"
    }
    print(f"  [{symbols.get(status, '•')}] {text}")

def create_directory_structure():
    """Create necessary directories"""
    print_step("Creating directory structure...", "INFO")
    
    directories = [
        "results",
        "results/plots",
        "config",
        "aggregator",
        "buildings",
        "communication",
        "simulation"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print_step("Directory structure created", "OK")

def create_init_files():
    """Create __init__.py files for Python packages"""
    print_step("Creating Python package structure...", "INFO")
    
    packages = ["aggregator", "buildings", "communication", "simulation"]
    
    for package in packages:
        init_file = os.path.join(package, "__init__.py")
        if not os.path.exists(init_file):
            open(init_file, 'w').close()
    
    print_step("Python package structure created", "OK")

def check_python_version():
    """Check if Python version is adequate"""
    print_step(f"Checking Python version...", "INFO")
    
    version = sys.version_info
    print_step(f"Python {version.major}.{version.minor}.{version.micro} detected", "INFO")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print_step("Python 3.8+ required!", "ERROR")
        return False
    
    print_step("Python version OK", "OK")
    return True

def check_config_files():
    """Verify configuration files exist"""
    print_step("Verifying configuration files...", "INFO")
    
    required_configs = [
        "config/system_config.yaml",
        "config/attack_scenarios.yaml"
    ]
    
    missing_files = []
    for config_file in required_configs:
        if not os.path.exists(config_file):
            missing_files.append(config_file)
    
    if missing_files:
        print_step(f"Missing configuration files:", "ERROR")
        for file in missing_files:
            print(f"      - {file}")
        return False
    
    print_step("Configuration files found", "OK")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print_step("Checking Python dependencies...", "INFO")
    
    # Try importing core packages
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'yaml': 'pyyaml',
        'casadi': 'casadi'
    }
    
    missing_packages = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print_step(f"Missing packages: {', '.join(missing_packages)}", "WARNING")
        
        if os.path.exists("requirements.txt"):
            print_step("Installing from requirements.txt...", "INFO")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
                ])
                print_step("Dependencies installed successfully", "OK")
                return True
            except subprocess.CalledProcessError:
                print_step("Failed to install dependencies", "ERROR")
                print_step("Please run manually: pip install -r requirements.txt", "INFO")
                return False
        else:
            print_step("requirements.txt not found", "ERROR")
            print_step("Please install packages manually:", "INFO")
            print(f"      pip install {' '.join(missing_packages)}")
            return False
    else:
        print_step("All core dependencies installed", "OK")
        return True

def check_optional_dependencies():
    """Check optional dependencies"""
    print_step("Checking optional dependencies...", "INFO")
    
    optional_packages = {
        'pyfmi': 'pyfmi (for FMU simulation)',
        'tensorflow': 'tensorflow (for Building B ANN models)',
        'deap': 'deap (for Building B genetic algorithm solver)'
    }
    
    missing_optional = []
    for module_name, description in optional_packages.items():
        try:
            __import__(module_name)
        except ImportError:
            missing_optional.append(description)
    
    if missing_optional:
        print_step("Optional packages not installed:", "WARNING")
        for pkg in missing_optional:
            print(f"      - {pkg}")
        print_step("Framework will work with limited functionality", "INFO")
    else:
        print_step("All optional dependencies installed", "OK")

def setup_environment_variables():
    """Check and provide guidance on environment variables"""
    print_step("Checking environment variables...", "INFO")
    
    # Check Dymola license
    dymola_license = os.environ.get('DYMOLA_RUNTIME_LICENSE')
    
    if dymola_license:
        print_step(f"DYMOLA_RUNTIME_LICENSE: {dymola_license}", "OK")
    else:
        print_step("DYMOLA_RUNTIME_LICENSE not set", "WARNING")
        print_step("If using Dymola FMUs, set this variable:", "INFO")
        
        if platform.system() == "Windows":
            print("      set DYMOLA_RUNTIME_LICENSE=C:\\ProgramData\\DassaultSystemes\\Dymola\\dymola.lic")
        else:
            print("      export DYMOLA_RUNTIME_LICENSE=/path/to/dymola.lic")

def verify_module_files():
    """Check if Python module files exist"""
    print_step("Verifying Python module files...", "INFO")
    
    required_files = [
        "run_hierarchical_mpc.py",
        "analyze_results.py",
        "communication/data_models.py",
        "communication/message_protocol.py",
        "aggregator/aggregator_mpc.py",
        "aggregator/attack_anticipator.py",
        "buildings/base_building.py",
        "buildings/building_a_interface.py",
        "buildings/building_b_interface.py",
        "simulation/coordinator.py",
        "simulation/metrics_collector.py"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print_step(f"Missing {len(missing_files)} module file(s):", "WARNING")
        for file in missing_files[:5]:  # Show first 5
            print(f"      - {file}")
        if len(missing_files) > 5:
            print(f"      ... and {len(missing_files) - 5} more")
        print_step("Please copy all module files from the framework", "INFO")
    else:
        print_step("All module files present", "OK")

def print_next_steps():
    """Print next steps"""
    print_header("Setup Complete!")
    print("\nNext Steps:")
    print("  1. Ensure all Python module files are in place")
    print("  2. Verify Building A and Building B FMU files:")
    print("      - buildingA_wo_TES/modelica_model/*.fmu")
    print("      - buildingB_w_TES/modelica_model/*.fmu")
    print("  3. Run simulation:")
    print("      python run_hierarchical_mpc.py --start-day 212 --duration-days 2")
    print("  4. Analyze results:")
    print("      python analyze_results.py")
    print("\nFor help, see README.md")
    print()

def main():
    """Main setup function"""
    print_header("Hierarchical Multi-Building Control Framework - Setup")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Run setup steps
    success = True
    
    success &= check_python_version()
    
    create_directory_structure()
    create_init_files()
    
    success &= check_config_files()
    success &= install_dependencies()
    
    check_optional_dependencies()
    setup_environment_variables()
    verify_module_files()
    
    if success:
        print_next_steps()
    else:
        print_header("Setup Incomplete")
        print_step("Please address the errors above before proceeding", "ERROR")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)