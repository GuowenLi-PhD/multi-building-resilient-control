# Import necessary modules from the provided scripts
from run_mpc_dnn import run_mpc  # Assuming the main function for running MPC is named 'run_mpc'
from plot_mpc_measurements import plot_results  # Assuming the main function for plotting is 'plot_results'
import os
#os.environ["MKL_THREADING_LAYER"] = "GNU"  # For NumPy with MKL to disable OpenMP features explicitly
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Allow multiple OpenMP runtimes; Avoid the OpenMP DLL conflict
os.environ["OMP_NUM_THREADS"] = "1"          # Restrict to one thread


# Define a function to run the tests for different PH values
def run_tests(PH_values, mIce_max, SOC_ini):
    # Loop through each test scenario based on PH
    for i, PH in enumerate(PH_values):
        print(f"Running test scenario {i + 1}: PH={PH}, mIce_max={mIce_max}, SOC_ini={SOC_ini}")

        # Call the MPC function with the current hyperparameters
        run_mpc(PH=PH, mIce_max=mIce_max, SOC_ini=SOC_ini)
        
        print(f"Post-processing and plotting {i + 1}\n")
        # After running MPC, perform the post-processing and plotting
        plot_results(PH=PH)  # Assuming PH is the parameter passed to 'plot_results'

        print(f"Completed test scenario {i + 1}\n")

if __name__ == "__main__":
    # Define the list of PH values you want to test
    #PH_values = [2, 4, 8, 10, 12, 16, 20, 24, 32, 40, 48]
    PH_values = [2, ]
    #PH_values = [1,] # used for calculate baseline RBC control (set lower bound of uMod = -1 during unoccupied to be storage-priority control)

    # Set the constant values for mIce_max and SOC_ini
    mIce_max = 3105*5.  # # 3105 L water per tank from NIST testbed, Trane Calmac Model 1082A
    #mIce_max = 1550. # Trane Calmac Model 1045C
    SOC_ini = 0.5  # initial State-of-Charge

    # Run the tests with the defined PH values and constant parameters
    run_tests(PH_values, mIce_max, SOC_ini)
