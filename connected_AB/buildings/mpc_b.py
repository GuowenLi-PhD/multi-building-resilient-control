'''
This script is used to formulate the MPC using DNN model.

Author: Guowen Li, Yangyang Fu
Email: guowenli@tamu.edu, yangyang.fu@tamu.edu 
Revisions:
    05/09/2024: CasADi optimization framework with diffential neural network models
'''
import casadi as ca
import gurobipy
import numpy as np
# import math
#from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from deap import base, creator, tools, algorithms
import random
import os
# os.environ['GUROBI_HOME'] = r'C:\gurobi1103\win64'
# os.environ['GUROBI_VERSION'] = r'65'
# os.environ['GRB_LICENSE_FILE'] = r'C:\gurobi1103\gurobi.lic'
# os.environ['PATH'] += r';C:\gurobi1103\win64\bin'
# os.environ['CASADIPATH'] = r'C:\ProgramData\anaconda3\envs\mpc-py38\Lib\site-packages\casadi'



class mpc_case():
    def __init__(self,PH,CH,time,dt,measurement,states,predictor):
        self.PH = PH # prediction horizon
        self.CH = CH # control horizon
        self.dt = dt # time step
        self.time = time # current time index
        self.measurement = measurement # measurement at current time step, dictionary
        self.predictor = predictor # price and outdoor air temperature for the future horizons
        self.states = states # dictionary
        self.number_zones = 5 # 5 zones
        self.occ_start = 6 # occupancy starts
        self.occ_end = 19 # occupancy ends
        self.T_upper = np.array([30.0 for i in range(24)]) # zone temperature bounds - need check with the high-fidelty model
        self.T_upper[self.occ_start:self.occ_end] = 25.0
        self.T_lower = np.array([18.0 for i in range(24)])
        self.T_lower[self.occ_start:self.occ_end] = 20.0

        # initialize optimiztion
        self.optimum = {}
        self.u_lb = [-1] # lower bounds
        # Reminder! Temperature: Celsius is used in prediction, but K is used in the Modelica model
        self.u_ub = [2]
        self.u_start = [-1]*self.PH # total number of control variables
        self.u_ini_fix = [-1]*self.PH # total number of control variables
        self.number_inputs = 1 # [uMod]
        self.w = [1., 1000., 1000.]  # weights in objective function: : Minimize energy cost + temperature violation + SOC penalty
        self.x_opt_0 = self.u_ub   # initialization of previous control actions

        # initialize zone arx model auto error term
        self._autoerror = {'core': 0,
                           'east': 0,
                           'north': 0,
                           'south': 0,
                           'west': 0}

    def optimize(self):
        occupied_ph = [0]*self.PH
        for i in range(self.PH):
            # future time        
            t = int( ( (self.time+i*self.dt) % 86400)/3600) # hour index 0~23           
            if t>=self.occ_start and t<self.occ_end:
                occupied_ph[i] = 1
        #print("occupied status with PH:",occupied_ph)

        ### get states and predictions at current time step
        # get predictions at current time step
        To_pred_ph = self.predictor['Toa'] # predicted outdoor air dry bulb temperature
        GHI_pred_ph = self.predictor['GHI'] # predicted relative humidity 
        price_ph = self.predictor['price'] # predicted electricity price (TOU)


        # get historical temperaure measurements
        Tz_core_his_meas = np.array(self.states['Tz_core_his_meas'][:]) # historical zone temperatur states for zone temperture prediction model, [t-1, t-2, t-3, t-4], Unit: Celsius
        Tz_east_his_meas = np.array(self.states['Tz_east_his_meas'][:])
        Tz_north_his_meas = np.array(self.states['Tz_north_his_meas'][:])
        Tz_south_his_meas = np.array(self.states['Tz_south_his_meas'][:])
        Tz_west_his_meas = np.array(self.states['Tz_west_his_meas'][:])
        Tz_ave_his_meas = np.array(self.states['Tz_ave_his_meas'][:])
        To_his_meas = np.array(self.states['To_his_meas'][:]) # historical outdoor air temperatur dry bulb states for zone temperture prediction model
        GHI_his_meas = np.array(self.states['GHI_his_meas'][:]) # historical global horizontal solar irradiation [W/m2]
        SOC_his_meas = np.array(self.states['SOC_his_meas'][:]) # historical state-of-charge
        P_his_meas = np.array(self.states['P_his_meas'][:]) # historical total power [W]
        #("\nTo_his_meas:",To_his_meas)

        # get historical predicted zone temperatures
        Tz_core_his_pred = np.array(self.states['Tz_core_his_pred'])
        Tz_east_his_pred = np.array(self.states['Tz_east_his_pred'])
        Tz_north_his_pred = np.array(self.states['Tz_north_his_pred'])
        Tz_south_his_pred = np.array(self.states['Tz_south_his_pred'])
        Tz_west_his_pred = np.array(self.states['Tz_west_his_pred'])

        # get previous control actions
        u_prev = self.x_opt_0

        ### formulate optimization problem for MPC
        ## declare symbolic variabels
        U = ca.MX.sym("U", self.number_inputs*self.PH)

        # # zone temperature bounds - need check with the high-fidelty model
        # T_upper = np.array([30.0 for i in range(24)])
        # T_upper[self.occ_start:self.occ_end] = 26.0
        # T_lower = np.array([18.0 for i in range(24)])
        # T_lower[self.occ_start:self.occ_end] = 22.0

        ## define objective
        # Get auto error from historical predictions and measurements: autocorrection term happens at the beginning of the PH. Calculation of such term inside PH is not efficient.
        n_Tz_his = len(Tz_core_his_meas)
        autoerror_core = 0
        autoerror_east = 0
        autoerror_north = 0
        autoerror_south = 0
        autoerror_west = 0

        for k in range(n_Tz_his):
            autoerror_core += (Tz_core_his_meas[k]-Tz_core_his_pred[k])/n_Tz_his
            autoerror_east += (Tz_east_his_meas[k]-Tz_east_his_pred[k])/n_Tz_his
            autoerror_north += (Tz_north_his_meas[k]-Tz_north_his_pred[k])/n_Tz_his
            autoerror_south += (Tz_south_his_meas[k]-Tz_south_his_pred[k])/n_Tz_his
            autoerror_west += (Tz_west_his_meas[k]-Tz_west_his_pred[k])/n_Tz_his

        # save for future use    
        self._autoerror = {'core':autoerror_core,
                        'east':autoerror_east,
                        'north':autoerror_north,
                        'south':autoerror_south,
                        'west':autoerror_west}
        
        # initialize outputs and intemediate variables before main loop
        Tz_core_pred_ph = [0.]*self.PH
        Tz_east_pred_ph = [0.]*self.PH
        Tz_north_pred_ph = [0.]*self.PH
        Tz_south_pred_ph = [0.]*self.PH
        Tz_west_pred_ph = [0.]*self.PH
        P_pred_ph = [0.]*self.PH
        SOC_pred_ph = [0.]*self.PH
        fval = []

        Tz_core_his_meas_k = [Tz for Tz in Tz_core_his_meas]
        Tz_east_his_meas_k = [Tz for Tz in Tz_east_his_meas]
        Tz_north_his_meas_k = [Tz for Tz in Tz_north_his_meas]
        Tz_south_his_meas_k = [Tz for Tz in Tz_south_his_meas]
        Tz_west_his_meas_k = [Tz for Tz in Tz_west_his_meas]
        Tz_ave_his_meas_k = [Tz for Tz in Tz_ave_his_meas]
        To_his_meas_k = [Tz for Tz in To_his_meas]
        GHI_his_meas_k = [Tz for Tz in GHI_his_meas]
        SOC_his_meas_k = [Tz for Tz in SOC_his_meas]
        P_his_meas_k = [Tz for Tz in P_his_meas]

        # load dnn models
        # Check if model paths were provided
        if hasattr(self, 'model_paths'):
            core_model_path = self.model_paths['core']
            east_model_path = self.model_paths['east']
            north_model_path = self.model_paths['north']
            south_model_path = self.model_paths['south']
            west_model_path = self.model_paths['west']
            SOC_model_path = self.model_paths['SOC']
            power_model_path = self.model_paths['power']
        else:
            core_model_path="system_identification/dnn_model_core_temperature.h5"
            east_model_path="system_identification/dnn_model_east_temperature.h5"
            north_model_path="system_identification/dnn_model_north_temperature.h5"
            south_model_path="system_identification/dnn_model_south_temperature.h5"
            west_model_path="system_identification/dnn_model_west_temperature.h5"
            SOC_model_path="system_identification/dnn_SOC_model.h5"
            power_model_path="system_identification/dnn_power_model.h5"
        SOC_dnn = self.SOC_dnn_tensorflow(SOC_model_path)
        power_dnn = self.power_dnn_tensorflow(power_model_path)
        core_temp_dnn = self.temp_dnn_tensorflow(core_model_path)
        east_temp_dnn = self.temp_dnn_tensorflow(east_model_path)
        north_temp_dnn = self.temp_dnn_tensorflow(north_model_path)
        south_temp_dnn = self.temp_dnn_tensorflow(south_model_path)
        west_temp_dnn = self.temp_dnn_tensorflow(west_model_path)

        # main loop
        for k in range(self.PH):
            ## arguments (1 control variabls + 1 slack variable): u[0:2] = [uMod, slack]
            u = U[k*self.number_inputs:(k+1)*self.number_inputs]
            
            ## Calculate the total power consumption and energy cost
            
            # SOC predcition [0,1]
            #SOC_pred_ph[k] = ca.fmin(ca.fmax(0.05, SOC_dnn(ca.vertcat(u[0], SOC_his_meas_k[0]))), 0.99)
            SOC_pred_ph[k] = SOC_dnn(ca.vertcat(u[0], SOC_his_meas_k[0])) 
            # quadratic terms matter
            #SOC_pred_ph[k] = SOC_his_meas_k[0] + ((u[0]-0)**2)*((u[0]-1)**2)*((u[0]-2)**2)*(0.08/6) + ((u[0]+1)**2)*((u[0]-0)**2)*((u[0]-2)**2)*(0.08/2) # charge 0.08/hr, discharge -0.08/hr
            #SOC_pred_ph[k] = SOC_his_meas_k[0] + (u[0]-0) * (u[0]-1) * (u[0]-2) * (0.08/6) + (u[0]+1) * (u[0]-0) * (u[0]-2) *(0.08/2) # charge 0.08/hr, discharge -0.08/hr

            
            # power consumption prediction
            # abs(uMod)*P_pred to enforce the correct action, where uMod = -1 (Charge TES); 0 (off); 1 (Discharge TES); 2 (Discharge chiller)
            # apply ca.fmax(X, 0.) to make power non-negative
            # P_pred_ph[k] = power_dnn(ca.vertcat(u[0], 
            #                                     Tz_ave_his_meas_k[0], 
            #                                     To_his_meas_k[0], To_pred_ph[k], 
            #                                     GHI_his_meas_k[0], GHI_pred_ph[k],
            #                                     P_his_meas_k[0])) #* ca.fabs(u[0])
            P_pred_ph[k] = power_dnn(ca.vertcat(u[0], To_pred_ph[k])) #* ca.fabs(u[0])
            # P_pred_ph[k] = (u[0]-0)*(u[0]-1)*(u[0]-2)*(-10000./6) +\
            #                (u[0]+1)*(u[0]-1)*(u[0]-2)*(0.) +\
            #                (u[0]+1)*(u[0]-0)*(u[0]-2)*(-5000./2) +\
            #                (u[0]+1)*(u[0]-0)*(u[0]-1)*(16000./6) # -1 (Charge TES 10000W); 0 (off 0W); 1 (Discharge TES 5000W); 2 (Discharge chiller 16000W)
            # P_pred_ph[k] = ((u[0]-0)**2) * ((u[0]-1)**2) * ((u[0]-2)**2) * (-10000./6) +\
            #                ((u[0]+1)**2) * ((u[0]-1)**2) * ((u[0]-2)**2) * (0.) +\
            #                ((u[0]+1)**2) * ((u[0]-0)**2) * ((u[0]-2)**2) * (-5000./2) +\
            #                ((u[0]+1)**2) * ((u[0]-0)**2) * ((u[0]-1)**2) * (16000./6) # -1 (Charge TES 10000W); 0 (off 0W); 1 (Discharge TES 5000W); 2 (Discharge chiller 16000W)


            # zonal temperature prediction
            Tz_core_pred_ph[k] = ca.fmin(ca.fmax(24, core_temp_dnn(ca.vertcat(u[0], Tz_core_his_meas_k[0])) + ca.MX(self._autoerror['core'])), 28)
            Tz_east_pred_ph[k] = ca.fmin(ca.fmax(24, east_temp_dnn(ca.vertcat(u[0], Tz_east_his_meas_k[0])) + ca.MX(self._autoerror['east'])), 28)
            Tz_north_pred_ph[k] = ca.fmin(ca.fmax(24, north_temp_dnn(ca.vertcat(u[0], Tz_north_his_meas_k[0])) + ca.MX(self._autoerror['north'])), 28)
            Tz_south_pred_ph[k] = ca.fmin(ca.fmax(24, south_temp_dnn(ca.vertcat(u[0], Tz_south_his_meas_k[0])) + ca.MX(self._autoerror['south'])), 28)
            Tz_west_pred_ph[k] = ca.fmin(ca.fmax(24, west_temp_dnn(ca.vertcat(u[0], Tz_west_his_meas_k[0])) + ca.MX(self._autoerror['west'])), 28)

            ## update the historical data
            Tz_core_his_meas_k.insert(0, Tz_core_pred_ph[k]) # Insert new datapoint at the beginning
            Tz_east_his_meas_k.insert(0, Tz_east_pred_ph[k])
            Tz_north_his_meas_k.insert(0, Tz_north_pred_ph[k])
            Tz_south_his_meas_k.insert(0, Tz_south_pred_ph[k])
            Tz_west_his_meas_k.insert(0, Tz_west_pred_ph[k])
            Tz_ave_his_meas_k.insert(0, (Tz_core_pred_ph[k]+Tz_east_pred_ph[k]+Tz_north_pred_ph[k]+Tz_south_pred_ph[k]+Tz_west_pred_ph[k])/self.number_zones)
            To_his_meas_k.insert(0, To_pred_ph[k])
            GHI_his_meas_k.insert(0, GHI_pred_ph[k])
            SOC_his_meas_k.insert(0, SOC_pred_ph[k])
            P_his_meas_k.insert(0, P_pred_ph[k])
            Tz_core_his_meas_k.pop() # Remove the last element
            Tz_east_his_meas_k.pop()
            Tz_north_his_meas_k.pop()
            Tz_south_his_meas_k.pop()
            Tz_west_his_meas_k.pop()
            Tz_ave_his_meas_k.pop()
            To_his_meas_k.pop()
            GHI_his_meas_k.pop()
            SOC_his_meas_k.pop()
            P_his_meas_k.pop()
            
            ## calculate the temperature violation
            t_hour = int(((self.time+k*self.dt) % 86400)/3600)
            Tz_low = self.T_lower[t_hour]
            Tz_upper = self.T_upper[t_hour]
            print("Tz_upper:", Tz_upper)
            delta_Tlow = ca.fmax(Tz_low-Tz_core_pred_ph[k],0.) + ca.fmax(Tz_low-Tz_east_pred_ph[k],0.) + ca.fmax(Tz_low-Tz_north_pred_ph[k],0.) + ca.fmax(Tz_low-Tz_south_pred_ph[k],0.) + ca.fmax(Tz_low-Tz_west_pred_ph[k],0.)
            delta_Thigh = ca.fmax(Tz_core_pred_ph[k]-Tz_upper,0.) + ca.fmax(Tz_east_pred_ph[k]-Tz_upper,0.) + ca.fmax(Tz_north_pred_ph[k]-Tz_upper,0.) + ca.fmax(Tz_south_pred_ph[k]-Tz_upper,0.) + ca.fmax(Tz_west_pred_ph[k]-Tz_upper,0.)
            
            ## calculate SOC penalty
            delta_SOClow = ca.fmax(0.2-SOC_pred_ph[k], 0.)
            #delta_SOChigh = ca.fmax(SOC_pred_ph[k]-0.9, 0.)
            
            # ## calculate the change of control action:
            # # control actions except slack
            # normalizer = [1/(self.u_ub[i]-self.u_lb[i]) for i in range(self.number_inputs)]
            # du_k = u - u_prev
            # u_prev = u  # update previous actions in PH
            # du_k_normalized = [normalizer[i]*du_k[i] for i in range(2,self.number_inputs-1)]
            # du_k_nom2 = ca.sumsqr(ca.vertcat(*du_k_normalized))/len(du_k_normalized)
            # #print(du_k_nom2)

            ## Objective Function: Minimize energy cost + temperature penalty + SOC penalty
            fo = ( self.w[0] * (price_ph[k] * P_pred_ph[k] * self.dt/3600./1000.) )**2 + \
                 ( self.w[1] * (delta_Thigh) )**2 #+\
                 #( self.w[2] * (delta_SOClow) ) # * (2-ca.fabs(u[0])): occupied - mode 2 (charge chiller), no penalty for SOC; unoccupied: encourage mode -1 charging TES
                 #self.w[1] * u[-1]**2 #slack variable
                 #self.w[2] * du_k_nom2
            fval.append(fo)
        
        fval_sum = ca.sum1(ca.vertcat(*fval))
        #obj=ca.Function('fval',[U],[fval_sum]) # this is the ultimate objective function
        obj=ca.Function('fval_sum',[U],[fval_sum])
        f = obj(U) # get the objective value

        ## get overshoot and undershoot for each step
        # current time step
        g = []
        lbg = []
        ubg = []
        u_lb_occ = []
        u_ub_occ = []
               
        for k in range(self.PH):
            # future time        
            t = int(self.time+k*self.dt)
            t = int((t % 86400)/3600)  # hour index 0~23
            if t>=self.occ_start and t<self.occ_end:
              u_lb_occ += [1] # [0]
              u_ub_occ += [2]
            else:
              u_lb_occ += [-1]
              u_ub_occ += [0]
            # # inequality constraints for slack variables of zonal temperature
            # eps = U[self.number_inputs*(k+1)-1] # eps: slack variable
            # g += [Tz_core_pred_ph[k]+eps,Tz_core_pred_ph[k]-eps, 
            #     Tz_east_pred_ph[k]+eps,Tz_east_pred_ph[k]-eps, 
            #     Tz_north_pred_ph[k]+eps,Tz_north_pred_ph[k]-eps, 
            #     Tz_south_pred_ph[k]+eps,Tz_south_pred_ph[k]-eps, 
            #     Tz_west_pred_ph[k]+eps,Tz_west_pred_ph[k]-eps]
            
            ## SOC    
            g += [SOC_pred_ph[k]]

            # # # constraints of zonal temperature: upper and lower T bounds
            # # lbg += [self.T_lower[t], 0.]*self.number_zones
            # # ubg += [ca.inf, self.T_upper[t]]*self.number_zones
            ## constrains of SOC
            lbg += [0.200]
            ubg += [0.999]           
        
        
        # initialize optimizer first guess
        #u_ini = self.u_start
        u_ini = u_ub_occ #u_lb_occ
        
        if False: # use the gradient-based solver: Bonmin (MINLP)
            # ## invoke the gradient-based solver: IPOPT (NLP)
            # options = {"print_time": True,"ipopt": {"max_iter": 200}} # "tol":0.01, "acceptable_tol": 0.01, "acceptable_obj_change_tol": 10**-2, "linear_solver": "mumps" from https://coin-or.github.io/Ipopt/OPTIONS.html
            # solver = ca.nlpsol("solver","ipopt",{"x":U, "f":f, 'g': ca.vertcat(*g)}, options)
            # res = solver(x0=u_ini, lbx=u_lb_occ, ubx=u_ub_occ, lbg=lbg, ubg=ubg)
            
            ## invoke the gradient-based solver: Bonmin (MINLP)
            solver_options = {
                            "bonmin.time_limit": 3600,  # Time limit in seconds
                            "bonmin.tol": 1e-2,  # Solution tolerance
                            "bonmin.max_iter": 1000,  # Increase max iterations if feasible
                            "bonmin.cutoff_decr": 1e-2,  # Increase the threshold for accepting better solutions
                            "bonmin.algorithm": 'B-BB', # default algorithm
                            "bonmin.resolve_on_small_infeasibility": True, 
                            "bonmin.solution_limit": 20, #abort after 20 feasible solutions have been found
                            "error_on_fail": False,
                            "discrete": [True]*self.PH
                            }
            # Bonmin features several algorithms: https://coin-or.github.io/Bonmin/options_list.html#sec:options_list
            # B-BB is a NLP-based branch-and-bound algorithm,
            # B-OA is an outer-approximation decomposition algorithm,
            # B-QG is an implementation of Quesada and Grossmann's branch-and-cut algorithm,
            # B-Hyb is a hybrid outer-approximation based branch-and-cut algorithm.
            solver = ca.nlpsol('nlp_solver', 'bonmin', {"x":U,"f":f, 'g': ca.vertcat(*g)}, solver_options)
            res = solver(x0=u_ini, lbx=u_lb_occ, ubx=u_ub_occ, lbg=lbg, ubg=ubg)
        
        if False: # use the gradient-based solver: Knitro (NLP)
            ## invoke the gradient-based solver: Knitro (NLP)
            # os.environ["KNITRODIR"] = "C:\Program Files\Artelys\Knitro 14.1.0" # Ensure Knitro's DLL or shared library is accessible
            # os.environ['CASADIPATH'] = r'C:\Users\guowenli\anaconda3\envs\mpc-py38\Lib\site-packages\casadi'
            import knitro
            solver_options = {
                            "knitro.maxtime_real": 120,  # Maximum allowable real time before termination
                            "knitro.ftol": 1e-3,  # Tolerance for stopping on small (feasible) changes to the objective
                            "knitro.xtol": 1e-3,  # Tolerance for stopping on small changes to the solution estimate
                            "knitro.feastol": 1e-3,  # Specifies the final relative stopping tolerance for the feasibility error
                            "knitro.opttol": 1e-3,  #Specifies the final relative stopping tolerance for the optimality error
                            "knitro.maxit": 1000,  # Maximum number of iterations
                            "knitro.algorithm": 0,  # Choose an appropriate algorithm based on the problem
                            #"knitro.ms_enable": 1,  # Enable multi-start (optional)
                            "knitro.mip_multistart":1, # enable a mixed-integer multistart heuristic at the branch-and-bound level
                            #"knitro.mip_restart":0, # Do not enable the MIP restart procedure
                            "error_on_fail": False,
                            "discrete": [True]*self.PH  # Set variables as discrete if needed
                            }
            # Create Knitro solver
            solver = ca.nlpsol('nlp_solver', 'knitro', {"x": U, "f": f, 'g': ca.vertcat(*g)}, solver_options)
            # Solve the problem
            res = solver(x0=u_ini, lbx=u_lb_occ, ubx=u_ub_occ, lbg=lbg, ubg=ubg)

        if False: # use the gradient-based solver: Gurobi (QP)
            ## invoke the gradient-based solver: Gurobi
            solver_options = {
                            "gurobi.NonConvex": 2, # Enable non-convex quadratic programming
                            #"gurobi.OutputFlag": 1,  # Enable solver output
                            "gurobi.TimeLimit": 600,  # Time limit in seconds
                            #"gurobi.MIPGap": 1e-2,  # Solution tolerance (MIP gap)
                            #"gurobi.MIPFocus": 1,  # Focus on finding feasible solutions
                            # "gurobi.Heuristics": 0.5,  # Heuristic parameter to improve performance
                            # "gurobi.Presolve": 2,  # Enable aggressive presolve
                            #"gurobi.NodeLimit": 10000,  # Limit on the number of nodes explored
                            "error_on_fail": False,
                            "discrete": [True]*self.PH,
                            }
            # Define the Gurobi solver for mixed-integer linear programming
            solver = ca.qpsol('solver', 'gurobi', {"x": U, "f": f, 'g': ca.vertcat(*g)}, solver_options) # gurobi may only used for QP linear problem in Casadi # qp solver for quatratic obj??
            # Solve the problem
            res = solver(x0=u_ini, lbx=u_lb_occ, ubx=u_ub_occ, lbg=lbg, ubg=ubg)
            
            #print(solver.stats())
            print("\nsolution x:",res['x'])
            print("Operation mode x = -1 (Charge TES); 0 (off); 1 (Discharge TES); 2 (Discharge chiller)")
            print("solution f:",res['f'])
            print("initial guess:",u_ini)
            print("u_lb:",u_lb_occ)
            print("u_ub:",u_ub_occ)
            print("lbg:",lbg)
            print("ubg:",ubg)
            print("\nsolver.stats()['return_status']:",solver.stats()['return_status'])
            
            return(res, solver.stats())
        
        if True: # use the heuristic solver: DEAP (gradient-free)
            ##=================heuristic solver=================
            # ## invoke the gradient-free solver: DEAP (https://github.com/DEAP/deap)
            ### Convert CasADi objective and constraints to Python functions for DEAP ###
        
            # Constraints (concatenate all constraints and create a CasADi function)
            g_constraints = ca.Function('constraints', [U], [ca.vertcat(*g)])  # Ensure g is a CasADi function
            
            def evaluate_individual(individual):
                # Ensure that values respect the bounds (lbx and ubx)
                individual = [max(lbx, min(round(val), ubx)) for val, lbx, ubx in zip(individual, u_lb_occ, u_ub_occ)]

                U_val = np.array(individual)
                
                # Evaluate objective using CasADi
                f = obj(U_val).full().item()  # Get the objective value
                
                # Evaluate constraints using CasADi
                g_values = g_constraints(U_val).full().flatten()  # Use the CasADi function for constraints
                
                # Apply penalties for constraint violations
                penalties = 0
                for i, g_val in enumerate(g_values):
                    if g_val < lbg[i]:
                        penalties += 10000 * (lbg[i] - g_val)  # Apply penalty for lower bound violation
                    elif g_val > ubg[i]:
                        penalties += 10000 * (g_val - ubg[i])  # Apply penalty for upper bound violation
                
                return f + penalties,  # Return the penalized objective

            ### Set up DEAP for solving the optimization problem ###

            # # Create the fitness function for minimization
            # creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            # creator.create("Individual", list, fitness=creator.FitnessMin)
            
            # Check if FitnessMin and Individual are already created which occurs when MPC re-run multiple times
            if not hasattr(creator, "FitnessMin"):
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize the objective

            if not hasattr(creator, "Individual"):
                creator.create("Individual", list, fitness=creator.FitnessMin)  # Create an individual

            # Define the DEAP toolbox
            toolbox = base.Toolbox()

            # Define how each attribute (control variable) is generated within bounds
            def bounded_attr():
                return random.uniform(-1.4, 2.4)  # Random initialization in [-1, 2], adjust for your problem

            # Create individuals (solution vectors)
            toolbox.register("individual", tools.initRepeat, creator.Individual, bounded_attr, n=self.PH)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            # Register genetic operators
            toolbox.register("mate", tools.cxBlend, alpha=0.5)
            
            # Define mutation to round values and clip to bounds [lbx, ubx]
            def discrete_mutation(individual):
                for i in range(len(individual)):
                    individual[i] = round(individual[i])  # Round to nearest integer
                    individual[i] = max(u_lb_occ[i], min(individual[i], u_ub_occ[i]))  # Clip to bounds
                return individual,

            toolbox.register("mutate", discrete_mutation)
            toolbox.register("select", tools.selTournament, tournsize=5) # tournsize=3

            # Register the evaluation function (objective + constraints)
            toolbox.register("evaluate", evaluate_individual)

            ### Run the genetic algorithm ###
            
            # Generate the initial population
            population = toolbox.population(n=200)

            # Run the genetic algorithm for 40 generations
            algorithms.eaSimple(population, toolbox, cxpb=0.8, mutpb=0.3, ngen=100, verbose=True)
            
            # # Ensure the best solutions are carried over to the next generation by adding elitism. This prevents losing good individuals.
            # if True:
            #     # Using an algorithm that retains the best individual (elitism)
            #     def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=__debug__):
            #         # Insert hall of fame (elitism)
            #         if halloffame is None:
            #             halloffame = tools.HallOfFame(1)  # Preserve the best 1 individual
                    
            #         # Evolve the population with elitism
            #         for gen in range(ngen):
            #             offspring = toolbox.select(population, len(population))
            #             offspring = list(map(toolbox.clone, offspring))

            #             # Apply crossover and mutation
            #             for child1, child2 in zip(offspring[::2], offspring[1::2]):
            #                 if random.random() < cxpb:
            #                     toolbox.mate(child1, child2)
            #                     del child1.fitness.values
            #                     del child2.fitness.values
            #             for mutant in offspring:
            #                 if random.random() < mutpb:
            #                     toolbox.mutate(mutant)
            #                     del mutant.fitness.values

            #             # Evaluate the fitness of individuals with invalid fitness
            #             invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            #             fitnesses = list(map(toolbox.evaluate, invalid_ind))
            #             for ind, fit in zip(invalid_ind, fitnesses):
            #                 ind.fitness.values = fit

            #             # Replace the old population with the new offspring
            #             population[:] = offspring

            #             # Update hall of fame (elitism)
            #             halloffame.update(population)

            #             # Print progress if needed
            #             if verbose:
            #                 print(f"Generation {gen + 1}, Best fitness: {halloffame[0].fitness.values[0]}")

            #         return population, halloffame

            #     # Apply elitism
            #     halloffame = tools.HallOfFame(1)  # Keep track of the best individual

            #     # Run the algorithm with elitism
            #     eaSimpleWithElitism(population, toolbox, cxpb=0.8, mutpb=0.3, ngen=100, halloffame=halloffame, verbose=True)

            #     # Print the best individual found
            #     print("hall of fame - Best individual:", halloffame[0])
            #     print("hall of fame - Best fitness:", halloffame[0].fitness.values[0])

            # Get the best individual (solution) from the population
            best_individual = tools.selBest(population, k=1)[0]

            # Ensure the best individual respects the bounds
            best_individual = [max(lbx, min(round(val), ubx)) for val, lbx, ubx in zip(best_individual, u_lb_occ, u_ub_occ)]

            best_obj_val = evaluate_individual(best_individual)[0]
            
            print("Best individual (original):", tools.selBest(population, k=1)[0])
            print("Objective value:", evaluate_individual(tools.selBest(population, k=1)[0])[0])
            print("Best individual (bounded integers):", best_individual)
            print("Objective value:", best_obj_val)

            res, solver_status = {}, {}
            res['x'] = best_individual
            res['f'] = best_obj_val
            solver_status['return_status'] = 'OPTIMAL'
            # print("\nsolver_status['return_status']:",solver_status['return_status'])

            return res, solver_status

    def set_time(self, time):
        
        self.time = time

    def set_measurement(self,measurement):
        """Set measurement at time t

        :param measurement: system measurement at time t
        :type measurement: pandas DataFrame
        """
        self.measurement = measurement
    
    def set_states(self,states):
        """Set states for current time step t

        :param states: values of states at t
        :type states: dict

            for example:

            states = {'Tz_his_t':[24,23,24,23]}

        """
        self.states = states

    def set_Tz_his_pred(self,Tz_his_pred):
        """set historical predicted zone temperature

        :param Tz_his_pred: values of states at t-4, t-3, t-2, t-1
        :type states: dict

            for example:

            Tz_his_pred = {'TCor_his_pred':[21,22,23,24]}

        """
        self.Tz_his_pred = Tz_his_pred

    def set_predictor(self, predictor):
        """Set predictor values for current time step t

        :param predictor: values of predictors from t to t+PH
        :type predictor: dict

            for example:

            predictor = {'energy_price':[1,3,4,5,6,7,8]}

        """
        self.predictor = predictor

    def get_u_start(self, optimum_prev):
        start = optimum_prev[self.number_inputs:]  # need to check
        start = np.append(start, self.u_lb) 
        #start = np.append(start, self.u_ub) # attention: it change the data type from ca.DM to np.ndarray
        #start = ca.vertcat(start, self.u_ub) # pay attention to it !!!
        
        return start

    def set_u_start(self,prev):
        """Set start value for design variables using previous optimization results
        """
        start = self.get_u_start(prev)
        self.u_start = start

    def set_u_prev(self, u_prev):
        """
        set control actions from previous step
        :param u_prev: previous control action vector
        :type u_prev: list
        """
        self.x_opt_0 = u_prev
        
    def get_open_loop_preds(self, u_opt_ph):
        occupied_ph = [0]*self.PH
        for i in range(self.PH):
            # future time        
            t = int( ( (self.time+i*self.dt) % 86400)/3600) # hour index 0~23           
            if t>=self.occ_start and t<self.occ_end:
                occupied_ph[i] = 1

        ### get states and predictions at current time step
        # get predictions at current time step
        To_pred_ph = self.predictor['Toa'] # predicted outdoor air dry bulb temperature
        GHI_pred_ph = self.predictor['GHI'] # predicted relative humidity 
        price_ph = self.predictor['price'] # predicted electricity price (TOU)

        # get historical temperaure measurements
        Tz_core_his_meas = np.array(self.states['Tz_core_his_meas'][:]) # historical zone temperatur states for zone temperture prediction model, [t-1, t-2, t-3, t-4], Unit: Celsius
        Tz_east_his_meas = np.array(self.states['Tz_east_his_meas'][:])
        Tz_north_his_meas = np.array(self.states['Tz_north_his_meas'][:])
        Tz_south_his_meas = np.array(self.states['Tz_south_his_meas'][:])
        Tz_west_his_meas = np.array(self.states['Tz_west_his_meas'][:])
        Tz_ave_his_meas = np.array(self.states['Tz_ave_his_meas'][:])
        To_his_meas = np.array(self.states['To_his_meas'][:]) # historical outdoor air temperatur dry bulb states for zone temperture prediction model
        GHI_his_meas = np.array(self.states['GHI_his_meas'][:]) # historical global horizontal solar irradiation [W/m2]
        SOC_his_meas = np.array(self.states['SOC_his_meas'][:]) # historical state-of-charge
        P_his_meas = np.array(self.states['P_his_meas'][:]) # historical total power [W]
        #("\nTo_his_meas:",To_his_meas)

        # get historical predicted zone temperatures
        Tz_core_his_pred = np.array(self.states['Tz_core_his_pred'])
        Tz_east_his_pred = np.array(self.states['Tz_east_his_pred'])
        Tz_north_his_pred = np.array(self.states['Tz_north_his_pred'])
        Tz_south_his_pred = np.array(self.states['Tz_south_his_pred'])
        Tz_west_his_pred = np.array(self.states['Tz_west_his_pred'])

        ## define objective
        # Get auto error from historical predictions and measurements: autocorrection term happens at the beginning of the PH. Calculation of such term inside PH is not efficient.
        n_Tz_his = len(Tz_core_his_meas)
        autoerror_core = 0
        autoerror_east = 0
        autoerror_north = 0
        autoerror_south = 0
        autoerror_west = 0

        for k in range(n_Tz_his):
            autoerror_core += (Tz_core_his_meas[k]-Tz_core_his_pred[k])/n_Tz_his
            autoerror_east += (Tz_east_his_meas[k]-Tz_east_his_pred[k])/n_Tz_his
            autoerror_north += (Tz_north_his_meas[k]-Tz_north_his_pred[k])/n_Tz_his
            autoerror_south += (Tz_south_his_meas[k]-Tz_south_his_pred[k])/n_Tz_his
            autoerror_west += (Tz_west_his_meas[k]-Tz_west_his_pred[k])/n_Tz_his

        # save for future use    
        self._autoerror = {'core':autoerror_core,
                        'east':autoerror_east,
                        'north':autoerror_north,
                        'south':autoerror_south,
                        'west':autoerror_west}

        # initialize outputs and intemediate variables before main loop
        Tz_core_pred_ph = [0.]*self.PH
        Tz_east_pred_ph = [0.]*self.PH
        Tz_north_pred_ph = [0.]*self.PH
        Tz_south_pred_ph = [0.]*self.PH
        Tz_west_pred_ph = [0.]*self.PH
        P_pred_ph = [0.]*self.PH
        SOC_pred_ph = [0.]*self.PH
        fval = []

        Tz_core_his_meas_k = [Tz for Tz in Tz_core_his_meas]
        Tz_east_his_meas_k = [Tz for Tz in Tz_east_his_meas]
        Tz_north_his_meas_k = [Tz for Tz in Tz_north_his_meas]
        Tz_south_his_meas_k = [Tz for Tz in Tz_south_his_meas]
        Tz_west_his_meas_k = [Tz for Tz in Tz_west_his_meas]
        Tz_ave_his_meas_k = [Tz for Tz in Tz_ave_his_meas]
        To_his_meas_k = [Tz for Tz in To_his_meas]
        GHI_his_meas_k = [Tz for Tz in GHI_his_meas]
        SOC_his_meas_k = [Tz for Tz in SOC_his_meas]
        P_his_meas_k = [Tz for Tz in P_his_meas]

        # load dnn models
        # Check if model paths were provided
        if hasattr(self, 'model_paths'):
            core_model_path = self.model_paths['core']
            east_model_path = self.model_paths['east']
            north_model_path = self.model_paths['north']
            south_model_path = self.model_paths['south']
            west_model_path = self.model_paths['west']
            SOC_model_path = self.model_paths['SOC']
            power_model_path = self.model_paths['power']
        else:
            core_model_path="system_identification/dnn_model_core_temperature.h5"
            east_model_path="system_identification/dnn_model_east_temperature.h5"
            north_model_path="system_identification/dnn_model_north_temperature.h5"
            south_model_path="system_identification/dnn_model_south_temperature.h5"
            west_model_path="system_identification/dnn_model_west_temperature.h5"
            SOC_model_path="system_identification/dnn_SOC_model.h5"
            power_model_path="system_identification/dnn_power_model.h5"
        SOC_dnn = self.SOC_dnn_tensorflow(SOC_model_path)
        power_dnn = self.power_dnn_tensorflow(power_model_path)
        core_temp_dnn = self.temp_dnn_tensorflow(core_model_path)
        east_temp_dnn = self.temp_dnn_tensorflow(east_model_path)
        north_temp_dnn = self.temp_dnn_tensorflow(north_model_path)
        south_temp_dnn = self.temp_dnn_tensorflow(south_model_path)
        west_temp_dnn = self.temp_dnn_tensorflow(west_model_path)
        
        print("\n========debugging problem formulation and solver optimization========")
        # main loop
        for k in range(self.PH):
            ## arguments (1 control variabls): u[0:1] = [uMod]
            u = u_opt_ph[k*self.number_inputs:(k+1)*self.number_inputs]
                      
            ## SOC predcition [0,1]
            #SOC_pred_ph[k] = ca.fmin(ca.fmax(0.05, SOC_dnn(ca.vertcat(u[0], SOC_his_meas_k[0]))), 0.99)
            SOC_pred_ph[k] = float(SOC_dnn(ca.vertcat(u[0], SOC_his_meas_k[0])))
            
            ## power consumption prediction
            # abs(uMod)*P_pred to enforce the correct action, where uMod = -1 (Charge TES); 0 (off); 1 (Discharge TES); 2 (Discharge chiller)
            # apply ca.fmax(X, 0.) to make power non-negative
            # P_pred_ph[k] = ca.fmin(20000, power_dnn(ca.vertcat(u[0], 
            #                                     Tz_ave_his_meas_k[0], 
            #                                     To_his_meas_k[0], To_pred_ph[k], 
            #                                     GHI_his_meas_k[0], GHI_pred_ph[k],
            #                                     P_his_meas_k[0]))  ) * ca.fabs(u[0])
            P_pred_ph[k] = float(power_dnn(ca.vertcat(u[0], To_pred_ph[k])) )#* ca.fabs(u[0]))
            
            ## zonal temperature prediction
            # Tz_core_pred_ph[k] = ca.fmin(ca.fmax(24, core_temp_dnn(ca.vertcat(u[0], Tz_core_his_meas_k[0])) + ca.MX(self._autoerror['core'])), 28)
            # Tz_east_pred_ph[k] = ca.fmin(ca.fmax(24, east_temp_dnn(ca.vertcat(u[0], Tz_east_his_meas_k[0])) + ca.MX(self._autoerror['east'])), 28)
            # Tz_north_pred_ph[k] = ca.fmin(ca.fmax(24, north_temp_dnn(ca.vertcat(u[0], Tz_north_his_meas_k[0])) + ca.MX(self._autoerror['north'])), 28)
            # Tz_south_pred_ph[k] = ca.fmin(ca.fmax(24, south_temp_dnn(ca.vertcat(u[0], Tz_south_his_meas_k[0])) + ca.MX(self._autoerror['south'])), 28)
            # Tz_west_pred_ph[k] = ca.fmin(ca.fmax(24, west_temp_dnn(ca.vertcat(u[0], Tz_west_his_meas_k[0])) + ca.MX(self._autoerror['west'])), 28)
            Tz_core_pred_ph[k] = float(core_temp_dnn(ca.vertcat(u[0], Tz_core_his_meas_k[0])) + ca.MX(self._autoerror['core']))
            Tz_east_pred_ph[k] = float(east_temp_dnn(ca.vertcat(u[0], Tz_east_his_meas_k[0])) + ca.MX(self._autoerror['east']))
            Tz_north_pred_ph[k] = float(north_temp_dnn(ca.vertcat(u[0], Tz_north_his_meas_k[0])) + ca.MX(self._autoerror['north']))
            Tz_south_pred_ph[k] = float(south_temp_dnn(ca.vertcat(u[0], Tz_south_his_meas_k[0])) + ca.MX(self._autoerror['south']))
            Tz_west_pred_ph[k] = float(west_temp_dnn(ca.vertcat(u[0], Tz_west_his_meas_k[0])) + ca.MX(self._autoerror['west']))
            
            ## update the historical data
            Tz_core_his_meas_k.insert(0, Tz_core_pred_ph[k]) # Insert new datapoint at the beginning
            Tz_east_his_meas_k.insert(0, Tz_east_pred_ph[k])
            Tz_north_his_meas_k.insert(0, Tz_north_pred_ph[k])
            Tz_south_his_meas_k.insert(0, Tz_south_pred_ph[k])
            Tz_west_his_meas_k.insert(0, Tz_west_pred_ph[k])
            Tz_ave_his_meas_k.insert(0, (Tz_core_pred_ph[k]+Tz_east_pred_ph[k]+Tz_north_pred_ph[k]+Tz_south_pred_ph[k]+Tz_west_pred_ph[k])/self.number_zones)
            To_his_meas_k.insert(0, To_pred_ph[k])
            GHI_his_meas_k.insert(0, GHI_pred_ph[k])
            SOC_his_meas_k.insert(0, SOC_pred_ph[k])
            P_his_meas_k.insert(0, P_pred_ph[k])
            Tz_core_his_meas_k.pop() # Remove the last element
            Tz_east_his_meas_k.pop()
            Tz_north_his_meas_k.pop()
            Tz_south_his_meas_k.pop()
            Tz_west_his_meas_k.pop()
            Tz_ave_his_meas_k.pop()
            To_his_meas_k.pop()
            GHI_his_meas_k.pop()
            SOC_his_meas_k.pop()
            P_his_meas_k.pop()

            ## calculate the temperature violation
            t_hour = int(((self.time+k*self.dt) % 86400)/3600)
            Tz_low = self.T_lower[t_hour]
            Tz_upper = self.T_upper[t_hour]
            delta_Tlow = ca.fmax(Tz_low-Tz_core_pred_ph[k],0.) + ca.fmax(Tz_low-Tz_east_pred_ph[k],0.) + ca.fmax(Tz_low-Tz_north_pred_ph[k],0.) + ca.fmax(Tz_low-Tz_south_pred_ph[k],0.) + ca.fmax(Tz_low-Tz_west_pred_ph[k],0.)
            delta_Thigh = ca.fmax(Tz_core_pred_ph[k]-Tz_upper,0.) + ca.fmax(Tz_east_pred_ph[k]-Tz_upper,0.) + ca.fmax(Tz_north_pred_ph[k]-Tz_upper,0.) + ca.fmax(Tz_south_pred_ph[k]-Tz_upper,0.) + ca.fmax(Tz_west_pred_ph[k]-Tz_upper,0.)
            
            ## calculate SOC penalty
            delta_SOClow = ca.fmax(0.2-SOC_pred_ph[k], 0.)
            delta_SOChigh = ca.fmax(SOC_pred_ph[k]-0.99, 0.)
            
            # ## calculate the change of control action:
            # # control actions except slack
            # normalizer = [1/(self.u_ub[i]-self.u_lb[i]) for i in range(self.number_inputs)]
            # du_k = u - u_prev
            # u_prev = u  # update previous actions in PH
            # du_k_normalized = [normalizer[i]*du_k[i] for i in range(2,self.number_inputs-1)]
            # du_k_nom2 = ca.sumsqr(ca.vertcat(*du_k_normalized))/len(du_k_normalized)
            # #print(du_k_nom2)

            ## Objective Function: Minimize energy cost + temperature penalty + SOC penalty
            fo = ( self.w[0] * (price_ph[k] * P_pred_ph[k] * self.dt/3600./1000.) ) + \
                 ( self.w[2] * (delta_SOClow)  ) # * (2-ca.fabs(u[0])): occupied - mode 2 (charge chiller), no penalty for SOC; unoccupied: encourage mode -1 charging TES
                 #( self.w[1] * (delta_Thigh) )**2 +\
                 #self.w[1] * u[-1]**2 #slack variable
                 #self.w[2] * du_k_nom2
            fval.append(float(fo))
            #print("objective value:",fo)
                        
        fval_sum = ca.sum1(ca.vertcat(*fval))
                
        degub_mode = False
        if degub_mode == True:
            print("SOC_pred_ph[k]:",SOC_pred_ph[k])
            print("P_pred_ph[k]:",P_pred_ph[k])
            #print("Tz_core_pred_ph[k]:",Tz_core_pred_ph[k])
            #print("Tz_low:", Tz_low)
            #print("Tz_upper:", Tz_upper)
            #print("delta_Tlow:", delta_Tlow)
            #print("delta_Thigh:", delta_Thigh)
            print("delta_SOClow:", delta_SOClow)
            #print("delta_SOChigh:", delta_SOChigh)
            print("power cost:", (self.w[0] * (price_ph[k] * P_pred_ph[k] * self.dt/3600./1000.))**2)
            print("SOC cost:", (self.w[2] * (delta_SOClow))**2 )
            print("objective function:", fval)
            #print("fval_sum:", fval_sum)
            
        print("total objectives:", sum(fval))

        return(P_pred_ph, SOC_pred_ph, Tz_core_pred_ph, Tz_east_pred_ph, Tz_north_pred_ph, Tz_south_pred_ph, Tz_west_pred_ph)    
                
    def get_core_temp_pred(self, u): # need to implement autoerror
        """Get predicted temperature of core zone using optimal control inputs
        """
        # Check if model paths were provided
        if hasattr(self, 'model_paths'):
            core_model_path = self.model_paths['core']
        else:
            core_model_path="system_identification/dnn_model_core_temperature.h5"
        core_temp_dnn = self.temp_dnn_tensorflow(core_model_path)
        Tz_pred = core_temp_dnn(ca.vertcat(u[0], self.states['Tz_core_his_meas'][0])) + self._autoerror['core']
        return Tz_pred

    def get_east_temp_pred(self, u):
        """Get predicted temperature of east zone using optimal control inputs
        """
        # Check if model paths were provided
        if hasattr(self, 'model_paths'):
            east_model_path = self.model_paths['east']
        else:
            east_model_path="system_identification/dnn_model_east_temperature.h5"
        east_temp_dnn = self.temp_dnn_tensorflow(east_model_path)
        Tz_pred = east_temp_dnn(ca.vertcat(u[0], self.states['Tz_east_his_meas'][0])) + self._autoerror['east']
        return Tz_pred

    def get_north_temp_pred(self, u):
        """Get predicted temperature of north zone using optimal control inputs
        """
                # Check if model paths were provided
        if hasattr(self, 'model_paths'):
            north_model_path = self.model_paths['north']
        else:
            north_model_path="system_identification/dnn_model_north_temperature.h5"
        north_temp_dnn = self.temp_dnn_tensorflow(north_model_path)
        Tz_pred = north_temp_dnn(ca.vertcat(u[0], self.states['Tz_north_his_meas'][0])) + self._autoerror['north']
        return Tz_pred

    def get_south_temp_pred(self, u):
        """Get predicted temperature of south zone using optimal control inputs
        """
                # Check if model paths were provided
        if hasattr(self, 'model_paths'):
            south_model_path = self.model_paths['south']
        else:
            south_model_path="system_identification/dnn_model_south_temperature.h5"
        south_temp_dnn = self.temp_dnn_tensorflow(south_model_path) 
        Tz_pred = south_temp_dnn(ca.vertcat(u[0], self.states['Tz_south_his_meas'][0])) + self._autoerror['south']
        return Tz_pred

    def get_west_temp_pred(self, u):
        """Get predicted temperature of west zone using optimal control inputs
        """
        # Check if model paths were provided
        if hasattr(self, 'model_paths'):
            west_model_path = self.model_paths['west']
        else:
            west_model_path="system_identification/dnn_model_west_temperature.h5"
        west_temp_dnn = self.temp_dnn_tensorflow(west_model_path)
        Tz_pred = west_temp_dnn(ca.vertcat(u[0], self.states['Tz_west_his_meas'][0])) + self._autoerror['west']
        return Tz_pred

    def get_power_pred(self, u):
        """Get predicted power consumption
        """
        # Check if model paths were provided
        if hasattr(self, 'model_paths'):
            power_model_path = self.model_paths['power']
        else:
            power_model_path="system_identification/dnn_power_model.h5"
        power_dnn = self.power_dnn_tensorflow(power_model_path)
        t = int((self.time%86400)/3600) # hour index 0~23           
        if t>=self.occ_start and t<self.occ_end:
            occupied = True
        else:
            occupied = False
        
        P = power_dnn(ca.vertcat(u[0], self.predictor['Toa'][0])) # self.states['Tz_ave_his_meas'][0], self.states['To_his_meas'][0], , u[1], occupied, self.states['SOC_his_meas'][0], 

        return ca.fmax(P, 0.)

    def get_SOC_pred(self, u):
        """Get predicted SOC
        """
        if hasattr(self, 'model_paths'):
            SOC_model_path = self.model_paths['SOC']
        else:
            SOC_model_path="system_identification/dnn_SOC_model.h5"
        SOC_dnn = self.SOC_dnn_tensorflow(SOC_model_path)
        
        SOC = SOC_dnn(ca.vertcat(u[0], self.states['SOC_his_meas'][0])) # u[1], occupied, , self.states['Tz_ave_his_meas'][0]

        return SOC

    def relu(self, x):
        '''A helper function named relu is defined for the ReLU activation function. 
        CasADi doesn't have a built-in ReLU function, so we create one using ca.fmax, 
        which computes the element-wise maximum with zero. 
        This effectively simulates the ReLU behavior: relu(x) = max(x, 0).
        '''
        return ca.fmax(x, 0)

    def softplus(self, x):
        '''Softplus activation function: smooth approximation of ReLU'''
        return ca.log(1 + ca.exp(x))

    def power_dnn_tensorflow(self, model_path="system_identification/dnn_power_model.h5"):
        '''Differentiable Neural Network model for total power consumption prediction
        '''
        # Load the TensorFlow model
        tf_model = load_model(model_path)
        layers = tf_model.layers

        # Start with the input shape from the model
        input_shape = tf_model.input_shape[1:]  # Assuming the first dimension is the batch size, Excluding the batch dimension
        nn_input = ca.MX.sym('nn_input', *input_shape)
                    
        # Assume the normalization layer is the first layer
        normalization_layer = tf_model.layers[0]
        mean = normalization_layer.mean.numpy()
        variance = normalization_layer.variance.numpy()
        std = np.sqrt(variance)
        
        # Normalize the input
        normalized_input = (nn_input - ca.DM(mean).T) / ca.DM(std).T
        x = normalized_input  # Begin with normalized input
        
        # Process each layer after the normalization layer
        for layer in tf_model.layers[1:]:  # Skip the first normalization layer
            if isinstance(layer, tf.keras.layers.Dense):  # Check if layer is Dense
                weights, biases = layer.get_weights()
                W = ca.DM(weights)
                b = ca.DM(biases)
                x = ca.mtimes(W.T, x) + b
                if 'relu' in layer.get_config()['activation']:
                    x = self.relu(x)  # Apply ReLU
                elif 'softplus' in layer.get_config()['activation']:
                    x = self.softplus(x)  # Apply Softplus activation

        # After the loop, 'x' holds the final layer's output
        nn_output = x  # This is the output of the neural network
        
        # A CasADi Function is created with the symbolic input nn_input and the network's final output nn_output. 
        # This Function object can be used to evaluate the neural network symbolically or numerically.
        # Define the Casadi function for the model
        nn_model_casadi = ca.Function('nn_model', [nn_input], [nn_output])
        
        # Finally, the CasADi function representing the neural network is returned.
        return nn_model_casadi

    def temp_dnn_tensorflow(self, model_path):
        '''Differentiable Neural Network model for core zone temperature prediction
        '''
        # Load the TensorFlow model
        tf_model = load_model(model_path)
        input_shape = tf_model.input_shape[1:]  # Assuming the first dimension is the batch size
        nn_input = ca.MX.sym('nn_input', *input_shape)
        
        # Assume the normalization layer is the first layer
        normalization_layer = tf_model.layers[0]
        mean = normalization_layer.mean.numpy()
        variance = normalization_layer.variance.numpy()
        std = np.sqrt(variance)
        
        # Normalize the input
        normalized_input = (nn_input - ca.DM(mean).T) / ca.DM(std).T
        x = normalized_input  # Begin with normalized input
        # Process each layer after the normalization layer
        for layer in tf_model.layers[1:]:  # Skip the first normalization layer
            if isinstance(layer, tf.keras.layers.Dense):  # Check if layer is Dense
                weights, biases = layer.get_weights()
                W = ca.DM(weights)
                b = ca.DM(biases)
                x = ca.mtimes(W.T, x) + b
                if 'relu' in layer.get_config()['activation']:
                    x = self.relu(x)  # Apply ReLU
                elif 'softplus' in layer.get_config()['activation']:
                    x = self.softplus(x)  # Apply Softplus activation
        nn_output = x  # This is the output of the neural network
        # Define the Casadi function for the model
        nn_model_casadi = ca.Function('nn_model', [nn_input], [nn_output])
        # Finally, the CasADi function representing the neural network is returned.
        return nn_model_casadi

    def SOC_dnn_tensorflow(self, model_path="system_identification/dnn_SOC_model.h5"):
        '''Differentiable Neural Network model for TES tank State-of-Charge TES prediction
        '''
        # Load the TensorFlow model
        tf_model = load_model(model_path)

        # Start with the input shape from the model
        input_shape = tf_model.input_shape[1:]  # Assuming the first dimension is the batch size, Excluding the batch dimension
        nn_input = ca.MX.sym('nn_input', *input_shape)
              
        # # Assume the normalization layer is the first layer
        # normalization_layer = tf_model.layers[0]
        # mean = normalization_layer.mean.numpy()
        # variance = normalization_layer.variance.numpy()
        # std = np.sqrt(variance)
        
        # # Normalize the input
        # normalized_input = (nn_input - ca.DM(mean).T) / ca.DM(std).T
        # x = normalized_input  # Begin with normalized input      
        # Process each layer after the normalization layer
        # for layer in tf_model.layers[1:]:  # Skip the first normalization layer
        
        x = nn_input # Begin with the input (no normalization)
        # Recreate the network using CasADi operations    
        for layer in tf_model.layers:  # Skip the first normalization layer
            if isinstance(layer, tf.keras.layers.Dense):  # Check if layer is Dense
                weights, biases = layer.get_weights()
                W = ca.DM(weights)
                b = ca.DM(biases)
                x = ca.mtimes(W.T, x) + b
                if 'relu' in layer.get_config()['activation']:
                    x = self.relu(x)  # Apply ReLU
                elif 'softplus' in layer.get_config()['activation']:
                    x = self.softplus(x)  # Apply Softplus activation
                #else 'linear' is then applied, which is f(x)=x

        # After the loop, 'x' holds the final layer's output
        nn_output = x  # This is the output of the neural network
        
        # Define the Casadi function for the model
        nn_model_casadi = ca.Function('nn_model', [nn_input], [nn_output])
        
        # Finally, the CasADi function representing the neural network is returned.
        return nn_model_casadi