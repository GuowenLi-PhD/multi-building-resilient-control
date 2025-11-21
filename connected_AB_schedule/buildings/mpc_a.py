'''
This script is used to formulate the baseline and adaptive MPC.

Author: Guowen Li, Yangyang Fu
Email: guowenli@tamu.edu, yangyang.fu@tamu.edu 
Revisions:
    02/15/2022: Add auto-correction in zone temperature model
    2023: Implement adaptive MPC for Device Reinitialization Attack on Core zone's VAV box
    2024: Add five-zone coordinated zonal temperature prediction models
    2025: wrap up for multi-building control
'''
import casadi as ca
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plts
import json
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class mpc_case():
    def __init__(self,PH,CH,time,dt,measurement,states,predictor, mpc_models, dos_attack_core_VAV=False):
        self.PH = PH # prediction horizon
        self.CH = CH # control horizon
        self.dt = dt # time step
        self.time = time # current time index
        self.measurement = measurement # measurement at current time step, dictionary
        self.predictor = predictor # price and outdoor air temperature for the future horizons
        self.states = states # dictionary
        self.number_zones = 5 # 5 zones
        self.occ_start = 7 # occupancy starts
        self.occ_end = 19 # occupancy ends

        # initialize optimiztion
        self.optimum = {}
        self.u_lb = [0, 0, 5, 15.6, 11.8, 0.23, 0.05, 0.05, 0.05, 0.04, 0]  # lower bounds following ASHRAE Guideline 36
        # Reminder! Temperature: Celsius is used in prediction, but K is used in the Modelica model
        self.u_ub = [1, 1, 10, 29.4, 18, 4.5, 0.90, 0.95, 0.95, 0.70, 0.1]
        self.u_start = [0, 0, 10, 23, 18, 0.23, 0.05, 0.05, 0.05, 0.04, 0.1]*PH # total number of 10 control variables and 1 slack variable (10+1) for each step
        self.number_inputs = 11
        #self.w = [1., 1., 100.]  # default weights in objective function: : Minimize energy cost + slack variable + action change rate
        self.x_opt_0 = self.u_lb   # initialization of previous control actions

        # initialize mpc model parameter
        self.params_fan = mpc_models['fan']
        self.params_fan_Tset22 = mpc_models['fan_Tset22']
        self.params_fan_Tset26 = mpc_models['fan_Tset26']
        self.params_chiller = mpc_models['chiller_plant']
        self.params_chiller_Tset22 = mpc_models['chiller_plant_Tset22']
        self.params_chiller_Tset26 = mpc_models['chiller_plant_Tset26']
        self.params_core = mpc_models['core']
        self.params_east = mpc_models['east']
        self.params_south = mpc_models['south']
        self.params_north = mpc_models['north']
        self.params_west= mpc_models['west']

        # initialize zone arx model auto error term
        self._autoerror = {'core': 0,
                           'east': 0,
                           'north': 0,
                           'south': 0,
                           'west': 0}
     
        # ============================================================
        # ADAPTIVE MPC: Attack-aware weight configuration
        # ============================================================
        self.dos_attack_core_VAV = dos_attack_core_VAV
        
        if self.dos_attack_core_VAV:
            # RESILIENT MODE: Prioritize thermal comfort over energy cost
            # Remove energy cost term (w[0]=0), increase comfort penalty weight
            self.w = [0., 100., 10.]  # [energy_cost, slack, slew_rate]
            print("\n" + "="*80)
            print("⚠️  ADAPTIVE MPC ACTIVATED - DoS Attack Detected on Core Zone VAV")
            print("="*80)
            print("Control Strategy Reconfigured:")
            print("  - Energy cost weight:     1.0  →  0.0   (DISABLED)")
            print("  - Comfort penalty weight: 1.0  →  100.0 (PRIORITY)")
            print("  - Slew rate weight:       100.0 → 10.0  (RELAXED)")
            print("  - Core zone airflow:      [0.23, 2.80] → [0.00, 0.01] m³/s (FROZEN)")
            print("  - Adjacent zones:         FULL AUTHORITY for compensatory control")
            print("="*80 + "\n")
        else:
            # NOMINAL MODE: Balance energy efficiency and comfort
            self.w = [1., 1., 100.]  # [energy_cost, slack, slew_rate]
            print("\n" + "="*80)
            print("✓ NOMINAL MPC MODE - Normal Operation")
            print("="*80)
            print("Control Strategy:")
            print("  - Energy cost weight:     1.0   (Balanced)")
            print("  - Comfort penalty weight: 1.0   (Balanced)")
            print("  - Slew rate weight:       100.0 (Smooth control)")
            print("="*80 + "\n")
        
        self.x_opt_0 = self.u_lb        

    def optimize(self, fixed_vars=None):
        """
        Optimize MPC control actions
        
        Parameters:
        -----------
        fixed_vars : dict or None
            Dictionary of scheduled variables to fix as hard constraints
            Example: {'bcp': 1, 'bahu': 1, 'Tsa': 13.0}
            If None, all variables are optimized
        
        Returns:
        --------
        res : dict with optimal control actions
        solver_status : dict with solver information
        """
        time = self.time
        
        # Handle fixed variables (scheduled controls)
        if fixed_vars is None:
            fixed_vars = {}
        
        # Variable name to index mapping
        var_names = ['bcp', 'bahu', 'Tchw', 'Tcw', 'Tsa', 'Vcore', 'Veast', 'Vnorth', 'Vsouth', 'Vwest', 'epsilon']
        var_index_map = {name: idx for idx, name in enumerate(var_names)}

        ### get states and predictions at current time step
        # get predictions at current time step
        To_pred_ph = self.predictor['Toa'] # predicted outdoor air temperature 
        RHo_pred_ph = self.predictor['RHoa'] # predicted relative humidity 
        price_ph = self.predictor['price'] # predicted electricity price (TOU)


        # get historical temperaure measurements
        Tz_core_his_meas = np.array(self.states['Tz_core_his_meas'][:])-273.15 # historical zone temperatur states for zone temperture prediction model
        Tz_east_his_meas = np.array(self.states['Tz_east_his_meas'][:])-273.15
        Tz_north_his_meas = np.array(self.states['Tz_north_his_meas'][:])-273.15
        Tz_south_his_meas = np.array(self.states['Tz_south_his_meas'][:])-273.15
        Tz_west_his_meas = np.array(self.states['Tz_west_his_meas'][:])-273.15
        To_his_meas = np.array(self.states['To_his_meas'][:]) # historical outdoor air temperatur states for zone temperture prediction model

        # get historical predicted zone temperatures
        Tz_core_his_pred = np.array(self.states['Tz_core_his_pred']) - 273.15
        Tz_east_his_pred = np.array(self.states['Tz_east_his_pred']) - 273.15
        Tz_north_his_pred = np.array(self.states['Tz_north_his_pred']) - 273.15
        Tz_south_his_pred = np.array(self.states['Tz_south_his_pred']) - 273.15
        Tz_west_his_pred = np.array(self.states['Tz_west_his_pred']) - 273.15

        # get previous control actions
        u_prev = self.x_opt_0

        ### formulate optimization problem for MPC
        ## declare symbolic variabels
        U = ca.MX.sym("U", self.number_inputs*self.PH)

        # zone temperature bounds - need check with the high-fidelty model
        T_upper = np.array([30.0 for i in range(24)])
        T_upper[self.occ_start:self.occ_end] = 24.0 # used to be 26.0, HIL for 24.0 C
        T_lower = np.array([18.0 for i in range(24)])
        T_lower[self.occ_start:self.occ_end] = 20.0 # 22.0

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
        
        # initialize optimizer first guess
        u_ini = self.u_start

        # initialize outputs and intemediate variables before main loop
        Tz_core_pred_ph = [0.]*self.PH
        Tz_east_pred_ph = [0.]*self.PH
        Tz_north_pred_ph = [0.]*self.PH
        Tz_south_pred_ph = [0.]*self.PH
        Tz_west_pred_ph = [0.]*self.PH
        P_pred_ph = [0.]*self.PH
        fval = []

        Tz_core_his_meas_k = [Tz for Tz in Tz_core_his_meas]
        Tz_east_his_meas_k = [Tz for Tz in Tz_east_his_meas]
        Tz_north_his_meas_k = [Tz for Tz in Tz_north_his_meas]
        Tz_south_his_meas_k = [Tz for Tz in Tz_south_his_meas]
        Tz_west_his_meas_k = [Tz for Tz in Tz_west_his_meas]
        To_his_meas_k = [To for To in To_his_meas]


        # main loop
        for k in range(self.PH):
            ## arguments (10 control variabls + 1 slack variable): u[0] ~ u[10] = [bcp, bahu, Tchw, Tcw, Tsa, Vcore, Veast, Vnorth, Vsouth, Vwest, \epsilon]
            u = U[k*self.number_inputs:(k+1)*self.number_inputs]
            
            ## Calculate the total power consumption and energy cost
            # get power model inputs
            Tz_avg_k = (Tz_core_his_meas_k[-1]+Tz_east_his_meas_k[-1]+Tz_north_his_meas_k[-1] +
                        Tz_south_his_meas_k[-1]+Tz_west_his_meas_k[-1])/self.number_zones

            P_pred_ph[k] = u[0]*self.ChillerPlantPower(self.params_chiller,u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],Tz_avg_k,To_pred_ph[k],RHo_pred_ph[k]) + \
                            u[1]*self.FanPower(self.params_fan,u[5],u[6],u[7],u[8],u[9]) # need to make sure chiller power is positive

            ## zonal temperature prediction
            Tz_core_pred_ph[k] = self.ZoneTemperature(self.params_core, u[4], Tz_east_his_meas_k[-1],Tz_north_his_meas_k[-1],Tz_south_his_meas_k[-1],Tz_west_his_meas_k[-1], u[5],u[6],u[7],u[8],u[9], Tz_core_his_meas_k[0], Tz_core_his_meas_k[1], Tz_core_his_meas_k[2], Tz_core_his_meas_k[3], 
                                To_his_meas_k[0], To_his_meas_k[1], To_his_meas_k[2], To_his_meas_k[3], u[0], u[1], self._autoerror['core'])
            
            Tz_east_pred_ph[k] = self.ZoneTemperature(self.params_east,u[4], Tz_core_his_meas_k[-1],Tz_north_his_meas_k[-1],Tz_south_his_meas_k[-1],Tz_west_his_meas_k[-1], u[6],u[5],u[7],u[8],u[9], Tz_east_his_meas_k[0], Tz_east_his_meas_k[1],Tz_east_his_meas_k[2],Tz_east_his_meas_k[3], 
                                To_his_meas_k[0],To_his_meas_k[1],To_his_meas_k[2],To_his_meas_k[3], u[0], u[1], self._autoerror['east'])
            
            Tz_north_pred_ph[k] = self.ZoneTemperature(self.params_north,u[4], Tz_core_his_meas_k[-1],Tz_east_his_meas_k[-1],Tz_south_his_meas_k[-1],Tz_west_his_meas_k[-1], u[7],u[5],u[6],u[8],u[9], Tz_north_his_meas_k[0],Tz_north_his_meas_k[1],Tz_north_his_meas_k[2],Tz_north_his_meas_k[3], 
                                To_his_meas_k[0],To_his_meas_k[1],To_his_meas_k[2],To_his_meas_k[3], u[0], u[1], self._autoerror['north'])
            
            Tz_south_pred_ph[k] = self.ZoneTemperature(self.params_south,u[4], Tz_core_his_meas_k[-1],Tz_east_his_meas_k[-1],Tz_north_his_meas_k[-1],Tz_west_his_meas_k[-1], u[8],u[5],u[6],u[7],u[9], Tz_south_his_meas_k[0],Tz_south_his_meas_k[1],Tz_south_his_meas_k[2],Tz_south_his_meas_k[3], 
                                To_his_meas_k[0], To_his_meas_k[1], To_his_meas_k[2], To_his_meas_k[3], u[0], u[1],self._autoerror['south'])
            
            Tz_west_pred_ph[k] = self.ZoneTemperature(self.params_west,u[4], Tz_core_his_meas_k[-1],Tz_east_his_meas_k[-1],Tz_north_his_meas_k[-1],Tz_south_his_meas_k[-1], u[9],u[5],u[6],u[7],u[8], Tz_west_his_meas_k[0],Tz_west_his_meas_k[1],Tz_west_his_meas_k[2],Tz_west_his_meas_k[3], 
                                To_his_meas_k[0],To_his_meas_k[1],To_his_meas_k[2], To_his_meas_k[3], u[0], u[1], self._autoerror['west'])
            
            ## updat the historical data
            Tz_core_his_meas_k.append(Tz_core_pred_ph[k])
            Tz_east_his_meas_k.append(Tz_east_pred_ph[k])
            Tz_north_his_meas_k.append(Tz_north_pred_ph[k])
            Tz_south_his_meas_k.append(Tz_south_pred_ph[k])
            Tz_west_his_meas_k.append(Tz_west_pred_ph[k])

            Tz_core_his_meas_k = Tz_core_his_meas_k[1:]
            Tz_east_his_meas_k = Tz_east_his_meas_k[1:]
            Tz_north_his_meas_k = Tz_north_his_meas_k[1:]
            Tz_south_his_meas_k = Tz_south_his_meas_k[1:]
            Tz_west_his_meas_k = Tz_west_his_meas_k[1:]

            ## calculate the change of control action: this is wrong: 
            # control actions except slack
            normalizer = [1/(self.u_ub[i]-self.u_lb[i]) for i in range(self.number_inputs)]
            du_k = u - u_prev
            u_prev = u  # update previous actions in PH
            du_k_normalized = [normalizer[i]*du_k[i] for i in range(2,self.number_inputs-1)]
            du_k_nom2 = ca.sumsqr(ca.vertcat(*du_k_normalized))/len(du_k_normalized)
            #print(du_k_nom2)
            # Objective Function: Minimize energy cost + slack variable + action change rate
            fo = self.w[0] * price_ph[k] * P_pred_ph[k] * self.dt/3600./1000. + \
                 self.w[1] * u[-1]**2 + \
                 self.w[2] * du_k_nom2
            fval.append(fo)
        
        fval_sum = ca.sum1(ca.vertcat(*fval))
        obj=ca.Function('fval',[U],[fval_sum]) # this is the ultimate objective function
        f = obj(U) # get the objective value

        ## get overshoot and undershoot for each step
        # current time step
        g = []
        lbg = []
        ubg = []
        u_lb = []
        u_ub = []
               
        for k in range(self.PH):
            # future time        
            t = int(time+k*self.dt)
            t = int((t % 86400)/3600)  # hour index 0~23           
            if t>=self.occ_start and t<self.occ_end:
                if self.dos_attack_core_VAV:
                    # resilient bounds under DR attack on VAV box of core zone
                    u_lb += [1, 1, 5, 15.6, 11.8, 0.00, 0.05, 0.05, 0.05, 0.04, 0.00]  # Zero lower bound due to DoS attack
                    u_ub += [1, 1, 10, 29.4, 18, 0.01, 0.90, 0.95, 0.95, 0.70, 0.10] # near zero upper bound due to DoS attack
                else:    
                    # nominal bounds under nominal operaiton
                    u_lb += [1, 1, 5, 15.6, 11.8, 0.23, 0.05, 0.05, 0.05, 0.04, 0.00]  # lower bound of zonal air flow rate is not zero because of minimum ventilation requirement
                    u_ub += [1, 1, 10, 29.4, 18, 2.80, 0.90, 0.95, 0.95, 0.70, 0.10]                     
            else:
              u_lb += [0, 0, 5, 20, 11.8, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00] 
              u_ub += [0, 0, 10, 20, 18, 0.01, 0.01, 0.01, 0.01, 0.01, 0.10]
              
            # inequality constraints
            eps = U[self.number_inputs*k+10] # eps: slack variable
            Tchw = U[self.number_inputs*k+2]
            Tcw = U[self.number_inputs*k+3]
            Tsa = U[self.number_inputs*k+4]

            g += [Tz_core_pred_ph[k]+eps,Tz_core_pred_ph[k]-eps, 
                Tz_east_pred_ph[k]+eps,Tz_east_pred_ph[k]-eps, 
                Tz_north_pred_ph[k]+eps,Tz_north_pred_ph[k]-eps, 
                Tz_south_pred_ph[k]+eps,Tz_south_pred_ph[k]-eps, 
                Tz_west_pred_ph[k]+eps,Tz_west_pred_ph[k]-eps]
            g += [(Tsa-11.8)/6.2*5+5 - Tchw] # Tsa is constrained by Tchw, -0.1 <= 0.8Tsa - 4.52 <= 1
            #g += [To_pred_ph[k] - Tcw] # Tcw is constraned by Toa, 0.1 <= To_pred_ph[k] - Tcw <= 7*(1-RHo_pred_ph[k])
            Td = To_pred_ph[k]
            RH = RHo_pred_ph[k]*100
            Twet = Td*math.atan(0.151977*(RH+8.313659)**0.5)+math.atan(Td+RH)-math.atan(RH-1.676331)+0.00391838*RH**1.5*math.atan(0.023101*RH)-4.686035
            g += [Tcw - Twet] # Tcw is constraned by Toa, 1.5 <= Tcw - Twet <= 3. C

            # get upper and lower T bound
            lbg += [T_lower[t], 0.]*self.number_zones
            lbg += [-0.1]
            lbg += [1.5]
            ubg += [ca.inf, T_upper[t]]*self.number_zones
            ubg += [1.]
            #ubg += [7.*(1-RHo_pred_ph[k])]
            ubg += [3.]
        
        # Apply fixed variables to bounds (scheduled controls as hard constraints)
        for k in range(self.PH):
            for var_name, var_value in fixed_vars.items():
                if var_name in var_index_map:
                    var_idx = var_index_map[var_name]
                    u_lb[k*self.number_inputs + var_idx] = var_value
                    u_ub[k*self.number_inputs + var_idx] = var_value
        
        ## invoke the solver: IPOPT (NLP) or Bonmin (MINLP)
        options = {"print_time": True,"ipopt": {"max_iter": 200}} # "tol":0.01, "acceptable_tol": 0.01, "acceptable_obj_change_tol": 10**-2, "linear_solver": "mumps" from https://coin-or.github.io/Ipopt/OPTIONS.html
        solver = ca.nlpsol("solver","ipopt",{"x":U, "f":f, 'g': ca.vertcat(*g)}, options)
        # discrete = [True,True,False,False,False,False,False,False,False,False,False]*self.PH
        # solver = nlpsol('nlp_solver', 'bonmin', {"x":U,"f":f, 'g': vertcat(*g)}, {"discrete": discrete}) # "bonmin": {"max_iter": 100}  "time_limit": 60
        ## need to add an equality constraint of bcp and bahu
        res = solver(x0=u_ini, lbx=u_lb, ubx=u_ub, lbg=lbg, ubg=ubg)
        #print(solver.stats())
        #print(res)
        print("\nsolution x:",res['x'])
        print("solution f:",res['f'])
        
        # Extract solver status
        solver_status = {
            'return_status': solver.stats().get('return_status', 'unknown'),
            'success': solver.stats().get('success', False)
        }
        
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
    
    def get_open_loop_preds(self, u_opt_ph: np.ndarray):
        """
        Get open-loop predictions over prediction horizon
        :param u_opt_ph: optimal control actions over prediction horizon
        """
        
        time = self.time

        ### get states and predictions at current time step
        # get predictions at current time step
        To_pred_ph = self.predictor['Toa'] # predicted outdoor air temperature 
        RHo_pred_ph = self.predictor['RHoa'] # predicted relative humidity 
        price_ph = self.predictor['price'] # predicted electricity price (TOU)


        # get historical temperaure measurements
        Tz_core_his_meas = np.array(self.states['Tz_core_his_meas'][:])-273.15 # historical zone temperatur states for zone temperture prediction model
        Tz_east_his_meas = np.array(self.states['Tz_east_his_meas'][:])-273.15
        Tz_north_his_meas = np.array(self.states['Tz_north_his_meas'][:])-273.15
        Tz_south_his_meas = np.array(self.states['Tz_south_his_meas'][:])-273.15
        Tz_west_his_meas = np.array(self.states['Tz_west_his_meas'][:])-273.15
        To_his_meas = np.array(self.states['To_his_meas'][:]) # historical outdoor air temperatur states for zone temperture prediction model

        # get historical predicted zone temperatures
        Tz_core_his_pred = np.array(self.states['Tz_core_his_pred']) - 273.15
        Tz_east_his_pred = np.array(self.states['Tz_east_his_pred']) - 273.15
        Tz_north_his_pred = np.array(self.states['Tz_north_his_pred']) - 273.15
        Tz_south_his_pred = np.array(self.states['Tz_south_his_pred']) - 273.15
        Tz_west_his_pred = np.array(self.states['Tz_west_his_pred']) - 273.15

        ## optimal control actions over prediction horizon
        U = u_opt_ph

        # zone temperature bounds - need check with the high-fidelty model
        T_upper = np.array([30.0 for i in range(24)])
        T_upper[self.occ_start:self.occ_end] = 24.0 # used to be 26.0, HIL for 24.0 C
        T_lower = np.array([18.0 for i in range(24)])
        T_lower[self.occ_start:self.occ_end] = 20.0 # 22.0

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
        fval = []

        Tz_core_his_meas_k = [Tz for Tz in Tz_core_his_meas]
        Tz_east_his_meas_k = [Tz for Tz in Tz_east_his_meas]
        Tz_north_his_meas_k = [Tz for Tz in Tz_north_his_meas]
        Tz_south_his_meas_k = [Tz for Tz in Tz_south_his_meas]
        Tz_west_his_meas_k = [Tz for Tz in Tz_west_his_meas]
        To_his_meas_k = [To for To in To_his_meas]


        # main loop
        for k in range(len(u_opt_ph)):
            ## arguments (10 control variabls + 1 slack variable): u[0] ~ u[10] = [bcp, bahu, Tchw, Tcw, Tsa, Vcore, Veast, Vnorth, Vsouth, Vwest, \epsilon]
            u = U[k*self.number_inputs:(k+1)*self.number_inputs]
            
            ## Calculate the total power consumption and energy cost
            # get power model inputs
            Tz_avg_k = (Tz_core_his_meas_k[-1]+Tz_east_his_meas_k[-1]+Tz_north_his_meas_k[-1] +
                        Tz_south_his_meas_k[-1]+Tz_west_his_meas_k[-1])/self.number_zones

            P_pred_ph[k] = u[0]*self.ChillerPlantPower(self.params_chiller,u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],Tz_avg_k,To_pred_ph[k],RHo_pred_ph[k]) + \
                            u[1]*self.FanPower(self.params_fan,u[5],u[6],u[7],u[8],u[9]) # need to make sure chiller power is positive

            ## zonal temperature prediction
            Tz_core_pred_ph[k] = self.ZoneTemperature(self.params_core, u[4], Tz_east_his_meas_k[-1],Tz_north_his_meas_k[-1],Tz_south_his_meas_k[-1],Tz_west_his_meas_k[-1], u[5],u[6],u[7],u[8],u[9], Tz_core_his_meas_k[0], Tz_core_his_meas_k[1], Tz_core_his_meas_k[2], Tz_core_his_meas_k[3], 
                                To_his_meas_k[0], To_his_meas_k[1], To_his_meas_k[2], To_his_meas_k[3], u[0], u[1], self._autoerror['core'])
            
            Tz_east_pred_ph[k] = self.ZoneTemperature(self.params_east,u[4], Tz_core_his_meas_k[-1],Tz_north_his_meas_k[-1],Tz_south_his_meas_k[-1],Tz_west_his_meas_k[-1], u[6],u[5],u[7],u[8],u[9], Tz_east_his_meas_k[0], Tz_east_his_meas_k[1],Tz_east_his_meas_k[2],Tz_east_his_meas_k[3], 
                                To_his_meas_k[0],To_his_meas_k[1],To_his_meas_k[2],To_his_meas_k[3], u[0], u[1], self._autoerror['east'])
            
            Tz_north_pred_ph[k] = self.ZoneTemperature(self.params_north,u[4], Tz_core_his_meas_k[-1],Tz_east_his_meas_k[-1],Tz_south_his_meas_k[-1],Tz_west_his_meas_k[-1], u[7],u[5],u[6],u[8],u[9], Tz_north_his_meas_k[0],Tz_north_his_meas_k[1],Tz_north_his_meas_k[2],Tz_north_his_meas_k[3], 
                                To_his_meas_k[0],To_his_meas_k[1],To_his_meas_k[2],To_his_meas_k[3], u[0], u[1], self._autoerror['north'])
            
            Tz_south_pred_ph[k] = self.ZoneTemperature(self.params_south,u[4], Tz_core_his_meas_k[-1],Tz_east_his_meas_k[-1],Tz_north_his_meas_k[-1],Tz_west_his_meas_k[-1], u[8],u[5],u[6],u[7],u[9], Tz_south_his_meas_k[0],Tz_south_his_meas_k[1],Tz_south_his_meas_k[2],Tz_south_his_meas_k[3], 
                                To_his_meas_k[0], To_his_meas_k[1], To_his_meas_k[2], To_his_meas_k[3], u[0], u[1],self._autoerror['south'])
            
            Tz_west_pred_ph[k] = self.ZoneTemperature(self.params_west,u[4], Tz_core_his_meas_k[-1],Tz_east_his_meas_k[-1],Tz_north_his_meas_k[-1],Tz_south_his_meas_k[-1], u[9],u[5],u[6],u[7],u[8], Tz_west_his_meas_k[0],Tz_west_his_meas_k[1],Tz_west_his_meas_k[2],Tz_west_his_meas_k[3], 
                                To_his_meas_k[0],To_his_meas_k[1],To_his_meas_k[2], To_his_meas_k[3], u[0], u[1], self._autoerror['west'])
            
            ## updat the historical data
            Tz_core_his_meas_k.append(Tz_core_pred_ph[k])
            Tz_east_his_meas_k.append(Tz_east_pred_ph[k])
            Tz_north_his_meas_k.append(Tz_north_pred_ph[k])
            Tz_south_his_meas_k.append(Tz_south_pred_ph[k])
            Tz_west_his_meas_k.append(Tz_west_pred_ph[k])

            Tz_core_his_meas_k = Tz_core_his_meas_k[1:]
            Tz_east_his_meas_k = Tz_east_his_meas_k[1:]
            Tz_north_his_meas_k = Tz_north_his_meas_k[1:]
            Tz_south_his_meas_k = Tz_south_his_meas_k[1:]
            Tz_west_his_meas_k = Tz_west_his_meas_k[1:]
        
        # Calculate comfort violation (degree-hours)
        comfort_violations = []
        violation_k = 0.0
        for k in range(len(u_opt_ph)):
            violation_k += max(0, Tz_core_pred_ph - self.T_upper) + max(0, self.T_lower - Tz_core_pred_ph)
            violation_k += max(0, Tz_east_pred_ph - self.T_upper) + max(0, self.T_lower - Tz_east_pred_ph)
            violation_k += max(0, Tz_north_pred_ph - self.T_upper) + max(0, self.T_lower - Tz_north_pred_ph)
            violation_k += max(0, Tz_south_pred_ph - self.T_upper) + max(0, self.T_lower - Tz_south_pred_ph)
            violation_k += max(0, Tz_west_pred_ph - self.T_upper) + max(0, self.T_lower - Tz_west_pred_ph)
        comfort_violations.append(violation_k * self.dt / 3600.0)
        
        return {
            'P_pred [W]': P_pred_ph,
            'total_energy_cost [kWh]': sum([price_ph[k] * P_pred_ph[k] * self.dt/3600./1000. for k in range(len(u_opt_ph))]),
            'Tz_core_pred [C]': Tz_core_pred_ph, 
            'Tz_east_pred [C]': Tz_east_pred_ph,
            'Tz_north_pred [C]': Tz_north_pred_ph,
            'Tz_south_pred [C]': Tz_south_pred_ph,
            'Tz_west_pred [C]': Tz_west_pred_ph,
            'comfort_violation [C*hour]': comfort_violations,
            'total_comfort_violation [C*hour]': sum(comfort_violations)
        }
        
    def get_core_temp_pred(self, u, autoerror):
        """Get predicted temperature of core zone using optimal control inputs
        """
        Tz_core_his = np.array(self.states['Tz_core_his_meas'][:])-273.15 # historical zone temperatur states for zone temperture prediction model
        Tz_east_his = np.array(self.states['Tz_east_his_meas'][:])-273.15
        Tz_north_his = np.array(self.states['Tz_north_his_meas'][:])-273.15
        Tz_south_his = np.array(self.states['Tz_south_his_meas'][:])-273.15
        Tz_west_his = np.array(self.states['Tz_west_his_meas'][:])-273.15
        To_mea_his = np.array(self.states['To_his_meas'][:]) # historical outdoor air temperatur states for zone temperture prediction model

        Tz_pred = self.ZoneTemperature(self.params_core,u[4], Tz_east_his[-1],Tz_north_his[-1],Tz_south_his[-1],Tz_west_his[-1], u[5],u[6],u[7],u[8],u[9], Tz_core_his[0],Tz_core_his[1],Tz_core_his[2],Tz_core_his[3], To_mea_his[0],To_mea_his[1],To_mea_his[2],To_mea_his[3], u[0], u[1], autoerror)

        return Tz_pred

    def get_east_temp_pred(self, u, autoerror):
        """Get predicted temperature of east zone using optimal control inputs
        """
        Tz_core_his = np.array(self.states['Tz_core_his_meas'][:])-273.15 # historical zone temperatur states for zone temperture prediction model
        Tz_east_his = np.array(self.states['Tz_east_his_meas'][:])-273.15
        Tz_north_his = np.array(self.states['Tz_north_his_meas'][:])-273.15
        Tz_south_his = np.array(self.states['Tz_south_his_meas'][:])-273.15
        Tz_west_his = np.array(self.states['Tz_west_his_meas'][:])-273.15
        To_mea_his = np.array(self.states['To_his_meas'][:]) # historical outdoor air temperatur states for zone temperture prediction model

        Tz_pred = self.ZoneTemperature(self.params_east,u[4], Tz_core_his[-1],Tz_north_his[-1],Tz_south_his[-1],Tz_west_his[-1], u[6],u[5],u[7],u[8],u[9], Tz_east_his[0],Tz_east_his[1],Tz_east_his[2],Tz_east_his[3], To_mea_his[0],To_mea_his[1],To_mea_his[2],To_mea_his[3], u[0], u[1],autoerror)

        return Tz_pred

    def get_north_temp_pred(self, u, autoerror):
        """Get predicted temperature of north zone using optimal control inputs
        """
        Tz_core_his = np.array(self.states['Tz_core_his_meas'][:])-273.15 # historical zone temperatur states for zone temperture prediction model
        Tz_east_his = np.array(self.states['Tz_east_his_meas'][:])-273.15
        Tz_north_his = np.array(self.states['Tz_north_his_meas'][:])-273.15
        Tz_south_his = np.array(self.states['Tz_south_his_meas'][:])-273.15
        Tz_west_his = np.array(self.states['Tz_west_his_meas'][:])-273.15
        To_mea_his = np.array(self.states['To_his_meas'][:]) # historical outdoor air temperatur states for zone temperture prediction model

        Tz_pred = self.ZoneTemperature(self.params_north, u[4], Tz_core_his[-1],Tz_east_his[-1],Tz_south_his[-1],Tz_west_his[-1], u[7],u[5],u[6],u[8],u[9], Tz_north_his[0],Tz_north_his[1], Tz_north_his[2],Tz_north_his[3], To_mea_his[0], To_mea_his[1], To_mea_his[2], To_mea_his[3], u[0], u[1], autoerror)

        return Tz_pred

    def get_south_temp_pred(self, u, autoerror):
        """Get predicted temperature of south zone using optimal control inputs
        """
        Tz_core_his = np.array(self.states['Tz_core_his_meas'][:])-273.15 # historical zone temperatur states for zone temperture prediction model
        Tz_east_his = np.array(self.states['Tz_east_his_meas'][:])-273.15
        Tz_north_his = np.array(self.states['Tz_north_his_meas'][:])-273.15
        Tz_south_his = np.array(self.states['Tz_south_his_meas'][:])-273.15
        Tz_west_his = np.array(self.states['Tz_west_his_meas'][:])-273.15
        To_mea_his = np.array(self.states['To_his_meas'][:]) # historical outdoor air temperatur states for zone temperture prediction model

        Tz_pred = self.ZoneTemperature(self.params_south,u[4], Tz_core_his[-1],Tz_east_his[-1],Tz_north_his[-1],Tz_west_his[-1], u[8],u[5],u[6],u[7],u[9], Tz_south_his[0],Tz_south_his[1],Tz_south_his[2],Tz_south_his[3], To_mea_his[0],To_mea_his[1],To_mea_his[2],To_mea_his[3], u[0], u[1], autoerror)

        return Tz_pred

    def get_west_temp_pred(self, u, autoerror):
        """Get predicted temperature of west zone using optimal control inputs
        """
        Tz_core_his = np.array(self.states['Tz_core_his_meas'][:])-273.15 # historical zone temperatur states for zone temperture prediction model
        Tz_east_his = np.array(self.states['Tz_east_his_meas'][:])-273.15
        Tz_north_his = np.array(self.states['Tz_north_his_meas'][:])-273.15
        Tz_south_his = np.array(self.states['Tz_south_his_meas'][:])-273.15
        Tz_west_his = np.array(self.states['Tz_west_his_meas'][:])-273.15
        To_mea_his = np.array(self.states['To_his_meas'][:]) # historical outdoor air temperatur states for zone temperture prediction model

        Tz_pred = self.ZoneTemperature(self.params_west,u[4], Tz_core_his[-1],Tz_east_his[-1],Tz_north_his[-1],Tz_south_his[-1], u[9],u[5],u[6],u[7],u[8], Tz_west_his[0],Tz_west_his[1],Tz_west_his[2],Tz_west_his[3], To_mea_his[0],To_mea_his[1],To_mea_his[2],To_mea_his[3], u[0], u[1], autoerror)

        return Tz_pred

    def get_power_pred(self, u):
        """Get predicted power consumption
        """
        Toa = self.predictor['Toa'] # predicted outdoor air temperature
        RHoa = self.predictor['RHoa'] # predicted relative humidity
        Tz_core_his = np.array(self.states['Tz_core_his_meas'][:])-273.15 # historical zone temperatur states for zone temperture prediction model
        Tz_East_his = np.array(self.states['Tz_east_his_meas'][:])-273.15
        Tz_North_his = np.array(self.states['Tz_north_his_meas'][:])-273.15
        Tz_South_his = np.array(self.states['Tz_south_his_meas'][:])-273.15
        Tz_West_his = np.array(self.states['Tz_west_his_meas'][:])-273.15
        Tz_avg = np.mean([Tz_core_his[0],Tz_East_his[0],Tz_North_his[0],Tz_South_his[0],Tz_West_his[0]])

        P = u[0]*self.ChillerPlantPower(self.params_chiller,u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],Tz_avg,Toa[0],RHoa[0]) + \
            u[1]*self.FanPower(self.params_fan,u[5],u[6],u[7],u[8],u[9])

        return P

    def upward_DF(self, u):
        """Get upward demand flexibility (decrease the zone air temperature setpoint by 2 C to gain the upward flexibility)
        """
        Toa = self.predictor['Toa'] # predicted outdoor air temperature
        RHoa = self.predictor['RHoa'] # predicted relative humidity
        Tz_core_his = np.array(self.states['Tz_core_his_meas'][:])-273.15 # historical zone temperatur states for zone temperture prediction model
        Tz_East_his = np.array(self.states['Tz_east_his_meas'][:])-273.15
        Tz_North_his = np.array(self.states['Tz_north_his_meas'][:])-273.15
        Tz_South_his = np.array(self.states['Tz_south_his_meas'][:])-273.15
        Tz_West_his = np.array(self.states['Tz_west_his_meas'][:])-273.15
        Tz_avg = -2. + np.mean([Tz_core_his[0],Tz_East_his[0],Tz_North_his[0],Tz_South_his[0],Tz_West_his[0]])

        P = u[0]*self.ChillerPlantPower(self.params_chiller_Tset22,u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],Tz_avg,Toa[0],RHoa[0]) + \
            u[1]*self.FanPower(self.params_fan_Tset22,u[5],u[6],u[7],u[8],u[9])
        return P

    def downward_DF(self, u):
        """Get downward demand flexibility (increase the zone air temperature setpoint by 2 C to gain the downward flexibility)
        """
        Toa = self.predictor['Toa'] # predicted outdoor air temperature
        RHoa = self.predictor['RHoa'] # predicted relative humidity
        Tz_core_his = np.array(self.states['Tz_core_his_meas'][:])-273.15 # historical zone temperatur states for zone temperture prediction model
        Tz_East_his = np.array(self.states['Tz_east_his_meas'][:])-273.15
        Tz_North_his = np.array(self.states['Tz_north_his_meas'][:])-273.15
        Tz_South_his = np.array(self.states['Tz_south_his_meas'][:])-273.15
        Tz_West_his = np.array(self.states['Tz_west_his_meas'][:])-273.15
        Tz_avg = 2. + np.mean([Tz_core_his[0],Tz_East_his[0],Tz_North_his[0],Tz_South_his[0],Tz_West_his[0]])

        P = u[0]*self.ChillerPlantPower(self.params_chiller_Tset26,u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],Tz_avg,Toa[0],RHoa[0]) + \
            u[1]*self.FanPower(self.params_fan_Tset26,u[5],u[6],u[7],u[8],u[9])
        return P

    def FanPower(self, params_fan, VCor, VEas, VNor, VSou, VWes,):
        """ AHU Fan Power Model
        """
        alpha = np.array(params_fan['alpha'])
        mz = (VCor+VEas+VNor+VSou+VWes)*1.29 #  m3/s to kg/s
        fan_pred = alpha[0]+alpha[1]*mz+alpha[2]*mz**2+alpha[3]*mz**3
        return fan_pred

    def ChillerPlantPower(self, params_chiller, TCHWSup, TCWSup, TSup, VCor, VEas, VNor, VSou, VWes, Tz_avg, Toa, RHoa):
        """ Chiller plant Power Model
        """
        mz = (VCor+VEas+VNor+VSou+VWes)*1.29 #  m3/s to kg/s
        alpha = np.array(params_chiller['alpha'])
        cp_pred = alpha[0]+alpha[1]*TCHWSup+alpha[2]*TCHWSup**2+alpha[3]*TCWSup+alpha[4]*TCWSup**2+alpha[5]*TCWSup*TCHWSup+\
            alpha[6]*(TSup-Tz_avg)*mz+\
            alpha[7]*Toa+alpha[8]*Toa**2+alpha[9]*RHoa+alpha[10]*RHoa**2
        return cp_pred

    def ZoneTemperature(self, params_temp, TSup, Tz_other_a,Tz_other_b,Tz_other_c,Tz_other_d, V_now,V_other_a,V_other_b,V_other_c,V_other_d, Tz_now_4,Tz_now_3,Tz_now_2,Tz_now_1, OA_4,OA_3,OA_2,OA_1,  bc, bahu, autoerror=0):
        """ Zone temperature Model: ARX model with an auto-correction term
        """
        TSup, Tz_five, mz_five, Tz_mea_his, To_mea_his, bc, bahu = TSup, [Tz_now_1,Tz_other_a,Tz_other_b,Tz_other_c,Tz_other_d], [V_now*1.29,V_other_a*1.29,V_other_b*1.29,V_other_c*1.29,V_other_d*1.29], [Tz_now_4,Tz_now_3,Tz_now_2,Tz_now_1], [OA_4,OA_3,OA_2,OA_1], bc, bahu
        alpha = params_temp['alpha'] # list
        beta = params_temp['beta'] # list
        gamma = params_temp['gamma'] # list
        n_alpha = len(alpha)
        n_beta = len(beta)
        n_gamma = len(beta)

        Tz_pred = ca.sum1(ca.vertcat(*[alpha[i]*Tz_mea_his[i] for i in range(n_alpha)])) \
                    + ca.sum1(ca.vertcat(*[beta[i]*To_mea_his[i] for i in range(n_beta)])) \
                    + ca.sum1(ca.vertcat(*[gamma[i]*bc*bahu*mz_five[i]*(TSup-Tz_five[i]) for i in range(n_gamma)])) \
                    ##+ ca.MX(autoerror)
        return Tz_pred
