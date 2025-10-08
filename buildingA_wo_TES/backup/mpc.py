'''
This script is used to formulate the baseline and adaptive MPC.

Author: Guowen Li, Yangyang Fu
Email: guowenli@tamu.edu, yangyang.fu@tamu.edu 
Revisions:
    02/15/2022: Add auto-correction in zone temperature model
    2023: Implement adaptive MPC for Device Reinitialization Attack on Core zone's VAV box
'''
import casadi as ca
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plts
import json

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
        self.w = [1., 1., 1.]  # weights in objective function: : Minimize energy cost + slack variable + action change rate
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
        
        # if dos attack happens on core zone VAV box, change the upper and lower bounds of core zone air flow rate
        self.dos_attack_core_VAV = dos_attack_core_VAV

    def optimize(self):
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

        # get previous control actions
        u_prev = self.x_opt_0

        ### formulate optimization problem for MPC
        ## declare symbolic variabels
        U = ca.MX.sym("U", self.number_inputs*self.PH)

        # zone temperature bounds - need check with the high-fidelty model
        T_upper = np.array([30.0 for i in range(24)])
        T_upper[self.occ_start:self.occ_end] = 24.0 # 26.0
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
            Tz_core_pred_ph[k] = self.ZoneTemperature(self.params_core, u[4], u[5], Tz_core_his_meas_k[0], Tz_core_his_meas_k[1], Tz_core_his_meas_k[2], Tz_core_his_meas_k[3], 
                                To_his_meas_k[0], To_his_meas_k[1], To_his_meas_k[2], To_his_meas_k[3], u[0], u[1], self._autoerror['core'])
            Tz_east_pred_ph[k] = self.ZoneTemperature(self.params_east,u[4],u[6],Tz_east_his_meas_k[0], Tz_east_his_meas_k[1],Tz_east_his_meas_k[2],Tz_east_his_meas_k[3], 
                                To_his_meas_k[0],To_his_meas_k[1],To_his_meas_k[2],To_his_meas_k[3], u[0], u[1], self._autoerror['east'])
            Tz_north_pred_ph[k] = self.ZoneTemperature(self.params_north,u[4],u[7],Tz_north_his_meas_k[0],Tz_north_his_meas_k[1],Tz_north_his_meas_k[2],Tz_north_his_meas_k[3], 
                                To_his_meas_k[0],To_his_meas_k[1],To_his_meas_k[2],To_his_meas_k[3], u[0], u[1], self._autoerror['north'])
            Tz_south_pred_ph[k] = self.ZoneTemperature(self.params_south,u[4],u[8],Tz_south_his_meas_k[0],Tz_south_his_meas_k[1],Tz_south_his_meas_k[2],Tz_south_his_meas_k[3], 
                                To_his_meas_k[0], To_his_meas_k[1], To_his_meas_k[2], To_his_meas_k[3], u[0], u[1],self._autoerror['south'])
            Tz_west_pred_ph[k] = self.ZoneTemperature(self.params_west,u[4],u[9],Tz_west_his_meas_k[0],Tz_west_his_meas_k[1],Tz_west_his_meas_k[2],Tz_west_his_meas_k[3], 
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
        return(res)

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
        
    def get_core_temp_pred(self, u, autoerror):
        """Get predicted temperature of core zone using optimal control inputs
        """
        Tz_core_his = np.array(self.states['Tz_core_his_meas'][:])-273.15 # historical zone temperatur states for zone temperture prediction model
        To_mea_his = np.array(self.states['To_his_meas'][:]) # historical outdoor air temperatur states for zone temperture prediction model

        Tz_pred = self.ZoneTemperature(self.params_core,u[4],u[5],Tz_core_his[0],Tz_core_his[1],Tz_core_his[2],Tz_core_his[3], To_mea_his[0],To_mea_his[1],To_mea_his[2],To_mea_his[3], u[0], u[1], autoerror)

        return Tz_pred

    def get_east_temp_pred(self, u, autoerror):
        """Get predicted temperature of east zone using optimal control inputs
        """
        Tz_East_his = np.array(self.states['Tz_east_his_meas'][:])-273.15
        To_mea_his = np.array(self.states['To_his_meas'][:]) # historical outdoor air temperatur states for zone temperture prediction model

        Tz_pred = self.ZoneTemperature(self.params_east,u[4],u[6],Tz_East_his[0],Tz_East_his[1],Tz_East_his[2],Tz_East_his[3], To_mea_his[0],To_mea_his[1],To_mea_his[2],To_mea_his[3], u[0], u[1],autoerror)

        return Tz_pred

    def get_north_temp_pred(self, u, autoerror):
        """Get predicted temperature of north zone using optimal control inputs
        """
        Tz_North_his = np.array(self.states['Tz_north_his_meas'][:])-273.15
        To_mea_his = np.array(self.states['To_his_meas'][:]) # historical outdoor air temperatur states for zone temperture prediction model

        Tz_pred = self.ZoneTemperature(self.params_north, u[4], u[7], Tz_North_his[0], Tz_North_his[1], Tz_North_his[2],
                                       Tz_North_his[3], To_mea_his[0], To_mea_his[1], To_mea_his[2], To_mea_his[3], u[0], u[1], autoerror)

        return Tz_pred

    def get_south_temp_pred(self, u, autoerror):
        """Get predicted temperature of south zone using optimal control inputs
        """
        Tz_South_his = np.array(self.states['Tz_south_his_meas'][:])-273.15
        To_mea_his = np.array(self.states['To_his_meas'][:]) # historical outdoor air temperatur states for zone temperture prediction model

        Tz_pred = self.ZoneTemperature(self.params_south,u[4],u[8],Tz_South_his[0],Tz_South_his[1],Tz_South_his[2],Tz_South_his[3], To_mea_his[0],To_mea_his[1],To_mea_his[2],To_mea_his[3], u[0], u[1], autoerror)

        return Tz_pred

    def get_west_temp_pred(self, u, autoerror):
        """Get predicted temperature of west zone using optimal control inputs
        """
        Tz_West_his = np.array(self.states['Tz_west_his_meas'][:])-273.15
        To_mea_his = np.array(self.states['To_his_meas'][:]) # historical outdoor air temperatur states for zone temperture prediction model

        Tz_pred = self.ZoneTemperature(self.params_west,u[4],u[9],Tz_West_his[0],Tz_West_his[1],Tz_West_his[2],Tz_West_his[3], To_mea_his[0],To_mea_his[1],To_mea_his[2],To_mea_his[3], u[0], u[1], autoerror)

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

    def ZoneTemperature(self, params_temp,arg0,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11, autoerror=0):
        """ Zone temperature Model: ARX model with an auto-correction term
        """
        TSup, V, Tz_mea_his, To_mea_his, bc, bahu = arg0,arg1,[arg2,arg3,arg4,arg5],[arg6,arg7,arg8,arg9],arg10,arg11
        mz = V*1.29
        alpha = params_temp['alpha'] # list
        beta = params_temp['beta'] # list
        gamma = params_temp['gamma'] # list
        n_alpha = len(alpha)
        n_beta = len(beta)

        Tz_pred = ca.sum1(ca.vertcat(*[alpha[i]*Tz_mea_his[i] for i in range(n_alpha)])) \
                    + ca.sum1(ca.vertcat(*[beta[i]*To_mea_his[i] for i in range(n_beta)])) \
                    + ca.MX(gamma*bc*bahu*mz*(TSup-Tz_mea_his[-1])) \
                    + ca.MX(autoerror)
        return Tz_pred
