import pulp
import pandas as pd
from typing import Dict

class LP:
    def __init__(
        self,
        cv_bfg:int = 930, 
        cv_cog:int = 3900, 
        cv_ldg:int = 1600, 
        cv_mxg1:int = 1050, 
        cv_mxg2:int = 2300
    ):    
        self.cv_bfg = cv_bfg
        self.cv_cog = cv_cog
        self.cv_ldg = cv_ldg
        self.cv_mxg1 = cv_mxg1
        self.cv_mxg2 = cv_mxg2
        self.__G = ['BFG','COG','LDG']
        self.__P = ['BFS','SMS','BAT','SP','LCP','HSM']
        self.__D = ['online','1','2','3','4']
        self.__D_except_1 = ['online','2','3','4']
        self.__C_elec = 5100
        self.__W_hh = 4300
        self.__W_h = 10
        self.__W_d = 0.4
        self.__W_l = 10
        self.__W = {'BFS':0.4, 'SMS':0.4, 'LCP':0.4, 'HSM':0.4, 'BAT':0.4 , 'SP':0.4}
        self.__H_cap = 150000
        self.k_ll = 0.07
        self.k_l = 0.13
        self.k_n = 0.4
        self.k_h = 0.87
        self.k_hh = 0.93
        self.__drum_cap = {'online':7190,'1':93540,'2':61980,'3':59245,'4':51600}
        self.__demand_min = {'SMS':0,'BAT':80}
        self.__demand_max = {'SMS':2,'BAT':97}
        self.__timedelta = 0.25
        self.__k = 1.161*10**(-6)
        self.__eta = 0.3126
        
    def find_decision_variable_output(
        self, 
        bfg_gen:float, 
        cog_gen:float, 
        bfs_demand:float, 
        sp_lcp_hsm_demand:float, 
        h_t_minus_1:float
    ) -> Dict[str,float]:
        cv_bfg = self.cv_bfg
        cv_cog = self.cv_cog
        cv_mxg1 = self.cv_mxg1
        cv_mxg2 = self.cv_mxg2
        G = self.__G
        P = self.__P
        D = self.__D
        D_except_1 = self.__D_except_1
        C_elec = self.__C_elec
        W_hh = self.__W_hh
        W_h = self.__W_h
        W_d = self.__W_d
        W_l = self.__W_l
        W = self.__W
        H_cap = self.__H_cap
        k_ll = self.k_ll
        k_l = self.k_l
        k_h = self.k_h
        k_n = self.k_n
        k_hh = self.k_hh
        drum_cap = self.__drum_cap
        demand_min = self.__demand_min
        demand_max = self.__demand_max
        timedelta = self.__timedelta
        k = self.__k
        eta = self.__eta
        
        # Decision Variables
        p_gen = pulp.LpVariable('Power_Generation', lowBound=0, cat='Continuous')
        sd_plus = pulp.LpVariable('Holder_Deviation_at_Positive_side', lowBound=0, upBound=298496, cat='Continuous')
        sd_minus = pulp.LpVariable('Holder_Deviation_at_Negative_side', lowBound=0, upBound=298496, cat='Continuous')
        sd_h = pulp.LpVariable('Holder_Deviation_above_high_level', lowBound=0, cat='Continuous')
        sd_l = pulp.LpVariable('Holder_Deviation_below_low_level', lowBound=0, cat='Continuous')
        f_hh_g = pulp.LpVariable.dicts('Energy_Flared',(g for g in G),lowBound=0, cat='Continuous')
        f_cpp_g = pulp.LpVariable.dicts('CPP_Consumption',(g for g in G),lowBound=0, cat='Continuous')
        f_bfs_g = pulp.LpVariable.dicts('BFS_Consumption',(g for g in G),lowBound=0, cat='Continuous')
        f_sms_cog = pulp.LpVariable('SMS_Consumption',lowBound=0, cat='Continuous')
        f_sp_mxg = pulp.LpVariable('MXG2_Consumption at SP',lowBound=0, cat='Continuous')
        f_lcp_mxg = pulp.LpVariable('MXG2_Consumption at LCP',lowBound=0, cat='Continuous')
        f_hsm_mxg = pulp.LpVariable('MXG2_Consumption at HSM',lowBound=0, cat='Continuous')
        f_bat_mxg = pulp.LpVariable('MXG1_Consumption at BAT',lowBound=0, cat='Continuous')
        h_t = pulp.LpVariable('Holder_Level_at_Time_T',lowBound=0, upBound=150000, cat='Continuous')
        p_k = pulp.LpVariable.dicts('Production_Achieved_at_Pant_K',(p for p in P), lowBound=0, cat='Continuous')
        f_drum_bfg = pulp.LpVariable.dicts('BFG_at_Drum',(d for d in D),lowBound=0,cat='Continuous')
        f_drum_cog = pulp.LpVariable.dicts('COG_at_Drum',(d for d in D),lowBound=0,cat='Continuous')
        
        # Define model
        model = pulp.LpProblem('Cost',pulp.LpMinimize)
        
        # Objective function
        model += pulp.lpSum(W_hh*f_hh_g[g] for g in G) + W_d*(sd_plus+sd_minus) \
            + W_h*sd_h + W_l*sd_l - C_elec*p_gen - pulp.lpSum(W[p]*p_k[p] for p in P)
        
        # BFG flow conservation
        model += bfg_gen*timedelta - (pulp.lpSum(f_drum_bfg[d] for d in D_except_1) \
                                      + f_bfs_g['BFG'] + f_drum_bfg['1'])*timedelta \
            - (h_t - h_t_minus_1*H_cap/100) == f_cpp_g['BFG']*timedelta
        
        # COG flow conservation
        model += cog_gen - (pulp.lpSum(f_drum_cog[d] for d in D_except_1) \
                            + f_bfs_g['COG'] + f_drum_bfg['1'] + f_sms_cog) \
            - f_cpp_g['COG'] == f_hh_g['COG']
        
        # Drum 1 energy conversion constraint
        model += f_drum_bfg["1"]*cv_bfg + f_drum_cog["1"]*cv_cog >= f_bat_mxg*cv_mxg1
        
        # Drum except 1 energy conversion
        model += pulp.lpSum(f_drum_bfg[d] for d in D_except_1)*cv_bfg \
            + pulp.lpSum(f_drum_cog[d] for d in D_except_1)*cv_cog \
                >= (f_sp_mxg + f_lcp_mxg + f_hsm_mxg)*cv_mxg2
        
        model += (f_cpp_g['BFG']*cv_bfg*k + f_cpp_g['COG']*cv_cog*k)*eta == p_gen
        
        # Flow rate to power conversion
        model += f_bfs_g['BFG']*cv_bfg*k + f_bfs_g['COG']*cv_cog*k == p_k['BFS']
        model += f_sms_cog*cv_cog*k == p_k['SMS']
        model += f_bat_mxg*cv_mxg1*k == p_k['BAT']
        model += f_sp_mxg*cv_mxg2*k == p_k['SP']
        model += f_lcp_mxg*cv_mxg2*k == p_k['LCP']
        model += f_hsm_mxg*cv_mxg2*k == p_k['HSM']
        
        # Demand meeting constraints in terms of MW
        model += p_k["BFS"] >= bfs_demand
        model += p_k["BFS"] <= bfs_demand*1.05
        model += p_k["SP"] + p_k["LCP"] + p_k["HSM"] >= sp_lcp_hsm_demand
        model += p_k["SP"] + p_k["LCP"] + p_k["HSM"] <= sp_lcp_hsm_demand*1.05
        model += p_k["BAT"] >= demand_min["BAT"]
        model += p_k["BAT"] <= demand_max["BAT"]
        model += p_k["SMS"] >= demand_min["SMS"]
        model += p_k["SMS"] <= demand_max["SMS"]
        
        # Drum capacity constraint
        for d in D:
            model += f_drum_bfg[d] + f_drum_cog[d] <= drum_cap[d]
            model += f_drum_bfg[d] + f_drum_cog[d] >= 0
        
        # Logical Relations
        model += h_t - H_cap*k_n == sd_plus - sd_minus
        model += h_t <= H_cap*k_hh + f_hh_g['BFG']
        model += h_t >= H_cap*k_ll
        model += h_t >= H_cap*(k_h - k_l)
        model += h_t <= H_cap*k_hh + sd_h
        
        # Solving model
        model.solve()
        
        # Save decision variables into dataframe
        return {
            "P_Gen":pulp.value(p_gen), 
            "Holder_Dev_Pos":pulp.value(sd_plus), 
            "Holder_Dev_Neg":pulp.value(sd_minus), 
            "Holder_Dev_High":pulp.value(sd_h), 
            "Holder_Dev_Low":pulp.value(sd_l), 
            "BFS": pulp.value(p_k["BFS"]), 
            "SP_LCP_HSM":pulp.value(p_k["SP"]) + pulp.value(p_k["LCP"]) + pulp.value(p_k["HSM"]), 
            "SMS": pulp.value(p_k["SMS"]), 
            "BAT": pulp.value(p_k["BAT"]), 
            "BFG_Flared":pulp.value(f_hh_g["BFG"])*k, 
            "COG_Flared":pulp.value(f_hh_g["COG"])*k, 
            "Holder_Level": pulp.value(h_t)*100/H_cap, 
            "Model_Status": pulp.LpStatus[model.status]
        }

# if __name__ == "__main__":
#     cv_bfg = 930
#     cv_cog = 3900
#     cv_ldg = 1600
#     cv_mxg1 = 1050
#     cv_mxg2 = 2300
#     bfg_gen = 350000
#     cog_gen = 50000
#     bfs_demand = 182
#     sp_lcp_hsm_demand = 162
#     h_t_minus_1 = 32
#     lp = LP(cv_bfg,cv_cog,cv_ldg,cv_mxg1,cv_mxg2)
#     dv = lp.find_decision_variable_output(bfg_gen, cog_gen, bfs_demand, sp_lcp_hsm_demand, h_t_minus_1)
#     print(dv)