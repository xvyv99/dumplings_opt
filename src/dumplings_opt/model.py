import time
import pulp as pl
from pulp import LpVariable, LpProblem, LpMaximize, lpSum, LpBinary
from typing import List, Tuple

import numpy.typing as npt
import numpy as np

import networkx as nx

from .data import DumplingsDataBasic, DumplingsMap, DumplingsSolutionBasic, DumplingsSolutionAdv

class DumplingsModelBasic:
    dumplings_data: DumplingsDataBasic
    
    lp_prob: LpProblem
    lp_var_x: dict
    lp_var_y: dict

    # soluton_demand_choice: npt.NDArray[np.uint64]

    def __init__(self, data: DumplingsDataBasic):
        self.dumplings_data = data
        
        self.lp_prob = LpProblem("DumplingsOpt"+str(int(time.time())), LpMaximize)
        
        I_num = self.dumplings_data.customer_num
        J_num = self.dumplings_data.truck_possible_num

        lp_var_x = LpVariable.dicts("x", range(J_num), cat=LpBinary)
        lp_var_y = LpVariable.dicts("y", (range(I_num), range(J_num)), cat=LpBinary)

        obj_expr = 0

        # Constrain 1: each customer served at most once
        for i in range(I_num):
            self.lp_prob += lpSum(lp_var_y[i][j] for j in range(J_num)) <= 1

        # Constrain 2: only assign if truck is open
        for j in range(J_num):
            for i in range(I_num):
                self.lp_prob += lp_var_y[i][j] <= lp_var_x[j]

        r, k, f= self.dumplings_data.r, self.dumplings_data.k, self.dumplings_data.f
        alpha = self.dumplings_data.preference_matrix
        d = self.dumplings_data.customer_demand

        for i in range(I_num):
            for j in range(J_num):
                obj_expr += (r-k)*alpha[i, j]*d[i]*lp_var_y[i][j]

        for j in range(J_num):
            obj_expr -= f*lp_var_x[j]

        self.lp_prob += obj_expr
        self.lp_var_x = lp_var_x
        self.lp_var_y = lp_var_y
        
    def solve(self):
        self.lp_prob.solve(pl.PULP_CBC_CMD(msg=False))
        return DumplingsSolutionBasic(self.dumplings_data, *self.get_var_np())

    def get_var_np(self) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        I_num = self.dumplings_data.customer_num
        J_num = self.dumplings_data.truck_possible_num

        lp_var_x_np = np.zeros(J_num, dtype=np.uint8)
        lp_var_y_np = np.zeros((I_num, J_num), dtype=np.uint8)
        
        for j in range(J_num):
            lp_var_x_np[j] = self.lp_var_x[j].value()
            for i in range(I_num):
                lp_var_y_np[i, j] = self.lp_var_y[i][j].value()

        return lp_var_x_np, lp_var_y_np
    
    def print_status(self):
        lp_var_x_np, lp_var_y_np = self.get_var_np()
        truck_ratio = np.sum(lp_var_x_np)/self.dumplings_data.truck_possible_num
        customer_ratio = np.sum(lp_var_y_np)/self.dumplings_data.customer_num
        print("Linear Programming Status: ", pl.LpStatus[self.lp_prob.status])
        print("Truck Setup Ratio: ", truck_ratio)
        print("Customer Served Ratio: ",customer_ratio)
        print("Object Value:", pl.value(self.lp_prob.objective))

class DumplingsModelAdv:
    dumplings_map: DumplingsMap
    
    lp_prob: LpProblem
    lp_var_x: dict
    lp_var_y: dict

    def __init__(self, map: DumplingsMap):
        self.dumplings_map = map
        
        self.lp_prob = LpProblem("DumplingsOptAdv"+str(int(time.time())), LpMaximize)
        
        I_num = self.dumplings_map.customer_num
        J_num = self.dumplings_map.truck_possible_num
        K_num = self.dumplings_map.timestep

        lp_var_x = LpVariable.dicts("x", (range(J_num), range(K_num)), cat=LpBinary)
        lp_var_y = LpVariable.dicts("y", (range(I_num), range(J_num), range(K_num)), cat=LpBinary)

        obj_expr = 0

        # Constrain 1: each customer served at most once
        for K in range(K_num):
            for i in range(I_num):
                self.lp_prob += lpSum(lp_var_y[i][j][K] for j in range(J_num)) <= 1

        # Constrain 2: only assign if truck is open
        for K in range(K_num):
            for j in range(J_num):
                for i in range(I_num):
                    self.lp_prob += lp_var_y[i][j][K] <= lp_var_x[j][K]

        r, k, f= self.dumplings_map.r, self.dumplings_map.k, self.dumplings_map.f
        alpha = self.dumplings_map.preference_matrix
        d = self.dumplings_map.customer_demand

        for K in range(K_num):
            for i in range(I_num):
                for j in range(J_num):
                    obj_expr += (r-k)*alpha[i, j, K]*d[i][K]*lp_var_y[i][j][K]

        for K in range(K_num):
            for j in range(J_num):
                obj_expr -= f*lp_var_x[j][K]

        self.lp_prob += obj_expr
        self.lp_var_x = lp_var_x
        self.lp_var_y = lp_var_y
        
    def solve(self):
        self.lp_prob.solve(pl.PULP_CBC_CMD(msg=False))
        return DumplingsSolutionAdv(self.dumplings_map, *self.get_var_np())

    def get_var_np(self) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        I_num = self.dumplings_map.customer_num
        J_num = self.dumplings_map.truck_possible_num
        K_num = self.dumplings_map.timestep

        lp_var_x_np = np.zeros((J_num, K_num), dtype=np.uint8)
        lp_var_y_np = np.zeros((I_num, J_num, K_num), dtype=np.uint8)
        
        for K in range(K_num):
            for j in range(J_num):
                lp_var_x_np[j, K] = self.lp_var_x[j][K].value()
                for i in range(I_num):
                    lp_var_y_np[i, j, K] = self.lp_var_y[i][j][K].value()

        return lp_var_x_np, lp_var_y_np
    
    def print_status(self):
        lp_var_x_np, lp_var_y_np = self.get_var_np()
        truck_ratio = np.sum(lp_var_x_np)/self.dumplings_map.truck_possible_num
        customer_ratio = np.sum(lp_var_y_np)/self.dumplings_map.customer_num
        print("Linear Programming Status: ", pl.LpStatus[self.lp_prob.status])
        print("Truck Setup Ratio: ", truck_ratio)
        print("Customer Served Ratio: ",customer_ratio)
        print("Object Value:", pl.value(self.lp_prob.objective))