import time
import pulp
from pulp import LpVariable, LpProblem, LpMaximize, lpSum, GLPK, LpBinary
from typing import Tuple

import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

from .data import DumplingsDataBasic

class DumplingsModel:
    dumplings_data: DumplingsDataBasic
    
    lp_prob: LpProblem
    lp_var_x: dict
    lp_var_y: dict
    
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

    def solve_bf(self):
        pass
        
    def solve(self):
        return self.lp_prob.solve(pulp.PULP_CBC_CMD(msg=False))

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
    
    def stat(self) -> Tuple[np.float64, np.float64]:
        lp_var_x_np, lp_var_y_np = self.get_var_np()
        return np.sum(lp_var_x_np)/self.dumplings_data.truck_possible_num, np.sum(lp_var_y_np)/self.dumplings_data.customer_num
    
    def print_status(self):
        truck_ratio, customer_ratio = self.stat()
        print("Linear Programming Status: ", pulp.LpStatus[self.lp_prob.status])
        print("Truck Setup Ratio: ", truck_ratio)
        print("Customer Served Ratio: ",customer_ratio)
        print("Object Value:", pulp.value(self.lp_prob.objective))
    
    def display_connection(self):
        G = nx.DiGraph()

        x_np, connection_mat = self.get_var_np()
        
        if np.sum(x_np)==0:
            print('There is no truck!')
            return 
        
        truck_nodes = [('T'+str(ind), {'type': 'T'}) for ind in range(self.dumplings_data.truck_possible_num)]
        customer_nodes = [('C'+str(ind), {'type': 'C'}) for ind in range(self.dumplings_data.customer_num)]

        G.add_nodes_from(truck_nodes + customer_nodes)
        
        correspond_truck_id = [np.where(row == 1)[0] for row in connection_mat]

        edge_lst = [('C'+str(customer_id), 'T'+str(truck_id[0])) for customer_id, truck_id in enumerate(correspond_truck_id)]
        G.add_edges_from(edge_lst)

        G_no_isolates = G.copy()
        G_no_isolates.remove_nodes_from(list(nx.isolates(G_no_isolates)))

        color_map = {
            "T": "gray",
            "C": "skyblue",
        }

        node_colors = [color_map[G_no_isolates.nodes[node]["type"]] for node in G_no_isolates.nodes()]
        
        pos = nx.spring_layout(G_no_isolates, k=1, iterations=256, seed=42)
        
        nx.draw_networkx_nodes(G_no_isolates, pos, node_color=node_colors, node_shape='s', node_size=700, edgecolors='black')
        nx.draw_networkx_edges(G_no_isolates, pos, width=1.5, alpha=0.7)
        nx.draw_networkx_labels(G_no_isolates, pos)