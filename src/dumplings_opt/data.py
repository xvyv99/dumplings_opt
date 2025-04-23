from typing import Tuple, Literal, List
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import numpy.typing as npt
import networkx as nx

from pathlib import Path

class DumplingsDataBasic:    
    customer_num: np.uint64
    truck_possible_num: np.uint64

    customer_demand: npt.NDArray[np.uint64]
    preference_matrix: npt.NDArray[np.float64]

    r: np.uint64
    k: np.uint64
    f: np.uint64

    def __init__(self, customer_num: np.uint64, truck_num: np.uint64):
        self.customer_num = customer_num
        self.truck_possible_num = truck_num

        self.customer_demand = np.random.randint(20, 80, size=(self.customer_num))
        self.preference_matrix = np.random.rand(self.customer_num, self.truck_possible_num)
        self.r = np.random.randint(3, 15)
        self.k = np.random.randint(3, 15)
        if self.r < self.k:
            self.r, self.k = self.k, self.r
        self.f = np.random.randint(100, 400)

    def print_info(self):
        print("Dumplings Data Basic Information:")
        print(f"- Number of customers: {self.customer_num}")
        print(f"- Number of possible trucks: {self.truck_possible_num}")
        print(f"- Customer demand range: [{np.min(self.customer_demand)}, {np.max(self.customer_demand)}]")
        print(f"- Total customer demand: {np.sum(self.customer_demand)}")
        print(f"- Preference matrix shape: {self.preference_matrix.shape}")
        print(f"- Preference value range: [{np.min(self.preference_matrix):.4f}, {np.max(self.preference_matrix):.4f}]")
        print(f"- r value (upper bound): {self.r}")
        print(f"- k value (lower bound): {self.k}")
        print(f"- f value (fixed cost): {self.f}")
    
    def calc(self, x_np: npt.NDArray[np.uint8], y_np: npt.NDArray[np.uint8]) -> np.float64:
        I_num = self.customer_num
        J_num = self.truck_possible_num
        
        assert y_np.shape == (I_num, J_num), f"{y_np.shape} != {(I_num, J_num)}"
        assert x_np.shape == (J_num,) , f"{x_np.shape} != {(J_num, )}"
        
        x_np, y_np = x_np.astype(np.uint64), y_np.astype(np.uint64) # Prevent overflow

        r, k, f= self.r, self.k, self.f
        alpha = self.preference_matrix
        d = self.customer_demand
        
        res = 0
        for i in range(I_num):
            for j in range(J_num):
                res += (r-k)*alpha[i, j]*d[i]*y_np[i, j]

        for j in range(J_num):
            res -= f*x_np[j]
        return res

class DumplingsMap(DumplingsDataBasic):
    days: np.uint64 # Opt day count
    te: np.uint64 # travel_expense

    customer_names: List[str]
    truck_names: List[str]
    
    truck_possible_location: npt.NDArray[np.float64]
    customer_location: npt.NDArray[np.float64]

    def __init__(self, 
        customer_num: np.uint64, # TODO: variable customer number with constant customer possible number
        truck_num: np.uint64,
        days: np.uint64 = 1,
        travel_expense: np.uint64|None = None
    ):
        super().__init__(customer_num, truck_num)
        self.days = days
        if not travel_expense:
            self.te = int(self.f / 2)
    
    @staticmethod
    def generate_preference_matrix() -> npt.NDArray[np.float64]:
        pass
    
    @staticmethod
    def from_random(size: Tuple[np.uint64, np.uint64], customer_num: np.uint64, truck_num: np.uint64):
        res_map = DumplingsMap(customer_num, truck_num)
        
        rng = np.random.default_rng()
        res_map.truck_possible_location = np.column_stack((
            rng.uniform(0, size[0], size=res_map.dumplings_data.truck_possible_num),
            rng.uniform(0, size[1], size=res_map.dumplings_data.truck_possible_num)
        ))
        res_map.customer_location = np.column_stack((
            rng.uniform(0, size[0], size=res_map.dumplings_data.customer_num),
            rng.uniform(0, size[1], size=res_map.dumplings_data.customer_num)
        ))
        
        res_map.customer_names = ['C'+str(ind) for ind in range(res_map.dumplings_data.customer_num)]
        res_map.truck_names = ['T'+str(ind) for ind in range(res_map.dumplings_data.truck_possible_num)]

        return res_map

    @staticmethod
    def from_official(day: Literal[1, 2, 3, 4, 5], data_folder: str='basic_model_data'):
        FILE_PREFIX = 'round1-day'+str(day)
        
        data_path = Path(data_folder) / Path('day_'+str(day))
        assert data_path.exists(), f'Folder {data_path.name} not found!'

        df_truck = pd.read_csv(data_path / (FILE_PREFIX+'_truck_node_data.csv'))
        df_customer = pd.read_csv(data_path / (FILE_PREFIX+'_demand_node_data.csv'))
        df_problem_set = pd.read_csv(data_path / (FILE_PREFIX+'_problem_data.csv'))
        df_demand = pd.read_csv(data_path / (FILE_PREFIX+'_demand_truck_data.csv'))

        base_col = df_demand['scaled_demand']
        df_demand['scaled_demand_norm'] = (base_col - base_col.min()) / (base_col.max() - base_col.min())

        truck_possible_num = df_truck.shape[0]
        customer_num = df_customer.shape[0]
        
        res_map = object.__new__(DumplingsMap)
        
        res_map.customer_num = customer_num
        res_map.truck_possible_num = truck_possible_num
        
        res_map.r = df_problem_set.iloc[0]['burrito_price']
        res_map.k = df_problem_set.iloc[0]['ingredient_cost']
        res_map.f = df_problem_set.iloc[0]['truck_cost']

        res_map.customer_names = df_customer.loc[:, 'index'].to_list()
        res_map.truck_names = df_truck.loc[:, 'index'].to_list()

        res_map.customer_location = df_customer.loc[:, ['x', 'y']].to_numpy()
        res_map.truck_possible_location = df_truck.loc[:, ['x', 'y']].to_numpy()

        res_map.customer_demand = df_customer.loc[:, 'demand'].to_numpy()
        
        res_map.preference_matrix = np.full((res_map.customer_num, res_map.truck_possible_num), -1, dtype=np.float64)
        for ind, r in df_demand.iterrows():
            i = res_map.customer_names.index(r['demand_node_index'])
            j = res_map.truck_names.index(r['truck_node_index'])
            res_map.preference_matrix[i, j] = r['scaled_demand_norm']
        assert np.all(res_map.preference_matrix >= 0)
        return res_map

    def draw_location(self):
        plt.scatter(self.customer_location[:,0], self.customer_location[:,1], color='blue')
        plt.scatter(self.truck_possible_location[:,0], self.truck_possible_location[:,1], color='red')

class DumplingsSolutionBasic:
    dumplings_data: DumplingsDataBasic
    truck_selection: npt.NDArray[np.uint8]
    customer2truck_selection: npt.NDArray[np.uint8]

    def __init__(self, data: DumplingsDataBasic, x_np: npt.NDArray[np.uint8], y_np: npt.NDArray[np.uint8]):
        self.dumplings_data = data
        assert x_np.shape == (data.truck_possible_num, )
        assert y_np.shape == (data.customer_num, data.truck_possible_num)
        self.truck_selection = x_np
        self.customer2truck_selection = y_np

    def calc(self) -> np.float64:
        I_num = self.dumplings_data.customer_num
        J_num = self.dumplings_data.truck_possible_num
        
        x_np, y_np = self.x_np.astype(np.uint64), self.y_np.astype(np.uint64) # Prevent overflow

        r, k, f= self.dumplings_data.r, self.dumplings_data.k, self.dumplings_data.f
        alpha = self.dumplings_data.preference_matrix
        d = self.dumplings_data.customer_demand
        
        res = 0
        for i in range(I_num):
            for j in range(J_num):
                res += (r-k)*alpha[i, j]*d[i]*y_np[i, j]

        for j in range(J_num):
            res -= f*x_np[j]
        return res

    @staticmethod
    def from_customer_choice(data: DumplingsDataBasic, demand_choice: npt.NDArray[np.uint64]):
        assert demand_choice.shape == (data.customer_num, ), f"Invaild shape {demand_choice}!"
        assert np.all(demand_choice < data.truck_possible_num), "Value should less than truck id!"
        truck_id = np.unique(demand_choice)

        I_num = data.customer_num
        J_num = data.truck_possible_num

        x_np = np.zeros(J_num, dtype=np.uint8)
        y_np = np.zeros((I_num, J_num), dtype=np.uint8)

        for x in truck_id:
            x_np[x] = 1
        for i, j in enumerate(demand_choice):
            y_np[i, j] = 1
        res_sol = DumplingsSolutionBasic(data, x_np, y_np)
        return res_sol
    
    def display_connection(self):
        G = nx.DiGraph()

        truck_selection, custom2truck_selection = self.truck_selection, self.customer2truck_selection
        
        assert np.sum(truck_selection)>0, 'There is no truck!'
        
        truck_nodes = [('T'+str(ind), {'type': 'T'}) for ind in range(self.dumplings_data.truck_possible_num)]
        customer_nodes = [('C'+str(ind), {'type': 'C'}) for ind in range(self.dumplings_data.customer_num)]

        G.add_nodes_from(truck_nodes + customer_nodes)
        
        correspond_truck_id = [np.where(row == 1)[0] for row in custom2truck_selection]

        # Consider the situation that customer do not fully served(Like data in day 1).
        customers_id: List[np.uint64] = []
        trucks_id: List[np.uint64] = []
        
        for ind, x in enumerate(correspond_truck_id):
            # assert x.size>0, f'Customter {ind} have not be served!'
            if x.size>0:
                customers_id.append(ind)
                trucks_id.append(x[0])

        edge_lst = [('C'+str(customer_id), 'T'+str(truck_id)) for customer_id, truck_id in zip(customers_id, trucks_id)]
        G.add_edges_from(edge_lst)

        G_no_isolates = G.copy()
        G_no_isolates.remove_nodes_from(list(nx.isolates(G_no_isolates)))

        color_map = {
            "T": "gray",
            "C": "skyblue",
        }

        node_colors = [color_map[G_no_isolates.nodes[node]["type"]] for node in G_no_isolates.nodes()]
        
        # pos = nx.spring_layout(G_no_isolates, k=1, iterations=256, seed=42)
        pos = nx.spring_layout(G_no_isolates, 
            k=1.5,  
            iterations=256, 
            weight=None
        )   

        nx.draw_networkx_nodes(G_no_isolates, pos, node_color=node_colors, node_shape='s', node_size=700, edgecolors='black')
        nx.draw_networkx_edges(G_no_isolates, pos, width=1.5, alpha=0.7)
        nx.draw_networkx_labels(G_no_isolates, pos)