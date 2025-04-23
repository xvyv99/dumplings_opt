from typing import Tuple, Literal, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy.typing as npt
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

    def eval(self, demand_choice: npt.NDArray[np.uint64]) -> np.float64:
        # `demand_choice` should like [1, 3, 4, 2, 4, ...], which index correspond to customer, value correspond to truck id.
        assert demand_choice.shape == (self.customer_num, ), f"Invaild shape {demand_choice}!"
        assert np.all(demand_choice < self.truck_possible_num), "Value should less than truck id!"
        truck_id = np.unique(demand_choice)

        I_num = self.customer_num
        J_num = self.truck_possible_num

        lp_var_x_np = np.zeros(J_num, dtype=np.uint8)
        lp_var_y_np = np.zeros((I_num, J_num), dtype=np.uint8)

        for x in truck_id:
            lp_var_x_np[x] = 1
        for i, j in enumerate(demand_choice):
            lp_var_y_np[i, j] = 1
        return self.calc(lp_var_x_np, lp_var_y_np)
    
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
    customer_names: List[str]
    truck_names: List[str]
    
    truck_possible_location: npt.NDArray[np.float64]
    customer_location: npt.NDArray[np.float64]

    def __init__(self, customer_num: np.uint64, truck_num: np.uint64):
        super().__init__(customer_num, truck_num)
    
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
        
        res_map.r = df_problem_set.iloc[0, 0]
        res_map.f = df_problem_set.iloc[0, 1]
        res_map.k = df_problem_set.iloc[0, 2]

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