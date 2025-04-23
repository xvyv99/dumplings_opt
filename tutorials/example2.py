from dumplings_opt import DumplingsModelAdv, DumplingsMap
import matplotlib.pyplot as plt

data_b1 = DumplingsMap.from_random(8 ,10, 2)
# data_b1 = DumplingsMap.from_official(2, 'data/basic_model_data')
data_b1.print_info()

model_b1 = DumplingsModelAdv(data_b1)
sol = model_b1.solve()

model_b1.print_status()
sol.display_connection()

plt.show()