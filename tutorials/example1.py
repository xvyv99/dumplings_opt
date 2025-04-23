from dumplings_opt import DumplingsModelBasic, DumplingsDataBasic, DumplingsMap
import matplotlib.pyplot as plt

data_b1 = DumplingsMap.from_official(2, 'data/basic_model_data')
data_b1.print_info()

model_b1 = DumplingsModelBasic(data_b1)
sol = model_b1.solve()

model_b1.print_status()
sol.display_connection()

plt.show()