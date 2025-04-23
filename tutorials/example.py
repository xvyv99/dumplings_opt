from dumplings_opt import DumplingsModel, DumplingsDataBasic, DumplingsMap
import matplotlib.pyplot as plt

data_b1 = DumplingsDataBasic(5, 10)
model_b1 = DumplingsModel(data_b1)
model_b1.solve()
model_b1.print_status()
model_b1.display_connection()

plt.show()