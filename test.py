import numpy as np
import os
from scipy.interpolate import griddata
from process_data import process_naca
from utils import extract_number
import matplotlib.pyplot as plt

area = [405, 425, 396, 404]
area = [i*128 for i in area]
cut = 1694107

data_dir = 'data/naca_new'
file_names = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
file_names.sort(key=extract_number)

print(area)
for file_name in file_names[25:26]:
    x, y, s = process_naca(os.path.join(data_dir, file_name), var="s", area=area, cut=cut, div128=False)
    print(x.shape, y.shape, s.shape)
    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

grid_size_x = 2000
grid_size_y = int(grid_size_x * (2 / 5))

xi, yi = np.mgrid[min(x):max(x):complex(0, grid_size_x), 
                          min(y):max(y):complex(0, grid_size_y)]

zi = griddata((x, y), s, (xi, yi), method='linear')
print(xi.shape, yi.shape, zi.shape)
plt.contourf(xi, yi, zi, levels=30)
plt.show()