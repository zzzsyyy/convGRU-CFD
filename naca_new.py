import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import os
import numpy
from matplotlib.path import Path
import utils
import process_data

plt.rcParams['savefig.dpi'] = 300

area = [405, 425, 396, 404]
cut = 1694107

def create_contour_plot(x, y, speed, file_name, fig_dir = None, naca='n0012.dat'):
    plt.figure(figsize=(8, 4))
    triangle = tri.Triangulation(x, y)
    levels = np.linspace(0, 0.1, 30)

    if naca != None:
        naca_filepath = os.path.join('data', naca)
        with open(naca_filepath, 'r') as infile:
            xx, yy = numpy.loadtxt(infile, dtype=float, unpack=True, skiprows=1)
    
        xx = xx * 16 + 391.75
        yy = yy * 16 + 400

        naca_path = Path(np.column_stack([xx, yy]))
        for i in range(len(x)):
            if naca_path.contains_point((x[i], y[i])):
                speed[i] = 0
        plt.plot(xx, yy, color='w', linestyle='-', linewidth=0.2)
        # plt.fill(xx, yy, color='white', alpha=1)

    plt.tricontourf(triangle, speed, levels=levels)
    plt.axis('equal')
    plt.xlim(area[0], area[1])
    plt.ylim(area[2], area[3]) 
    plt.colorbar(label='v')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Velocity at time step {file_name}')
    if fig_dir==None:
        plt.show()
    else:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, f'{file_name}.png'))
        plt.clf()

data_dir = 'data/naca_new'

file_names = [f for f in os.listdir(data_dir) if f.endswith('.dat')]
file_names.sort(key=utils.extract_number)

plot_files = []
fig_dir="fig/naca_new"

if __name__ == '__main__':
    for file_name in file_names[25:26]:
        x, y, s = process_data.process_naca(os.path.join(data_dir, file_name), var="s", area=area, cut=cut)
        print(x.shape, y.shape, s.shape)
        # create_contour_plot(x, y, s, file_name.split('.')[0], fig_dir=fig_dir)
        create_contour_plot(x, y, s, file_name.split('.')[0])
