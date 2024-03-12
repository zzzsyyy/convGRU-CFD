import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.interpolate import griddata
import os

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 192

def process_cylinder(file_name, var="s", area = None, cut = None):
    '''
        (x,y,u,v) or (x,y,p)
    '''
    cols = (0, 1, 2, 3) if var in ["v", "u", "s"]  else (0, 1, 5)
    data = np.loadtxt(file_name, skiprows=2, usecols=cols, delimiter=' ', max_rows=cut)
    print(data.shape)
    filtered_data = data[(data[:, 0] >= area[0]) & (data[:, 0] <= area[1]) & (data[:, 1] >= area[2]) & (data[:, 1] <= area[3])]
    if var in ["s", "u", "v"]:
        x, y, u, v = filtered_data.T
    else:
        x, y, p = filtered_data.T
    if var == "s":
        s = np.sqrt(u**2 + v**2)
        ret = (x, y, s)
    elif var == "u":
        ret = (x, y, u)
    elif var == "v":
        ret= (x, y, v)
    else:
        ret = (x, y, p)
    print(f"File {file_name.split('.')[0]} processed")
    return ret

def process_naca(file_name, var = "v", area = None, cut = None, div128=True):
    '''
        (x,y,u,v) or (x,y,p)
    '''
    cols = (0, 1, 2, 3) if var in ["v", "u", "s"]  else (0, 1, 5)
    data = np.loadtxt(file_name, skiprows=2, usecols=cols, delimiter=' ', max_rows=cut)
    if div128:
        data[:, 0] /= 128
        data[:, 1] /= 128
    filtered_data = data[(data[:, 0] >= area[0]) & (data[:, 0] <= area[1]) & (data[:, 1] >= area[2]) & (data[:, 1] <= area[3])]
    if var in ["s", "u", "v"]:
        x, y, u, v = filtered_data.T
    else:
        x, y, p = filtered_data.T
    if var == "s":
        s = np.sqrt(u**2 + v**2)
        ret = (x, y, s)
    elif var == "u":
        ret = (x, y, u)
    elif var == "v":
        ret= (x, y, v)
    else:
        ret = (x, y, p)
    print(f"File {file_name.split('.')[0]} processed")
    return ret

def create_contour_plot(x, y, s, file_name, show=False, regrid = None, var = "v", area=[]):
    # levels = np.linspace(0, 0.2, 40)
    plt.figure(figsize=(25, 20))
    if regrid:
        xi = np.linspace(area[0], area[1], regrid[0])
        yi = np.linspace(area[2], area[3], regrid[1])
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), s, (xi, yi), method='linear')
        plt.contourf(xi, yi, zi, levels=14)
    else:
        triangle = tri.Triangulation(x, y)
        plt.tricontourf(triangle, s, levels=20, cmap=plt.cm.viridis)

    plt.axis('equal')
    plt.colorbar(label=var)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'{var} Contour Plot at {file_name}')
    if show:
        plt.show()
    else:
        plt.savefig(f'{file_name}.png')
        plt.close()

def main(type = "cylinder", idx = 47, var="v"):
    if type == "cylinder":
        area = [235, 260, 230, 250]
        area = [i * 32 for i in area]
        cut = 1101852
        data_dir = 'data_cylinder/2D_Cylinder_MachineLearning'
        file_name = format(f'{idx}.dat')
        x, y, s = process_cylinder(os.path.join(data_dir, file_name), var=var, area=area, cut=cut)
        create_contour_plot(x, y, s, file_name.split('.')[0], show=False, area=area)
        return (x, y, s)
    else:
        pass


if __name__ == '__main__':
    main()
