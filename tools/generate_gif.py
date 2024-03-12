import os
import imageio
import utils

fig_dir = "fig/naca_new"
gif_filename = 'contours.gif'
if "__name__" == "__main__":
    with imageio.get_writer(gif_filename, mode='I', duration=0.5) as writer:
        plot_names = [f for f in os.listdir(fig_dir) if f.endswith('.png')]
        plot_files = sorted(plot_names, key=utils.extract_number)
        for filename in plot_files:
            image = imageio.imread(os.path.join(fig_dir, filename))
            writer.append_data(image)