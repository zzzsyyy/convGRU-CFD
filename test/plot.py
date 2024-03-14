import matplotlib.pyplot as plt
import numpy as np
cmap='RdBu'

plt.rcParams['savefig.dpi'] = 300


def process_data(data, j ,n, cut):
    data = data[j, n, 0].cpu().numpy()[cut[0]:cut[2], cut[1]:cut[3]]
    data_p=(data-data.min())/(data.max() - data.min())
    mean = np.mean(data_p)
    std = np.std(data_p)
    return (data, data.max(), data.min(), mean, std)

def precess_pred(pred, j, n, cut, true_min, true_max, true_mean, true_std):
    pred = pred[j, n, 0].cpu().detach().numpy()[cut[0]:cut[2], cut[1]:cut[3]]
    pred=(pred-pred.min())/(pred.max() - pred.min())
    pred = (pred - np.mean(pred))/np.std(pred)
    my_pred = (pred*true_std + true_mean)*(true_max - true_min) + true_min
    return my_pred

def plot_line(axarr, l,n, data, draw_area):
    im=axarr[l][n].imshow(data, origin='lower', cmap=cmap)
    # axarr[l][n].set_aspect('equal')
    axarr[l][n].set_xlim(draw_area[0], draw_area[2])
    axarr[l][n].set_ylim(draw_area[3], draw_area[1])
    axarr[l][n].axis("off")
    return im

def plot_diff_line(ax, tar_img, out_img,cut,cutline, cmap_=None):
    assert cmap_ is None
    if cutline[0] == 'y':
        axis = np.arange(cut[3]-cut[1])
        true_im = tar_img[cutline[1],:]
        pred_im = out_img[cutline[1],:]
    if cutline[0] == 'x':
        axis = np.arange(cut[2]-cut[0])
        true_im = tar_img[:,cutline[1]]
        pred_im = out_img[:,cutline[1]]
    ax.plot(axis, true_im, linestyle='-', color='blue', linewidth=0.8)
    ax.plot(axis, pred_im, linestyle='-', color='red', linewidth=0.8)
    ax.legend(['True', 'Predicted'], fontsize=8)

def plot_diff_gray(ax, tar_img, out_img,draw_area, cmap_='gray'):
    diff = np.abs(tar_img - out_img)
    # cond = (diff>=-0.005) & (diff<=0.005)
    # print("cond", cond.shape)
    # diff_img = np.zeros_like(diff)
    # diff_img[cond] = 1;
    # im2 = axarr[2][n].contourf(diff_img, origin="lower", llevels=2)
    # axarr[2][n].set_aspect('equal')
    # print(tar_img.max(), out_img.max())
    im=ax.imshow(diff, origin="lower", cmap=cmap_)
    ax.set_xlim(draw_area[0], draw_area[2])
    ax.set_ylim(draw_area[3], draw_area[1])
    ax.axis("off")
    return im


def plot_true_pred_diff_base(pred, true, batch_idx, cut=(5,5,195,495), draw_area=[0,0,500,200], cutline=('y', 100), func=plot_diff_line, break_=False):
    for j in range(5):
        print(f"Plotting Sample {j} in batch {batch_idx}...")
        _, axarr = plt.subplots(3, 5, figsize=(6*6, 3*5), constrained_layout=True)
        plt.subplots_adjust(wspace=0.2, hspace=-0.8)
        for n in range(5):
            if n == 0:
                axarr[0][n].set_title("Target")
                axarr[1][n].set_title("Prediction")
                axarr[2][n].set_title("Diff")
            tar_img,true_max, true_min, true_mean, true_std = process_data(true, j, n, cut)
            im0 = plot_line(axarr, 0,n, tar_img, draw_area)
            
            out_img = precess_pred(pred, j, n, cut, true_min, true_max, true_mean, true_std)
            im1 = plot_line(axarr, 1,n, out_img, draw_area)

            if func == plot_diff_gray:
                func(axarr[2][n], tar_img, out_img, draw_area, 'gray')
            elif func == plot_diff_line:
                func(axarr[2][n], tar_img, out_img, cut, cutline)
        plt.colorbar(im1, ax=axarr[1, :], location='right', shrink=0.6)
        plt.colorbar(im0, ax=axarr[0, :], location='right', shrink=0.6)
        if func == plot_diff_gray:
            plt.colorbar(im1, ax=axarr[2, :], location='right', shrink=0.6)
        plt.show()
        if break_:
            break

def plot_true_pred_diff_gray(pred, true, batch_idx, cut=(5,5,195,495), draw_area=[0,0,500,200], cutline=('y', 100), break_=False):
    plot_true_pred_diff_base(pred, true, batch_idx, cut, draw_area, cutline, plot_diff_gray, break_)

def plot_true_pred_diff_plot(pred, true, batch_idx, cut=(5,5,195,495), draw_area=[0,0,500,200], cutline=('y', 100), break_=False):
    plot_true_pred_diff_base(pred, true, batch_idx, cut, draw_area, cutline, plot_diff_line, break_)

def plot_save(pred, true, batch_idx, cut=(5,5,195,495), cutline=('y', 100), break_=False):
    for j in range(5):
        for n in range(5):
            tar_img,true_max, true_min, true_mean, true_std = process_data(true, j, n, cut)
            out_img = precess_pred(pred, j, n, cut, true_min, true_max, true_mean, true_std)
            if cutline[0] == 'y':
                axis = np.arange(cut[3]-cut[1])
                true_im = tar_img[cutline[1],:]
                pred_im = out_img[cutline[1],:]
            if cutline[0] == 'x':
                axis = np.arange(cut[2]-cut[0])
                true_im = tar_img[:,cutline[1]]
                pred_im = out_img[:,cutline[1]]
            plt.plot(axis, true_im, linestyle='-', color='blue', linewidth=0.8)
            plt.plot(axis, pred_im, linestyle='-', color='red', linewidth=0.8)
            plt.plot(axis, np.abs(true_im-pred_im), linestyle='-', color='orange', linewidth=0.8)
            plt.legend(['True', 'Predicted'], fontsize=8)
            # plt.xlabel(f'{'x' if cutline[0]=='y' else 'x'} coordinates')
            plt.xlabel(f'{"x" if cutline[0]=="y" else "y"} coordinates')
            plt.ylabel('Velocity')
            plt.title(f'Targets vs. Preds at {cutline[0]}={cutline[1]}')
            if n == 0:
                plt.savefig(f'diff_{cutline[0]}_{cutline[1]}.png')
            # plt.show()
            plt.clf()
            if break_:
                break
        if break_:
            break