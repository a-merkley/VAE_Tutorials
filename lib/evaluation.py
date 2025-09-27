import numpy as np
import os
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA


def check_dir_exists(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def make_save_path(runs, descr, extension=".png"):
    '''
    Create the folder and file names to save image file to
    :param runs: list of int. All desired runs to put on same plot
    :param descr: string. Description of what the plot is plotting
    extension: string. Type of file extension to save
    :return:
    '''
    runs_name = ["_run" + str(r) for r in runs]
    fname = descr + ''.join(runs_name) + extension
    path = os.path.join(os.getcwd(), "results")
    check_dir_exists(path)
    return os.path.join(path, fname)


# Plot standard loss (reconstruction + kl)
def plot_loss(runs, model, show_fig=True, save_fig=True):
    loss_paths = [os.path.join(model+"_runs", "run_"+str(r).zfill(2), "loss.npz") for r in runs]
    fig, ax = plt.subplots(nrows=2, ncols=1)

    for i, root in enumerate(loss_paths):
        losses = np.load(root)
        kl = losses['kl']
        recon = losses['recon']
        ax[0].plot(recon, label="Recon" + str(runs[i]))
        ax[1].plot(kl, '--', label="KL" + str(runs[i]))
    ax[0].set_title("Training reconstruction loss")
    ax[0].legend()
    ax[0].set_xlabel("Iterations (batches)")
    ax[1].set_title("Training KL loss")
    ax[1].legend()
    ax[1].set_xlabel("Iterations (batches)")
    plt.tight_layout()

    if save_fig:
        save_path = make_save_path(runs, "train_loss")
        fig.savefig(save_path)
    if show_fig:
        plt.show()


# Plot hierarchical KL loss
def plot_hier_kl_loss(runs, model, num_plots, show_fig=True, save_fig=True):
    loss_paths = [os.path.join(model+"_runs", "run_"+str(r).zfill(2), "loss.npz") for r in runs]
    nrows = np.ceil(np.sqrt(num_plots)).astype(int)
    ncols = np.ceil(num_plots / nrows).astype(int)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True)

    for i, root in enumerate(loss_paths):
        losses = np.load(root)
        kl_losses = [f for f in losses if "kl_h" in f]
        num_hier = len(kl_losses)

        for h in range(num_hier):
            kl_h = losses['kl_h'+str(h).zfill(2)]

            if num_plots <= 2:
                axs[h].plot(kl_h, label="Run " + str(runs[i]))
                axs[h].set_title("Latent level " + str(h))
                axs[h].legend()
            else:
                row_h, col_h = np.unravel_index(h, (nrows, ncols))
                axs[row_h, col_h].plot(kl_h, label="Run " + str(runs[i]))
                axs[row_h, col_h].set_title("Latent level " + str(h))
                axs[row_h, col_h].legend()

    plt.suptitle("KL loss for each hierarchy")
    plt.tight_layout()

    if save_fig:
        save_path = make_save_path(runs, "train_loss")
        fig.savefig(save_path)
    if show_fig:
        plt.show()


# Plot metrics
def plot_metrics(runs, model, metric_lst, show_fig=True, save_fig=True):
    num_metrics = len(metric_lst)
    metric_paths = [os.path.join(model+"_runs", "run_" + str(r).zfill(2), "metrics.npz") for r in runs]

    nrows = np.ceil(np.sqrt(num_metrics)).astype(int)
    ncols = np.ceil(num_metrics / nrows).astype(int)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True)

    for i, root in enumerate(metric_paths):
        metrics = np.load(root)

        for j in range(num_metrics):
            metric_j = metrics[metric_lst[j]]
            idx1, idx2 = np.unravel_index(j, (nrows, ncols))

            axs[idx1, idx2].plot(metric_j, label="Run "+str(runs[i]))
            axs[idx1, idx2].set_title(metric_lst[j])
            axs[idx1, idx2].legend()

    plt.tight_layout()
    if save_fig:
        save_path = make_save_path(runs, "metrics")
        fig.savefig(save_path)
    if show_fig:
        plt.show()


