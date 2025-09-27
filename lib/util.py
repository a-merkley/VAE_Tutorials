import torch
from torch.nn.functional import normalize
from torchvision import transforms, datasets
from torch.utils.data import TensorDataset
import numpy as np
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import math

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


# This function is specifically for training VAE models in Jupyter
def train_vae(model, optimizer, data_loader, device, **kwargs):
    data_src = kwargs.get('data_src', 'MNIST')
    epochs = kwargs.get('epochs', 10)

    recon_arr = []
    kl_arr = []
    mi_arr = []

    for epoch in tqdm(range(epochs)):
        for batch in data_loader:
            # Forward pass
            x, u = get_batch(batch, data_src, device)
            mu_q, logvar_q = model(x)

            # Compute metrics
            zs = model.get_z_samples()
            ami = approximate_mi(mu_q, logvar_q, zs)
            mi_arr.append(ami)

            # Update model, compute loss
            optimizer.zero_grad()
            losses = model.loss(x)
            loss = losses[0] + losses[1]
            loss.backward()
            optimizer.step()

            # Save losses
            recon_arr.append(losses[0].item())
            kl_arr.append(losses[1].item())

    metrics = {'ami': mi_arr}
    return recon_arr, kl_arr, metrics


# This function is specifically for training iVAE models in Jupyter
def train_ivae(model, optimizer, data_loader, device, **kwargs):
    data_src = kwargs.get('data_src', 'ivae1')
    epochs = kwargs.get('epochs', 20)
    one_hot = kwargs.get('one_hot', True)

    recon_arr = []
    kl_arr = []
    mi_arr = []

    for epoch in tqdm(range(epochs)):
        for batch in data_loader:
            x, u = get_batch(batch, data_src, device)
            if data_src == "MNIST" or not one_hot:
                u = u.view(-1, 1)  # add another dimension for 1d labels
            output = model(x, u)

            # Compute metrics
            zs = model.get_z_samples()
            ami = approximate_mi(output[-2], output[-1], zs)
            mi_arr.append(ami)

            optimizer.zero_grad()
            losses = model.loss(x)
            recon_loss = -losses[0]
            kl_loss = losses[1] - losses[2]

            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()

            recon_arr.append(recon_loss.item())
            kl_arr.append(kl_loss.item())

    metrics = {'ami': mi_arr}
    return recon_arr, kl_arr, metrics


# This function is specifically for training LVAE models in Jupyter
def train_lvae(model, optimizer, data_loader, device, **kwargs):
    data_src = kwargs.get('data_src', 'MNIST')
    epochs = kwargs.get('epochs', 20)

    recon_arr = []
    kl_arr = []
    kl_hier_arr = []
    au_lst = []  # Store average of active unit variance per hierarchy per epoch
    mi_arr = []

    pz = kwargs.get('pz', [1, 2, 3])
    num_hier = len(pz)

    for epoch in tqdm(range(epochs)):
        epoch_lst = []
        for batch in data_loader:
            x, u = get_batch(batch, data_src, device)

            # === Forward pass
            output = model(x)

            # === Latent space performance metrics
            mu_q = output[2]
            logvar_q = output[3]
            zs = model.get_z_samples()
            mi_temp = []
            for kk in range(num_hier):
                # Active units
                au, au_var = active_units(mu_q[kk])
                epoch_lst.append(au_var)
                # Mutual information
                ami = approximate_mi(mu_q[kk], logvar_q[kk], zs[kk])
                mi_temp.append(ami)
            mi_arr.append(mi_temp)

            # === Update model, compute loss
            optimizer.zero_grad()
            beta = model.anneal(epoch)
            losses = model.loss(x, output, beta)
            loss = losses[0] + losses[1]
            loss.backward()
            optimizer.step()

            recon_arr.append(losses[0].item())
            kl_arr.append(losses[1].item())
            kl_hier_item = [kl.item() for kl in losses[2]]
            kl_hier_arr.append(kl_hier_item)

        for ii in range(num_hier):
            au_lst.append(np.array(epoch_lst[ii::num_hier]).mean(0))

    au_overall = []  # list of averaged np arrays of active units, shape (num_epochs, dim)
    for ii in range(num_hier):
        au_overall.append(np.array(au_lst[ii::num_hier]))

    # Add all metrics here
    metrics = {'au': au_overall, 'ami': mi_arr}

    return recon_arr, kl_arr, np.array(kl_hier_arr), metrics


def sample(mu, log_var):
    z = mu + (0.5 * log_var).exp() * torch.randn_like(mu)
    return z

def kl_normal(mu0, log_var0, mu1, log_var1):
    dim = mu0.shape[1]
    trace = log_var0.exp() / log_var1.exp()
    quad = (mu0 - mu1).pow(2) / log_var1.exp()
    logdet = log_var1 - log_var0
    kl_div = (trace + quad + logdet).sum(-1) - dim
    return 0.5*kl_div

def normal_ll(x, mu, logvar):
    # Log likelihood of multivariate gaussian with diagonal covariance
    if torch.cuda.is_available():
        x = x.cuda()
        mu = mu.cuda()
        logvar = logvar.cuda()

    const = x.shape[-1] * torch.tensor(2*torch.pi).log()
    ll = -0.5 * (const + (logvar + (x - mu).pow(2) / logvar.exp()).sum(-1))
    return ll

def compound_normal(mu0, log_var0, mu1, log_var1):
    # Calculate parameters of compound distribution (product of two normal distributions, diagonal covariance)
    # N(mu2, var2) = alpha * N(mu0, var0) * N(mu1, var1), for constant alpha
    var0_inv = log_var0.exp().reciprocal()
    var1_inv = log_var1.exp().reciprocal()
    var2 = (var0_inv + var1_inv).reciprocal()
    mu2 = var2 * (var0_inv*mu0 + var1_inv*mu1)
    return mu2, var2


def get_dataset(**kwargs):
    data_src = kwargs.get('data_src')
    if data_src == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
        data = datasets.MNIST(root="../../data", download=True, transform=transform)
        kx = 784
    elif data_src == "ivae1":
        data, kx = get_ivae_dataset(**kwargs)
    return data, kx


def get_ivae_dataset(**kwargs):
    # Load settings
    num_u = kwargs.get('num_u', -1)
    N_trunc = kwargs.get('N_trunc', -1)
    one_hot = kwargs.get('one_hot', True)
    zscore_u = kwargs.get('zscore_u', True)

    data = np.load("../../data/data_ivae1.npz")
    N = kwargs.get('N')
    x = data['x'][:N, :]
    u = data['u'][:N, :]
    z = data['s'][:N, :]

    raw_idx = np.argmax(u, axis=1)

    # Preprocess u if need be
    if num_u > 0:
        u_unq = np.random.choice(u.shape[1], size=num_u, replace=False)
        u_idx = []
        for uval in u_unq:
            u_idx.append(np.where(raw_idx == uval)[0])
        u_idx = np.array(u_idx).flatten()
        raw_idx = raw_idx[u_idx]
        x = x[u_idx, :]
        u = u[u_idx, :]
    else:
        u_idx = []
        for uval in np.unique(raw_idx):
            location = np.where(raw_idx == uval)[0]
            if N_trunc > 0:
                u_idx.append(location[:N_trunc])
            else:
                u_idx.append(location)
        u_idx = np.array(u_idx).flatten()
        raw_idx = raw_idx[u_idx]
        x = x[u_idx, :]
        u = u[u_idx, :]

    if not one_hot:
        u = raw_idx
        if zscore_u:
            u = (u - u.mean()) / u.std()

    data = TensorDataset(torch.Tensor(x), torch.Tensor(u), torch.Tensor(z))
    kx = x.shape[1]
    return data, kx


def get_batch(batch, data_src, device, return_z=False):
    if data_src == 'MNIST':
        x, u = batch[0].to(device), batch[1].to(device)
        u = u.to(torch.float32)
        return x, u
    elif data_src == "ivae1":
        x, u, z = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        if return_z:
            return x, u, z
        else:
            return x, u


############### VAE EVALUATION METRICS ###############
def active_units(mu_q, thresh=1e-3):
    au_var = mu_q.var(0)
    au = au_var > thresh
    return au.detach().cpu().numpy(), au_var.detach().cpu().numpy()


def approximate_mi(mu, logvar, z):
    # I_q(X;Z) = E_x[ q(z|x) log q(z|x)/q(z)] = E_x[q(z|x) (log q(z|x) - log q(z))]
    # log q(z) = -log|X| + log sum_x q(z|x)  <-- employ logsumexp trick here, |X| is batch size
    k = mu.shape[0]
    z = z[:, None, :]
    mu = mu[None, :, :]
    logvar = logvar[None, :, :]

    log_qz_x = normal_ll(z, mu, logvar)
    log_qz = torch.logsumexp(log_qz_x, dim=1) - np.log(k)
    log_qz_x_diag = log_qz_x.diag()

    ami = (log_qz_x_diag - log_qz).mean()
    return ami.item()


def approximate_mi_msg(mu, logvar, z, m):
    # I_p(M;Z) = E_m[ sum_z p(z|m) log p(z|m)/p(m)] = E_m[ sum_z q(z|m) (log p(z|m) - log p(m))]

    # Set up
    batch_size, dim = mu.shape
    z0 = z - mu

    one_idx = m.nonzero()[:, 1]  # NOTE: assumes one-hot encoding for message var
    unique_m = one_idx.unique()
    mi_per_m = torch.zeros(len(unique_m))

    for i, mval in enumerate(unique_m):
        # Select all z samples from a given m
        all_i = (one_idx == mval).nonzero().squeeze(dim=1)
        zi = z0[all_i, :]
        logvar_qi = logvar[all_i, :]

        # Calculate terms for: log p(z|m) - log p(z)
        log_post = dim * np.log(2 * np.pi) + logvar_qi + zi.pow(2) * logvar_qi.exp().reciprocal()
        log_post = -0.5 * log_post.sum(1)
        posterior = log_post.exp()
        aggreg_post = torch.logsumexp(log_post, dim=0) - np.log(batch_size)

        # Calculate: p(z|m) (log p(z|m) - log p(z))
        inside_exp = posterior * (log_post - aggreg_post)

        # Calculate: sum_z p(z|m) (log p(z|m) - log p(z))
        mi_per_m[i] = inside_exp.sum().item()

    # Calculate: E_m[ sum_z p(z|m) (log p(z|m) - log p(z)) ]
    approx_mi = mi_per_m.mean().item()
    return approx_mi


def mcc(z_true, mu_q, logvar_q):
    # Mean correlation coefficient
    dim = mu_q.shape[1]
    z_hat = mu_q + torch.exp(0.5*logvar_q) * torch.randn_like(mu_q)
    z_hat = z_hat.detach().cpu().numpy()
    z_true = z_true.detach().cpu().numpy()
    mcc_mat = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            mcc_mat[i, j] = np.corrcoef(z_true[:, j], z_hat[:, i])[0, 1]

    row_ind, col_ind = linear_sum_assignment(1-abs(mcc_mat))
    print("Optimal assignment: ", list(zip(row_ind, col_ind)))
    print("Total cost: ", mcc_mat[row_ind, col_ind].sum())
    print(mcc_mat)
    return mcc_mat


def decoding_acc_lbl(model, x, m_events, N=100):
    batch_size, kx = x.shape
    ll_all = []

    for mval in range(m_events):
        # Create copies of the specific m value across all samples
        m = torch.zeros(m_events)
        m[mval] = 1  # make one-hot encoding
        m = m.expand(batch_size, m_events)
        mu_prior = model.prior_mu_net(m)
        logvar_prior = model.prior_net(m)

        # Draw N samples from p(z|m)
        mu_prior = mu_prior.unsqueeze(0).expand(N, -1, -1)
        logvar_prior = logvar_prior.unsqueeze(0).expand(N, -1, -1)
        z = mu_prior + (0.5*logvar_prior).exp() * torch.randn(N, batch_size, 2)

        # Input these z samples into the decoder to get mu_p, sigma_p
        z_flat = z.reshape(N*batch_size, -1)
        mu_p = model.decoder(z_flat).reshape(N, batch_size, -1)
        logvar_p = model.decoder_logvar(z_flat).reshape(N, batch_size, -1)

        # Calculate log likelihood with each (mu_p, sigma_p) pair
        x_exp = x.unsqueeze(0)
        ll = normal_ll(x_exp, mu_p, logvar_p)
        ll_total = torch.logsumexp(ll, dim=0) - torch.log(torch.tensor(N, dtype=ll.dtype))
        ll_all.append(ll_total)#.detach().cpu().numpy())

    logliks = torch.stack(ll_all, dim=0).transpose(1, 0)
    log_p_m_given_x = torch.nn.functional.log_softmax(logliks, dim=0)
    answers = log_p_m_given_x.argmax(1)
    return answers, logliks


def get_num_parameters(model, verbose=True):
    n_parameters = sum(param.numel() for param in model.parameters())
    if verbose:
        print("Number of parameters: " + str(n_parameters))
    return n_parameters

def plot_pca(x, n_components=2, plt_lst=None, lbls=None):
    x = x.detach().numpy()

    pca = PCA(n_components=n_components)
    x0 = pca.fit_transform(x)

    plt_dict = {'marker': '.'}

    if plt_lst is None:
        plt_lst = [0, 1]
    if lbls is not None:
        plt_dict['c'] = lbls
        plt_dict['cmap'] = 'tab10'

    plt.scatter(x0[:, plt_lst[0]], x0[:, plt_lst[1]], **plt_dict)
    if lbls is not None:
        plt.legend()
    plt.show()


def plot_multiple_pca(z_lst, n_components=2, plt_lst=None, lbls=None, title_lst=None):
    num_plots = len(z_lst)
    nrows = np.ceil(np.sqrt(num_plots)).astype(int)
    ncols = np.ceil(num_plots / nrows).astype(int)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

    plt_dict = {'marker': '.'}

    if plt_lst is None:
        plt_lst = [0, 1]
    if lbls is not None:
        plt_dict['c'] = lbls
        plt_dict['cmap'] = plt.cm.get_cmap('nipy_spectral', len(lbls))

    for i in range(num_plots):
        z = z_lst[i].detach().cpu().numpy()
        pca = PCA(n_components=n_components)
        z0 = pca.fit_transform(z)

        if num_plots == 1:
            ax_i = axs
        elif num_plots == 2:
            ax_i = axs[i]
        else:
            row_i, col_i = np.unravel_index(i, (nrows, ncols))
            ax_i = axs[row_i, col_i]

        sc = ax_i.scatter(z0[:, plt_lst[0]], z0[:, plt_lst[1]], **plt_dict)
        if title_lst is not None:
            ax_i.set_title(title_lst[i])

    # Set legend
    if lbls is not None:
        handles, labels = sc.legend_elements(num=None)
        fig.legend(handles, labels, loc="center right")

    plt.tight_layout()
    plt.show()

