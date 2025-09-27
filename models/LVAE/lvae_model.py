import numpy as np
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from torch.utils.data import DataLoader, random_split

import lib.util as util
from lib.logger import Logger


######################### CLASS DEFINITIONS ############################
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, mu_flag=True, var_flag=True, **kwargs):
        super().__init__()
        self.mu_flag = mu_flag
        self.var_flag = var_flag

        activation = kwargs.get('activation')
        slope = kwargs.get('lrelu_slope')
        num_layers = kwargs.get('num_layers')  # number of (linear, activation) blocks before output
        norm_type = kwargs.get('norm_type')

        hidden_dims = [in_dim] + num_layers * [hidden]

        # Define middle layers
        layer_lst = []
        for n in range(num_layers):
            # Linear layer (+ normalization)
            if norm_type == 'none':
                layer_lst.append(nn.Linear(hidden_dims[n], hidden_dims[n + 1]))
            elif norm_type == 'weight':
                layer_lst.append(weight_norm(nn.Linear(hidden_dims[n], hidden_dims[n + 1])))
            elif norm_type == 'batch':
                layer_lst.append(nn.Linear(hidden_dims[n], hidden_dims[n + 1]))
                layer_lst.append(nn.BatchNorm1d(hidden_dims[n+1]))
            elif norm_type == 'batch_weight':
                layer_lst.append(weight_norm(nn.Linear(hidden_dims[n], hidden_dims[n + 1])))
                layer_lst.append(nn.BatchNorm1d(hidden_dims[n + 1]))

            # Activation type
            if activation == 'leaky_relu':
                layer_lst.append(nn.LeakyReLU(slope))
            elif activation == 'relu':
                layer_lst.append(nn.ReLU())
            elif activation == 'tanh':
                layer_lst.append(nn.Tanh())
            elif activation == 'sigmoid':
                layer_lst.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layer_lst)

        # Define output layers
        if self.mu_flag:
            self.mu = nn.Linear(hidden_dims[-1], out_dim)
        if self.var_flag:
            self.log_var = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x, return_intermed=False):
        out = self.layers(x)

        output = []
        if return_intermed:
            output.append(out)
        if self.mu_flag:
            output.append(self.mu(out))
        if self.var_flag:
            output.append(self.log_var(out))
        return output


class LVAE(nn.Module):
    def __init__(self, kx, **kwargs):
        super().__init__()

        self.kx = kx

        self.pz = kwargs.get('pz')    # cannot have default value
        self.num_hier = len(self.pz)  # number of hierarchical layers, must be at least 2
        self.hidden_dim = kwargs.get('hidden_dim')
        self.n_warmup = kwargs.get('n_warmup')  # number of epochs for KL loss annealing

        if len(self.hidden_dim) != len(self.pz):
            print("ERROR: Length of hidden dim list must be same as length of latent dim list")

        self.encoders = nn.ParameterList()
        self.decoders = nn.ParameterList()

        self.encoders.append(MLP(self.kx, self.pz[0], self.hidden_dim[0], **kwargs))
        for i in range(1, self.num_hier):
            mlp_enc = MLP(self.hidden_dim[i - 1], self.pz[i], self.hidden_dim[i], **kwargs)
            self.encoders.append(mlp_enc)
            mlp_dec = MLP(self.pz[self.num_hier-i], self.pz[self.num_hier-i-1], self.hidden_dim[-i], **kwargs)
            self.decoders.append(mlp_dec)
        self.decoders.append(MLP(self.pz[0], self.kx, self.hidden_dim[0], **kwargs))

    def forward(self, x):
        # hat = deterministic encoder layer, qs = stochastic encoder layer, ps = decoder layer
        mu_hats, mu_qs, mu_ps = [], [], []
        logvar_hats, logvar_qs, logvar_ps = [], [], []

        # === Encoder pass
        val = x
        for i in range(self.num_hier):
            val, mu_i_hat, logvar_i_hat = self.encoders[i](val, return_intermed=True)
            mu_hats.append(mu_i_hat)
            logvar_hats.append(logvar_i_hat)

        # === Decoder pass
        self.z_vals = []
        z_val = util.sample(mu_hats[-1], logvar_hats[-1])
        self.z_vals.append(z_val)
        # Add q(z_L|x)
        mu_qs.append(mu_hats[-1])
        logvar_qs.append(logvar_hats[-1])
        # Add prior p(z_L) ~ N(0, I)
        mu_ps.append(torch.zeros_like(z_val))
        logvar_ps.append(torch.zeros_like(z_val))

        for i in range(self.num_hier):
            # Get p(z_i|z_i+1) or p(x|z_1)
            mu_p, logvar_p = self.decoders[i](z_val)
            mu_ps.append(mu_p)
            logvar_ps.append(logvar_p)

            if i < self.num_hier-1:
                # Construct compound distribution
                idx = self.num_hier - i - 2  # iterate backwards on (mu_hat, logvar_hat), exclude first set outside loop
                mu_q, logvar_q = util.compound_normal(mu_hats[idx], logvar_hats[idx], mu_p, logvar_p)
                mu_qs.append(mu_q)
                logvar_qs.append(logvar_q)
                # Sample
                z_val = util.sample(mu_q, logvar_q)
                self.z_vals.append(z_val)

        return mu_hats, logvar_hats, mu_qs, logvar_qs, mu_ps, logvar_ps

    def loss(self, x, params, beta=1):
        mu_hats, logvar_hats, mu_qs, logvar_qs, mu_ps, logvar_ps = params

        recon = -1 * util.normal_ll(x, mu_ps[-1], logvar_ps[-1])
        kl_div = 0
        kl_per_hier = []
        for i in range(self.num_hier):
            kl = util.kl_normal(mu_qs[i], logvar_qs[i], mu_ps[i], logvar_ps[i])
            kl_per_hier.append(kl.mean())
            kl_div += kl

        return recon.mean(), beta * kl_div.mean(), kl_per_hier

    def anneal(self, iteration):
        if self.n_warmup <= 0:
            return 1
        beta = np.min([iteration / self.n_warmup, 1])
        return beta

    def get_z_samples(self):
        return self.z_vals



######################### TRAINING & EVALUATION ############################
if __name__ == "__main__":
    # Settings
    config_path = "config.yaml"
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    train_cfg = config['train']
    model_cfg = config['model']

    # Set random seed
    torch.manual_seed(train_cfg['rng'])
    np.random.seed(train_cfg['rng'])

    # Load CPU or GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Initialize logger
    logger = Logger("LVAE_runs")
    logger.record_model(config)

    # Dataset
    data, kx = util.get_dataset(**train_cfg)
    generator = torch.Generator().manual_seed(train_cfg['rng'])
    train_data, valid_data, test_data = random_split(data, train_cfg['data_split'], generator=generator)
    train_loader = DataLoader(train_data, train_cfg['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, train_cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    # Model definition
    model = LVAE(kx, **model_cfg).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

    num_hier = len(model_cfg['pz'])

    # Training loop
    loss_epoch = []
    for epoch in tqdm(range(train_cfg['epochs'])):
        loss_batch = []

        for batch in train_loader:
            x, u = util.get_batch(batch, train_cfg['data_src'], device)

            # === Forward pass
            output = model(x)

            # === Latent space performance metrics
            mu_q = output[2]
            logvar_q = output[3]
            zs = model.get_z_samples()

            au_lst, au_var_lst, ami_lst = [], [], []
            for kk in range(num_hier):
                au, au_var = util.active_units(mu_q[kk])
                au_lst.append(au)
                au_var_lst.append(au_var)
                ami = util.approximate_mi(mu_q[kk], logvar_q[kk], zs[kk])
                ami_lst.append(ami)

            # === Update model, compute loss
            optimizer.zero_grad()
            beta = model.anneal(epoch)
            losses = model.loss(x, output, beta)
            loss = losses[0] + losses[1]
            loss.backward()
            optimizer.step()

            loss_batch.append(loss.item())

            # === Validation
            if train_cfg['validation']:
                model.eval()
                with torch.no_grad():
                    for batch_valid in valid_loader:
                        x, u = util.get_batch(batch_valid, train_cfg['data_src'], device)
                        output = model(x)

                        losses_valid = model.loss(x, output)
                        loss_valid = -losses_valid[0] + losses_valid[1]
                    scheduler.step(loss_valid)

            # === Save metrics & parameters
            header_order = ["mu_hat", "logvar_hat", "mu_q", "logvar_q", "mu_p", "logvar_p"]
            for ii in range(6):
                subheaders = []
                subout = []
                for n in range(len(output[ii])):
                    subheaders.append(header_order[ii] + "_z" + str(n).zfill(2))
                    temp = output[ii][n]
                    subout.append(output[ii][n])
                logger.record_params(subout, subheaders)
            logger.record_loss([losses[0].item(), losses[1].item()], ["recon", "kl"])

            for jj in range(num_hier):
                ami_header = "ami_h"+str(jj).zfill(2)
                logger.record_metrics([ami_lst[jj]], [ami_header])
                l_header = "kl_h"+str(jj).zfill(2)
                logger.record_loss([losses[2][jj].item()], [l_header])
                au_var_j = au_var_lst[jj]
                for ii in range(len(au_var_j)):
                    au_header = "active_units_h" + str(jj).zfill(2) + "_z" + str(ii).zfill(2)
                    logger.record_metrics([au_var_j[ii]], [au_header])

        loss_epoch.append(np.mean(loss_batch))
        print("Epoch " + str(epoch) + ": " + str(np.mean(loss_batch)))

    plt.plot(loss_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Overall loss")
    plt.show()

    logger.save()


    ######################### TEST PERFORMANCE ############################
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x, u = util.get_batch(batch, train_cfg['data_src'], device)
            output = model(x)

            losses = model.loss(x, output)
            print("Test loss (recon): " + str(-losses[0].item()))
            print("Test loss (KL): " + str(losses[1].item()))

            mu_q = output[2]
            logvar_q = output[3]
            zs = model.get_z_samples()

            au_lst, au_var_lst, ami_lst = [], [], []
            for kk in range(num_hier):
                au, au_var = util.active_units(mu_q[kk])
                au_lst.append(au)
                au_var_lst.append(au_var)

                ami = util.approximate_mi(mu_q[kk], logvar_q[kk], zs[kk])
                print("Approximate MI hierarchy " + str(kk) + ": " + str(ami))
                ami_lst.append(ami)

        # Plot latents at each level
        u = u.detach().cpu()
        util.plot_multiple_pca(zs, n_components=2, lbls=u)
