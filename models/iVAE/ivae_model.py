import numpy as np
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import lib.util as util
from lib.logger import Logger


######################### CLASS DEFINITIONS ############################
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, mu_flag=True, var_flag=True, **kwargs):
        super().__init__()
        self.mu_flag = mu_flag
        self.var_flag = var_flag

        activation = kwargs.get('activation')
        slope = kwargs.get('slope')
        hidden = kwargs.get('hidden_dim')
        num_layers = kwargs.get('num_layers')  # number of (linear, activation) blocks before output

        hidden_dims = num_layers * [hidden]
        hidden_dims = [in_dim] + hidden_dims

        layer_lst = []
        for n in range(num_layers):
            layer_lst.append(nn.Linear(hidden_dims[n], hidden_dims[n+1]))
            if activation == 'leaky_relu':
                layer_lst.append(nn.LeakyReLU(slope))
            elif activation == 'relu':
                layer_lst.append(nn.ReLU())
            elif activation == 'tanh':
                layer_lst.append(nn.Tanh())
            elif activation == 'sigmoid':
                layer_lst.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layer_lst)
        if self.mu_flag:
            self.mu = nn.Linear(hidden_dims[-1], out_dim)
        if self.var_flag:
            self.log_var = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x):
        out = self.layers(x)
        if self.mu_flag and self.var_flag:
            return self.mu(out), self.log_var(out)
        if self.mu_flag:
            return self.mu(out)
        if self.var_flag:
            return self.log_var(out)


class iVAE(nn.Module):
    def __init__(self, kx, ku, **kwargs):
        super().__init__()
        self.kx = kx
        self.ku = ku

        # Default values are for original iVAE
        self.pz = kwargs.get('pz', 2)
        self.shared_net = kwargs.get('shared_net', False)
        self.var_p_noise = kwargs.get('var_p_noise', 0.01)
        self.learn_mu_prior = kwargs.get('learn_mu_prior', False)
        self.learn_var_p = kwargs.get('learn_var_p', False)
        self.activation = kwargs.get('activation', 'leaky_relu')
        self.slope = kwargs.get('lrelu_slope', 0.01)
        self.hidden_dim = kwargs.get('hidden_dim', 50)
        self.num_layers = kwargs.get('num_layers', 2)  # number of (linear, activation) blocks before output
        self.ivae_loss = kwargs.get('ivae_loss', True)
        self.compound_posterior = kwargs.get('compound_posterior', False)

        net_args = {'activation': self.activation, 'slope': self.slope, 'hidden_dim': self.hidden_dim,
                    'num_layers': self.num_layers}

        # prior
        if self.shared_net:
            self.prior_net = MLP(ku, self.pz, mu_flag=self.learn_mu_prior, **net_args)
        else:
            self.prior_net = MLP(ku, self.pz, mu_flag=False, **net_args)  # this is for logvar
            if self.learn_mu_prior:
                self.prior_mu_net = MLP(ku, self.pz, var_flag=False, **net_args)
        if not self.learn_mu_prior:
            self.mu_prior = torch.zeros(self.pz)

        # decoder
        if self.shared_net:
            self.decoder = MLP(self.pz, kx, var_flag=self.learn_var_p, **net_args)
        else:
            self.decoder = MLP(self.pz, kx, var_flag=False, **net_args)  # this is for mu
            if self.learn_var_p:
                self.decoder_logvar = MLP(self.pz, kx, mu_flag=False, **net_args)
        if not self.learn_var_p:
            noise = torch.tensor(self.var_p_noise)
            self.log_var_p = noise.log() * torch.ones(kx)

        # encoder
        if self.shared_net:
            self.encoder = MLP(kx + ku, self.pz, **net_args)
        else:
            self.encoder_mu = MLP(kx + ku, self.pz, var_flag=False, **net_args)
            self.encoder_logvar = MLP(kx + ku, self.pz, mu_flag=False, **net_args)

    def forward(self, x, u):
        xu = torch.cat((x, u), 1)
        # Prior parameters
        if self.learn_mu_prior:
            if self.shared_net:
                self.mu_prior, self.log_var_prior = self.prior_net(u)
            else:
                self.mu_prior = self.prior_mu_net(u)
                self.log_var_prior = self.prior_net(u)
        else:
            self.log_var_prior = self.prior_net(u)

        # Encoder parameters
        if self.shared_net:
            self.mu_q, self.log_var_q = self.encoder(xu)
        else:
            self.mu_q = self.encoder_mu(xu)
            self.log_var_q = self.encoder_logvar(xu)

        if self.compound_posterior:
            self.mu_q, self.log_var_q = util.compound_normal(self.mu_q, self.log_var_q, self.mu_prior, self.log_var_prior)

        # Sampling
        z = util.sample(self.mu_q, self.log_var_q)
        self.z = z

        # Decoder parameters
        if self.learn_var_p:
            if self.shared_net:
                self.mu_p, self.log_var_p = self.decoder(z)
            else:
                self.mu_p = self.decoder(z)
                self.log_var_p = self.decoder_logvar(z)
        else:
            self.mu_p = self.decoder(z)

        return self.mu_p, self.log_var_p, self.mu_prior, self.log_var_prior, self.mu_q, self.log_var_q

    def loss(self, x):
        mu_prior = self.mu_prior.expand(x.shape[0], -1)
        if self.ivae_loss:
            px_z = util.normal_ll(x, self.mu_p, self.log_var_p)
            qz_xm = util.normal_ll(self.z, self.mu_q, self.log_var_q)
            pz_m = util.normal_ll(self.z, mu_prior, self.log_var_prior)
            return px_z.mean(), qz_xm.mean(), pz_m.mean()
        else:
            kl_div = util.kl_normal(self.mu_q, self.log_var_q, mu_prior, self.log_var_prior)
            recon = -1 * util.normal_ll(x, self.mu_p, self.log_var_p)
            return recon.mean(), kl_div.mean()

    def get_z_samples(self):
        return self.z



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

    # Initialize logger
    logger = Logger("iVAE_runs")
    logger.record_model(config)

    # Load CPU or GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Prepare dataset
    data, kx = util.get_dataset(**train_cfg)
    generator = torch.Generator().manual_seed(train_cfg['rng'])
    train_data, valid_data, test_data = random_split(data, train_cfg['data_split'], generator=generator)
    train_loader = DataLoader(train_data, train_cfg['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, train_cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    u = data[1]
    ku = u.shape[1] if train_cfg['one_hot'] else 1
    u_events = ku if train_cfg['one_hot'] else len(u)


    # Define model
    model = iVAE(kx, ku, **model_cfg).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

    # Training loop
    iteration = 0
    for epoch in tqdm(range(train_cfg['epochs'])):

        for batch in train_loader:
            # === Forward pass
            x, u = util.get_batch(batch, train_cfg['data_src'], device)
            if not train_cfg['one_hot']:
                u = u.view(-1, 1)
            output = model(x, u)

            # === Latent space performance metrics
            # I(X;Z)
            zs = model.get_z_samples()
            mu_q = output[-2]
            logvar_q = output[-1]
            ami_xz = util.approximate_mi(mu_q, logvar_q, zs)
            # Active units
            au, au_var = util.active_units(mu_q)

            # === Save metrics & parameters
            # Loop for storing active units per latent
            for ii in range(len(au_var)):
                au_header = "active_units_z" + str(ii).zfill(2)
                logger.record_metrics([au_var[ii]], [au_header])
            logger.record_metrics([ami_xz],  ['approx_mi_xz'])

            # === Update model, compute loss
            optimizer.zero_grad()
            losses = model.loss(x)
            if model_cfg['ivae_loss']:
                loss = -losses[0] + losses[1] - losses[2]
                logger.record_loss([-losses[0].item(), (losses[1] - losses[2]).item()], ["recon", "kl"])
            else:
                loss = losses[0] + losses[1]
                logger.record_loss([losses[0].item(), losses[1].item()], ["recon", "kl"])
            loss.backward()
            optimizer.step()

            # === Validation/test
            if train_cfg['validation']:
                model.eval()
                with torch.no_grad():
                    for batch in valid_loader:
                        x = batch[0]
                        u = batch[1]
                        output = model(x, u)

                        losses_valid = model.loss(x)
                        if model_cfg['ivae_loss']:
                            loss_valid = -losses[0] + losses[1] - losses[2]
                        else:
                            loss_valid = losses[0] + losses[1]
                scheduler.step(loss_valid)

            iteration += 1

        logger.record_params([iteration], ['epoch_end'])

    logger.save()



    ######################### TEST PERFORMANCE ############################
    # NOTE: this only works if you have ground-truth latents, e.g. data_ivae1, not MNIST
    if train_cfg['data_src'] == 'ivae1':
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                x, u, z = util.get_batch(batch, train_cfg['data_src'], device, return_z=True)

                if not train_cfg['one_hot']:
                    u = u.view(-1, 1)
                output = model(x, u)

                zs = model.get_z_samples()
                mu_q = output[-2]
                logvar_q = output[-1]
                mcc = util.mcc(z, mu_q, logvar_q)
                # acc = util.decoding_acc_lbl(model, x, u_events)
                ami_valid = util.approximate_mi(mu_q, logvar_q, zs)

                losses = model.loss(x)
                print("Test recon loss: " + str(losses[0].item()))
                print("Test KL loss: " + str(losses[1].item()))
                print("Approximate mutual information: " + str(ami_valid))

                # Show f mixing function activations
                plt.imshow(output[0].detach().cpu().numpy(), aspect=0.005)
                plt.show()
