# This provides an ablation study on iVAE by testing a number of different configurations & model assumptions
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from matplotlib import pyplot as plt
from tqdm import tqdm
import lib.util as util
from lib.logger import Logger



class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, mu_flag=True, var_flag=True, **kwargs):
        super().__init__()
        self.mu_flag = mu_flag
        self.var_flag = var_flag

        activation = kwargs.get('activation')
        slope = kwargs.get('slope')
        hidden = kwargs.get('hidden_dim')
        num_layers = kwargs.get('num_layers')  # number of (linear, activation) blocks before output
        layer_growth = kwargs.get('layer_growth')

        if layer_growth == 'const':
            hidden_dims = num_layers * [hidden]
            hidden_dims = [in_dim] + hidden_dims
        else:
            hidden_dims = np.linspace(in_dim, out_dim, num_layers+2).astype(int)[:-1]  # FIXME: CHECK THIS

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


class ivaeReplica(nn.Module):
    def __init__(self, kx, km, **kwargs):
        super().__init__()
        self.kx = kx
        self.km = km

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
        self.layer_growth = kwargs.get('layer_growth', 'const')
        self.ivae_loss = kwargs.get('ivae_loss', True)
        self.compound_posterior = kwargs.get('compound_posterior', False)

        net_args = {'activation': self.activation, 'slope': self.slope, 'hidden_dim': self.hidden_dim,
                    'num_layers': self.num_layers, 'layer_growth': self.layer_growth}

        # prior
        if self.shared_net:
            self.prior_net = MLP(km, self.pz, mu_flag=self.learn_mu_prior, **net_args)
        else:
            self.prior_net = MLP(km, self.pz, mu_flag=False, **net_args)  # this is for logvar
            if self.learn_mu_prior:
                self.prior_mu_net = MLP(km, self.pz, var_flag=False, **net_args)
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
            self.encoder = MLP(kx + km, self.pz, **net_args)
        else:
            self.encoder_mu = MLP(kx + km, self.pz, var_flag=False, **net_args)
            self.encoder_logvar = MLP(kx + km, self.pz, mu_flag=False, **net_args)

    def forward(self, x, m):
        xm = torch.cat((x, m), 1)
        # Prior parameters
        if self.learn_mu_prior:
            if self.shared_net:
                self.mu_prior, self.log_var_prior = self.prior_net(m)
            else:
                self.mu_prior = self.prior_mu_net(m)
                self.log_var_prior = self.prior_net(m)
        else:
            self.log_var_prior = self.prior_net(m)

        # Encoder parameters
        if self.shared_net:
            self.mu_q, self.log_var_q = self.encoder(xm)
        else:
            self.mu_q = self.encoder_mu(xm)
            self.log_var_q = self.encoder_logvar(xm)

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
            px_z = util.gauss_ll(x, self.mu_p, self.log_var_p)
            qz_xm = util.gauss_ll(self.z, self.mu_q, self.log_var_q)
            pz_m = util.gauss_ll(self.z, mu_prior, self.log_var_prior)
            return torch.mean(px_z), torch.mean(qz_xm), torch.mean(pz_m)
        else:
            kl = util.kl_normal(self.mu_q, self.log_var_q, mu_prior, self.log_var_prior)
            recon = -1 * util.gauss_ll(x, self.mu_p, self.log_var_p)
            return torch.mean(recon), torch.mean(kl)

    def get_z_samples(self):
        return self.z




#####################################################
# FIXME: MOVE THE PLOTTING STUFF SOMEWHERE ELSE!!!
# loss_roots = ["iVAE_runs\\run_008\\loss.npz", "iVAE_runs\\run_009\\loss.npz", "iVAE_runs\\run_010\\loss.npz"]
metric_roots = ["iVAE_runs\\run_000\\metrics.npz"]#, "iVAE_runs\\run_001\\metrics.npz"]
# # Plot loss
# for i, root in enumerate(loss_roots):
#     losses = np.load(root)
#     kl = losses['kl']
#     recon = losses['recon']
#     plt.plot(recon, label="Recon" + str(i))
#     # plt.plot(kl, '--', label="KL" + str(i))
# plt.legend()
# plt.show()
# # Plot metrics
# for i, root in enumerate(metric_roots):
#     metrics = np.load(root)
#     ami = metrics['approx_mi']
#     au = metrics['active_units']
#     plt.plot(ami, label="Approx MI "+str(i))
#     # plt.plot(au[:, 0], label="Latent 0")
#     # plt.plot(au[:, 1], label="Latent 1")
# plt.legend()
# plt.show()


settings = {
    # Training parameters
    'batch_size': 64,
    'epochs': 1,
    'pz': 2,
    'rng': 1,
    # Network settings
    'learn_mu_prior': True,
    'learn_var_p': True,
    'shared_net': False,
    'lrelu_slope': 0.01,
    'hidden_dim': 50,
    'num_layers': 2,
    'layer_growth': 'const',  # 'const' or 'linear'
    # Auxiliary data settings
    'one_hot': True,
    'num_m': -1,  # if > 0, choose this many unique m values, else choose all
    'zscore_m': True,  # only used when M is not one-hot encoding
    'N': 40000,
    'N_trunc': -1,
    # Loss function settings
    'ivae_loss': False,  # true means compute loss as the sum of 3 log likelihood terms, false means use KL divergence
    'compound_posterior': True,
}

# Init logger
logger = Logger("iVAE_runs")
logger.record_model(settings)

torch.manual_seed(settings['rng'])
np.random.seed(settings['rng'])

# Create dataset
data = np.load("data/data_ivae1.npz")
x = data['x'][:settings['N'], :]
m = data['u'][:settings['N'], :]
z = data['s'][:settings['N'], :]
raw_idx = np.argmax(m, axis=1)

# Preprocess M if need be
if settings['num_m'] > 0:
    m_unq = np.random.choice(m.shape[1], size=settings['num_m'], replace=False)
    m_idx = []
    for mval in m_unq:
        m_idx.append(np.where(raw_idx == mval)[0])
    m_idx = np.array(m_idx).flatten()
    raw_idx = raw_idx[m_idx]
    x = x[m_idx, :]
    m = m[m_idx, :]
else:
    m_idx = []
    for mval in np.unique(raw_idx):
        location = np.where(raw_idx == mval)[0]
        if settings['N_trunc'] > 0:
            m_idx.append(location[:settings['N_trunc']])
        else:
            m_idx.append(location)
    m_idx = np.array(m_idx).flatten()
    raw_idx = raw_idx[m_idx]
    x = x[m_idx, :]
    m = m[m_idx, :]

if not settings['one_hot']:
    m = raw_idx
    if settings['zscore_m']:
        m = (m - m.mean()) / m.std()

# Prepare dataset
kx = x.shape[1]
km = m.shape[1] if settings['one_hot'] else 1
m_events = km if settings['one_hot'] else len(m)
data = TensorDataset(torch.Tensor(x), torch.Tensor(m), torch.Tensor(z))
generator = torch.Generator().manual_seed(settings['rng'])
train_data, valid_data, test_data = random_split(data, [0.8, 0.1, 0.1], generator=generator)
train_loader = DataLoader(train_data, settings['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_data, settings['batch_size'], shuffle=False)


# Define model
model = ivaeReplica(kx, km, **settings)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

loss_ces = []  # FIXME: INTEGRATE INTO CODE
num_matches = []

iteration = 0
for epoch in tqdm(range(settings['epochs'])):

    for i, batch in enumerate(train_loader):
        x = batch[0]
        m = batch[1]
        if not settings['one_hot']:
            m = m.view(-1, 1)
        output = model(x, m)

        # # Latent space performance metrics
        # m_pred, lls = util.lbl_decode_perf(model, x, m_events, N=10)
        # m_true = np.argmax(m, axis=1)
        # match = len(torch.where(m_pred == m_true)[0])
        # num_matches.append(match / m_events)
        # # crit = torch.nn.CrossEntropyLoss()
        # # loss_ce = crit(lls, m_true)
        # # loss_ces.append(loss_ce.detach().numpy())
        # mu_q = output[-2]
        # logvar_q = output[-1]
        # au, au_var = util.active_units(mu_q)
        # zs = model.get_z_samples()
        # ami = util.approx_mi(mu_q, logvar_q, zs)
        # logger.record_metrics([au_var, ami], ['active_units', 'approx_mi'])

        # TODO: parameters saved as mean over batch, may not be appropriate for different M's
        logger.record_params(output, ["mu_p", "logvar_p", "mu_prior", "logvar_prior", "mu_q", "logvar_q"])

        optimizer.zero_grad()
        losses = model.loss(x)
        if settings['ivae_loss']:
            loss = -losses[0] + losses[1] - losses[2]
            logger.record_loss([-losses[0], losses[1] - losses[2]], ["recon", "kl"])
        else:
            loss = losses[0] + losses[1]
            logger.record_loss(losses, ["recon", "kl"])
        loss.backward()
        optimizer.step()

        # # Validation/test
        # model.eval()
        # with torch.no_grad():
        #     for batch in valid_loader:
        #         x = batch[0]
        #         m = batch[1]
        #         output = model(x, m)
        #
        #         losses_valid = model.loss(x)
        #         if settings['ivae_loss']:
        #             loss_valid = -losses[0] + losses[1] - losses[2]
        #         else:
        #             loss_valid = losses[0] + losses[1]
        #
        # scheduler.step(loss_valid)

        iteration += 1

    logger.record_params([iteration], ['epoch_end'])

# # plt.plot(loss_ces, label="Cross entropy")
# plt.plot(num_matches, label="Raw accu")
# plt.title("Supposed cross entropy loss of decoder")
# plt.legend()
# plt.show()

logger.save()



# Test performance
x, m, z = zip(*test_data)
x = torch.stack(list(x))
m = torch.stack(list(m))
z = torch.stack(list(z))

model.eval()
with torch.no_grad():
    if not settings['one_hot']:
        m = m.view(-1, 1)
    output = model(x, m)

    mu_q = output[-2]
    logvar_q = output[-1]
    mcc = util.mcc(z, mu_q, logvar_q)
    acc = util.decoding_acc_lbl(model, x, m_events)

    losses = model.loss(x)
    print("Test recon loss: " + str(losses[0].detach().cpu().numpy()))
    print("Test KL loss: " + str(losses[1].detach().cpu().numpy()))

    # Show f mixing function activations
    plt.imshow(output[0].detach().cpu().numpy(), aspect=0.005)
    plt.show()


