import numpy as np
import yaml
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from torch.nn.utils.parametrizations import weight_norm

import lib.util as util
from lib.logger import Logger


######################### CLASS DEFINITIONS ############################
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        activation = kwargs.get('activation')
        slope = kwargs.get('lrelu_slope')
        hidden = kwargs.get('hidden_dim')
        num_layers = kwargs.get('num_layers')
        norm_type = kwargs.get('norm_type')

        hidden_dims = num_layers * [hidden]
        hidden_dims = [in_dim] + hidden_dims

        # Define middle layers
        layer_lst = []
        for n in range(num_layers):
            # Linear layer (possibly w normalization)
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
        self.mu = nn.Linear(hidden_dims[-1], out_dim)
        self.log_var = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x):
        out = self.layers(x)
        return self.mu(out), self.log_var(out)


class VAE(nn.Module):
    def __init__(self, kx, **kwargs):
        super().__init__()
        self.kx = kx

        self.pz = kwargs.get('pz')

        self.encoder = MLP(kx, self.pz, **kwargs)
        self.decoder = MLP(self.pz, kx, **kwargs)

    def forward(self, x):
        self.mu_e, self.logvar_e = self.encoder(x)
        z = util.sample(self.mu_e, self.logvar_e)
        self.mu_d, self.logvar_d = self.decoder(z)

    def loss(self, x):
        recon = -1 * util.normal_ll(x, self.mu_d, self.logvar_d)
        kl_div = util.kl_normal(self.mu_e, self.logvar_e, torch.zeros_like(self.mu_e), torch.ones_like(self.logvar_e))

        return recon.mean(), kl_div.mean()



######################### TRAINING & EVALUATION ############################
if __name__=="__main__":
    # Settings
    config_path = "config.yaml"
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    train_cfg = config['train']
    model_cfg = config['model']

    # Load CPU or GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Initialize logger
    logger = Logger("VAE_runs")
    logger.record_model(config)

    # Create train/test datasets
    data, kx = util.get_dataset(**train_cfg)
    generator = torch.Generator().manual_seed(train_cfg['rng'])
    train_data, valid_data, test_data = random_split(data, train_cfg['data_split'], generator=generator)
    train_loader = DataLoader(train_data, train_cfg['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_data, train_cfg['batch_size'], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    # Initialize model
    model = VAE(kx, **model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])

    # Training loop
    loss_epoch = []
    for epoch in tqdm(range(train_cfg['epochs'])):
        loss_batch = []

        for batch in train_loader:
            x, u = util.get_batch(batch, train_cfg['data_src'], device)

            # === Forward pass
            model(x)

            # === Update model, compute loss
            optimizer.zero_grad()
            losses = model.loss(x)
            loss = losses[0] + losses[1]
            loss.backward()
            optimizer.step()

            loss_batch.append(loss.item())

            logger.record_loss([losses[0].item(), losses[1].item()], ["recon", "kl"])

        loss_epoch.append(np.mean(loss_batch))

    logger.save()

    plt.plot(loss_epoch)
    plt.xlabel("Epoch")
    plt.ylabel("Overall loss")
    plt.show()
