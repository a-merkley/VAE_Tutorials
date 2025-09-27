from sklearn.datasets import load_digits
import util
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(1e-2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(1e-2)
        )
        self.mu = nn.Linear(hidden, out_dim)
        self.logvar = nn.Linear(hidden, out_dim)

    def forward(self, out):
        out = self.layers(out)
        return self.mu(out), self.logvar(out)

# Class for deterministic network for bottom-up inputs
class DNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(1e-2),
            nn.Linear(hidden, out_dim),
            nn.LeakyReLU(1e-2)
        )

    def forward(self, x):
        return self.layers(x)


class HVAE(nn.Module):
    def __init__(self, kx, kz1, kz2, hidden=30):
        super().__init__()

        self.d1_net = DNet(kx, kz1, hidden)  # d1 = f1(x) -> input to encoder1
        self.d2_net = DNet(kz1, kz2, hidden)  # d2 = f2(d1) -> input to encoder2

        # FIXME: CHECK INPUT DIMENSIONS ON ENCODERS
        self.encoder1 = MLP(kz1, kz1, hidden)  # q_b(z1|x)=N(eta_b, sigma_b), bottom-up correction
        self.encoder2 = MLP(kz2, kz2, hidden)  # q(z2|x)=N(eta2, C2), posterior of z2
        self.prior = MLP(kz2, kz1, hidden)  # p(z1|z2)=N(mu1, sigma1)
        # Intermediate in here is a compound distribution for q(z1|z2,x)=q_b(z1|x)*p(z1|z2)
        self.decoder = MLP(kz1, kx, hidden)  # p(x|z1)=N(mu_d, sigma_d) TODO: MAYBE MAKE THIS BERNOULLI

    def forward(self, x):
        # NOTE: (eta, C) are parameters of posteriors q; (mu, var) are parameters of true distribution p
        d1 = self.d1_net(x)
        d2 = self.d2_net(d1)

        self.eta_b, self.log_Cb = self.encoder1(d1)
        self.eta2, self.log_C2 = self.encoder2(d2)

        self.z2 = util.sample(self.eta2, self.log_C2)
        self.mu1, self.logvar1 = self.prior(self.z2)

        self.eta1, C1 = util.compound_normal(self.eta_b, self.log_Cb, self.mu1, self.logvar1)
        self.log_C1 = C1.log()

        self.z1 = util.sample(self.eta1, self.log_C1)
        self.mu_x, self.logvar_x = self.decoder(self.z1)

        return (self.eta1, self.log_C1, self.eta2, self.log_C2,
                self.mu1, self.logvar1, self.eta_b, self.log_Cb,
                self.mu_x, self.logvar_x)

    def loss(self, x):
        recon = -1 * util.gauss_ll(x, self.mu_x, self.logvar_x)  # FIXME: NEED ANY ADDITIONAL SAMPLING?
        kl1 = util.kl_normal(self.eta1, self.log_C1, self.mu1, self.logvar1)
        kl2 = util.kl_normal(self.eta2, self.log_C2, torch.zeros_like(self.eta2), torch.zeros_like(self.log_C2))
        return torch.mean(recon), torch.mean(kl1), torch.mean(kl2)




if __name__ == "__main__":
    settings = {
        'epochs': 100,
        'batch_size': 32,
        'rng': 88,
        'hidden': 30
    }

    torch.manual_seed(settings['rng'])
    np.random.seed(settings['rng'])

    # Get raw data
    digits = load_digits()  # TODO: possibly make a CNN?
    x = digits.data
    m = digits.target

    # Model parameters
    kx = x.shape[1]
    kz1 = 2
    kz2 = kz1

    # Make dataset
    data = TensorDataset(torch.Tensor(x), torch.Tensor(m))
    generator = torch.Generator().manual_seed(settings['rng'])
    train_data, test_data = random_split(data, [0.8, 0.2], generator=generator)
    train_loader = DataLoader(train_data, settings['batch_size'], shuffle=True)

    # Define model
    model = HVAE(kx, kz1, kz2, settings['hidden'])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3)

    # Capture losses
    recon_loss = []
    kl1_loss = []
    kl2_loss = []

    # Training loop
    for epoch in tqdm(range(settings['epochs'])):
        for batch in train_loader:
            x = batch[0]
            # output: eta1, log_C1, eta2, log_C2, mu1, logvar1, eta_b, log_Cb, mu_x, logvar_x
            output = model(x)

            losses = model.loss(x)
            loss = losses[0] + losses[1] + losses[2]

            optimizer.zero_grad()  # FIXME: POTENTIALLY MOVE THIS ABOVE LOSSES
            loss.backward()
            optimizer.step()

            recon_loss.append(losses[0].item())
            kl1_loss.append(losses[1].item())
            kl2_loss.append(losses[2].item())


    from matplotlib import pyplot as plt
    plt.plot(recon_loss, label="Recon")
    plt.plot(kl1_loss, label="KL 1")
    plt.plot(kl2_loss, label="KL 2")
    plt.legend()
    plt.show()

