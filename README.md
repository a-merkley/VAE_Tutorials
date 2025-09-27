# VAE Tutorials
This repository contains examples of VAE models and what happens to them for changes in architecture, dataset, and loss function. The models considered are:
- **Vanilla VAE**: this is the original VAE introduced in 2013 by Kingma et al and is a baseline for understanding the key structure of other VAE models.
- **iVAE**: identifiable VAE was introduced in 2020 by Khemakhem et al to obtain a meaningful latent structure by adding an auxiliary variable that guarantees model identifiability.
- **LVAE**: Ladder VAE is an early proposal for stable training of hierarchical VAE models, appearing in the work of Sonderby et al in 2016.

The Jupyter tutorials study these models both in practice and explain their conceptual motivations. There are also ready-to-run model .py files in each model folder that can be run outside of Jupyter.
