import torch
import numpy as np

SIGMA_H2O = torch.tensor([0.00, 0.02, 0.15, 0.70, 0.08, 0.65, 0.12, 0.85, 0.35, 0.02, 0.05, 0.08, 0.45])
SIGMA_CO2 = torch.tensor([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.15, 0.00, 0.02, 1.00, 0.04, 0.00])
SIGMA_SO2 = torch.tensor([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00])
SIGMA_CO  = torch.tensor([0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.18, 0.00, 0.00, 0.00, 0.00, 1.00, 0.04])
RAYLEIGH  = torch.tensor([0.35, 0.20, 0.08, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])

def generate_spectrum_batch(batch_size):
    log_h2o = torch.empty(batch_size, 1).uniform_(-5.5, -1.5)
    log_co2 = torch.empty(batch_size, 1).uniform_(-8.5, -1.5)
    log_so2 = torch.empty(batch_size, 1).uniform_(-9.5, -3.0)
    log_co  = torch.empty(batch_size, 1).uniform_(-8.5, -1.5)
    temp    = torch.empty(batch_size, 1).uniform_(650.0, 1700.0)
    d_base  = torch.empty(batch_size, 1).uniform_(0.0180, 0.0235)
    gravity = torch.empty(batch_size, 1).uniform_(2.5, 9.5)

    theta = torch.cat([log_h2o, log_co2, log_so2, log_co, temp, d_base, gravity], dim=1)

    x_h2o = 10 ** log_h2o
    x_co2 = 10 ** log_co2
    x_so2 = 10 ** log_so2
    x_co  = 10 ** log_co

    tau = (
        1.0 + 
        RAYLEIGH.unsqueeze(0) +
        x_h2o * SIGMA_H2O.unsqueeze(0) * 1.8e5 +
        x_co2 * SIGMA_CO2.unsqueeze(0) * 7.14e6 +
        x_so2 * SIGMA_SO2.unsqueeze(0) * 1.85e7 +
        x_co  * SIGMA_CO.unsqueeze(0)  * 2.1e5
    )

    h_eff = 0.000185 * (temp / 1000.0) * (4.3 / gravity)
    spectrum = d_base + h_eff * torch.log(tau)

    noise = torch.randn_like(spectrum) * 3.5e-5
    observed = spectrum + noise
    model_input = torch.cat([observed, gravity], dim=1)

    return theta[:, :5], model_input
