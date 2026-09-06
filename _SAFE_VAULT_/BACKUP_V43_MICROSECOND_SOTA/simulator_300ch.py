import torch
import torch.nn.functional as F

WAVELENGTHS = torch.linspace(0.6, 5.3, 300)

def _build_cross_section_matrix(wl):
    sigma_h2o = (
        0.20 * torch.exp(-((wl - 0.94) ** 2) / (2 * (0.04 ** 2))) +
        0.35 * torch.exp(-((wl - 1.15) ** 2) / (2 * (0.05 ** 2))) +
        0.85 * torch.exp(-((wl - 1.40) ** 2) / (2 * (0.07 ** 2))) +
        0.75 * torch.exp(-((wl - 1.88) ** 2) / (2 * (0.08 ** 2))) +
        0.90 * torch.exp(-((wl - 2.70) ** 2) / (2 * (0.12 ** 2)))
    )
    sigma_co2 = (
        0.10 * torch.exp(-((wl - 2.01) ** 2) / (2 * (0.04 ** 2))) +
        0.30 * torch.exp(-((wl - 2.72) ** 2) / (2 * (0.05 ** 2))) +
        1.00 * torch.exp(-((wl - 4.33) ** 2) / (2 * (0.09 ** 2)))
    )
    sigma_so2 = 1.00 * torch.exp(-((wl - 4.05) ** 2) / (2 * (0.035 ** 2)))
    sigma_co = 1.00 * torch.exp(-((wl - 4.67) ** 2) / (2 * (0.07 ** 2)))
    sigma_ch4 = (
        0.30 * torch.exp(-((wl - 1.66) ** 2) / (2 * (0.04 ** 2))) +
        0.60 * torch.exp(-((wl - 2.30) ** 2) / (2 * (0.06 ** 2))) +
        0.95 * torch.exp(-((wl - 3.31) ** 2) / (2 * (0.08 ** 2)))
    )
    rayleigh = 0.025 * (0.6 / wl) ** 4
    return sigma_h2o, sigma_co2, sigma_so2, sigma_co, sigma_ch4, rayleigh

SIGMA_H2O, SIGMA_CO2, SIGMA_SO2, SIGMA_CO, SIGMA_CH4, RAYLEIGH = _build_cross_section_matrix(WAVELENGTHS)

def generate_base_spectrum(params, g, log_pcloud=1.0, device="cuda"):
    """
    Belirli fiziksel parametrelerden tek bir 300-kanallı spektrum üretir.
    """
    if isinstance(g, torch.Tensor):
        g = float(g.item())
    if isinstance(log_pcloud, torch.Tensor):
        log_pcloud = float(log_pcloud.item())

    wl_h2o = SIGMA_H2O.to(device)
    wl_co2 = SIGMA_CO2.to(device)
    wl_so2 = SIGMA_SO2.to(device)
    wl_co  = SIGMA_CO.to(device)
    wl_ch4 = SIGMA_CH4.to(device)
    wl_ray = RAYLEIGH.to(device)

    temp = params["temp"]
    tau = (
        1.0 +
        (wl_ray * (temp / 1000.0)) +
        (10 ** params["log_h2o"]) * wl_h2o * 2.2e5 +
        (10 ** params["log_co2"]) * wl_co2 * 6.8e6 +
        (10 ** params["log_so2"]) * wl_so2 * 2.5e7 +
        (10 ** params["log_co"])  * wl_co  * 2.4e5 +
        (10 ** params["log_ch4"]) * wl_ch4 * 4.1e5
    )

    h_eff = 0.000195 * (temp / 1000.0) * (4.2 / g)
    clear_spec = params["d_base"] + h_eff * torch.log(tau)
    
    max_cloud_height = params["d_base"] + h_eff * float(max(0.2, min(5.0, log_pcloud + 4.0)))
    return torch.min(clear_spec, torch.tensor(max_cloud_height, device=device))

def extract_features(spectrum, gravity):
    spec_3d = spectrum.unsqueeze(1) if spectrum.dim() == 2 else spectrum.unsqueeze(0).unsqueeze(1)
    continuum = F.avg_pool1d(spec_3d, kernel_size=31, stride=1, padding=15).squeeze(1)
    if continuum.dim() == 1:
        continuum = continuum.unsqueeze(0)
    
    spec_2d = spectrum if spectrum.dim() == 2 else spectrum.unsqueeze(0)
    detrended = spec_2d - continuum

    det_mean = detrended.mean(dim=1, keepdim=True)
    det_std = detrended.std(dim=1, keepdim=True) + 1e-7
    norm_detrended = (detrended - det_mean) / det_std

    q90 = torch.quantile(spec_2d, 0.90, dim=1, keepdim=True)
    q10 = torch.quantile(spec_2d, 0.10, dim=1, keepdim=True)
    robust_amp = (q90 - q10) * 1000.0

    h2o_contrast = (spec_2d[:, 51:52] - spec_2d[:, 41:42]) * 1000.0
    co2_contrast = (spec_2d[:, 237:238] - spec_2d[:, 220:221]) * 1000.0
    curvature = (spec_2d[:, 5:6] - 2.0 * spec_2d[:, 25:26] + spec_2d[:, 45:46]) * 1000.0

    grav_t = gravity if isinstance(gravity, torch.Tensor) else torch.tensor([[gravity]], device=spec_2d.device)
    if grav_t.dim() == 1:
        grav_t = grav_t.unsqueeze(1)
    norm_g = (grav_t - 6.25) / 2.25

    return torch.cat([norm_detrended, robust_amp, h2o_contrast, co2_contrast, curvature, norm_g], dim=1)
