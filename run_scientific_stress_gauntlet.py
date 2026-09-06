import warnings
import logging
import time
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from simulator_300ch import WAVELENGTHS, _build_cross_section_matrix, extract_features

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print("     ASTROFIZIKSEL STRES TESTI: THE SCIENTIFIC GAUNTLET V2.2            ")
print(f"--> MODEL: 300-Kanal High-Pass Invariant MAF | DONANIM: {device.upper()}")
print("=========================================================================")

posterior = torch.load("posterior_300ch.pt", map_location=device, weights_only=False)

SIGMA_H2O, SIGMA_CO2, SIGMA_SO2, SIGMA_CO, SIGMA_CH4, RAYLEIGH = _build_cross_section_matrix(WAVELENGTHS)

def generate_base_spectrum(params, g, log_pcloud=1.0):
    wl_h2o = SIGMA_H2O.to(device)
    wl_co2 = SIGMA_CO2.to(device)
    wl_so2 = SIGMA_SO2.to(device)
    wl_co  = SIGMA_CO.to(device)
    wl_ch4 = SIGMA_CH4.to(device)
    wl_ray = RAYLEIGH.to(device)

    tau = (
        1.0 +
        (wl_ray * (params["temp"] / 1000.0)) +
        (10 ** params["log_h2o"]) * wl_h2o * 2.2e5 +
        (10 ** params["log_co2"]) * wl_co2 * 6.8e6 +
        (10 ** params["log_so2"]) * wl_so2 * 2.5e7 +
        (10 ** params["log_co"])  * wl_co  * 2.4e5 +
        (10 ** params["log_ch4"]) * wl_ch4 * 4.1e5
    )
    h_eff = 0.000195 * (params["temp"] / 1000.0) * (4.2 / g)
    clear_spec = params["d_base"] + h_eff * torch.log(tau)
    
    max_cloud_height = params["d_base"] + h_eff * float(np.clip(log_pcloud + 4.0, 0.2, 5.0))
    return torch.min(clear_spec, torch.tensor(max_cloud_height, device=device))

# TEST 1: YILDIZ LEKESI KIRLENMESI (TLSE)
print("\n[STRES TESTI 1/4]: YILDIZ LEKESI KIRLENMESI (TLSE - Rackham et al.)")
p_wasp39 = {"log_h2o": -3.35, "log_co2": -3.40, "log_so2": -4.95, "log_co": -3.60, "log_ch4": -6.50, "temp": 1050.0, "d_base": 0.0210}
raw_spec_39 = generate_base_spectrum(p_wasp39, 4.1, log_pcloud=1.0)
spot_distortion = 0.00008 * ((0.6 / WAVELENGTHS.to(device)) ** 1.2)
corrupted_spec_39 = (raw_spec_39 + spot_distortion + torch.randn_like(raw_spec_39) * 2.2e-5).unsqueeze(0)

x_input_39 = extract_features(corrupted_spec_39, torch.tensor([[4.1]], device=device))
samples_39 = posterior.sample((3000,), x=x_input_39, show_progress_bars=False)

co2_mean, co2_std = samples_39[:, 1].mean().item(), samples_39[:, 1].std().item()
so2_mean, so2_std = samples_39[:, 2].mean().item(), samples_39[:, 2].std().item()
temp_mean_39, temp_std_39 = samples_39[:, 5].mean().item(), samples_39[:, 5].std().item()

z_co2 = abs(co2_mean - (-3.40)) / np.sqrt(co2_std**2 + 0.25**2)
z_so2 = abs(so2_mean - (-4.95)) / np.sqrt(so2_std**2 + 0.35**2)
z_t39 = abs(temp_mean_39 - 1050.0) / np.sqrt(temp_std_39**2 + 60.0**2)
pass_t1 = (z_co2 <= 2.0) and (z_so2 <= 2.0) and (z_t39 <= 2.0)

print(f"--> log(CO2) : {co2_mean:.2f} +- {co2_std:.2f} (Z = {z_co2:.2f} sigma | Gercek: -3.40)")
print(f"--> log(SO2) : {so2_mean:.2f} +- {so2_std:.2f} (Z = {z_so2:.2f} sigma | Gercek: -4.95)")
print(f"--> Temp (K) : {temp_mean_39:.1f} +- {temp_std_39:.1f} K (Z = {z_t39:.2f} sigma | Gercek: 1050 K)")
print(f"--> STRES TESTI 1 KARARI : {'BASARILI (Leke Etkisi Filtrelendi)' if pass_t1 else 'BASARISIZ'}")

# TEST 2: OPAK BULUT TABAKASI (10 mbar / logP = -2.0)
print("\n[STRES TESTI 2/4]: OPAK BULUT TABAKASI (Benneke & Seager 2012 / Sing et al. 2016)")
p_wasp96 = {"log_h2o": -2.45, "log_co2": -6.00, "log_so2": -7.50, "log_co": -6.00, "log_ch4": -6.00, "temp": 1285.0, "d_base": 0.0215}
cloudy_spec_96 = (generate_base_spectrum(p_wasp96, 8.2, log_pcloud=-2.0) + torch.randn(300, device=device) * 2.2e-5).unsqueeze(0)

x_input_96 = extract_features(cloudy_spec_96, torch.tensor([[8.2]], device=device))
samples_96 = posterior.sample((3000,), x=x_input_96, show_progress_bars=False)

h2o_mean_96, h2o_std_96 = samples_96[:, 0].mean().item(), samples_96[:, 0].std().item()
temp_mean_96, temp_std_96 = samples_96[:, 5].mean().item(), samples_96[:, 5].std().item()
pcloud_mean_96 = samples_96[:, 6].mean().item()

z_h2o_96 = abs(h2o_mean_96 - (-2.45)) / np.sqrt(h2o_std_96**2 + 0.30**2)
z_t96 = abs(temp_mean_96 - 1285.0) / np.sqrt(temp_std_96**2 + 75.0**2)
pass_t2 = (z_h2o_96 <= 2.0) and (z_t96 <= 2.0)

print(f"--> log(H2O)    : {h2o_mean_96:.2f} +- {h2o_std_96:.2f} (Z = {z_h2o_96:.2f} sigma | Gercek: -2.45)")
print(f"--> Temp (K)    : {temp_mean_96:.1f} +- {temp_std_96:.1f} K (Z = {z_t96:.2f} sigma | Gercek: 1285 K)")
print(f"--> log(P_cloud): {pcloud_mean_96:.2f} bar (Tespit: Yuksek Bulut Tabakasi)")
print(f"--> STRES TESTI 2 KARARI : {'BASARILI (Bulutlu Rejim Dogrulandi)' if pass_t2 else 'BASARISIZ'}")

# TEST 3: KORELE KIRMIZI GURULTU (1/f Pink Noise)
print("\n[STRES TESTI 3/4]: KORELE KIRMIZI GURULTU (JWST 1/f Pink Noise)")
p_wasp17 = {"log_h2o": -3.10, "log_co2": -5.50, "log_so2": -7.50, "log_co": -4.80, "log_ch4": -6.50, "temp": 1420.0, "d_base": 0.0218}
raw_spec_17 = generate_base_spectrum(p_wasp17, 3.16, log_pcloud=1.0)
white_noise = np.random.normal(0, 2.2e-5, size=300)
red_noise_np = gaussian_filter1d(white_noise, sigma=3.0) * np.sqrt(3.0)
correlated_spec_17 = (raw_spec_17 + torch.tensor(red_noise_np, dtype=torch.float32, device=device)).unsqueeze(0)

x_input_17 = extract_features(correlated_spec_17, torch.tensor([[3.16]], device=device))
samples_17 = posterior.sample((3000,), x=x_input_17, show_progress_bars=False)

h2o_mean_17, h2o_std_17 = samples_17[:, 0].mean().item(), samples_17[:, 0].std().item()
temp_mean_17, temp_std_17 = samples_17[:, 5].mean().item(), samples_17[:, 5].std().item()

z_h2o_17 = abs(h2o_mean_17 - (-3.10)) / np.sqrt(h2o_std_17**2 + 0.35**2)
z_t17 = abs(temp_mean_17 - 1420.0) / np.sqrt(temp_std_17**2 + 80.0**2)
pass_t3 = (z_h2o_17 <= 2.0) and (z_t17 <= 2.0)

print(f"--> log(H2O) : {h2o_mean_17:.2f} (Z = {z_h2o_17:.2f} sigma) | Temp : {temp_mean_17:.1f} K (Z = {z_t17:.2f} sigma)")
print(f"--> STRES TESTI 3 KARARI : {'BASARILI' if pass_t3 else 'BASARISIZ'}")

# TEST 4: KOR EKSTREM DEV (WASP-121b Ultra-Hot Jupiter)
print("\n[STRES TESTI 4/4]: KOR EKSTREM DEV (WASP-121b - 1650 K, g=9.4)")
p_wasp121 = {"log_h2o": -3.20, "log_co2": -5.50, "log_so2": -7.50, "log_co": -2.90, "log_ch4": -7.00, "temp": 1650.0, "d_base": 0.0225}
raw_spec_121 = (generate_base_spectrum(p_wasp121, 9.4, log_pcloud=1.0) + torch.randn(300, device=device) * 2.5e-5).unsqueeze(0)

x_input_121 = extract_features(raw_spec_121, torch.tensor([[9.4]], device=device))
samples_121 = posterior.sample((3000,), x=x_input_121, show_progress_bars=False)

h2o_mean_121, h2o_std_121 = samples_121[:, 0].mean().item(), samples_121[:, 0].std().item()
co_mean_121, co_std_121   = samples_121[:, 3].mean().item(), samples_121[:, 3].std().item()
temp_mean_121, temp_std_121 = samples_121[:, 5].mean().item(), samples_121[:, 5].std().item()

z_h2o_121 = abs(h2o_mean_121 - (-3.20)) / np.sqrt(h2o_std_121**2 + 0.35**2)
z_co_121  = abs(co_mean_121 - (-2.90)) / np.sqrt(co_std_121**2 + 0.35**2)
z_t121    = abs(temp_mean_121 - 1650.0) / np.sqrt(temp_std_121**2 + 90.0**2)
pass_t4 = (z_h2o_121 <= 2.0) and (z_co_121 <= 2.0) and (z_t121 <= 2.0)

print(f"--> log(H2O) : {h2o_mean_121:.2f} | log(CO) : {co_mean_121:.2f} | Temp : {temp_mean_121:.1f} K (Z = {z_t121:.2f} sigma)")
print(f"--> STRES TESTI 4 KARARI : {'BASARILI' if pass_t4 else 'BASARISIZ'}")

print("\n=========================================================================")
passed_stress = sum([pass_t1, pass_t2, pass_t3, pass_t4])
print(f"--> TOPLAM GECILEN STRES TESTI: {passed_stress} / 4")
if passed_stress == 4:
    print("[KUSURSUZ BILIMSEL DAYANIKLILIK (4/4 TAM BASARI)]:")
    print("Model; leke ayristirma, bulut dejenerasyonu, dedektor kirmizi gurultusu")
    print("ve ekstrem termal rejim senaryolarinin dordunde de 2-Sigma icinde kalarak")
    print("gercek teleskop bozulmalarina karsi kurşun geçirmezligini kanitlamistir.")
else:
    print(f"[EKSIK]: {4 - passed_stress} test elendi.")
print("=========================================================================")
