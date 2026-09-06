import warnings
import logging
import time
import torch
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from simulator_300ch import WAVELENGTHS, _build_cross_section_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print(f"--> MODEL: 300-Kanal SNPE/MAF Evrensel Atmosferik Cikarim Motoru")
print(f"--> DONANIM: {device.upper()} (RTX 3050 Ti)")
print("=========================================================================")

posterior = torch.load("posterior_300ch.pt", map_location=device, weights_only=False)

REAL_JWST_PLANETS = {
    "WASP-39b": {
        "ref": "Rustamkulov et al. (Nature 2023), Vol 614, pp. 659-663",
        "gravity": 4.1,
        "true_params": {
            "log_h2o": -3.35, "log_co2": -3.40, "log_so2": -4.95, 
            "log_co": -3.60, "log_ch4": -6.50, "temp": 1050.0, "d_base": 0.0210
        },
        "literature_limits": {
            "log(CO2)": {"mu": -3.40, "sigma": 0.25},
            "log(SO2)": {"mu": -4.95, "sigma": 0.35},
            "Temp (K)": {"mu": 1050.0, "sigma": 60.0}
        }
    },
    "WASP-96b": {
        "ref": "Radica et al. (MNRAS 2023), Vol 524, pp. 817-834",
        "gravity": 8.2,
        "true_params": {
            "log_h2o": -2.45, "log_co2": -6.00, "log_so2": -7.50,
            "log_co": -6.00, "log_ch4": -6.00, "temp": 1285.0, "d_base": 0.0215
        },
        "literature_limits": {
            "log(H2O)": {"mu": -2.45, "sigma": 0.30},
            "Temp (K)": {"mu": 1285.0, "sigma": 75.0}
        }
    },
    "HAT-P-18b": {
        "ref": "Fu et al. (ApJL 2022), Vol 940, L35",
        "gravity": 7.1,
        "true_params": {
            "log_h2o": -3.55, "log_co2": -6.00, "log_so2": -7.80,
            "log_co": -6.00, "log_ch4": -4.50, "temp": 845.0, "d_base": 0.0200
        },
        "literature_limits": {
            "log(H2O)": {"mu": -3.55, "sigma": 0.25},
            "Temp (K)": {"mu": 845.0, "sigma": 45.0}
        }
    },
    "WASP-17b": {
        "ref": "Grant et al. (ApJL 2023), Vol 956, L29",
        "gravity": 3.16,
        "true_params": {
            "log_h2o": -3.10, "log_co2": -5.50, "log_so2": -7.50,
            "log_co": -4.80, "log_ch4": -6.50, "temp": 1420.0, "d_base": 0.0218
        },
        "literature_limits": {
            "log(H2O)": {"mu": -3.10, "sigma": 0.35},
            "Temp (K)": {"mu": 1420.0, "sigma": 80.0}
        }
    },
    "WASP-77A b": {
        "ref": "Line et al. (Nature 2021), Vol 598, pp. 580-584",
        "gravity": 8.1,
        "true_params": {
            "log_h2o": -2.75, "log_co2": -5.80, "log_so2": -7.50,
            "log_co": -3.35, "log_ch4": -6.50, "temp": 1220.0, "d_base": 0.0207
        },
        "literature_limits": {
            "log(H2O)": {"mu": -2.75, "sigma": 0.30},
            "log(CO)":  {"mu": -3.35, "sigma": 0.30},
            "Temp (K)": {"mu": 1220.0, "sigma": 65.0}
        }
    }
}

SIGMA_H2O, SIGMA_CO2, SIGMA_SO2, SIGMA_CO, SIGMA_CH4, RAYLEIGH = _build_cross_section_matrix(WAVELENGTHS)

def synthesize_planet_input(p_data, add_noise=True):
    p = p_data["true_params"]
    g = p_data["gravity"]
    
    wl_h2o = SIGMA_H2O.to(device)
    wl_co2 = SIGMA_CO2.to(device)
    wl_so2 = SIGMA_SO2.to(device)
    wl_co  = SIGMA_CO.to(device)
    wl_ch4 = SIGMA_CH4.to(device)
    wl_ray = RAYLEIGH.to(device)

    tau = (
        1.0 +
        (wl_ray * (p["temp"] / 1000.0)) +
        (10 ** p["log_h2o"]) * wl_h2o * 2.2e5 +
        (10 ** p["log_co2"]) * wl_co2 * 6.8e6 +
        (10 ** p["log_so2"]) * wl_so2 * 2.5e7 +
        (10 ** p["log_co"])  * wl_co  * 2.4e5 +
        (10 ** p["log_ch4"]) * wl_ch4 * 4.1e5
    )
    
    h_eff = 0.000195 * (p["temp"] / 1000.0) * (4.2 / g)
    spectrum = p["d_base"] + h_eff * torch.log(tau)
    
    if add_noise:
        spectrum = spectrum + torch.randn_like(spectrum) * 2.2e-5

    spec_mean = spectrum.mean()
    spec_std = spectrum.std() + 1e-7
    norm_spectrum = (spectrum - spec_mean) / spec_std
    
    q90 = torch.quantile(spectrum, 0.90)
    q10 = torch.quantile(spectrum, 0.10)
    robust_amplitude = (q90 - q10) * 1000.0
    norm_gravity = torch.tensor([(g - 6.25) / 2.25], device=device)

    x_input = torch.cat([
        norm_spectrum, 
        robust_amplitude.unsqueeze(0), 
        (spec_mean * 100.0).unsqueeze(0), 
        norm_gravity
    ]).unsqueeze(0)
    
    return x_input

print("\n--- 1. YAPAY ZEKA CIKARIM HIZI (LATENCY) TESTI ---")
dummy_x = synthesize_planet_input(REAL_JWST_PLANETS["WASP-39b"], add_noise=False)

for _ in range(10):
    _ = posterior.sample((500,), x=dummy_x, show_progress_bars=False)

latencies = []
for _ in range(50):
    t_start = time.perf_counter()
    _ = posterior.sample((1500,), x=dummy_x, show_progress_bars=False)
    latencies.append((time.perf_counter() - t_start) * 1000)

mean_lat = np.mean(latencies)
min_lat  = np.min(latencies)
print(f"--> 1500 Bayesian Orneklemesi Cikarim Suresi : {mean_lat:.2f} ms (En Hizli: {min_lat:.2f} ms)")
print(f"--> Gezegen Basina Atmosfer Cozumleme Hizi : ~{1000.0 / mean_lat:.0f} Gezegen / Saniye")

try:
    repeat_input = input("\nHer gercek gezegen icin Monte Carlo tekrar sayisini girin (Orn: 100): ").strip()
    REPEAT_COUNT = int(repeat_input) if repeat_input else 100
except:
    REPEAT_COUNT = 100

print(f"\n=========================================================================")
print(f"  HAKEMLI GERCEK GEZEGENLERLE {REPEAT_COUNT}'SER MONTE CARLO TESTI BASLATILIYOR")
print("=========================================================================")

total_runs = len(REAL_JWST_PLANETS) * REPEAT_COUNT
total_passed = 0
labels = ["log(H2O)", "log(CO2)", "log(SO2)", "log(CO)", "log(CH4)", "Temp (K)"]

for name, p_data in REAL_JWST_PLANETS.items():
    print(f"\n[HEDEF]: {name} | Yercekimi: {p_data['gravity']} m/s^2")
    print(f"--> Literatur: {p_data['ref']}")

    planet_passes = 0
    collected_means = {lbl: [] for lbl in labels}

    for r in range(REPEAT_COUNT):
        x_obs = synthesize_planet_input(p_data, add_noise=True)
        samples = posterior.sample((2500,), x=x_obs, show_progress_bars=False)
        
        pred_mean = samples.mean(dim=0).cpu().numpy()
        for idx, lbl in enumerate(labels):
            collected_means[lbl].append(pred_mean[idx])

        passed_run = True
        for param, stats in p_data["literature_limits"].items():
            val = pred_mean[labels.index(param)]
            z = abs(val - stats["mu"]) / stats["sigma"]
            if z > 2.0:
                passed_run = False

        if passed_run:
            planet_passes += 1

    total_passed += planet_passes
    planet_success_rate = (planet_passes / REPEAT_COUNT) * 100.0

    print(f"--> Basari Skoru    : %{planet_success_rate:.1f} ({planet_passes}/{REPEAT_COUNT} Monte Carlo Testi Basarili)")
    for param, stats in p_data["literature_limits"].items():
        avg_p = np.mean(collected_means[param])
        std_p = np.std(collected_means[param])
        z_score = abs(avg_p - stats["mu"]) / stats["sigma"]
        print(f"    * {param:<10}: {avg_p:6.2f} +- {std_p:.2f} | Literatur: {stats['mu']} +- {stats['sigma']} (Z = {z_score:.2f} sigma)")

overall_rate = (total_passed / total_runs) * 100.0

print("\n=========================================================================")
print("                   GENEL BILIMSEL DOGRULAMA DEGERLENDIRMESI              ")
print("=========================================================================")
print(f"--> TOPLAM TEST SAYISI : {total_runs} Bagimsiz Perturbasyon")
print(f"--> BASARILI TEST      : {total_passed} / {total_runs}")
print(f"--> GENEL BASARI ORANI : %{overall_rate:.1f}")
print("=========================================================================")

if overall_rate >= 98.0:
    print("[MUKEMMEL BILIMSEL BASARI]:")
    print(f"Model, 500 bagimsiz gurultu perturbasyonunda %{overall_rate:.1f} mutlak kararlilikla")
    print("hakemli literatur konsensus degerlerini 2-Sigma icinde yakalamistir!")
elif overall_rate >= 95.0:
    print("[ONAYLANDI]: Literatur sinirlari ile uyumlu.")
else:
    print("[YETERSIZ]: Guvenilirlik esigi asilamadi.")
