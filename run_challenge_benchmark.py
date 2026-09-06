import warnings
import logging
import torch
import numpy as np

# Sari uyarilari ve loglama ciktilarini tamamen susturur
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from sbi.inference import SNPE
from sbi.utils import BoxUniform
from simulator import generate_spectrum_batch
from test_planets_data import CHALLENGE_DATABASE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Donanim: {device.upper()} (Monster Abra A5 / RTX 3050 Ti)")

prior = BoxUniform(
    low=torch.tensor([-5.5, -8.5, -9.5, -8.5, 650.0], device=device),
    high=torch.tensor([-1.5, -1.5, -3.0, -1.5, 1700.0], device=device)
)

print("\n--- Model Egitiliyor (25.000 Simulasyonluk Fizik Havuzu) ---")
theta_train, x_train = generate_spectrum_batch(25000)
theta_train, x_train = theta_train.to(device), x_train.to(device)

inferer = SNPE(prior=prior, density_estimator="maf", device=device)
density_estimator = inferer.append_simulations(theta_train, x_train).train(
    training_batch_size=512, 
    max_num_epochs=45
)
posterior = inferer.build_posterior(density_estimator)

REPEAT_COUNT = 20
print(f"\n=========================================================================")
print(f"   HAKEMLI LITERATURLE 2-SIGMA BAYESIAN UYUM TESTI (20 MONTE CARLO)      ")
print(f"=========================================================================")

final_success = True

for name, data in CHALLENGE_DATABASE.items():
    raw_spectrum = torch.tensor(data["spectrum"], dtype=torch.float32, device=device)
    gravity_val = torch.tensor([data["gravity"]], dtype=torch.float32, device=device)
    
    runs_h2o, runs_co2, runs_so2, runs_co, runs_temp = [], [], [], [], []
    runs_passed = 0
    
    print(f"\n[HEDEF]: {name} | Yercekimi (g): {data['gravity']} m/s^2 | Kaynak: {data['ref']}")
    
    for r in range(REPEAT_COUNT):
        perturbed_spectrum = raw_spectrum + torch.randn_like(raw_spectrum) * 4.0e-5
        x_obs = torch.cat([perturbed_spectrum, gravity_val]).unsqueeze(0)
        
        samples = posterior.sample((1500,), x=x_obs, show_progress_bars=False)
        
        h2o = samples[:, 0].mean().item()
        co2 = samples[:, 1].mean().item()
        so2 = samples[:, 2].mean().item()
        co  = samples[:, 3].mean().item()
        temp = samples[:, 4].mean().item()
        
        runs_h2o.append(h2o)
        runs_co2.append(co2)
        runs_so2.append(so2)
        runs_co.append(co)
        runs_temp.append(temp)
        
        # 2-Sigma (Z <= 2.0 / %95.4 guven) kontrolu
        passed = True
        current_vals = {"H2O": h2o, "CO2": co2, "SO2": so2, "CO": co, "T": temp}
        
        for param, stats in data["distributions"].items():
            val = current_vals[param]
            z_score = abs(val - stats["mu"]) / stats["sigma"]
            if z_score > 2.0:
                passed = False
                
        if "upper_limits" in data:
            for param, limit in data["upper_limits"].items():
                if current_vals[param] > limit:
                    passed = False
                    
        if passed:
            runs_passed += 1

    mean_h2o, std_h2o = np.mean(runs_h2o), np.std(runs_h2o)
    mean_co2, std_co2 = np.mean(runs_co2), np.std(runs_co2)
    mean_so2, std_so2 = np.mean(runs_so2), np.std(runs_so2)
    mean_co, std_co   = np.mean(runs_co), np.std(runs_co)
    mean_temp, std_temp = np.mean(runs_temp), np.std(runs_temp)
    
    success_rate = (runs_passed / REPEAT_COUNT) * 100
    
    print(f"--> Bilimsel Basari : %{success_rate:.1f} ({runs_passed}/{REPEAT_COUNT} Test 2-Sigma Icinde)")
    print(f"--> log(H2O)        : {mean_h2o:.2f} +- {std_h2o:.2f}")
    print(f"--> log(CO2)        : {mean_co2:.2f} +- {std_co2:.2f}")
    print(f"--> log(SO2)        : {mean_so2:.2f} +- {std_so2:.2f}")
    print(f"--> log(CO)         : {mean_co:.2f} +- {std_co:.2f}")
    print(f"--> T_eff           : {mean_temp:.1f} +- {std_temp:.1f} K")
    
    for param, stats in data["distributions"].items():
        val = {"H2O": mean_h2o, "CO2": mean_co2, "SO2": mean_so2, "CO": mean_co, "T": mean_temp}[param]
        z = abs(val - stats["mu"]) / stats["sigma"]
        print(f"    * {param} Z-Skoru: {z:.2f} sigma (Literatur: {stats['mu']} +- {stats['sigma']})")

    if success_rate < 85.0:
        final_success = False
        print(f"    [!] Guvenilirlik esigi (%85) asilamadi!")

print("\n=========================================================================")
print("                   GENEL DOGRULAMA DEGERLENDIRMESI                       ")
print("=========================================================================")

if final_success:
    print("[MUKEMMEL BILIMSEL ONAY]:")
    print("Model; ekstrem gezegenlerin tamaminda 2-Sigma (%95.4 guven) sinirlari icinde uyum saglamistir.")
else:
    print("[BASARISIZ]: Istatistiksel uyumsuzluk tespit edildi.")
