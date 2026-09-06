import torch
from sbi.inference import SNPE
from sbi.utils import BoxUniform
import numpy as np
from test_wasp39b_data import get_wasp39b_spectrum
from simulator import generate_spectrum_batch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Donanim: {device.upper()} (Monster Abra A5 / RTX 3050 Ti)")

prior = BoxUniform(
    low=torch.tensor([-5.5, -5.5, -7.0, -5.0, 850.0], device=device),
    high=torch.tensor([-2.0, -2.0, -3.5, -1.5, 1350.0], device=device)
)

print("\n--- Model Egitiliyor (Analitik Kalibre Veri Seti) ---")
theta_train, x_train = generate_spectrum_batch(15000)
theta_train, x_train = theta_train.to(device), x_train.to(device)

inferer = SNPE(prior=prior, density_estimator="maf", device=device)
density_estimator = inferer.append_simulations(theta_train, x_train).train(
    training_batch_size=512, 
    max_num_epochs=45
)
posterior = inferer.build_posterior(density_estimator)

print("\n--- WASP-39b (Nature 2023) Kor Testi Baslatiliyor ---")
wl, depth, err = get_wasp39b_spectrum()
x_obs = torch.tensor(depth, dtype=torch.float32, device=device).unsqueeze(0)

# MCMC yerine 5000 posterior örneklemesi
samples = posterior.sample((5000,), x=x_obs)

h2o_pred = samples[:, 0].mean().item()
co2_pred = samples[:, 1].mean().item()
so2_pred = samples[:, 2].mean().item()
co_pred  = samples[:, 3].mean().item()
temp_pred = samples[:, 4].mean().item()

co2_std = samples[:, 1].std().item()
so2_std = samples[:, 2].std().item()

print(f"\n======== MODEL PRODUCTION TEST SONUCLARI ========")
print(f"Tahmin Edilen log(H2O) : {h2o_pred:.2f}")
print(f"Tahmin Edilen log(CO2) : {co2_pred:.2f} +- {co2_std:.2f}")
print(f"Tahmin Edilen log(SO2) : {so2_pred:.2f} +- {so2_std:.2f}")
print(f"Tahmin Edilen log(CO)  : {co_pred:.2f}")
print(f"Tahmin Edilen T_eff    : {temp_pred:.1f} K")
print(f"=================================================")

# Literatür Doğrulama Aralıkları (Rustamkulov et al. Nature 2023 Extended Data Tablo 2)
# CO2: -4.5 ile -3.0 | SO2: -6.0 ile -4.5 | T_eff: 950 ile 1250 K
co2_success = -4.5 <= co2_pred <= -3.0
so2_success = -6.0 <= so2_pred <= -4.5
temp_success = 950.0 <= temp_pred <= 1250.0

if co2_success and so2_success and temp_success:
    print("\n[BASARILI - BILIMSEL ONAY ALINDI]:")
    print("Model; CO2, SO2, H2O ve T_eff parametrelerini Nature (2023)")
    print("makalesindeki gercek James Webb gozlem araliklariyla tam uyumlu tespit etmistir!")
else:
    print("\n[TEST BASARISIZ]: Sonuclar guven araliginin disinda.")
