import torch
import time
from sbi.inference import SNPE
from sbi.utils import BoxUniform
import numpy as np
from simulator import generate_spectrum_batch
from test_planets_data import PLANET_DATABASE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Donanim: {device.upper()} (Monster Abra A5 / RTX 3050 Ti)")

# Prior: [log_H2O, log_CO2, log_SO2, log_CO, T_eff, D_base]
prior = BoxUniform(
    low=torch.tensor([-5.5, -5.5, -7.5, -5.0, 700.0, 0.0185], device=device),
    high=torch.tensor([-2.0, -2.0, -3.5, -1.5, 1450.0, 0.0230], device=device)
)

print("\n--- Evrensel Model Egitiliyor (20.000 Simulasyon, 6 Boyutlu Fizik) ---")
start_train = time.time()
theta_train, x_train = generate_spectrum_batch(20000)
theta_train, x_train = theta_train.to(device), x_train.to(device)

inferer = SNPE(prior=prior, density_estimator="maf", device=device)
density_estimator = inferer.append_simulations(theta_train, x_train).train(
    training_batch_size=512, 
    max_num_epochs=45
)
posterior = inferer.build_posterior(density_estimator)
print(f"--> Egitim Tamamlandi: {time.time() - start_train:.2f} saniye.")

print("\n=========================================================================")
print("                   HAKEMLI LITERATUR KOR TEST MATRISI                    ")
print("=========================================================================")

all_passed = True

for name, data in PLANET_DATABASE.items():
    x_obs = torch.tensor(data["spectrum"], dtype=torch.float32, device=device).unsqueeze(0)
    
    t0 = time.time()
    samples = posterior.sample((4000,), x=x_obs)
    infer_time = (time.time() - t0) * 1000 # milisaniye
    
    h2o = samples[:, 0].mean().item()
    co2 = samples[:, 1].mean().item()
    so2 = samples[:, 2].mean().item()
    temp = samples[:, 4].mean().item()
    
    print(f"\n[HEDEF]: {name} | Kaynak: {data['ref']}")
    print(f"--> Cikarim Suresi : {infer_time:.1f} ms")
    print(f"--> log(H2O)       : {h2o:.2f}")
    print(f"--> log(CO2)       : {co2:.2f}")
    print(f"--> log(SO2)       : {so2:.2f}")
    print(f"--> T_eff          : {temp:.1f} K")
    
    planet_pass = True
    for param, (low, high) in data["expected"].items():
        val = {"H2O": h2o, "CO2": co2, "SO2": so2, "T": temp}[param]
        if not (low <= val <= high):
            planet_pass = False
            all_passed = False
            print(f"    [!] {param} beklenen ({low}, {high}) araliginin disinda: {val:.2f}")
            
    status = "ONAYLANDI (LITERATURLE UYUMLU)" if planet_pass else "REDDEDILDI"
    print(f"--> Test Karari    : {status}")

print("\n=========================================================================")
print("                50'LIK MONTE CARLO GENELLESTIRME TESTI                   ")
print("=========================================================================")

theta_test, x_test = generate_spectrum_batch(50)
theta_test, x_test = theta_test.to(device), x_test.to(device)

errors = []
for i in range(50):
    s = posterior.sample((1000,), x=x_test[i].unsqueeze(0), show_progress_bars=False)
    pred_mean = s.mean(dim=0)
    err = torch.abs(pred_mean - theta_test[i]).cpu().numpy()
    errors.append(err)

errors = np.array(errors)
mae_h2o = np.mean(errors[:, 0])
mae_co2 = np.mean(errors[:, 1])
mae_so2 = np.mean(errors[:, 2])
mae_temp = np.mean(errors[:, 4])

print(f"50 Kor Test Gezegeninde Ortalama Mutlak Hata (MAE):")
print(f"--> log(H2O) Hata Payi : +- {mae_h2o:.2f} dex")
print(f"--> log(CO2) Hata Payi : +- {mae_co2:.2f} dex")
print(f"--> log(SO2) Hata Payi : +- {mae_so2:.2f} dex")
print(f"--> Sicaklik Hata Payi : +- {mae_temp:.1f} K")

if all_passed and mae_co2 < 0.45 and mae_temp < 70.0:
    print("\n[GENEL SONUC: MUKEMMEL]")
    print("Model hem 3 farkli gercek JWST gezegeninde hem de 50 rastgele")
    print("kor testte istatistiksel ve fiziksel dogrulugunu kanitlamistir.")
else:
    print("\n[GENEL SONUC: EKSIKLIKLER VAR]")
