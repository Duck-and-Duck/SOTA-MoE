import torch
import numpy as np
from sbi.inference import SNPE
from sbi.utils import BoxUniform
from simulator import generate_spectrum_batch
from test_planets_data import PLANET_DATABASE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Donanım: {device.upper()} (Monster Abra A5 / RTX 3050 Ti)")

prior = BoxUniform(
    low=torch.tensor([-5.5, -5.5, -9.5, -5.0, 700.0], device=device),
    high=torch.tensor([-1.5, -2.0, -3.5, -1.5, 1450.0], device=device)
)

print("\n--- Evrensel Model Eğitiliyor (Sıfır Rejection Kısıtı) ---")
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
print(f"   {REPEAT_COUNT} TEKRARLI GÜRÜLTÜ PERTÜRBASYONLU KÖR TEST (MONTE CARLO)    ")
print(f"=========================================================================")

final_success = True

for name, data in PLANET_DATABASE.items():
    raw_spectrum = torch.tensor(data["spectrum"], dtype=torch.float32, device=device)
    gravity_val = torch.tensor([data["gravity"]], dtype=torch.float32, device=device)
    
    runs_h2o, runs_co2, runs_so2, runs_temp = [], [], [], []
    runs_passed = 0
    
    print(f"\n[HEDEF]: {name} | Yerçekimi (g): {data['gravity']} m/s^2")
    
    for r in range(REPEAT_COUNT):
        perturbed_spectrum = raw_spectrum + torch.randn_like(raw_spectrum) * 4.0e-5
        x_obs = torch.cat([perturbed_spectrum, gravity_val]).unsqueeze(0)
        
        samples = posterior.sample((1500,), x=x_obs, show_progress_bars=False)
        
        h2o = samples[:, 0].mean().item()
        co2 = samples[:, 1].mean().item()
        so2 = samples[:, 2].mean().item()
        temp = samples[:, 4].mean().item()
        
        runs_h2o.append(h2o)
        runs_co2.append(co2)
        runs_so2.append(so2)
        runs_temp.append(temp)
        
        passed = True
        for param, (low, high) in data["expected"].items():
            val = {"H2O": h2o, "CO2": co2, "SO2": so2, "T": temp}[param]
            if not (low <= val <= high):
                passed = False
        if passed:
            runs_passed += 1

    mean_h2o, std_h2o = np.mean(runs_h2o), np.std(runs_h2o)
    mean_co2, std_co2 = np.mean(runs_co2), np.std(runs_co2)
    mean_so2, std_so2 = np.mean(runs_so2), np.std(runs_so2)
    mean_temp, std_temp = np.mean(runs_temp), np.std(runs_temp)
    
    success_rate = (runs_passed / REPEAT_COUNT) * 100
    
    print(f"--> Başarı Oranı : %{success_rate:.1f} ({runs_passed}/{REPEAT_COUNT} Test Başarılı)")
    print(f"--> log(H2O)     : {mean_h2o:.2f} ± {std_h2o:.2f}")
    print(f"--> log(CO2)     : {mean_co2:.2f} ± {std_co2:.2f}")
    print(f"--> log(SO2)     : {mean_so2:.2f} ± {std_so2:.2f}")
    print(f"--> T_eff        : {mean_temp:.1f} ± {std_temp:.1f} K")
    
    if success_rate < 85.0:
        final_success = False
        print(f"    [!] Güvenilirlik eşiği (%85) aşılamadı!")

print("\n=========================================================================")
print("                   GENEL DOĞRULAMA DEĞERLENDİRMESİ                       ")
print("=========================================================================")

if final_success:
    print("[MÜKEMMEL BİLİMSEL ONAY]:")
    print("Model; 3 farklı gerçek JWST gezegeninin tamamında Monte Carlo")
    print("gürültü pertürbasyonlarına rağmen en az %85 kararlılıkla")
    print("hakemli literatür konsensus değerlerini yakalamıştır.")
else:
    print("[BAŞARISIZ]: Bazı hedefler istatistiksel sınırların altında kaldı.")
