import warnings
import logging
import time
import torch

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from sbi.inference import SNPE
from sbi.utils import BoxUniform
from simulator_300ch import generate_300ch_spectrum_batch, WAVELENGTHS

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print(f"--> MODEL: SNPE-C / MAF (High-Pass Invariant Architecture)")
print(f"--> DONANIM: {device.upper()} (RTX 3050 Ti)")
print(f"--> GIRIS: 300 Detrended Kanal + 5 Fiziksel Baglam = 305 Boyut")
print("=========================================================================")

prior = BoxUniform(
    low=torch.tensor([-5.5, -7.5, -8.0, -7.5, -7.5, 650.0, -4.0], device=device),
    high=torch.tensor([-1.5, -1.5, -2.5, -1.5, -1.5, 1750.0, 1.0], device=device)
)

SIMULATION_COUNT = 60000
print(f"\n1. {SIMULATION_COUNT} Adet Detrended + Kontrastli Spektrum Uretiliyor...")
t0 = time.time()
theta_train, x_train = generate_300ch_spectrum_batch(SIMULATION_COUNT, device=device)
print(f"--> Uretim Tamamlandi: {time.time() - t0:.2f} saniye. X Sekli: {x_train.shape}")

print("\n2. MAF Agi Egitiliyor (Leke ve Bulut Invariant Egitim)...")
t_train = time.time()

inferer = SNPE(prior=prior, density_estimator="maf", device=device)
density_estimator = inferer.append_simulations(theta_train, x_train).train(
    training_batch_size=256,
    learning_rate=4.5e-4,
    max_num_epochs=60,
    stop_after_epochs=9
)
posterior = inferer.build_posterior(density_estimator)
print(f"--> Egitim Tamamlandi: {time.time() - t_train:.2f} saniye.")

torch.save(posterior, "posterior_300ch.pt")
print("--> Agirliklar 'posterior_300ch.pt' dosyasina basariyla kaydedildi.")
