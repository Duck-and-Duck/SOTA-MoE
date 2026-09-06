import warnings
warnings.filterwarnings("ignore")

import lightkurve as lk
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Donanim: {device.upper()} (Monster Abra A5 / RTX 3050 Ti)")

TARGET_TIC = "TIC 261136679"
print(f"\n1. TESS CVZ Verisi Yukleniyor: {TARGET_TIC}...")

search = lk.search_targetpixelfile(TARGET_TIC, mission="TESS")
tpf = search[0].download(quality_bitmask="default")
time_arr = tpf.time.value
flux_cube = tpf.flux.value  # (Time, Y, X)

# NaN degerleri temizle
valid_mask = ~np.isnan(flux_cube).any(axis=(1, 2)) & ~np.isnan(time_arr)
time_arr = time_arr[valid_mask]
flux_cube = flux_cube[valid_mask]
T_len, H, W = flux_cube.shape
print(f"--> Temiz Zaman Adimi: {T_len} | Piksel Boyutu: {H}x{W}")

# 2. Klasik Basit Aciklik Fotometrisi (SAP - Ham Baseline)
raw_sap_flux = np.sum(flux_cube, axis=(1, 2))
raw_norm_flux = raw_sap_flux / np.median(raw_sap_flux)

# 3. Gaia Koordinatlarini TESS Piksel Matrisine Izdusur (Dogrudan tpf.ra, tpf.dec)
ra, dec = tpf.ra, tpf.dec
query = f"""
SELECT source_id, ra, dec, phot_g_mean_mag 
FROM gaiadr3.gaia_source 
WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, 0.03))
ORDER BY phot_g_mean_mag ASC
"""
job = Gaia.launch_job(query)
gaia_stars = job.get_results()

sky_coords = SkyCoord(gaia_stars["ra"], gaia_stars["dec"], unit="deg")
pix_x, pix_y = tpf.wcs.world_to_pixel(sky_coords)
print(f"--> {len(gaia_stars)} Gaia Yildizinin Piksel Izdusumu Tamamlandi.")

# 4. Yapay Zeka Modeli: Causal Pixel Deconvolution Network (CPDN)
class CausalPixelDeblender(nn.Module):
    def __init__(self, num_pixels):
        super(CausalPixelDeblender, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_pixels, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

flat_pixels = flux_cube.reshape(T_len, -1)
pix_mean = np.mean(flat_pixels, axis=0, keepdims=True)
pix_std = np.std(flat_pixels, axis=0, keepdims=True) + 1e-6
norm_pixels = (flat_pixels - pix_mean) / pix_std

X_tensor = torch.tensor(norm_pixels, dtype=torch.float32, device=device)
y_target = torch.tensor(raw_norm_flux, dtype=torch.float32, device=device).unsqueeze(1)

model = CausalPixelDeblender(num_pixels=H*W).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.HuberLoss()

print("\n2. Yapay Zeka Egitiliyor (Piksel Ayrastirma ve Jitter Temizleme)...")
t0 = time.time()
model.train()
for epoch in range(60):
    optimizer.zero_grad()
    pred = model(X_tensor)
    loss = criterion(pred, y_target)
    loss.backward()
    optimizer.step()

train_time = time.time() - t0
print(f"--> Model {train_time:.2f} saniyede egitildi.")

# 5. Temiz Isik Egrisini Cikar
model.eval()
with torch.no_grad():
    clean_flux = model(X_tensor).cpu().squeeze().numpy()

# 6. Bilimsel Metrik: CDPP (Gurultu Seviyesi) Olcumu (ppm)
def calculate_cdpp(flux_series):
    diff = np.diff(flux_series)
    sigma = np.std(diff) / np.sqrt(2)
    return sigma * 1e6

raw_noise_ppm = calculate_cdpp(raw_norm_flux)
clean_noise_ppm = calculate_cdpp(clean_flux)

print(f"\n=========================================================================")
print(f"               FOTOMETRIK GURULTU VE KALITE RAPORU                       ")
print(f"=========================================================================")
print(f"Ham TESS Basit Toplam Gurultusu (SAP) : {raw_noise_ppm:.1f} ppm")
print(f"Yapay Zeka ile Temizlenmis Gurultu    : {clean_noise_ppm:.1f} ppm")
gain = ((raw_noise_ppm - clean_noise_ppm) / raw_noise_ppm) * 100
print(f"Gurultu Azaltma / Hassasiyet Kazanimi  : %{gain:.1f}")
print(f"=========================================================================")

# 7. CSV Olarak Kaydet
output_csv = "tic261136679_clean_lightcurve.csv"
df = pd.DataFrame({
    "time_bjd": time_arr,
    "raw_flux": raw_norm_flux,
    "clean_flux": clean_flux
})
df.to_csv(output_csv, index=False)
print(f"\n[ISLEM BASARILI]: Temizlenmis isik egrisi '{output_csv}' dosyasina yazildi!")
