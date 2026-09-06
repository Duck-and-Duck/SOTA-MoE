import warnings
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*85)
print("   ACIMASIZ ASTROFIZIKSEL EGITIM MOTORU (RUTHLESS STELLAR NOISE SYNTHESIZER)")
print(f"--> DONANIM: {device.upper()} | PARAZITLER: Harvey Red Noise + Spots + Flares + Grazing")
print("="*85)

# =========================================================================
# 1. ACIMASIZ FİZİKSEL PARAZİT JENERATÖRLERİ
# =========================================================================
def generate_harvey_red_noise(N_pts=3000, dt=0.00913):
    phi = np.exp(-dt / 0.45) # 11 saatlik otokorelasyon
    white = np.random.normal(0.0, 0.00035, N_pts)
    red = np.zeros(N_pts)
    for t in range(1, N_pts):
        red[t] = phi * red[t-1] + np.sqrt(1 - phi**2) * white[t]
    return red

def generate_spot_modulation(time_base):
    # McQuillan et al. (2014): 2 ila 15 günlük yıldız lekesi rotasyon dalgası
    p_rot = np.random.uniform(2.5, 14.0)
    amp = np.random.uniform(0.0005, 0.0035) # 500 - 3500 ppm dalgalanma
    phase_offset = np.random.uniform(0, 2*np.pi)
    return amp * np.sin(2 * np.pi * time_base / p_rot + phase_offset)

def generate_random_flares(time_base):
    # Davenport (2016): Ani parlama ve üstel sönümlenme
    flare_curve = np.zeros_like(time_base)
    num_flares = np.random.randint(0, 3)
    for _ in range(num_flares):
        t_flare = np.random.uniform(time_base[0], time_base[-1])
        amp = np.random.uniform(0.001, 0.006)
        dur = np.random.uniform(0.08, 0.25)
        mask = time_base >= t_flare
        flare_curve[mask] += amp * np.exp(-(time_base[mask] - t_flare) / dur)
    return flare_curve

def generate_mandel_agol(time_base, p, dur, depth, impact_b=0.3):
    phase = ((time_base + 0.5 * p) % p) - (0.5 * p)
    in_tr = np.abs(phase) < (dur / 2.0)
    tr_flux = np.ones_like(time_base)
    if not np.any(in_tr):
        return tr_flux
    x = phase[in_tr] / (dur / 2.0)
    z = np.sqrt(x**2 + impact_b**2)
    mu = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, z**2)))
    limb_dark = 1.0 - 0.32 * (1.0 - mu) - 0.28 * ((1.0 - mu)**2)
    profile = depth * (limb_dark / 1.0) * np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, x**2)))
    tr_flux[in_tr] -= profile
    return tr_flux

def generate_eclipsing_binary(time_base, p, dur, depth):
    phase = ((time_base + 0.5 * p) % p) - (0.5 * p)
    eb_flux = np.ones_like(time_base)
    # Primer (V-apeks)
    x_prim = phase / (dur / 2.0)
    prim_m = np.abs(phase) < (dur / 2.0)
    eb_flux[prim_m] -= depth * np.maximum(0.0, 1.0 - np.abs(x_prim[prim_m]))
    # Sekonder Tutulma (Faz 0.5)
    sec_ph = ((time_base) % p) - (0.5 * p)
    x_sec = sec_ph / (dur / 2.0)
    sec_m = np.abs(sec_ph) < (dur / 2.0)
    sec_depth = depth * np.random.uniform(0.35, 0.85)
    eb_flux[sec_m] -= sec_depth * np.maximum(0.0, 1.0 - np.abs(x_sec[sec_m]))
    return eb_flux

# =========================================================================
# 2. VERİ HAVUZU (6000 ACIMASIZ ÖRNEK)
# =========================================================================
N = 6000
time_base = np.linspace(0, 27.4, 3000)
t_gpu = torch.tensor(time_base, dtype=torch.float32, device=device)

def interp1d_gpu(x_query, x_ref, y_ref):
    indices = torch.searchsorted(x_ref.contiguous(), x_query.contiguous())
    indices = torch.clamp(indices, 1, len(x_ref) - 1)
    x0, x1 = x_ref[indices - 1], x_ref[indices]
    y0, y1 = y_ref[indices - 1], y_ref[indices]
    weight = (x_query - x0) / (x1 - x0 + 1e-8)
    return y0 + weight * (y1 - y0)

g_tensors = torch.zeros(N, 1, 201, device=device)
l_tensors = torch.zeros(N, 1, 61, device=device)
labels = torch.zeros(N, 1, device=device)

print(f"1. {N} Adet Leke + Flare + Harvey Gürültülü Örnek Sentezleniyor...")
np.random.seed(42)

for i in range(N):
    p = np.random.uniform(1.8, 8.5)
    dur = np.random.uniform(0.07, 0.17)
    # EŞİT DERİNLİK (Kopya çekmek imkansız: 800 - 3500 ppm)
    depth = np.random.uniform(0.0008, 0.0035)

    # Parazitlerin tamamını üst üste bindir
    stellar_noise = generate_harvey_red_noise(3000) + generate_spot_modulation(time_base) + generate_random_flares(time_base)

    if i < N // 2:
        # SINIF 1: Mandel-Agol U-Gezegen
        impact = np.random.uniform(0.0, 0.70)
        flux = generate_mandel_agol(time_base, p, dur, depth, impact_b=impact) + stellar_noise
        labels[i] = 1.0
    else:
        # SINIF 0: İkili Yıldız, Ters Transit, veya Saf Leke/Flare
        sub = i % 3
        if sub == 0:
            flux = generate_eclipsing_binary(time_base, p, dur, depth) + stellar_noise
        elif sub == 1:
            # TERS-TRANSIT (Anti-hallucination güvencesi)
            clean_tr = generate_mandel_agol(time_base, p, dur, depth, impact_b=0.3)
            flux = (2.0 - clean_tr) + stellar_noise
        else:
            # Saf Gürültü ve Yıldız Lekesi
            flux = 1.0 + stellar_noise
        labels[i] = 0.0

    f_gpu = torch.tensor(flux, dtype=torch.float32, device=device)
    phase_t = ((t_gpu + 0.5 * p) % p) - (0.5 * p)
    s_idx = torch.argsort(phase_t)
    s_ph = phase_t[s_idx]
    s_fl = f_gpu[s_idx]

    g_raw = interp1d_gpu(torch.linspace(-0.5 * p, 0.5 * p, 201, device=device), s_ph, s_fl)
    half_dur = max(dur * 2.5, p * 0.035)
    l_raw = interp1d_gpu(torch.linspace(-half_dur, half_dur, 61, device=device), s_ph, s_fl)

    # Z-Score Standardizasyonu
    g_tensors[i, 0] = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_tensors[i, 0] = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)

# =========================================================================
# 3. MODEL EĞİTİMİ (AstroNetHQ)
# =========================================================================
class AstroNetHQ(nn.Module):
    def __init__(self):
        super(AstroNetHQ, self).__init__()
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(15),
            nn.Flatten()
        )
        self.local_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(15),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 15 + 32 * 15, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )

    def forward(self, g, l):
        return self.fc(torch.cat([self.global_conv(g), self.local_conv(l)], dim=1))

dataset = TensorDataset(g_tensors, l_tensors, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = AstroNetHQ().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=22)
criterion = nn.BCEWithLogitsLoss()

print("2. Model Tavizsiz Olarak Eğitiliyor (22 Epoch)...")
t0 = time.perf_counter()
model.train()
for epoch in range(22):
    total_loss = 0.0
    for gb, lb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(gb, lb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 5 == 0 or epoch == 21:
        print(f"    * Epoch [{epoch+1:02d}/22] -> Kayıp (Loss): {total_loss/len(loader):.4f}")

if device == "cuda":
    torch.cuda.synchronize()

print(f"--> Eğitim Tamamlandı: {time.perf_counter() - t0:.2f} saniye.")
torch.save(model.state_dict(), "astronet_hq.pt")
print("--> 'astronet_hq.pt' acımasız parametrelerle güncellendi ve mühürlendi.\n")