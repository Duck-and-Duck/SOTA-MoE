import warnings
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from hq_data_utils import generate_mandel_agol_transit, generate_realistic_binary, generate_harvey_red_noise

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*85)
print("   TRAINING ADV-Net V4: TRUE END-TO-END AUTONOMOUS CONVOLUTIONAL MODEL")
print(f"--> COMPUTATION: {device.upper()} (RTX) | STANDART: NASA Kepler/TESS SPOC SOTA")
print("="*85)

# NASA EXOMINER / GOOGLE ASTRONET SAF EVRİŞİMSEL DERİN MİMARİSİ
class TrueAutonomousAstroNet(nn.Module):
    def __init__(self):
        super(TrueAutonomousAstroNet, self).__init__()
        # 1. Global Kol: 201 nokta (Tüm yörüngede sekonder tutulmayı ve eksantrik tutulmaları arar)
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Flatten() # 32 * 50 = 1600 Boyut
        )
        # 2. Local Kol: 61 nokta (U-tabanı vs V-apeksinin tüm morfolojisini öğrenir)
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
            nn.MaxPool1d(2),
            nn.Flatten() # 32 * 15 = 480 Boyut
        )
        # 3 Sınıflı Doğrudan Çıkarım: 0: NON_PLANET, 1: PLANET, 2: BINARY
        self.classifier = nn.Sequential(
            nn.Linear(1600 + 480, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 3)
        )

    def forward(self, g, l):
        g_feat = self.global_conv(g)
        l_feat = self.local_conv(l)
        return self.classifier(torch.cat([g_feat, l_feat], dim=1))

# EKSANTRİK VE TEMASLI İKİLİ SİMÜLATÖRÜ (Twicken et al. 2018 / Prsa et al. 2022)
def generate_advanced_binary(time_arr, period, t0, duration, depth, ecc=0.0, is_contact=False):
    phase = ((time_arr - t0 + 0.5 * period) % period) - (0.5 * period)
    flux = np.ones_like(time_arr)

    # Primer V-Tutulma (Faz 0.0)
    x_prim = phase / (duration / 2.0)
    prim_m = np.abs(phase) < (duration / 2.0)
    flux[prim_m] -= depth * np.maximum(0.0, 1.0 - np.abs(x_prim[prim_m]))

    # Eksantrik Sekonder Tutulma (Faz kayması: ecc * 0.35 civarı)
    phase_sec_offset = 0.5 + (ecc * 0.35 if ecc > 0 else 0.0)
    sec_phase = ((time_arr - t0 - phase_sec_offset * period + 0.5 * period) % period) - (0.5 * period)
    x_sec = sec_phase / (duration / 2.0)
    sec_m = np.abs(sec_phase) < (duration / 2.0)
    sec_depth = depth * np.random.uniform(0.30, 0.85)
    flux[sec_m] -= sec_depth * np.maximum(0.0, 1.0 - np.abs(x_sec[sec_m]))

    # Temaslı ikili gravitasyonel elipsoidal dalgası
    if is_contact:
        flux += (depth * 0.25) * np.cos(4 * np.pi * phase / period)
    return flux

def gpu_box_bin(query_bins, sorted_phase, sorted_flux):
    half_w = 0.5 * (query_bins[1] - query_bins[0])
    l_edge = query_bins - half_w
    r_edge = query_bins + half_w
    idx_l = torch.searchsorted(sorted_phase, l_edge)
    idx_r = torch.searchsorted(sorted_phase, r_edge)
    f_cumsum = F.pad(torch.cumsum(sorted_flux, dim=0), (1, 0))
    bin_sums = f_cumsum[idx_r] - f_cumsum[idx_l]
    bin_counts = idx_r - idx_l
    mask = bin_counts > 0
    res = torch.zeros_like(query_bins)
    res[mask] = bin_sums[mask] / bin_counts[mask].float()
    if not torch.all(mask):
        empty_idx = torch.clamp(torch.searchsorted(sorted_phase, query_bins[~mask]), 0, len(sorted_flux) - 1)
        res[~mask] = sorted_flux[empty_idx]
    return res

def phase_fold_aligned(t_t, f_t, p, t0, dur):
    phase = ((t_t - t0 + 0.5 * p) % p) - (0.5 * p)
    s_idx = torch.argsort(phase)
    s_ph, s_fl = phase[s_idx], f_t[s_idx]

    g_bins = torch.linspace(-0.5 * p, 0.5 * p, 201, device=device)
    g_raw = gpu_box_bin(g_bins, s_ph, s_fl)

    win_local = max(dur * 2.0, p * 0.04)
    l_bins = torch.linspace(-win_local, win_local, 61, device=device)
    l_raw = gpu_box_bin(l_bins, s_ph, s_fl)

    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)
    return g_norm, l_norm

# 30.000 HEDEFLİK GERÇEKÇİ VE HASMANE VERİ HAVUZU
N = 30000
time_base = np.linspace(0, 27.4, 3000)
t_gpu = torch.tensor(time_base, dtype=torch.float32, device=device)

g_tensors = torch.zeros(N, 1, 201, device=device)
l_tensors = torch.zeros(N, 1, 61, device=device)
labels = torch.zeros(N, dtype=torch.long, device=device)

np.random.seed(42)
print(f"--> 1. {N} Adet Eksantrik, Seyreltilmis ve Dengesiz Veri Sentezleniyor...")

for i in range(N):
    target_class = i % 3 # 0: NON_PLANET, 1: PLANET, 2: BINARY
    p = np.random.uniform(1.2, 14.0)
    dur = np.random.uniform(0.06, 0.18)
    t0_sim = np.random.uniform(0.1, 1.5)

    noise = generate_harvey_red_noise(3000) + 0.0004 * np.sin(2 * np.pi * time_base / np.random.uniform(3, 14))

    if target_class == 1:
        # GEZEGEN (U-şekli, log-uniform derinlik 200 - 15.000 ppm)
        depth = 10 ** np.random.uniform(np.log10(0.00020), np.log10(0.0150))
        impact = np.random.uniform(0.0, 0.85)
        flux = generate_mandel_agol_transit(time_base, p, t0_sim, dur, depth, impact_b=impact) + noise
        labels[i] = 1

    elif target_class == 2:
        # İKİLİ YILDIZ (Eksantrik sekonderler, temaslı ikililer ve teğet V-şekilleri)
        depth = 10 ** np.random.uniform(np.log10(0.0010), np.log10(0.0350))
        eccentricity = np.random.uniform(0.0, 0.5) if (i % 3 == 0) else 0.0
        contact_flag = (i % 5 == 0)
        flux = generate_advanced_binary(time_base, p, t0_sim, dur, depth, ecc=eccentricity, is_contact=contact_flag) + noise
        labels[i] = 2

    else:
        # NON_PLANET (Düz yıldız gürültüsü, ters transitler ve süper-flare'ler)
        sub = i % 4
        if sub in [0, 1]:
            flux = 1.0 + noise # %50 SAF DÜZ GÜRÜLTÜ
        elif sub == 2:
            depth = 10 ** np.random.uniform(np.log10(0.0005), np.log10(0.0080))
            tr = generate_mandel_agol_transit(time_base, p, t0_sim, dur, depth, impact_b=0.3)
            flux = (2.0 - tr) + noise # Ters Transit
        else:
            flare = np.random.uniform(0.005, 0.035) * np.exp(-np.linspace(0, 3, 3000))
            flux = 1.0 + noise + flare # Flare
        labels[i] = 0

    f_gpu = torch.tensor(flux, dtype=torch.float32, device=device)
    g_n, l_n = phase_fold_aligned(t_gpu, f_gpu, p, t0_sim, dur)
    g_tensors[i, 0] = g_n
    l_tensors[i, 0] = l_n

dataset = TensorDataset(g_tensors, l_tensors, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = TrueAutonomousAstroNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=24)
criterion = nn.CrossEntropyLoss()

print("--> 2. True ADV-Net V4 Eğitiliyor (24 Epoch | Saf Konvolüsyon)...")
t0 = time.perf_counter()
for epoch in range(24):
    model.train()
    total_loss = 0.0
    for gb, lb, yb in loader:
        optimizer.zero_grad()
        out = model(gb, lb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 6 == 0 or epoch == 23:
        print(f"    * Epoch [{epoch+1:02d}/24] -> Loss: {total_loss/len(loader):.4f}")

if device == "cuda": torch.cuda.synchronize()
print(f"--> Egitim Tamamlandi: {time.perf_counter() - t0:.2f} sn.")
torch.save(model.state_dict(), "astronet_autonomous.pt")
print("--> 'astronet_autonomous.pt' Saf Evrişim Ağı Kaydedildi.\n")