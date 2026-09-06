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
print("   TRAINING ADV-Net V2: MORPHOLOGY-PRESERVING SOTA AUTONOMOUS MODEL (30.000 SAMPLES)")
print(f"--> COMPUTATION: {device.upper()} (RTX) | RESOLUTION PRESERVED (NO ADAPTIVE POOL CRUSH)")
print("="*85)

# SHALLUE & VANDERBURG (2018) STANDARDI YÜKSEK ÇÖZÜNÜRLÜKLÜ MİMARİ
class HighResAutonomousAstroNet(nn.Module):
    def __init__(self):
        super(HighResAutonomousAstroNet, self).__init__()
        # Global Kol: 201 nokta -> MaxPool(2) -> 100 -> MaxPool(2) -> 50
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Flatten() # 32 * 50 = 1600 Boyut (Sekonder tutulmaları ASLA silmez)
        )
        # Local Kol: 61 nokta -> MaxPool(2) -> 30 -> MaxPool(2) -> 15
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
            nn.Flatten() # 32 * 15 = 480 Boyut (V-apeksini ASLA yuvarlatmaz)
        )
        # 3 Sınıflı Çıkarım: 0: NON_PLANET, 1: PLANET, 2: BINARY
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

# 30.000 HEDEFLİK DENGELİ EĞİTİM SETİ
N = 30000
time_base = np.linspace(0, 27.4, 3000)
t_gpu = torch.tensor(time_base, dtype=torch.float32, device=device)

g_tensors = torch.zeros(N, 1, 201, device=device)
l_tensors = torch.zeros(N, 1, 61, device=device)
labels = torch.zeros(N, dtype=torch.long, device=device)

np.random.seed(42)
print(f"--> 1. {N} Adet Dengeli ve Çözünürlüğü Korunmuş Veri Sentezleniyor...")

for i in range(N):
    target_class = i % 3  # 0: NON_PLANET, 1: PLANET, 2: BINARY
    p = np.random.uniform(1.2, 12.0)
    dur = np.random.uniform(0.06, 0.17)
    t0_sim = np.random.uniform(0.1, 1.5)

    noise = generate_harvey_red_noise(3000) + 0.0003 * np.sin(2 * np.pi * time_base / np.random.uniform(4, 14))

    if target_class == 1:
        # SINIF 1: GEZEGEN (U-şekli, düz tabanlı, log-uniform derinlik)
        depth = 10 ** np.random.uniform(np.log10(0.00025), np.log10(0.0120))
        impact = np.random.uniform(0.0, 0.80)
        flux = generate_mandel_agol_transit(time_base, p, t0_sim, dur, depth, impact_b=impact) + noise
        labels[i] = 1

    elif target_class == 2:
        # SINIF 2: İKİLİ YILDIZ (Keskin V-apeksi veya Faz 0.5 sekonderi)
        depth = 10 ** np.random.uniform(np.log10(0.0015), np.log10(0.0350))
        flux = generate_realistic_binary(time_base, p, t0_sim, dur, depth, is_contact=(i % 5 == 0)) + noise
        labels[i] = 2

    else:
        # SINIF 0: NON_PLANET (DÜZ ÇİZGİ KORUMASI EKLENDİ!)
        sub = i % 4
        if sub in [0, 1]:
            # %50 SAF DÜZ YILDIZ GÜRÜLTÜSÜ (Transit YOK - Sessiz Yıldız Koruması)
            flux = 1.0 + noise
        elif sub == 2:
            # Ters Transit (Anti-Hallucination)
            depth = 10 ** np.random.uniform(np.log10(0.0005), np.log10(0.0080))
            tr = generate_mandel_agol_transit(time_base, p, t0_sim, dur, depth, impact_b=0.3)
            flux = (2.0 - tr) + noise
        else:
            # Flare veya Değişken Yıldız
            flare = np.random.uniform(0.005, 0.030) * np.exp(-np.linspace(0, 3, 3000))
            flux = 1.0 + noise + flare
        labels[i] = 0

    f_gpu = torch.tensor(flux, dtype=torch.float32, device=device)
    g_n, l_n = phase_fold_aligned(t_gpu, f_gpu, p, t0_sim, dur)
    g_tensors[i, 0] = g_n.squeeze()
    l_tensors[i, 0] = l_n.squeeze()

dataset = TensorDataset(g_tensors, l_tensors, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = HighResAutonomousAstroNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=24)
criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

print("--> 2. ADV-Net V2 Eğitiliyor (24 Epoch | Keskin Morfoloji Kilitleniyor)...")
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
print("--> 'astronet_autonomous.pt' Yüksek Çözünürlüklü Ağırlıklarla Güncellendi.\n")