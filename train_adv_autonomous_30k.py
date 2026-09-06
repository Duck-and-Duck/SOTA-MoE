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
print("   TRAINING ADV-Net: AUTONOMOUS 3-CLASS DEEP VETTER (30.000 SAMPLES)")
print(f"--> COMPUTATION: {device.upper()} (RTX) | ZERO RULE-BASED CRUTCHES")
print("="*85)

class AutonomousAstroNet(nn.Module):
    def __init__(self):
        super(AutonomousAstroNet, self).__init__()
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=7, padding=3),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten()
        )
        self.local_conv = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm1d(24),
            nn.LeakyReLU(0.1),
            nn.Conv1d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten()
        )
        # 3 Sınıflı Doğrudan Çıkarım: 0: NON_PLANET, 1: PLANET, 2: BINARY
        self.classifier = nn.Sequential(
            nn.Linear(48 * 16 + 48 * 16, 96),
            nn.BatchNorm1d(96),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.30),
            nn.Linear(96, 32),
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

# 30.000 HEDEFLİK DEVASA ADVERSARIAL EĞİTİM SETİ
N = 30000
time_base = np.linspace(0, 27.4, 3000)
t_gpu = torch.tensor(time_base, dtype=torch.float32, device=device)

g_tensors = torch.zeros(N, 1, 201, device=device)
l_tensors = torch.zeros(N, 1, 61, device=device)
labels = torch.zeros(N, dtype=torch.long, device=device)

np.random.seed(42)
print(f"--> 1. {N} Adet Ekstrem Kombinasyonlu Veri Sentezleniyor...")

for i in range(N):
    target_class = i % 3  # 0: NON_PLANET, 1: PLANET, 2: BINARY
    p = np.random.uniform(1.0, 14.0)
    dur = np.random.uniform(0.05, 0.18)
    t0_sim = np.random.uniform(0.1, 1.5)

    noise = generate_harvey_red_noise(3000) + 0.0003 * np.sin(2 * np.pi * time_base / np.random.uniform(3, 14))

    if target_class == 1:
        # SINIF 1: GEZEGEN (U-Profili, log-uniform sığ ve derin geçişler)
        depth = 10 ** np.random.uniform(np.log10(0.00025), np.log10(0.0120))
        impact = np.random.uniform(0.0, 0.82)
        if np.random.rand() > 0.7: noise += 0.004 * np.exp(-np.linspace(0, 4, 3000)) # Flare altında transit
        flux = generate_mandel_agol_transit(time_base, p, t0_sim, dur, depth, impact_b=impact) + noise
        labels[i] = 1

    elif target_class == 2:
        # SINIF 2: İKİLİ YILDIZ (V-Apeks, sekonder tutulmalı, temaslı ve teğet)
        depth = 10 ** np.random.uniform(np.log10(0.0010), np.log10(0.0350))
        is_contact = (i % 6 == 0)
        flux = generate_realistic_binary(time_base, p, t0_sim, dur, depth, is_contact=is_contact) + noise
        labels[i] = 2

    else:
        # SINIF 0: NON_PLANET (Sessiz gürültü, ters-transit, dev flare, leke)
        sub = i % 3
        depth = 10 ** np.random.uniform(np.log10(0.0005), np.log10(0.0080))
        if sub == 0:
            # Ters transit (Anti-hallucination)
            tr = generate_mandel_agol_transit(time_base, p, t0_sim, dur, depth, impact_b=0.3)
            flux = (2.0 - tr) + noise
        elif sub == 1:
            # Dev süper-flare (AU Mic benzeri patlamalar)
            flare = np.random.uniform(0.008, 0.040) * np.exp(-np.linspace(0, 3, 3000))
            flux = 1.0 + noise + flare
        else:
            # Saf yıldız lekesi ve aletsel eğim
            spot = np.random.uniform(0.001, 0.005) * np.sin(2 * np.pi * time_base / np.random.uniform(3, 8))
            flux = 1.0 + noise + spot
        labels[i] = 0

    f_gpu = torch.tensor(flux, dtype=torch.float32, device=device)
    g_n, l_n = phase_fold_aligned(t_gpu, f_gpu, p, t0_sim, dur)
    g_tensors[i, 0] = g_n.squeeze()
    l_tensors[i, 0] = l_n.squeeze()

dataset = TensorDataset(g_tensors, l_tensors, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = AutonomousAstroNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=22)
criterion = nn.CrossEntropyLoss(label_smoothing=0.04)

print("--> 2. ADV-Net Eğitiliyor (22 Epoch | 3 Sınıflı Softmax)...")
t0 = time.perf_counter()
for epoch in range(22):
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
    if (epoch + 1) % 5 == 0 or epoch == 21:
        print(f"    * Epoch [{epoch+1:02d}/22] -> Loss: {total_loss/len(loader):.4f}")

if device == "cuda": torch.cuda.synchronize()
print(f"--> Egitim Tamamlandi: {time.perf_counter() - t0:.2f} sn.")
torch.save(model.state_dict(), "astronet_autonomous.pt")
print("--> 'astronet_autonomous.pt' Otonom Modeli Mühürlendi.\n")