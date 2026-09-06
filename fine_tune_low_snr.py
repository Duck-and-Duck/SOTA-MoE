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
print("   LOW-SNR FINE-TUNING PROTOCOL (SHALLOW TRANSIT RECOVERY SPECIALIST)")
print(f"--> COMPUTATION: {device.upper()} (RTX) | STANDART: Shallue & Vanderburg (2018)")
print("="*85)

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

model = AstroNetHQ().to(device)
model.load_state_dict(torch.load("astronet_hq.pt", map_location=device))

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

    # 3-Noktalı Morfolojik Düzleştirme (Audit ile birebir kilitli)
    l_smooth = F.avg_pool1d(l_raw.view(1, 1, -1), kernel_size=3, stride=1, padding=1).squeeze()

    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_smooth - torch.median(l_smooth)) / (torch.std(l_smooth) + 1e-7)
    return g_norm, l_norm

# 8.000 HEDEFLİK ÖZEL SIĞ TRANSİT ADAPTASYON HAVUZU
N_ft = 8000
time_base = np.linspace(0, 27.4, 3000)
t_gpu = torch.tensor(time_base, dtype=torch.float32, device=device)

g_tensors = torch.zeros(N_ft, 1, 201, device=device)
l_tensors = torch.zeros(N_ft, 1, 61, device=device)
labels = torch.zeros(N_ft, 1, device=device)

np.random.seed(777)
print("--> 1. Hedefe Yonelik Sig Transit Havuzu Olusturuluyor (200 - 1500 ppm)...")

for i in range(N_ft):
    p = np.random.uniform(1.2, 12.0)
    dur = np.random.uniform(0.06, 0.17)
    t0_sim = np.random.uniform(0.1, 1.5)

    # Düşük derinlik rejimi (200 - 2500 ppm)
    depth = 10 ** np.random.uniform(np.log10(0.00020), np.log10(0.00250))
    noise = generate_harvey_red_noise(3000) + 0.0003 * np.sin(2 * np.pi * time_base / np.random.uniform(4, 14))

    if i < N_ft // 2:
        impact = np.random.uniform(0.0, 0.75)
        flux = generate_mandel_agol_transit(time_base, p, t0_sim, dur, depth, impact_b=impact) + noise
        labels[i] = 1.0
    else:
        sub = i % 3
        if sub == 0:
            flux = generate_realistic_binary(time_base, p, t0_sim, dur, depth, is_contact=False) + noise
        elif sub == 1:
            tr = generate_mandel_agol_transit(time_base, p, t0_sim, dur, depth, impact_b=0.3)
            flux = (2.0 - tr) + noise
        else:
            flux = 1.0 + noise
        labels[i] = 0.0

    f_gpu = torch.tensor(flux, dtype=torch.float32, device=device)
    g_n, l_n = phase_fold_aligned(t_gpu, f_gpu, p, t0_sim, dur)
    g_tensors[i, 0] = g_n.squeeze()
    l_tensors[i, 0] = l_n.squeeze()

dataset = TensorDataset(g_tensors, l_tensors, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Düşük öğrenme oranlı adaptasyon (Omurgayı bozmadan sığ transit hassasiyeti kazandırır)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.8e-4, weight_decay=1e-3)
criterion = nn.BCEWithLogitsLoss()

print("--> 2. AstroNet-HQ Sig Transit Adaptasyonu (8 Epoch Fine-Tuning)...")
t0 = time.perf_counter()
model.train()
for epoch in range(8):
    total_loss = 0.0
    for gb, lb, yb in loader:
        optimizer.zero_grad()
        pred = model(gb, lb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"    * Adaptasyon Epoch [{epoch+1:02d}/08] -> Loss: {total_loss/len(loader):.4f}")

if device == "cuda":
    torch.cuda.synchronize()

print(f"--> Fine-Tuning Bitti: {time.perf_counter() - t0:.2f} sn.")
torch.save(model.state_dict(), "astronet_hq.pt")
print("--> 'astronet_hq.pt' SOTA agirliklariyla muhurlendi.\n")