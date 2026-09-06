import warnings
import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*85)
print("   DOYGUNLUKSUZ VE DENGELI MASTER AGITIMI (LABEL SMOOTHING + ANTI-SATURATION)")
print(f"--> DONANIM: {device.upper()} | REFERANS: Guo et al. (2017) Calibrated Networks")
print("="*85)

class AstroNetHQ(nn.Module):
    def __init__(self):
        super(AstroNetHQ, self).__init__()
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.20),
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
            nn.Dropout(0.20),
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
            nn.Dropout(0.35),
            nn.Linear(64, 1)
        )

    def forward(self, g, l):
        return self.fc(torch.cat([self.global_conv(g), self.local_conv(l)], dim=1))

def interp1d_gpu(x_query, x_ref, y_ref):
    indices = torch.searchsorted(x_ref.contiguous(), x_query.contiguous())
    indices = torch.clamp(indices, 1, len(x_ref) - 1)
    x0, x1 = x_ref[indices - 1], x_ref[indices]
    y0, y1 = y_ref[indices - 1], y_ref[indices]
    weight = (x_query - x0) / (x1 - x0 + 1e-8)
    return y0 + weight * (y1 - y0)

N = 5000
time_base = np.linspace(0, 27.4, 3000)
t_gpu = torch.tensor(time_base, dtype=torch.float32, device=device)

g_tensors = torch.zeros(N, 1, 201, device=device)
l_tensors = torch.zeros(N, 1, 61, device=device)
labels = torch.zeros(N, 1, device=device)

np.random.seed(42)
print(f"1. {N} Adet Dengeli (U-Transit, V-Ikili, Sessiz Yildiz, Flare) Sentezleniyor...")

for i in range(N):
    p = np.random.uniform(1.5, 8.5)
    dur = np.random.uniform(0.06, 0.16)
    depth = np.random.uniform(0.0008, 0.0040)
    noise = np.random.normal(0.0, 0.0004, 3000) + 0.0006 * np.sin(2 * np.pi * time_base / np.random.uniform(4, 12))

    phase = ((time_base + 0.5 * p) % p) - (0.5 * p)
    in_tr = np.abs(phase) < (dur / 2.0)
    x_val = phase[in_tr] / (dur / 2.0)

    if i < N // 2:
        # GEZEGEN (U-Profili, Mandel-Agol)
        flux = 1.0 + noise
        u_prof = depth * (1.0 - 0.25 * (x_val**2)) * np.sqrt(np.maximum(0.0, 1.0 - (x_val**2)))
        flux[in_tr] -= u_prof
        labels[i] = 0.92  # Label Smoothing (Asla 1.0 verme)
    else:
        # GEZEGEN DISI (Ikili Yildiz V-profili, Flare, veya Bos Yildiz)
        flux = 1.0 + noise
        sub = i % 3
        if sub == 0:
            # Keskin V-Profili İkili
            v_prof = depth * np.maximum(0.0, 1.0 - np.abs(x_val))
            flux[in_tr] -= v_prof
        elif sub == 1:
            # Pozitif Parlama (Flare)
            flux[100:130] += depth * 1.5 * np.exp(-np.linspace(0, 3, 30))
        else:
            # Saf Dalgalanma (Sessiz yıldız modülasyonu)
            pass
        labels[i] = 0.08  # Label Smoothing (Asla 0.0 verme)

    f_gpu = torch.tensor(flux, dtype=torch.float32, device=device)
    phase_t = ((t_gpu + 0.5 * p) % p) - (0.5 * p)
    s_idx = torch.argsort(phase_t)
    s_ph, s_fl = phase_t[s_idx], f_gpu[s_idx]

    g_raw = interp1d_gpu(torch.linspace(-0.5 * p, 0.5 * p, 201, device=device), s_ph, s_fl)
    half_dur = max(dur * 2.5, p * 0.035)
    l_raw = interp1d_gpu(torch.linspace(-half_dur, half_dur, 61, device=device), s_ph, s_fl)

    g_tensors[i, 0] = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_tensors[i, 0] = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)

dataset = TensorDataset(g_tensors, l_tensors, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = AstroNetHQ().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
criterion = nn.BCEWithLogitsLoss()

print("2. Model Eğitiliyor (Doygunluk Önleyici Düzenleme ile)...")
t0 = time.perf_counter()
model.train()
for epoch in range(16):
    for gb, lb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(gb, lb), yb)
        loss.backward()
        optimizer.step()

if device == "cuda":
    torch.cuda.synchronize()

print(f"--> Egitim Tamamlandi: {time.perf_counter() - t0:.2f} saniye.")
torch.save(model.state_dict(), "astronet_hq.pt")
print("--> 'astronet_hq.pt' kalibre edilmis agirliklarla guncellendi.\n")