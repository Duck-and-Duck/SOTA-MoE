import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np
from hq_data_utils import generate_mandel_agol_transit, generate_realistic_binary, generate_harvey_red_noise

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print(f"--> ANTI-HALLUCINATION ASTRONET EGITIMI | BIRIM: {device.upper()}")
print("=========================================================================")

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

def interp1d_gpu(x_query, x_ref, y_ref):
    indices = torch.searchsorted(x_ref.contiguous(), x_query.contiguous())
    indices = torch.clamp(indices, 1, len(x_ref) - 1)
    x0, x1 = x_ref[indices - 1], x_ref[indices]
    y0, y1 = y_ref[indices - 1], y_ref[indices]
    weight = (x_query - x0) / (x1 - x0 + 1e-8)
    return y0 + weight * (y1 - y0)

N = 6000
print(f"1. {N} Adet Ornek (Gezegen vs Ikili vs TERS-TRANSIT/FLARE) Uretiliyor...")
np.random.seed(42)

g_tensors = torch.zeros(N, 1, 201, device=device)
l_tensors = torch.zeros(N, 1, 61, device=device)
labels = torch.zeros(N, 1, device=device)
time_base = np.linspace(0, 27.4, 3000)
t_gpu = torch.tensor(time_base, dtype=torch.float32, device=device)

for i in range(N):
    p = np.random.uniform(2.0, 7.5)
    dur = np.random.uniform(0.08, 0.16)
    depth = np.random.uniform(0.0006, 0.0030)
    noise = generate_harvey_red_noise(3000) + 0.0004 * np.sin(2 * np.pi * time_base / np.random.uniform(5, 12))
    
    if i < N // 2:
        # SINIF 1: MANDEL-AGOL GEZEGENLERI (Negatif U-Cukuru) -> 1.0
        impact = np.random.uniform(0.0, 0.75)
        flux = generate_mandel_agol_transit(time_base, p, 0.0, dur, depth, impact_b=impact) + noise
        labels[i] = 1.0
    else:
        # SINIF 0: IKILILER VE TERS-TRANSITLER -> 0.0
        sub_type = i % 4
        if sub_type in [0, 1]:
            # İkili Yıldız Tutulması
            flux = generate_realistic_binary(time_base, p, 0.0, dur, depth, is_contact=(sub_type == 1)) + noise
        elif sub_type == 2:
            # TERS-TRANSİT (Pozitif Tepe): Ağın halüsinasyon görmesini engeller!
            transit_clean = generate_mandel_agol_transit(time_base, p, 0.0, dur, depth, impact_b=0.3)
            flux = (2.0 - transit_clean) + noise
        else:
            # Asimetrik Yıldız Parlaması (Flare)
            flare = 0.003 * np.exp(-np.linspace(0, 5, 3000))
            flux = 1.0 + noise + flare
        labels[i] = 0.0
        
    f_gpu = torch.tensor(flux, dtype=torch.float32, device=device)
    phase_t = ((t_gpu + 0.5 * p) % p) - (0.5 * p)
    s_idx = torch.argsort(phase_t)
    s_ph = phase_t[s_idx]
    s_fl = f_gpu[s_idx]
    
    g_raw = interp1d_gpu(torch.linspace(-0.5 * p, 0.5 * p, 201, device=device), s_ph, s_fl)
    l_raw = interp1d_gpu(torch.linspace(-p * 0.05, p * 0.05, 61, device=device), s_ph, s_fl)
    
    g_tensors[i, 0] = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_tensors[i, 0] = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)

dataset = TensorDataset(g_tensors, l_tensors, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = AstroNetHQ().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18)
criterion = nn.BCEWithLogitsLoss()

print("2. Model Egitiliyor (Ters-Transit / Flare Bagisikligi Kazandiriliyor)...")
t0 = time.perf_counter()
model.train()
for epoch in range(18):
    for g_b, l_b, y_b in loader:
        optimizer.zero_grad()
        loss = criterion(model(g_b, l_b), y_b)
        loss.backward()
        optimizer.step()
    scheduler.step()

if device == "cuda":
    torch.cuda.synchronize()

print(f"--> Egitim Tamamlandi: {time.perf_counter() - t0:.2f} saniye.")
torch.save(model.state_dict(), "astronet_hq.pt")
print("--> 'astronet_hq.pt' Guncellendi.")
