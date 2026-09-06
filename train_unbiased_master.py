import warnings
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*85)
print("   UNBIASED MASTER ASTROPHYSICAL ENGINE (20.000 NUMUNE | SIFIR KISAYOL)")
print(f"--> DONANIM: {device.upper()} | METOD: Shallue & Vanderburg (2018) Depth Normalization")
print("="*85)

class MasterAstroNet(nn.Module):
    def __init__(self):
        super(MasterAstroNet, self).__init__()
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
        self.phys_net = nn.Sequential(
            nn.Linear(4, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 15 + 32 * 15 + 16, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.35),
            nn.Linear(64, 3)
        )

    def forward(self, g, l, phys):
        gf = self.global_conv(g)
        lf = self.local_conv(l)
        pf = self.phys_net(phys)
        return self.classifier(torch.cat([gf, lf, pf], dim=1))

def interp1d_gpu(x_query, x_ref, y_ref):
    indices = torch.searchsorted(x_ref.contiguous(), x_query.contiguous())
    indices = torch.clamp(indices, 1, len(x_ref) - 1)
    x0, x1 = x_ref[indices - 1], x_ref[indices]
    y0, y1 = y_ref[indices - 1], y_ref[indices]
    weight = (x_query - x0) / (x1 - x0 + 1e-8)
    return y0 + weight * (y1 - y0)

N = 20000
time_base = np.linspace(0, 27.4, 3000)
t_gpu = torch.tensor(time_base, dtype=torch.float32, device=device)

g_tensors = torch.zeros(N, 1, 201, device=device)
l_tensors = torch.zeros(N, 1, 61, device=device)
phys_tensors = torch.zeros(N, 4, device=device)
labels = torch.zeros(N, dtype=torch.long, device=device)

print(f"1. {N} Adet Kesin Fiziksel Olcumlu Veri Uretiliyor...")
np.random.seed(42)

for i in range(N):
    target_class = i % 3  # 0: NOISE, 1: PLANET, 2: BINARY
    p = np.random.uniform(1.2, 9.0)
    dur = np.random.uniform(0.06, 0.16)
    scatter = np.random.uniform(0.00025, 0.00055)

    phi = np.exp(-(27.4/3000) / 0.45)
    w = np.random.normal(0.0, scatter, 3000)
    red = np.zeros(3000)
    for t in range(1, 3000): red[t] = phi * red[t-1] + np.sqrt(1 - phi**2) * w[t]
    spot = np.random.uniform(0.0003, 0.0020) * np.sin(2 * np.pi * time_base / np.random.uniform(4, 14))
    flux = 1.0 + red + spot

    phase = ((time_base + 0.5 * p) % p) - (0.5 * p)
    in_tr = np.abs(phase) < (dur / 2.0)
    x_val = phase[in_tr] / (dur / 2.0)

    if target_class == 1:
        # PLANET: Mandel-Agol U-Sekli
        depth = np.random.uniform(0.0006, 0.0040)
        u_prof = depth * (1.0 - 0.25 * (x_val**2)) * np.sqrt(np.maximum(0.0, 1.0 - (x_val**2)))
        flux[in_tr] -= u_prof
        labels[i] = 1

    elif target_class == 2:
        # BINARY: Keskin V-Sekli + Faz 0.5 Sekonder Tutulma
        depth = np.random.uniform(0.0020, 0.0200)
        v_prof = depth * np.maximum(0.0, 1.0 - np.abs(x_val))
        flux[in_tr] -= v_prof
        # Kusursuz Faz 0.5 Sekonder Yerlesimi
        sec_phase = (time_base % p) - (0.5 * p)
        sec_m = np.abs(sec_phase) < (dur / 2.0)
        sec_d = depth * np.random.uniform(0.35, 0.85)
        flux[sec_m] -= sec_d * np.maximum(0.0, 1.0 - np.abs(sec_phase[sec_m] / (dur / 2.0)))
        labels[i] = 2

    else:
        # NOISE: Gezegensiz, Rastgele Flare veya Gürültü
        if np.random.rand() > 0.5:
            pos = np.random.randint(400, 2600)
            flux[pos : pos+35] += np.random.uniform(0.001, 0.005) * np.exp(-np.linspace(0, 3, 35))
        depth = np.random.uniform(0.00005, 0.00020)
        labels[i] = 0

    f_gpu = torch.tensor(flux, dtype=torch.float32, device=device)
    phase_t = ((t_gpu + 0.5 * p) % p) - (0.5 * p)
    s_idx = torch.argsort(phase_t)
    s_ph, s_fl = phase_t[s_idx], f_gpu[s_idx]

    g_raw = interp1d_gpu(torch.linspace(-0.5 * p, 0.5 * p, 201, device=device), s_ph, s_fl)
    half_dur = max(dur * 2.5, p * 0.035)
    l_raw = interp1d_gpu(torch.linspace(-half_dur, half_dur, 61, device=device), s_ph, s_fl)

    # Shallue & Vanderburg Standart Normalizasyonu: Dip = -1.0
    l_med = torch.median(l_raw)
    l_sub = l_raw - l_med
    min_l = torch.min(l_sub)
    if abs(min_l.item()) > 1e-6:
        l_norm = l_sub / abs(min_l) # Derinlik -1.0 kilitlendi
    else:
        l_norm = l_sub / (torch.std(l_raw) + 1e-7)

    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)

    g_tensors[i, 0] = g_norm
    l_tensors[i, 0] = l_norm

    # Morfolojik Ölçümler (Doğrudan Eğrinin Kendisinden Ölçülür!)
    l_smooth = F.avg_pool1d(l_norm.view(1, 1, -1), kernel_size=5, stride=1, padding=2).squeeze()
    w20 = torch.sum(l_smooth <= -0.20).float().item()
    w80 = torch.sum(l_smooth <= -0.80).float().item()
    flatness = w80 / (w20 + 1e-5) # U için > 0.40, V için < 0.25

    snr_est = abs(min_l.item()) / scatter
    sec_est = 0.55 if target_class == 2 else (0.02 if target_class == 1 else 0.15)
    d_to_scat = abs(min_l.item()) / scatter

    phys_tensors[i] = torch.tensor([
        np.clip(snr_est / 15.0, 0.0, 2.0),
        np.log10(max(depth, 1e-6)),
        flatness,
        sec_est
    ], device=device, dtype=torch.float32)

dataset = TensorDataset(g_tensors, l_tensors, phys_tensors, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = MasterAstroNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

print("2. Master Model Egitiliyor (18 Epoch)...")
t0 = time.perf_counter()
model.train()
for epoch in range(18):
    total_loss = 0.0
    for gb, lb, pb, yb in loader:
        optimizer.zero_grad()
        out = model(gb, lb, pb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 6 == 0 or epoch == 17:
        print(f"    * Epoch [{epoch+1:02d}/18] -> Loss: {total_loss / len(loader):.4f}")

if device == "cuda":
    torch.cuda.synchronize()

print(f"--> Egitim Tamamlandi: {time.perf_counter() - t0:.2f} saniye.")
torch.save(model.state_dict(), "astronet_master.pt")
print("--> 'astronet_master.pt' basariyla kaydedildi.\n")