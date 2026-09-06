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
print("   MULTI-MODAL ASTRONET MASTER EGITIMI (15.000 FIZIKSEL NUMUNE | 3 SINIF)")
print(f"--> HESAPLAMA BIRIMI: {device.upper()} | REFERANS: Shallue & Vanderburg / Coughlin et al.")
print("="*85)

# =========================================================================
# 1. 3-SINIFLI MULTI-MODAL ASTRONET MIMARISI
# =========================================================================
class MultiModalAstroNet(nn.Module):
    def __init__(self):
        super(MultiModalAstroNet, self).__init__()
        # 1. Global Kol (201 pt)
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.15),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(15),
            nn.Flatten()
        )
        # 2. Local Kol (61 pt)
        self.local_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.15),
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
        # 3. Fiziksel Skaler Kolu (5 Metrik)
        # [SNR, Log10(Depth), Odd_Even_Mismatch, Secondary_Ratio, Depth_to_Scatter]
        self.phys_dense = nn.Sequential(
            nn.Linear(5, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1)
        )
        # 4. Bilesik Karar Katmani (3 Sinif: 0=NOISE, 1=PLANET, 2=BINARY)
        self.classifier = nn.Sequential(
            nn.Linear(32 * 15 + 32 * 15 + 16, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.30),
            nn.Linear(64, 3)
        )

    def forward(self, g, l, phys):
        g_feat = self.global_conv(g)
        l_feat = self.local_conv(l)
        p_feat = self.phys_dense(phys)
        return self.classifier(torch.cat([g_feat, l_feat, p_feat], dim=1))

# =========================================================================
# 2. 15.000 NUMUNELIK KAPSAMLI FIZIKSEL VERI URETIMI
# =========================================================================
N = 15000
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
phys_tensors = torch.zeros(N, 5, device=device)
labels = torch.zeros(N, dtype=torch.long, device=device)

print(f"1. {N} Adet Fiziksel Olarak Zenginlestirilmis Egitim Seti Olusturuluyor...")
np.random.seed(42)

for i in range(N):
    target_class = i % 3  # 0: NOISE, 1: PLANET, 2: BINARY
    p = np.random.uniform(1.2, 9.0)
    dur = np.random.uniform(0.05, 0.17)
    base_scatter = np.random.uniform(0.0002, 0.0006)

    # Harvey Kırmızı Gürültüsü + Leke Rotasyonu
    phi = np.exp(-(27.4/3000) / 0.45)
    w_noise = np.random.normal(0.0, base_scatter, 3000)
    red_noise = np.zeros(3000)
    for t in range(1, 3000):
        red_noise[t] = phi * red_noise[t-1] + np.sqrt(1 - phi**2) * w_noise[t]
    spot_wave = np.random.uniform(0.0003, 0.0025) * np.sin(2 * np.pi * time_base / np.random.uniform(4, 15))
    flux = 1.0 + red_noise + spot_wave

    phase = ((time_base + 0.5 * p) % p) - (0.5 * p)
    in_tr = np.abs(phase) < (dur / 2.0)
    x_val = phase[in_tr] / (dur / 2.0)

    if target_class == 1:
        # PLANET: Mandel-Agol U-Sekli
        depth = np.random.uniform(0.0006, 0.0035)
        u_prof = depth * (1.0 - 0.25 * (x_val**2)) * np.sqrt(np.maximum(0.0, 1.0 - (x_val**2)))
        flux[in_tr] -= u_prof
        labels[i] = 1
        snr_val = depth / (base_scatter / np.sqrt(max(1, np.sum(in_tr))))
        odd_even_val = np.random.uniform(0.0, 0.12)
        sec_ratio_val = np.random.uniform(0.0, 0.05)
        d_to_scatter = depth / base_scatter

    elif target_class == 2:
        # BINARY: V-Sekli + Faz 0.5 Sekonder Tutulma
        depth = np.random.uniform(0.0025, 0.015)
        v_prof = depth * np.maximum(0.0, 1.0 - np.abs(x_val))
        flux[in_tr] -= v_prof
        sec_ph = ((time_base) % p) - (0.5 * p)
        sec_m = np.abs(sec_ph) < (dur / 2.0)
        sec_d = depth * np.random.uniform(0.35, 0.80)
        flux[sec_m] -= sec_d * np.maximum(0.0, 1.0 - np.abs(sec_ph[sec_m] / (dur / 2.0)))
        labels[i] = 2
        snr_val = depth / (base_scatter / np.sqrt(max(1, np.sum(in_tr))))
        odd_even_val = np.random.uniform(0.30, 0.85)
        sec_ratio_val = sec_d / depth
        d_to_scatter = depth / base_scatter

    else:
        # NOISE: Saf Yildiz Gurultusu / Flare / Gecis Yok
        if np.random.rand() > 0.5:
            # Flare ekle
            flare_pos = np.random.randint(500, 2500)
            flux[flare_pos : flare_pos+40] += np.random.uniform(0.001, 0.005) * np.exp(-np.linspace(0, 3, 40))
        depth = np.random.uniform(0.00005, 0.00025)
        labels[i] = 0
        snr_val = np.random.uniform(1.0, 4.5)
        odd_even_val = np.random.uniform(0.10, 0.90)
        sec_ratio_val = np.random.uniform(0.0, 0.50)
        d_to_scatter = depth / base_scatter

    f_gpu = torch.tensor(flux, dtype=torch.float32, device=device)
    phase_t = ((t_gpu + 0.5 * p) % p) - (0.5 * p)
    s_idx = torch.argsort(phase_t)
    s_ph, s_fl = phase_t[s_idx], f_gpu[s_idx]

    g_raw = interp1d_gpu(torch.linspace(-0.5 * p, 0.5 * p, 201, device=device), s_ph, s_fl)
    half_dur = max(dur * 2.5, p * 0.035)
    l_raw = interp1d_gpu(torch.linspace(-half_dur, half_dur, 61, device=device), s_ph, s_fl)

    g_tensors[i, 0] = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_tensors[i, 0] = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)
    phys_tensors[i] = torch.tensor([
        np.clip(snr_val / 20.0, 0.0, 2.0),
        np.log10(max(depth, 1e-6)),
        odd_even_val,
        sec_ratio_val,
        np.clip(d_to_scatter / 10.0, 0.0, 2.0)
    ], device=device, dtype=torch.float32)

dataset = TensorDataset(g_tensors, l_tensors, phys_tensors, labels)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = MultiModalAstroNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

print("2. Multi-Modal Nöral Ağ Eğitiliyor (20 Epoch)...")
t0 = time.perf_counter()
model.train()
for epoch in range(20):
    total_loss = 0.0
    for gb, lb, pb, yb in loader:
        optimizer.zero_grad()
        out = model(gb, lb, pb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    if (epoch + 1) % 5 == 0 or epoch == 19:
        print(f"    * Epoch [{epoch+1:02d}/20] -> Loss: {total_loss / len(loader):.4f}")

if device == "cuda":
    torch.cuda.synchronize()

print(f"--> Egitim Tamamlandi: {time.perf_counter() - t0:.2f} saniye.")
torch.save(model.state_dict(), "astronet_multimodal.pt")
print("--> 'astronet_multimodal.pt' basariyla muhurlendi.\n")