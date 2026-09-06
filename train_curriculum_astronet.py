import warnings
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("="*85)
print("   CURRICULUM LEARNING ASTRONET (FAZLI ZORLASTIRMA PROTOKOLU)")
print(f"--> HESAPLAMA BIRIMI: {device.upper()} (RTX) | METOD: Bengio et al. (2009)")
print("="*85)

# =========================================================================
# 1. STANDART ASTRONET-HQ MIMARISI
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

# =========================================================================
# 2. FIZIKSEL SINYAL VE PARAZIT URETECLERI
# =========================================================================
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

def generate_binary(time_base, p, dur, depth, is_contact=False):
    phase = ((time_base + 0.5 * p) % p) - (0.5 * p)
    eb_flux = np.ones_like(time_base)
    x_prim = phase / (dur / 2.0)
    prim_m = np.abs(phase) < (dur / 2.0)
    eb_flux[prim_m] -= depth * np.maximum(0.0, 1.0 - np.abs(x_prim[prim_m]))

    sec_ph = ((time_base) % p) - (0.5 * p)
    x_sec = sec_ph / (dur / 2.0)
    sec_m = np.abs(sec_ph) < (dur / 2.0)
    sec_d = depth * np.random.uniform(0.35, 0.75)
    eb_flux[sec_m] -= sec_d * np.maximum(0.0, 1.0 - np.abs(x_sec[sec_m]))

    if is_contact:
        eb_flux += (depth * 0.25) * np.cos(4 * np.pi * phase / p)
    return eb_flux

def generate_harvey_noise(N_pts=3000, dt=0.00913):
    phi = np.exp(-dt / 0.45)
    white = np.random.normal(0.0, 0.00030, N_pts)
    red = np.zeros(N_pts)
    for t in range(1, N_pts):
        red[t] = phi * red[t-1] + np.sqrt(1 - phi**2) * white[t]
    return red

def interp1d_gpu(x_query, x_ref, y_ref):
    indices = torch.searchsorted(x_ref.contiguous(), x_query.contiguous())
    indices = torch.clamp(indices, 1, len(x_ref) - 1)
    x0, x1 = x_ref[indices - 1], x_ref[indices]
    y0, y1 = y_ref[indices - 1], y_ref[indices]
    weight = (x_query - x0) / (x1 - x0 + 1e-8)
    return y0 + weight * (y1 - y0)

def build_phase_views(flux_arr, p, time_base, t_gpu):
    f_gpu = torch.tensor(flux_arr, dtype=torch.float32, device=device)
    phase_t = ((t_gpu + 0.5 * p) % p) - (0.5 * p)
    s_idx = torch.argsort(phase_t)
    s_ph, s_fl = phase_t[s_idx], f_gpu[s_idx]

    g_raw = interp1d_gpu(torch.linspace(-0.5 * p, 0.5 * p, 201, device=device), s_ph, s_fl)
    l_raw = interp1d_gpu(torch.linspace(-p * 0.05, p * 0.05, 61, device=device), s_ph, s_fl)

    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)
    return g_norm, l_norm

# =========================================================================
# 3. KADEMELI VERI URETIMI (3 AYRI FAZ)
# =========================================================================
N = 6000
time_base = np.linspace(0, 27.4, 3000)
t_gpu = torch.tensor(time_base, dtype=torch.float32, device=device)

def create_dataset(phase_level):
    np.random.seed(42 + phase_level * 10)
    g_t = torch.zeros(N, 1, 201, device=device)
    l_t = torch.zeros(N, 1, 61, device=device)
    y_t = torch.zeros(N, 1, device=device)

    for i in range(N):
        p = np.random.uniform(2.0, 7.5)
        dur = np.random.uniform(0.08, 0.16)
        depth = np.random.uniform(0.0008, 0.0035)

        if phase_level == 1:
            # FAZ 1: Saf Morfoloji (Düşük Beyaz Gürültü)
            noise = np.random.normal(0.0, 0.00010, 3000)
        elif phase_level == 2:
            # FAZ 2: Harvey Kırmızı Gürültüsü + Standart Gürültü
            noise = generate_harvey_noise(3000)
        else:
            # FAZ 3: Tam Gerçekçilik (Harvey + Yıldız Lekesi + Flare)
            spot = 0.0005 * np.sin(2 * np.pi * time_base / np.random.uniform(5, 12))
            noise = generate_harvey_noise(3000) + spot

        if i < N // 2:
            # GEZEGEN: U-Şekli
            impact = np.random.uniform(0.0, 0.70)
            flux = generate_mandel_agol(time_base, p, dur, depth, impact_b=impact) + noise
            y_t[i] = 1.0
        else:
            # GEZEGEN DIŞI: V-İkili Tutulması, Ters Transit veya Flare
            sub = i % 4
            if sub in [0, 1]:
                flux = generate_binary(time_base, p, dur, depth, is_contact=(sub == 1)) + noise
            elif sub == 2 and phase_level == 3:
                # Ters-Transit (Anti-Hallucination)
                clean_tr = generate_mandel_agol(time_base, p, dur, depth, impact_b=0.3)
                flux = (2.0 - clean_tr) + noise
            else:
                # Flare veya Değişken Yıldız
                flare = 0.0025 * np.exp(-np.linspace(0, 4, 3000)) if phase_level == 3 else 0.0
                flux = 1.0 + noise + flare
            y_t[i] = 0.0

        g_norm, l_norm = build_phase_views(flux, p, time_base, t_gpu)
        g_t[i, 0] = g_norm
        l_t[i, 0] = l_norm

    return DataLoader(TensorDataset(g_t, l_t, y_t), batch_size=128, shuffle=True)

# =========================================================================
# 4. MODEL EGITIM DONGUSU (CURRICULUM SCHEDULE)
# =========================================================================
model = AstroNetHQ().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

stages = [
    {"level": 1, "epochs": 8, "desc": "FAZ 1: Saf U vs V Morfolojik Temel Egitimi"},
    {"level": 2, "epochs": 8, "desc": "FAZ 2: Harvey Kirmizi Gurultusu ve Doku Dayanikliligi"},
    {"level": 3, "epochs": 8, "desc": "FAZ 3: Yildiz Lekeleri + Flare + Anti-Hallucination Sertlestirme"}
]

t_total = time.perf_counter()

for stage in stages:
    print(f"\n--> {stage['desc']} (8 Epoch)...")
    loader = create_dataset(stage["level"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage["epochs"])

    for epoch in range(stage["epochs"]):
        model.train()
        total_loss = 0.0
        for gb, lb, yb in loader:
            optimizer.zero_grad()
            pred = model(gb, lb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        if (epoch + 1) % 4 == 0:
            print(f"    * Epoch [{epoch+1:02d}/{stage['epochs']}] -> Kayip (Loss): {total_loss/len(loader):.4f}")

if device == "cuda":
    torch.cuda.synchronize()

print(f"\n--> Curriculum Egitimi Basariyla Bitti: {time.perf_counter() - t_total:.2f} saniye.")
torch.save(model.state_dict(), "astronet_hq.pt")
print("--> 'astronet_hq.pt' mufredat agirliklariyla kaydedildi.\n")