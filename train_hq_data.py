import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print(f"--> YUKSEK KALITELI ASTROFIZIKSEL VERI MOTORU | BIRIM: {device.upper()}")
print("=========================================================================")

# =========================================================================
# 1. MANDEL & AGOL (2002) VE KUADRATIK KENAR KARARMALI GECIS FIZIGI
# =========================================================================
def generate_mandel_agol_transit(time_arr, period, t0, duration, depth, impact_b=0.3):
    """
    Claret (2017) TESS bant katsayilariyla (u1=0.32, u2=0.28) Mandel-Agol U-profili
    """
    phase = ((time_arr - t0 + 0.5 * period) % period) - (0.5 * period)
    in_transit = np.abs(phase) < (duration / 2.0)
    
    transit_flux = np.ones_like(time_arr)
    if not np.any(in_transit):
        return transit_flux
    
    # Boyutsuz normalize zaman koordinatı
    x = phase[in_transit] / (duration / 2.0)
    # Geometrik uzaklık z^2 = x^2 + b^2
    z = np.sqrt(x**2 + impact_b**2)
    
    # Kuadratik kenar kararmalı U-şekilli emilim faktörü
    # Merkezde düzleşen, ingress/egress'te omuz veren tam fiziksel profil
    mu = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, z**2)))
    limb_dark = 1.0 - 0.32 * (1.0 - mu) - 0.28 * ((1.0 - mu)**2)
    profile = depth * (limb_dark / 1.0) * np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, x**2)))
    
    transit_flux[in_transit] -= profile
    return transit_flux

# =========================================================================
# 2. GERCEKCI IKILI YILDIZ MOTORU (SEKONDER TUTULMA + ELIPSOIDAL VARYASYON)
# =========================================================================
def generate_realistic_binary(time_arr, period, t0, duration, depth, is_contact=False):
    """
    Prsa et al. (2011) Kepler/TESS EB katalogu fizigi:
    Primer Tutulma + Faz 0.5 Sekonder Tutulma + Gelgit Elipsoidal Dalgasi
    """
    phase = ((time_arr - t0 + 0.5 * period) % period) - (0.5 * period)
    binary_flux = np.ones_like(time_arr)
    
    # 1. Primer Tutulma (V-Şekilli Apeks, Faz 0.0)
    x_prim = phase / (duration / 2.0)
    prim_mask = np.abs(phase) < (duration / 2.0)
    binary_flux[prim_mask] -= depth * np.maximum(0.0, 1.0 - np.abs(x_prim[prim_mask]))
    
    # 2. Sekonder Tutulma (Faz 0.5 civarı, daha sığ yoldaş yıldız tutulması)
    sec_phase = ((time_arr - t0) % period) - (0.5 * period)
    x_sec = sec_phase / (duration / 2.0)
    sec_mask = np.abs(sec_phase) < (duration / 2.0)
    sec_depth = depth * np.random.uniform(0.35, 0.75) # Sıcaklık farkı
    binary_flux[sec_mask] -= sec_depth * np.maximum(0.0, 1.0 - np.abs(x_sec[sec_mask]))
    
    # 3. Elipsoidal Değişim (Temaslı ikililerde yerçekimsel basıklık dalgalanması)
    if is_contact:
        ellip_amp = depth * 0.25
        binary_flux += ellip_amp * np.cos(4 * np.pi * phase / period)
        
    return binary_flux

# =========================================================================
# 3. HARVEY (1985) MODELI KIRMIZI GURULTU VE GRANULASYON
# =========================================================================
def generate_harvey_red_noise(N_pts=3000, dt=0.00913):
    phi = np.exp(-dt / 0.45) # 11 saatlik granülasyon korelasyonu
    white = np.random.normal(0.0, 0.00030, N_pts)
    red = np.zeros(N_pts)
    for t in range(1, N_pts):
        red[t] = phi * red[t-1] + np.sqrt(1 - phi**2) * white[t]
    return red

# =========================================================================
# 4. YUKSEK KALITELI VERI HAVUZU URETIMI (6000 ORNEK)
# =========================================================================
N = 6000
print(f"1. {N} Adet NASA Standartlarinda Yuksek Kaliteli Veri Uretiliyor...")
np.random.seed(42)

g_tensors = torch.zeros(N, 1, 201, device=device)
l_tensors = torch.zeros(N, 1, 61, device=device)
labels = torch.zeros(N, 1, device=device)

time_base = np.linspace(0, 27.4, 3000)

def interp1d_gpu(x_query, x_ref, y_ref):
    indices = torch.searchsorted(x_ref.contiguous(), x_query.contiguous())
    indices = torch.clamp(indices, 1, len(x_ref) - 1)
    x0, x1 = x_ref[indices - 1], x_ref[indices]
    y0, y1 = y_ref[indices - 1], y_ref[indices]
    weight = (x_query - x0) / (x1 - x0 + 1e-8)
    return y0 + weight * (y1 - y0)

t_gpu = torch.tensor(time_base, dtype=torch.float32, device=device)

t0_start = time.perf_counter()
for i in range(N):
    p = np.random.uniform(2.0, 7.5)
    dur = np.random.uniform(0.08, 0.16)
    depth = np.random.uniform(0.0006, 0.0030)
    
    # Gerçekçi yıldız gürültüsü: Granülasyon + Leke Salınımı
    noise = generate_harvey_red_noise(3000) + 0.0004 * np.sin(2 * np.pi * time_base / np.random.uniform(5, 12))
    
    if i < N // 2:
        # SINIF 1: MANDEL-AGOL GEZEGENLERI (Limb-Darkened U-Profili)
        impact = np.random.uniform(0.0, 0.75) # Gerçekçi darbe parametresi
        flux = generate_mandel_agol_transit(time_base, p, 0.0, dur, depth, impact_b=impact) + noise
        labels[i] = 1.0
    else:
        # SINIF 0: GERCEKCI IKILI YILDIZLAR (Sekonder Tutulmali V-Profili)
        contact_flag = (i % 3 == 0)
        flux = generate_realistic_binary(time_base, p, 0.0, dur, depth, is_contact=contact_flag) + noise
        labels[i] = 0.0
        
    f_gpu = torch.tensor(flux, dtype=torch.float32, device=device)
    
    # NASA SPOC Ön-İşleme: GPU Faz Katlama ve Z-Score
    phase_t = ((t_gpu + 0.5 * p) % p) - (0.5 * p)
    s_idx = torch.argsort(phase_t)
    s_ph = phase_t[s_idx]
    s_fl = f_gpu[s_idx]
    
    g_raw = interp1d_gpu(torch.linspace(-0.5 * p, 0.5 * p, 201, device=device), s_ph, s_fl)
    l_raw = interp1d_gpu(torch.linspace(-p * 0.05, p * 0.05, 61, device=device), s_ph, s_fl)
    
    # Z-Score Standardizasyonu
    g_tensors[i, 0] = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_tensors[i, 0] = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)

print(f"--> Veri Havuzu Hazirlandi: {time.perf_counter() - t0_start:.2f} saniye.")

# =========================================================================
# 5. MODEL MIMARISI VE EGITIM (ASTRONET HIGH-QUALITY)
# =========================================================================
class AstroNetHQ(nn.Module):
    def __init__(self):
        super(AstroNetHQ, self).__init__()
        # Global Kol: 201 Nokta (SEKONDER TUTULMAYI YAKALAYAN GENIS ALAN)
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
        # Local Kol: 61 Nokta (LIMB DARKENING U-TABANINI YAKALAYAN DETAYLI ALAN)
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
        g_feat = self.global_conv(g)
        l_feat = self.local_conv(l)
        return self.fc(torch.cat([g_feat, l_feat], dim=1))

val_split = int(N * 0.8)
train_dataset = TensorDataset(g_tensors[:val_split], l_tensors[:val_split], labels[:val_split])
val_dataset = TensorDataset(g_tensors[val_split:], l_tensors[val_split:], labels[val_split:])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

model = AstroNetHQ().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
criterion = nn.BCEWithLogitsLoss()

print("2. Model Yüksek Kaliteli Veriyle Egitiliyor (20 Epoch)...")
t_tr0 = time.perf_counter()

for epoch in range(20):
    model.train()
    tr_loss = 0.0
    for g_b, l_b, y_b in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(g_b, l_b), y_b)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()
    scheduler.step()
    
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for g_b, l_b, y_b in val_loader:
            preds = model(g_b, l_b)
            val_loss += criterion(preds, y_b).item()
            correct += ((torch.sigmoid(preds) >= 0.50).float() == y_b).sum().item()
            
    val_acc = correct / len(val_dataset) * 100.0
    if (epoch + 1) % 5 == 0:
        print(f"    * Epoch [{epoch+1:02d}/20] -> Train Loss: {tr_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: %{val_acc:.1f}")

if device == "cuda":
    torch.cuda.synchronize()

print(f"--> Egitim Tamamlandi: {time.perf_counter() - t_tr0:.2f} saniye.")
torch.save(model.state_dict(), "astronet_hq.pt")
print("--> Kaliteli Model 'astronet_hq.pt' mühürlendi.")
