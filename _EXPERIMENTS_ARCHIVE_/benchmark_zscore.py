import warnings
import logging
import time
import torch
import torch.nn as nn
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print("      GERCEKCI VE TAVIZSIZ BILANCO V16 (Z-SCORE STANDARDIZASYONU)        ")
print(f"--> DONANIM: {device.upper()} (RTX 3050 Ti)")
print("=========================================================================")

class AstroNetVetter(nn.Module):
    def __init__(self):
        super(AstroNetVetter, self).__init__()
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        self.local_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 50 + 32 * 15, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, g, l):
        return self.fc(torch.cat([self.global_conv(g), self.local_conv(l)], dim=1))

vetter = AstroNetVetter().to(device)
vetter.load_state_dict(torch.load("astronet_zscore.pt", map_location=device))
vetter.eval()

def interp1d_gpu(x_query, x_ref, y_ref):
    indices = torch.searchsorted(x_ref.contiguous(), x_query.contiguous())
    indices = torch.clamp(indices, 1, len(x_ref) - 1)
    x0, x1 = x_ref[indices - 1], x_ref[indices]
    y0, y1 = y_ref[indices - 1], y_ref[indices]
    weight = (x_query - x0) / (x1 - x0 + 1e-8)
    return y0 + weight * (y1 - y0)

def phase_fold_and_zscore(time_t, flux_t, period, t0):
    # 1. Faz Katlama
    phase = ((time_t - t0 + 0.5 * period) % period) - (0.5 * period)
    sorted_indices = torch.argsort(phase)
    sorted_phase = phase[sorted_indices]
    sorted_flux = flux_t[sorted_indices]

    global_bins = torch.linspace(-0.5 * period, 0.5 * period, 201, device=device)
    g_raw = interp1d_gpu(global_bins, sorted_phase, sorted_flux)

    dur_approx = period * 0.05
    local_bins = torch.linspace(-dur_approx, dur_approx, 61, device=device)
    l_raw = interp1d_gpu(local_bins, sorted_phase, sorted_flux)

    # 2. Z-Score Standardizasyonu (Shallue & Vanderburg 2018)
    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)

    return g_norm.view(1, 1, 201).contiguous(), l_norm.view(1, 1, 61).contiguous()

# =========================================================================
# TEST 1: GERÇEK NASA TESS VERİSİ (WASP-18 b)
# =========================================================================
print("\n--- 1. GERCEK NASA TESS VERISI (WASP-18 b) ---")
import lightkurve as lk
try:
    search = lk.search_lightcurve("TIC 100100827", mission="TESS", sector=2)
    lc = search[0].download(quality_bitmask="hardest").remove_nans()
    flat_lc = lc.flatten(window_length=101)
    t_val = torch.tensor(flat_lc.time.value[:3000], dtype=torch.float32, device=device)
    f_val = torch.tensor(flat_lc.flux.value[:3000], dtype=torch.float32, device=device)
    
    g_w18, l_w18 = phase_fold_and_zscore(t_val, f_val, 0.94145, 1354.45)
    with torch.no_grad():
        prob_w18 = torch.sigmoid(vetter(g_w18, l_w18)).item()
    
    print(f"--> WASP-18 b Olasılığı : %{prob_w18 * 100:.2f}")
    print(f"--> Karar                : {'ONAYLANDI (Gezegen)' if prob_w18 >= 0.50 else 'REDDEDILDI'}")
except Exception as e:
    print(f"[HATA]: {e}")

# =========================================================================
# TEST 2: 200 HEDEFLİK KÖR TEST MATRİSİ (BİREBİR AYNI 1200 ppm DERİNLİKTE)
# =========================================================================
print("\n--- 2. 200 HEDEFLIK KOR TEST MATRISI (EŞİT 1200 ppm DERİNLİKTE) ---")
print("--> 100 Gezegen (U-Tipi) ve 100 İkili Yıldız (V-Tipi) taranıyor...")

TP, FP, TN, FN = 0, 0, 0, 0
prob_planets = []
prob_binaries = []

t_sim = torch.linspace(0, 27.4, 3000, device=device)
p_test = 3.5
dur_test = 0.10

# 100 Gezegen Testi (1200 ppm, U-şekli)
for _ in range(100):
    noise = torch.randn(3000, device=device) * 0.00035
    phase = ((t_sim + 0.5 * p_test) % p_test) - (0.5 * p_test)
    mask = torch.abs(phase) < (dur_test / 2.0)
    
    x_t = (phase[mask] / (dur_test / 2.0)).cpu().numpy()
    u_prof = 0.0012 * (1.0 - 0.2 * (x_t**2)) * np.sqrt(np.maximum(0.0, 1.0 - (x_t**2)))
    noise[mask] -= torch.tensor(u_prof, device=device, dtype=torch.float32)
    
    g, l = phase_fold_and_zscore(t_sim, noise, p_test, 0.0)
    with torch.no_grad():
        p = torch.sigmoid(vetter(g, l)).item()
    prob_planets.append(p)
    if p >= 0.50:
        TP += 1
    else:
        FN += 1

# 100 İkili Yıldız Testi (BİREBİR AYNI 1200 ppm, V-şekli)
for _ in range(100):
    noise = torch.randn(3000, device=device) * 0.00035
    phase = ((t_sim + 0.5 * p_test) % p_test) - (0.5 * p_test)
    mask = torch.abs(phase) < (dur_test / 2.0)
    
    x_v = (phase[mask] / (dur_test / 2.0)).cpu().numpy()
    v_prof = 0.0012 * np.maximum(0.0, 1.0 - np.abs(x_v))
    noise[mask] -= torch.tensor(v_prof, device=device, dtype=torch.float32)
    
    g, l = phase_fold_and_zscore(t_sim, noise, p_test, 0.0)
    with torch.no_grad():
        p = torch.sigmoid(vetter(g, l)).item()
    prob_binaries.append(p)
    if p >= 0.50:
        FP += 1
    else:
        TN += 1

accuracy = (TP + TN) / 200.0 * 100.0
precision = TP / (TP + FP + 1e-7) * 100.0
recall = TP / (TP + FN + 1e-7) * 100.0

print("\n=========================================================================")
print("               DURUST VE GERCEKCI BILIMSEL BILANCO                       ")
print("=========================================================================")
print(f"--> Doğru Pozitif (TP) [Gezegeni Bildi]   : {TP:3d} / 100")
print(f"--> Yanlış Negatif (FN) [Gezegeni Kaçırdı] : {FN:3d} / 100")
print(f"--> Doğru Negatif (TN) [İkiliyi Eledi]     : {TN:3d} / 100")
print(f"--> Yanlış Pozitif (FP) [İkiliyi Yuttu]    : {FP:3d} / 100")
print("-------------------------------------------------------------------------")
print(f"--> GERÇEK DOĞRULUK (ACCURACY) : %{accuracy:.1f}")
print(f"--> KESİNLİK (PRECISION)       : %{precision:.1f}")
print(f"--> YAKALAMA ORANI (RECALL)    : %{recall:.1f}")
print(f"--> Ortalama Gezegen Olasılığı : %{np.mean(prob_planets)*100:.1f}")
print(f"--> Ortalama İkili Olasılığı   : %{np.mean(prob_binaries)*100:.1f}")
print("=========================================================================")
