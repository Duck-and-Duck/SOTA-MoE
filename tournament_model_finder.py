import warnings
import logging
import time
import os
import torch
import torch.nn as nn
import numpy as np
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*75)
print("       MODEL GLADYATOR LIGI: EN YUKSEK IQ'LU ASTRONET MODELI SECIMI      ")
print(f"--> DONANIM: {device.upper()} | GERCEK NASA TESS VERILERI ILE KARSILASTIRMA")
print("="*75)

# 1. STANDART MIMARI (astronet_production, honest, zscore, weights)
class AstroNetStandard(nn.Module):
    def __init__(self):
        super(AstroNetStandard, self).__init__()
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

    def forward(self, g, l, phys=None):
        return self.fc(torch.cat([self.global_conv(g), self.local_conv(l)], dim=1))

# 2. HQ MIMARISI (astronet_hq)
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

    def forward(self, g, l, phys=None):
        return self.fc(torch.cat([self.global_conv(g), self.local_conv(l)], dim=1))

def interp1d_gpu(x_query, x_ref, y_ref):
    indices = torch.searchsorted(x_ref.contiguous(), x_query.contiguous())
    indices = torch.clamp(indices, 1, len(x_ref) - 1)
    x0, x1 = x_ref[indices - 1], x_ref[indices]
    y0, y1 = y_ref[indices - 1], y_ref[indices]
    weight = (x_query - x0) / (x1 - x0 + 1e-8)
    return y0 + weight * (y1 - y0)

def prepare_views(t_val, f_val, period, t0):
    t_t = torch.tensor(t_val, dtype=torch.float32, device=device)
    f_t = torch.tensor(f_val, dtype=torch.float32, device=device)
    phase = ((t_t - t0 + 0.5 * period) % period) - (0.5 * period)
    s_idx = torch.argsort(phase)
    s_ph = phase[s_idx]
    s_fl = f_t[s_idx]

    g_raw = interp1d_gpu(torch.linspace(-0.5 * period, 0.5 * period, 201, device=device), s_ph, s_fl)
    l_raw = interp1d_gpu(torch.linspace(-period * 0.05, period * 0.05, 61, device=device), s_ph, s_fl)

    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)
    return g_norm.view(1, 1, 201).contiguous(), l_norm.view(1, 1, 61).contiguous()

# GERCEK NASA TEST HEDEFLERI
BENCHMARK_DATA = [
    {"name": "WASP-18 b", "tic": "TIC 100100827", "sector": 2, "period": 0.94145, "t0": 1354.45, "type": "PLANET"},
    {"name": "L 98-59 c", "tic": "TIC 260128333", "sector": 2, "period": 3.69060, "t0": 1362.74, "type": "PLANET"},
    {"name": "TESS EB 1", "tic": "TIC 339607421", "sector": 2, "period": 1.25820, "t0": 1354.10, "type": "BINARY"},
    {"name": "TESS EB 2", "tic": "TIC 48227288",  "sector": 2, "period": 1.95630, "t0": 1355.20, "type": "BINARY"}
]

print("--> 1. Gercek NASA Isik Egrileri On-Bellegine Aliniyor...")
cached_targets = []
for tgt in BENCHMARK_DATA:
    try:
        search = lk.search_lightcurve(tgt["tic"], mission="TESS", sector=tgt["sector"])
        lc = search[0].download(quality_bitmask="hardest").remove_nans().flatten(window_length=101)
        g_v, l_v = prepare_views(lc.time.value[:3000], lc.flux.value[:3000], tgt["period"], tgt["t0"])
        cached_targets.append({"name": tgt["name"], "type": tgt["type"], "g": g_v, "l": l_v})
        print(f"    * {tgt['name']} hazir.")
    except Exception as e:
        print(f"    * {tgt['name']} alinamadi: {e}")

# Yarisacak modeller
CANDIDATE_MODELS = [
    {"path": "astronet_production.pt", "class": AstroNetStandard},
    {"path": "astronet_hq.pt",         "class": AstroNetHQ},
    {"path": "astronet_honest.pt",     "class": AstroNetStandard},
    {"path": "astronet_zscore.pt",     "class": AstroNetStandard},
    {"path": "astronet_weights.pt",    "class": AstroNetStandard}
]

print("\n--> 2. Modeller Yarisiyor (Gercek Dunya Skoru Hesaplaniyor)...")
leaderboard = []

for cand in CANDIDATE_MODELS:
    p = cand["path"]
    if not os.path.exists(p):
        continue
    try:
        m = cand["class"]().to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()

        correct = 0
        total = len(cached_targets)
        probs_log = []

        with torch.no_grad():
            for t in cached_targets:
                prob = torch.sigmoid(m(t["g"], t["l"])).item()
                pred = "PLANET" if prob >= 0.50 else "BINARY"
                if pred == t["type"]:
                    correct += 1
                probs_log.append(f"{t['name'][:6]}:%{prob*100:.0f}")

        acc = (correct / total) * 100.0 if total > 0 else 0.0
        leaderboard.append({"path": p, "acc": acc, "correct": correct, "total": total, "log": " | ".join(probs_log)})
    except Exception as e:
        print(f"    [!] {p} calistirilamadi: {e}")

leaderboard.sort(key=lambda x: x["acc"], reverse=True)

print("\n" + "="*75)
print("                  MODEL GLADYATOR LIGI RESMI PUAN TABLOSU                 ")
print("="*75)
for idx, r in enumerate(leaderboard):
    badge = "★ SAMPIYON MODEL" if idx == 0 and r["acc"] >= 75.0 else ""
    print(f"{idx+1}. {r['path']:<22} | Isabet: %{r['acc']:5.1f} ({r['correct']}/{r['total']}) | {r['log']} {badge}")
print("="*75)

best_model_path = leaderboard[0]["path"]
print(f"\n--> EN IYI MODEL SECILDI: '{best_model_path}' (Dogruluk: %{leaderboard[0]['acc']:.1f})")