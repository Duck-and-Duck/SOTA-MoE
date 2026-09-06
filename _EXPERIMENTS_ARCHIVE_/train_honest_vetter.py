import warnings
import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print(f"--> DURUST VE DOYGUNLUKSUZ ASTRONET EGITIMI | BIRIM: {device.upper()}")
print("=========================================================================")

class AstroNetVetter(nn.Module):
    def __init__(self):
        super(AstroNetVetter, self).__init__()
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Dropout(0.25),
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
            nn.Dropout(0.25),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 50 + 32 * 15, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.35),
            nn.Linear(64, 1)
        )

    def forward(self, g, l):
        return self.fc(torch.cat([self.global_conv(g), self.local_conv(l)], dim=1))

# 4000 Örnek: GEZEGEN VE IKILI YILDIZLAR TAMAMEN AYNI DERINLIKTE (500 - 3000 ppm)
N = 4000
print(f"1. {N} Adet Eşit-Derinlikli Sentetik Set Üretiliyor (Kopya Çekme Engellendi)...")
np.random.seed(42)

g_np = np.random.normal(1.0, 0.00035, size=(N, 1, 201)).astype(np.float32)
l_np = np.random.normal(1.0, 0.00035, size=(N, 1, 61)).astype(np.float32)
labels_np = np.zeros((N, 1), dtype=np.float32)

for i in range(N // 2):
    # Gezegen: 500 - 3000 ppm arası U-profili
    depth = np.random.uniform(0.0005, 0.0030)
    dur_half = np.random.randint(6, 12)
    x_axis = np.linspace(-1.0, 1.0, dur_half * 2 + 1)
    u_shape = depth * (1.0 - 0.2 * (x_axis**2)) * np.sqrt(np.maximum(0.0, 1.0 - (x_axis**2)))
    g_np[i, 0, 100 - dur_half : 100 + dur_half + 1] -= u_shape
    l_np[i, 0, 30 - dur_half : 30 + dur_half + 1] -= u_shape
    labels_np[i] = 0.95  # Label Smoothing: Asla 1.0 verme!

for i in range(N // 2, N):
    if i % 2 == 0:
        # İkili Yıldız: BİREBİR AYNI DERİNLİKTE (500 - 3000 ppm) Keskin V-Profili
        v_depth = np.random.uniform(0.0005, 0.0030)
        v_shape = np.maximum(0.0, 1.0 - np.abs(np.linspace(-1, 1, 25))) * v_depth
        g_np[i, 0, 100 - 12 : 100 + 13] -= v_shape
        l_np[i, 0, 30 - 12 : 30 + 13] -= v_shape
    else:
        # Gürültü / Flare
        f_amp = np.random.uniform(0.0005, 0.0025)
        f_shape = np.exp(-np.linspace(0, 3, 20)) * f_amp
        g_np[i, 0, 100 : 120] += f_shape
    labels_np[i] = 0.05  # Label Smoothing: Asla 0.0 verme!

dataset = TensorDataset(torch.from_numpy(g_np), torch.from_numpy(l_np), torch.from_numpy(labels_np))
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = AstroNetVetter().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
criterion = nn.BCEWithLogitsLoss()

print("2. Model Eğitiliyor (Doygunluk Önleyici Dropout ve Weight Decay ile)...")
t0 = time.perf_counter()
model.train()
for epoch in range(18):
    for g_b, l_b, y_b in loader:
        g_b, l_b, y_b = g_b.to(device), l_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(g_b, l_b), y_b)
        loss.backward()
        optimizer.step()

if device == "cuda":
    torch.cuda.synchronize()

print(f"--> Egitim Tamamlandi: {time.perf_counter() - t0:.2f} saniye.")
torch.save(model.state_dict(), "astronet_honest.pt")
print("--> Model 'astronet_honest.pt' olarak kaydedildi.")
