import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import time
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("=========================================================================")
print(f"--> Z-SCORE NORMALIZASYONLU ASTRONET EGITIMI | BIRIM: {device.upper()}")
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

N = 3600
print(f"1. {N} Adet Z-Score Normalize Edilmis Ornek Uretiliyor (DC Ofset Sifirlandi)...")
np.random.seed(42)

g_np = np.zeros((N, 1, 201), dtype=np.float32)
l_np = np.zeros((N, 1, 61), dtype=np.float32)
labels_np = np.zeros((N, 1), dtype=np.float32)

# Yarısı Gezegen (U-Tipi), Yarısı İkili Yıldız (V-Tipi)
for i in range(N):
    # Taban gürültüsü std=1.0, mean=0.0
    g_raw = np.random.normal(0.0, 1.0, 201).astype(np.float32)
    l_raw = np.random.normal(0.0, 1.0, 61).astype(np.float32)
    
    # Derinlik sinyal-gürültü oranı cinsinden: 2.5 sigma ile 5.5 sigma arası
    snr_depth = np.random.uniform(2.5, 5.5)
    dur_half = np.random.randint(6, 12)

    if i < N // 2:
        # GEZEGEN (U-Şekli, Düz Taban) -> Sınıf 1
        x_axis = np.linspace(-1.0, 1.0, dur_half * 2 + 1)
        u_profile = snr_depth * (1.0 - 0.2 * (x_axis**2)) * np.sqrt(np.maximum(0.0, 1.0 - (x_axis**2)))
        g_raw[100 - dur_half : 100 + dur_half + 1] -= u_profile
        l_raw[30 - dur_half : 30 + dur_half + 1] -= u_profile
        labels_np[i] = 1.0
    else:
        # İKİLİ YILDIZ (V-Şekli, Keskin Apeks) -> Sınıf 0
        v_profile = snr_depth * np.maximum(0.0, 1.0 - np.abs(np.linspace(-1.0, 1.0, 25)))
        g_raw[100 - 12 : 100 + 13] -= v_profile
        l_raw[30 - 12 : 30 + 13] -= v_profile
        labels_np[i] = 0.0

    # Z-Score Standardizasyonu (Google AstroNet Kuralı)
    g_np[i, 0] = (g_raw - np.median(g_raw)) / (np.std(g_raw) + 1e-7)
    l_np[i, 0] = (l_raw - np.median(l_raw)) / (np.std(l_raw) + 1e-7)

dataset = TensorDataset(torch.from_numpy(g_np), torch.from_numpy(l_np), torch.from_numpy(labels_np))
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = AstroNetVetter().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

print("2. Model Egitiliyor (Loss Takibi ile)...")
t0 = time.perf_counter()
model.train()
for epoch in range(15):
    total_loss = 0.0
    for g_b, l_b, y_b in loader:
        g_b, l_b, y_b = g_b.to(device), l_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(g_b, l_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0:
        print(f"    * Epoch [{epoch+1:02d}/15] -> Loss: {total_loss / len(loader):.4f}")

if device == "cuda":
    torch.cuda.synchronize()

print(f"--> Egitim Tamamlandi: {time.perf_counter() - t0:.2f} saniye.")
torch.save(model.state_dict(), "astronet_zscore.pt")
print("--> Model 'astronet_zscore.pt' olarak basariyla mühürlendi.")
