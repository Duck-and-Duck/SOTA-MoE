import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*95)
print("  OSTE-MoE V45: ABSOLUTE SCIENTIFIC CALIBRATION & ANTI-HALLUCINATION TRAINING")
print(f"--> DONANIM: {device.upper()} | 24,000 FIZIKSEL KALIBRE ORNEK | DUZ YILDIZ & SIG TRANSIT ODAKLI")
print("="*95)

# 1. 1D-CNN ASTRONET-HQ MIMARISI
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

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * ((1 - pt) ** self.gamma)
        return torch.mean(focal_weight * bce)

# 2. FİZİKSEL OLARAK DOĞRU 24.000 SENTETİK ÖRNEK ÜRETİMİ
print("--> Bilimsel Egitim Evreni Uretiliyor (Null Hypothesis + Sığ Transitler + EBs)...")
np.random.seed(2026)

N_HALF = 12000
G_LEN = 201
L_LEN = 61

# Global Faz: [-0.5, 0.5], Yerel Faz: [-2.0, 2.0] (dur cinsinden normalize)
g_phase = np.linspace(-0.5, 0.5, G_LEN)
l_phase = np.linspace(-2.0, 2.0, L_LEN)

g_data = np.zeros((2 * N_HALF, 1, G_LEN), dtype=np.float32)
l_data = np.zeros((2 * N_HALF, 1, L_LEN), dtype=np.float32)
labels = np.zeros((2 * N_HALF, 1), dtype=np.float32)

# A. POZİTİF SINIF: GERÇEK GEZEGENLER (M-cüce Süper-Dünyaları ve Jüpiterler)
for i in range(N_HALF):
    dur_frac = np.random.uniform(0.015, 0.060) # Yörüngenin %1.5 - %6'sı
    # Sığ transitlere ağırlık ver: 250 ppm ile 20,000 ppm arası logaritmik
    depth = 10 ** np.random.uniform(np.log10(0.00025), np.log10(0.0200))
    
    # Global U-Transit
    g_tr = np.zeros(G_LEN)
    m_g = np.abs(g_phase) < (dur_frac / 2.0)
    g_tr[m_g] = -depth * (1.0 - 0.20 * (2.0 * g_phase[m_g] / dur_frac)**2)
    
    # Local U-Transit: Yerel eksen [-2, 2] normalize olduğundan transit daima [-0.5, 0.5] aralığındadır!
    l_tr = np.zeros(L_LEN)
    m_l = np.abs(l_phase) < 0.5
    l_tr[m_l] = -depth * (1.0 - 0.20 * (2.0 * l_phase[m_l])**2)
    
    # Leke ve Flare Asisi (Ağ bu eğimlerin altındaki U-transiti okumayı öğrenir)
    if i % 2 == 0:
        spot = np.random.uniform(0.0003, 0.0025) * np.sin(2 * np.pi * g_phase * np.random.uniform(1, 3))
        g_tr += spot
        l_tr += spot[100 - L_LEN//2 : 100 - L_LEN//2 + L_LEN]
    
    if i % 4 == 0:
        flare = np.random.uniform(0.003, 0.015) * np.exp(-np.linspace(0, 3, G_LEN))
        g_tr += flare
        l_tr += flare[100 - L_LEN//2 : 100 - L_LEN//2 + L_LEN]
        
    noise = np.random.normal(0, np.random.uniform(0.00008, 0.00035), G_LEN)
    g_tr += noise
    l_tr += noise[:L_LEN]
    
    g_data[i, 0, :] = (g_tr - np.median(g_tr)) / (np.std(g_tr) + 1e-7)
    l_data[i, 0, :] = (l_tr - np.median(l_tr)) / (np.std(l_tr) + 1e-7)
    labels[i, 0] = 1.0

# B. NEGATİF SINIF: DENGELİ KONTROL EVRENİ (DÜZ YILDIZLAR + ÇİFT YILDIZLAR + FLARE)
for j in range(N_HALF):
    idx = N_HALF + j
    sub_type = j % 5
    
    g_tr = np.zeros(G_LEN)
    l_tr = np.zeros(L_LEN)
    
    if sub_type == 0 or sub_type == 1:
        # KATEGORİ 1: SESSİZ DURAĞAN YILDIZ VE KIRMIZI GÜRÜLTÜ (NULL HYPOTHESIS - %40 NEGATİF)
        # Transit yok! Model düz çizgileri gezegen sanmayı burada bırakır.
        pass
        
    elif sub_type == 2:
        # KATEGORİ 2: SEKONDER TUTULMALI İKİLİ YILDIZ (Faz 0.5 ve Kenar Çukurları)
        d_p = np.random.uniform(0.008, 0.030)
        d_s = d_p * np.random.uniform(0.30, 0.85)
        dur_f = np.random.uniform(0.03, 0.08)
        
        m_p = np.abs(g_phase) < (dur_f / 2.0)
        g_tr[m_p] = -d_p
        m_lp = np.abs(l_phase) < 0.5
        l_tr[m_lp] = -d_p
        
        # Sekonder Tutulma (Faz -0.5 ve +0.5 uçlarında)
        g_tr[g_phase < (-0.5 + dur_f/2.0)] = -d_s
        g_tr[g_phase > (0.5 - dur_f/2.0)] = -d_s
        
    elif sub_type == 3:
        # KATEGORİ 3: SAF FLARE VE LEKE MODÜLASYONU (TRANSİT YOK!)
        flare = np.random.uniform(0.008, 0.035) * np.exp(-np.linspace(0, 4, G_LEN))
        spot = np.random.uniform(0.001, 0.005) * np.sin(2 * np.pi * g_phase * np.random.uniform(1, 4))
        g_tr = flare + spot
        l_tr = g_tr[100 - L_LEN//2 : 100 - L_LEN//2 + L_LEN]
        
    else:
        # KATEGORİ 4: DERİN V-ŞEKİLLİ KONTAK İKİLİ VEYA SES SPİKE
        d_cont = np.random.uniform(0.025, 0.070)
        m_p = np.abs(g_phase) < 0.03
        g_tr[m_p] = -d_cont * (1.0 - np.abs(g_phase[m_p] / 0.03))
        m_lp = np.abs(l_phase) < 0.5
        l_tr[m_lp] = -d_cont * (1.0 - np.abs(l_phase[m_lp] / 0.5))

    noise = np.random.normal(0, np.random.uniform(0.00008, 0.00035), G_LEN)
    g_tr += noise
    l_tr += noise[:L_LEN]

    g_data[idx, 0, :] = (g_tr - np.median(g_tr)) / (np.std(g_tr) + 1e-7)
    l_data[idx, 0, :] = (l_tr - np.median(l_tr)) / (np.std(l_tr) + 1e-7)
    labels[idx, 0] = 0.0

# 3. MODELİ EĞİT VE KAYDET
tensor_g = torch.tensor(g_data, device=device)
tensor_l = torch.tensor(l_data, device=device)
tensor_y = torch.tensor(labels, device=device)

dataset = TensorDataset(tensor_g, tensor_l, tensor_y)
train_size = int(0.85 * len(dataset))
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

model = AstroNetHQ().to(device)
criterion = BinaryFocalLoss(gamma=2.0, alpha=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18)

print("--> Model Kalibrasyon Egitimi Baslatiliyor (18 Epoch)...")
best_loss = 999.0

for epoch in range(1, 19):
    model.train()
    t_loss = 0.0
    for b_g, b_l, b_y in train_loader:
        optimizer.zero_grad()
        preds = model(b_g, b_l)
        loss = criterion(preds, b_y)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
    scheduler.step()
    
    model.eval()
    val_loss = 0.0
    corr, tot = 0, 0
    with torch.no_grad():
        for b_g, b_l, b_y in val_loader:
            preds = model(b_g, b_l)
            val_loss += criterion(preds, b_y).item()
            corr += ((torch.sigmoid(preds) >= 0.5) == b_y).sum().item()
            tot += b_y.size(0)
            
    v_loss = val_loss / len(val_loader)
    v_acc = corr / tot * 100.0
    if v_loss < best_loss:
        best_loss = v_loss
        torch.save(model.state_dict(), "astronet_hq.pt")
        
    if epoch % 3 == 0 or epoch == 18:
        print(f"    Epoch [{epoch:02d}/18] | Val Loss: {v_loss:.5f} | Val Accuracy: %{v_acc:.2f}")

print(f"\n[✓ BASARI]: Kusursuz Kalibre Edilmis Agirliklar 'astronet_hq.pt' Olarak Kaydedildi.")
print("="*95)
