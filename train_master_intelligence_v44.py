import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*90)
print("  OSTE-MoE V44: SOTA MASTER MODEL TRAINING (ANTI-SHORTCUT & STELLAR ROBUST)")
print(f"--> DONANIM: {device.upper()} | 24,000 ASTROFIZIKSEL ORNEK | FOCAL LOSS & ADVERSARIAL EB")
print("="*90)

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

# FOCAL LOSS: ZOR ORNEKLERE ODAKLANAN KAYIP FONKSIYONU
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.6):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (self.alpha * targets + (1 - self.alpha) * (1 - targets)) * ((1 - pt) ** self.gamma)
        return torch.mean(focal_weight * bce)

# 2. SENTETIK ASTROFIZIKSEL VERI URETIMI (24,000 ORNEK)
print("--> Sentetik Astrofiziksel Egitim Evreni Uretiliyor (24,000 Ornek)...")
np.random.seed(42)

N_HALF = 12000
G_LEN = 201
L_LEN = 61

# Global ve Local Faz Izgaralari
g_phase = np.linspace(-0.5, 0.5, G_LEN)
l_phase = np.linspace(-0.06, 0.06, L_LEN)

g_data = np.zeros((2 * N_HALF, 1, G_LEN), dtype=np.float32)
l_data = np.zeros((2 * N_HALF, 1, L_LEN), dtype=np.float32)
labels = np.zeros((2 * N_HALF, 1), dtype=np.float32)

# A. POZITIF SINIF: GERCEK GEZEGENLER (Limb-Darkened U-Shape, Lekeli ve Flareli)
for i in range(N_HALF):
    dur = np.random.uniform(0.04, 0.12)
    depth = 10 ** np.random.uniform(np.log10(0.0003), np.log10(0.0200)) # 300 ppm - 20000 ppm
    
    # Global U-Transit
    g_tr = np.zeros(G_LEN)
    mask_g = np.abs(g_phase) < (dur / 2.0)
    g_tr[mask_g] = -depth * (1.0 - 0.25 * (2.0 * g_phase[mask_g] / dur)**2)
    
    # Local U-Transit
    l_tr = np.zeros(L_LEN)
    mask_l = np.abs(l_phase) < (dur / 2.0)
    l_tr[mask_l] = -depth * (1.0 - 0.25 * (2.0 * l_phase[mask_l] / dur)**2)
    
    # Leke / Sinus dalgasi ve Kirmizi Gurultu Asisi
    if i % 2 == 0:
        spot_wave = np.random.uniform(0.0005, 0.0035) * np.sin(2 * np.pi * g_phase * np.random.uniform(1.0, 3.0))
        g_tr += spot_wave
        l_tr += spot_wave[100 - L_LEN//2 : 100 - L_LEN//2 + L_LEN]
    
    # Flare Asisi (AU Mic Direnci)
    if i % 3 == 0:
        flare = np.random.uniform(0.005, 0.025) * np.exp(-np.linspace(0, 3, G_LEN))
        g_tr += flare
        l_tr += flare[100 - L_LEN//2 : 100 - L_LEN//2 + L_LEN]

    # Gurultu
    noise_lvl = np.random.uniform(0.0001, 0.0006)
    g_tr += np.random.normal(0, noise_lvl, G_LEN)
    l_tr += np.random.normal(0, noise_lvl, L_LEN)
    
    # Z-Score Normalizasyonu
    g_data[i, 0, :] = (g_tr - np.median(g_tr)) / (np.std(g_tr) + 1e-7)
    l_data[i, 0, :] = (l_tr - np.median(l_tr)) / (np.std(l_tr) + 1e-7)
    labels[i, 0] = 1.0

# B. NEGATIF SINIF: KONTROLLU ADVERSARIAL IKILI YILDIZLAR VE GUZEL TUZAKLAR
for j in range(N_HALF):
    idx = N_HALF + j
    sub_type = j % 4
    
    g_tr = np.zeros(G_LEN)
    l_tr = np.zeros(L_LEN)
    
    if sub_type == 0:
        # TUZAK 1: Sekonder Tutulmali Ikili Yildiz (Faz 0.5'te derin cukur)
        d_p = np.random.uniform(0.010, 0.030)
        d_s = d_p * np.random.uniform(0.25, 0.75) # Belirgin sekonder
        dur_eb = np.random.uniform(0.06, 0.14)
        
        # Primer (Faz 0)
        m_p = np.abs(g_phase) < (dur_eb / 2.0)
        g_tr[m_p] = -d_p * (1.0 - (2.0 * g_phase[m_p] / dur_eb)**2)
        m_lp = np.abs(l_phase) < (dur_eb / 2.0)
        l_tr[m_lp] = -d_p * (1.0 - (2.0 * l_phase[m_lp] / dur_eb)**2)
        
        # Sekonder (Faz -0.5 ve +0.5 uclarinda)
        m_s1 = g_phase < (-0.5 + dur_eb / 2.0)
        m_s2 = g_phase > (0.5 - dur_eb / 2.0)
        g_tr[m_s1] = -d_s
        g_tr[m_s2] = -d_s

    elif sub_type == 1:
        # TUZAK 2: Asimetrik V-Sekilli Derin Kontak Ikili
        d_cont = np.random.uniform(0.025, 0.060)
        dur_cont = np.random.uniform(0.08, 0.16)
        m_p = np.abs(g_phase) < (dur_cont / 2.0)
        g_tr[m_p] = -d_cont * (1.0 - np.abs(2.0 * g_phase[m_p] / dur_cont)) # Keskin V
        m_lp = np.abs(l_phase) < (dur_cont / 2.0)
        l_tr[m_lp] = -d_cont * (1.0 - np.abs(2.0 * l_phase[m_lp] / dur_cont))
        
    elif sub_type == 2:
        # TUZAK 3: Saf Flare ve Değişen Leke (Transitsiz Sahte Alarm)
        flare = np.random.uniform(0.010, 0.040) * np.exp(-np.linspace(0, 4, G_LEN))
        spot = np.random.uniform(0.002, 0.006) * np.sin(2 * np.pi * g_phase * np.random.uniform(1.0, 4.0))
        g_tr = flare + spot
        l_tr = g_tr[100 - L_LEN//2 : 100 - L_LEN//2 + L_LEN]
        
    else:
        # TUZAK 4: Tekil Gurultu Cukuru (SES/MES Spikes)
        rand_pt = np.random.randint(20, G_LEN - 20)
        g_tr[rand_pt-2:rand_pt+3] = -np.random.uniform(0.003, 0.010)

    # Gurultu
    noise_lvl = np.random.uniform(0.0001, 0.0006)
    g_tr += np.random.normal(0, noise_lvl, G_LEN)
    l_tr += np.random.normal(0, noise_lvl, L_LEN)

    g_data[idx, 0, :] = (g_tr - np.median(g_tr)) / (np.std(g_tr) + 1e-7)
    l_data[idx, 0, :] = (l_tr - np.median(l_tr)) / (np.std(l_tr) + 1e-7)
    labels[idx, 0] = 0.0

# 3. VERI SETI VE MODEL EGITIMI
tensor_g = torch.tensor(g_data, device=device)
tensor_l = torch.tensor(l_data, device=device)
tensor_y = torch.tensor(labels, device=device)

dataset = TensorDataset(tensor_g, tensor_l, tensor_y)
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

model = AstroNetHQ().to(device)
criterion = BinaryFocalLoss(gamma=2.0, alpha=0.6)
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

print("--> Model Egitimi Baslatiliyor (20 Epoch / Focal Loss)...")
best_val_acc = 0.0

for epoch in range(1, 21):
    model.train()
    total_loss = 0.0
    for b_g, b_l, b_y in train_loader:
        optimizer.zero_grad()
        preds = model(b_g, b_l)
        loss = criterion(preds, b_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    
    # Dogrulama
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b_g, b_l, b_y in val_loader:
            preds = model(b_g, b_l)
            predicted = (torch.sigmoid(preds) >= 0.50).float()
            correct += (predicted == b_y).sum().item()
            total += b_y.size(0)
    
    val_acc = (correct / total) * 100.0
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "astronet_hq.pt")
    
    if epoch % 4 == 0 or epoch == 20:
        print(f"    Epoch [{epoch:02d}/20] | Loss: {total_loss/len(train_loader):.4f} | Val Accuracy: %{val_acc:.2f} (En Iyi: %{best_val_acc:.2f})")

print(f"\n[✓ BASARI]: En Akilli Model Agirliklari 'astronet_hq.pt' Olarak Kaydedildi (Val Acc: %{best_val_acc:.2f})")
print("="*90)
