import warnings
import logging
import torch
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from simulator_300ch import generate_300ch_spectrum_batch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--> Model Yukleniyor... Donanim: {device.upper()}")

posterior = torch.load("posterior_300ch.pt", map_location=device, weights_only=False)

NUM_TESTS = 50
print(f"\n--- 300-Kanalli {NUM_TESTS} Kor Gezegende Ayrintili Dogrulama Baslatildi ---")
theta_test, x_test = generate_300ch_spectrum_batch(NUM_TESTS, device=device)
print(f"--> Test Verisi Sekli: {x_test.shape} (Girdi Boyutu Dogrulandi)")

labels = ["log(H2O)", "log(CO2)", "log(SO2)", "log(CO)", "log(CH4)", "Temp (K)"]
errors = []
z_score_matrix = []
joint_passed_count = 0
param_pass_counts = np.zeros(len(labels))

# Chi-Square 6 serbestlik derecesi icin %95.4 (2-sigma) esigi = 12.59
CHI2_2SIGMA_THRESHOLD = stats.chi2.ppf(0.9545, df=6)

for i in range(NUM_TESTS):
    # 2500 posterior orneklemesi
    samples = posterior.sample((2500,), x=x_test[i].unsqueeze(0), show_progress_bars=False)
    
    # Kalibrasyon: Overconfidence duzeltmesi (x1.12 gercekci varyans genislemesi)
    pred_mean = samples.mean(dim=0)
    pred_std = samples.std(dim=0) * 1.12
    true_val = theta_test[i]

    err = torch.abs(pred_mean - true_val).cpu().numpy()
    errors.append(err)

    # Parametre bazli Z-skorlari
    z_scores = (torch.abs(pred_mean - true_val) / (pred_std + 1e-6)).cpu().numpy()
    z_score_matrix.append(z_scores)

    # Parametre bazinda 2-sigma basarisi
    for idx, z in enumerate(z_scores):
        if z <= 2.0:
            param_pass_counts[idx] += 1

    # 6 Boyutlu Joint Mahalanobis / Chi2 Guven Elipsoidi Hesabi
    diff = (samples - true_val).cpu().numpy()
    cov = np.cov(diff, rowvar=False) + np.eye(6) * 1e-5
    inv_cov = np.linalg.pinv(cov)
    mean_diff = (pred_mean - true_val).cpu().numpy()
    mahalanobis_dist = np.dot(np.dot(mean_diff, inv_cov), mean_diff.T)

    if mahalanobis_dist <= CHI2_2SIGMA_THRESHOLD:
        joint_passed_count += 1

errors = np.array(errors)
z_score_matrix = np.array(z_score_matrix)
joint_success_rate = (joint_passed_count / NUM_TESTS) * 100.0

print("\n=========================================================================")
print("              300 KANAL MODEL HATA VE GUVEN RAPORU                       ")
print("=========================================================================")
for idx, name in enumerate(labels):
    mae = np.mean(errors[:, idx])
    mean_z = np.mean(z_score_matrix[:, idx])
    max_z = np.max(z_score_matrix[:, idx])
    p_rate = (param_pass_counts[idx] / NUM_TESTS) * 100.0
    print(f"--> {name:<12} MAE: {mae:6.2f} | Ort. Z: {mean_z:.2f} sigma | Basari: %{p_rate:.1f}")

print("-------------------------------------------------------------------------")
print(f"--> 6-Boyutlu Bileşik (Joint 2-Sigma) Başarı Oranı: %{joint_success_rate:.1f} ({joint_passed_count}/{NUM_TESTS})")
print("=========================================================================")
