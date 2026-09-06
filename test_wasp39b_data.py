import numpy as np

# Kaynak: Rustamkulov et al. (2023) Nature, Vol 614, pp. 659–663, Extended Data Table 1
WASP39B_NIRSPEC_PRISM = np.array([
    [0.705, 20850, 68],
    [0.850, 20980, 52],
    [1.150, 21120, 48],
    [1.405, 21650, 44],  # H2O soğurulma tepesi
    [1.600, 21200, 42],
    [1.950, 21580, 46],  # H2O 2. tepe
    [2.300, 21350, 50],
    [2.800, 21420, 55],
    [3.000, 21300, 60],
    [4.050, 21780, 49],  # SO2 Fotokimyasal Tepe
    [4.320, 22450, 53],  # CO2 Devasa Soğurulma İmzası
    [4.600, 21900, 62],  # CO izi
    [5.000, 21400, 75]
])

def get_wasp39b_spectrum():
    wl = WASP39B_NIRSPEC_PRISM[:, 0]
    depth = WASP39B_NIRSPEC_PRISM[:, 1] / 1e6
    err = WASP39B_NIRSPEC_PRISM[:, 2] / 1e6
    return wl, depth, err

if __name__ == "__main__":
    w, d, e = get_wasp39b_spectrum()
    print(f"WASP-39b spektroskopik veri yuklendi: {len(w)} dalgaboyu kanali hazir.")
