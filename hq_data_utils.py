import numpy as np

def generate_mandel_agol_transit(time_arr, period, t0, duration, depth, impact_b=0.3):
    phase = ((time_arr - t0 + 0.5 * period) % period) - (0.5 * period)
    in_transit = np.abs(phase) < (duration / 2.0)
    transit_flux = np.ones_like(time_arr)
    if not np.any(in_transit):
        return transit_flux
    x = phase[in_transit] / (duration / 2.0)
    z = np.sqrt(x**2 + impact_b**2)
    mu = np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, z**2)))
    limb_dark = 1.0 - 0.32 * (1.0 - mu) - 0.28 * ((1.0 - mu)**2)
    profile = depth * (limb_dark / 1.0) * np.sqrt(np.maximum(0.0, 1.0 - np.minimum(1.0, x**2)))
    transit_flux[in_transit] -= profile
    return transit_flux

def generate_realistic_binary(time_arr, period, t0, duration, depth, is_contact=False):
    phase = ((time_arr - t0 + 0.5 * period) % period) - (0.5 * period)
    binary_flux = np.ones_like(time_arr)
    x_prim = phase / (duration / 2.0)
    prim_mask = np.abs(phase) < (duration / 2.0)
    binary_flux[prim_mask] -= depth * np.maximum(0.0, 1.0 - np.abs(x_prim[prim_mask]))
    
    sec_phase = ((time_arr - t0) % period) - (0.5 * period)
    x_sec = sec_phase / (duration / 2.0)
    sec_mask = np.abs(sec_phase) < (duration / 2.0)
    sec_depth = depth * np.random.uniform(0.35, 0.75)
    binary_flux[sec_mask] -= sec_depth * np.maximum(0.0, 1.0 - np.abs(x_sec[sec_mask]))
    
    if is_contact:
        binary_flux += (depth * 0.25) * np.cos(4 * np.pi * phase / period)
    return binary_flux

def generate_harvey_red_noise(N_pts=3000, dt=0.00913):
    phi = np.exp(-dt / 0.45)
    white = np.random.normal(0.0, 0.00030, N_pts)
    red = np.zeros(N_pts)
    for t in range(1, N_pts):
        red[t] = phi * red[t-1] + np.sqrt(1 - phi**2) * white[t]
    return red
