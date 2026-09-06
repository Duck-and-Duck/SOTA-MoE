import numpy as np
import torch

# KOPPARAPU ET AL. (2013, 2014) YAŞANABİLİR BÖLGE KATSAYILARI (G, K, M Cüceleri İçin)
# Güneş tipi yıldız akısına oranla (S_eff_Sun = 1.0)
def check_habitable_zone(insolation, teff_star):
    t_diff = teff_star - 5780.0
    # Runaway Greenhouse (İç Sınır)
    s_inner = 1.0512 + 1.3242e-4 * t_diff + 1.5418e-8 * (t_diff**2)
    # Maximum Greenhouse (Dış Sınır)
    s_outer = 0.3438 + 5.8942e-5 * t_diff + 1.1731e-8 * (t_diff**2)
    
    if s_outer <= insolation <= s_inner:
        return "OPTIMAL YASANABILIR BOLGE (Habitable Zone - Sivi Su Mumkun)"
    elif insolation > s_inner:
        return "SICAK BÖLGE (Asiri Sicak / Sera Etkisi)"
    else:
        return "SOGUK BÖLGE (Donmus / Buzul Rejimi)"

# CHEN & KIPPING (2017) KÜTLE-YARIÇAP NÖRAL / ANALİTİK PROBABİLİSTİK MODELİ
def predict_mass_from_radius(radius_earth):
    if radius_earth <= 1.23:
        # Kayalık Bölge (Terran: M ~ R^3.68)
        mass = (radius_earth ** 3.68)
    elif radius_earth <= 14.3:
        # Gaz / Uçucu Zengin (Neptün Benzeri: M ~ R^1.70)
        mass = 0.97 * (radius_earth ** 1.70)
    else:
        # Gaz Devi (Jovian: Kütleçekimsel sıkışma)
        mass = 317.8 * (radius_earth / 11.2) ** 0.5
    return max(0.1, mass)

def characterize_discovered_planet(P_days, depth_fraction, dur_days, r_star_solar=1.0, m_star_solar=1.0, t_star_k=5778.0):
    G = 6.67430e-11
    M_sun = 1.98847e30
    R_sun = 6.957e8
    R_earth = 6.371e6
    M_earth = 5.9722e24
    AU = 1.495978707e11
    sigma_sb = 5.670374419e-8

    # 1. YÖRÜNGE ÖZELLİKLERİ
    P_sec = P_days * 86400.0
    # Yarı Büyük Eksen (Kepler 3. Yasası: a^3 = G * M_* * P^2 / 4pi^2)
    a_m = ((G * (m_star_solar * M_sun) * (P_sec**2)) / (4 * (np.pi**2))) ** (1.0 / 3.0)
    a_AU = a_m / AU

    # Yörünge Eğikliği (i) ve Darbe Parametresi (b) (Seager & Mallen-Ornelas 2003)
    k = np.sqrt(max(depth_fraction, 1e-6))
    r_star_m = r_star_solar * R_sun
    v_orb = 2 * np.pi * a_m / P_sec
    chord_len = v_orb * (dur_days * 86400.0)
    b_impact = np.sqrt(max(0.0, ((1 + k)**2) - (chord_len / r_star_m)**2))
    b_impact = min(0.95, b_impact)
    cos_i = b_impact * (r_star_m / a_m)
    inclination_deg = np.arccos(min(1.0, max(0.0, cos_i))) * (180.0 / np.pi)

    # 2. FİZİKSEL ÖZELLİKLER
    # Gezegen Yarıçapı (R_p)
    R_p_m = k * r_star_m
    R_p_earth = R_p_m / R_earth

    # Tahmini Kütle (M_p) - Chen & Kipping (2017)
    M_p_earth = predict_mass_from_radius(R_p_earth)
    M_p_kg = M_p_earth * M_earth

    # Ortalama Yoğunluk (rho)
    vol_m3 = (4.0 / 3.0) * np.pi * (R_p_m**3)
    density_g_cm3 = (M_p_kg / vol_m3) / 1000.0

    # Yüzey Yerçekimi (log g)
    g_surf_m_s2 = G * M_p_kg / (R_p_m**2)
    log_g = np.log10(g_surf_m_s2 * 100.0) # cgs

    # 3. ÇEVRESEL VE ENERJİ ÖZELLİKLERİ
    # Yıldız Aydınlatma Gücü (Luminosity: L = 4pi * R_*^2 * sigma * T_*^4)
    L_star = 4 * np.pi * (r_star_m**2) * sigma_sb * (t_star_k**4)
    L_sun = 3.828e26
    L_star_solar = L_star / L_sun

    # Işınım Akısı (Insolation Flux: S_p = L_* / a^2)
    insolation_earth = L_star_solar / (a_AU**2)

    # Bond Albedosu (Heng & Demory 2013)
    if R_p_earth < 2.0:
        albedo = 0.30 # Kayalık Dünya benzeri
    elif R_p_earth < 6.0:
        albedo = 0.25 # Neptün benzeri
    else:
        albedo = 0.10 # Sıcak Jüpiter benzeri (Koyu gaz devleri)

    # Denge Sıcaklığı (T_eq)
    T_eq = t_star_k * np.sqrt(r_star_m / (2 * a_m)) * ((1 - albedo)**0.25)

    # Yaşanabilir Bölge Değerlendirmesi
    hz_status = check_habitable_zone(insolation_earth, t_star_k)

    # 4. ATMOSFERİK BASINÇ VE SKALA YÜKSEKLİĞİ (Scale Height: H = k_B * T / mu * g)
    k_B = 1.380649e-23
    mu_mol = 2.3 * 1.66054e-27 if R_p_earth >= 3.0 else 28.0 * 1.66054e-27 # Gaz devi (H/He) veya Kayalık (N2/CO2)
    H_scale_m = (k_B * T_eq) / (mu_mol * g_surf_m_s2)
    H_scale_km = H_scale_m / 1000.0

    return {
        "Period_days": P_days,
        "SemiMajorAxis_AU": a_AU,
        "Eccentricity": 0.0, # Dairesel kabul
        "Inclination_deg": inclination_deg,
        "Impact_b": b_impact,
        "Radius_Earth": R_p_earth,
        "Mass_Earth": M_p_earth,
        "Density_g_cm3": density_g_cm3,
        "Surface_Gravity_log_g": log_g,
        "Gravity_m_s2": g_surf_m_s2,
        "Insolation_Earth": insolation_earth,
        "Albedo": albedo,
        "T_eq_K": T_eq,
        "Habitable_Zone_Status": hz_status,
        "Scale_Height_km": H_scale_km
    }