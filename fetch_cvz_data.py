import warnings
warnings.filterwarnings("ignore")

import lightkurve as lk
from astroquery.gaia import Gaia
import numpy as np

TARGET_TIC = "TIC 261136679"

print(f"--> TESS CVZ Hedefi Araniyor: {TARGET_TIC}")
search_result = lk.search_targetpixelfile(TARGET_TIC, mission="TESS")
print(f"Bulunan Gozlem Sektoru Sayisi: {len(search_result)}")

# 1. Sektorun piksel matrisini indir
tpf = search_result[0].download(quality_bitmask="default")
print(f"--> Ham Piksel Matrisi Indirildi: {tpf.shape} (Zaman x Y x X)")

# Koordinatlar dogrudan ra ve dec ozelliklerinden alinir
ra, dec = tpf.ra, tpf.dec
print(f"--> Gaia DR3 Koordinati Taraniyor: RA={ra:.4f}, DEC={dec:.4f}")

query = f"""
SELECT source_id, ra, dec, phot_g_mean_mag 
FROM gaiadr3.gaia_source 
WHERE 1=CONTAINS(
    POINT('ICRS', ra, dec), 
    CIRCLE('ICRS', {ra}, {dec}, 0.03)
)
ORDER BY phot_g_mean_mag ASC
"""
job = Gaia.launch_job(query)
gaia_stars = job.get_results()

print(f"\n[FIZYON BASARILI]: Bu TESS pikselinin icine dusen Gaia Yildiz Sayisi: {len(gaia_stars)}")
print("En parlak 3 yildiz:")
for i in range(min(3, len(gaia_stars))):
    print(f"  * Yildiz {i+1}: Gaia ID={gaia_stars['source_id'][i]} | Parlaklik (G)={gaia_stars['phot_g_mean_mag'][i]:.2f}")

print("\n--> TESS CVZ + Gaia Fuzyon Verisi Bellege Alindi.")
