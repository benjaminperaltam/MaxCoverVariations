import numpy as np
import pandas as pd
import geopandas as gpd
import os
from unidecode import unidecode

import sys
sys.path.insert(1, '/Users/benja/OneDrive/Documentos/U/MaxCover')
from parameters import comunas_report
from Survey_OD import Viajes_auto
files_path = "..\\MaxCover"

comunas = gpd.read_file(os.path.join(files_path, 'COMUNA/COMUNAS_2020.shp'))
santiago_map = comunas[comunas.COMUNA.isin(comunas_report)]
santiago_map['COMUNA'] = [unidecode(s).lower() for s in santiago_map['COMUNA']]
santiago_map.to_crs("EPSG:32718", inplace=True)
Nvehic = pd.DataFrame({n: d for (n, d) in zip(['Comuna', 'Nvehic'],
                                              np.unique(np.concatenate((Viajes_auto['ComunaOrigen'],
                                                                        Viajes_auto['ComunaDestino'])).astype(str), return_counts=True))})
Nvehic = Nvehic[Nvehic['Comuna'] != 'nan'] 
Nvehic['pp'] = Nvehic['Nvehic'] / Nvehic['Nvehic'].sum()
Nvehic['Comuna'] = [unidecode(s).lower() for s in Nvehic['Comuna']]
Nvehic = Nvehic.merge(santiago_map[['COMUNA', 'SUPERFICIE']],
                      left_on='Comuna',
                      right_on='COMUNA').drop('COMUNA', axis=1)
Nvehic['vehic/surf'] = Nvehic['Nvehic']/Nvehic['SUPERFICIE']
santiago_map = santiago_map[santiago_map['COMUNA'].isin(Nvehic[Nvehic['vehic/surf'] > 10].Comuna)]