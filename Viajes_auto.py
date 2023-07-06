import pandas as pd
import numpy as np
from unidecode import unidecode
import sys

sys.path.insert(1, '/Users/benja/OneDrive/Documentos/U/MaxCover')
from Survey_OD import Viajes_auto

Viajes_auto['Manhattan'] = Viajes_auto['Manhattan'] / 1e3
Viajes_auto['dt'] = np.abs(Viajes_auto['HoraFin'] - Viajes_auto['HoraIni']).dt.total_seconds().astype(float) / (60 * 60)
Viajes_auto['dt'] = Viajes_auto.apply(lambda x: x['dt'] + 0.8 if (x['dt'] == 0) else x['dt'], axis=1)
Viajes_auto = Viajes_auto[~pd.isna(Viajes_auto['dt'])]
Viajes_auto['dx/dt'] = Viajes_auto['Manhattan'] / Viajes_auto['dt']
Viajes_auto = Viajes_auto[Viajes_auto['dx/dt'] > 0]
Viajes_auto['ComunaOrigen'] = [unidecode(c).lower() for c in Viajes_auto['ComunaOrigen'].astype(str)]
Viajes_auto['ComunaDestino'] = [unidecode(c).lower() for c in Viajes_auto['ComunaDestino'].astype(str)]
df_viajes = Viajes_auto[['OrigenCoordX', 'OrigenCoordY', 'ComunaOrigen', 'DestinoCoordX', 'DestinoCoordY', 'ComunaDestino', 'dt']]
df_viajes.columns = ['x_origen', 'y_origen', 'Comuna_origen', 'x_destino', 'y_destino', 'Comuna_destino', 'dt']