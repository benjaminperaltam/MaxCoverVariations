import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from Viajes_auto import Viajes_auto
import numpy as np

save = True

fig, ax = plt.subplots(figsize=(8, 6))
hours = mdates.HourLocator(interval = 2)
h_fmt = mdates.DateFormatter('%H:%M')
arr = Viajes_auto['HoraIni'].astype('datetime64[ns]')
weights = np.ones(len(arr))/len(arr)
hist = ax.hist(arr, bins=48, edgecolor='black', weights=weights)
ax.xaxis.set_major_locator(hours)
ax.xaxis.set_major_formatter(h_fmt)
ax.set_title('Trips per hour (normalized)')
plt.xticks(rotation = 45);
Tp = hist[0][::2] + hist[0][1::2]

if save:
    np.save('Inputs/Tp.npy', Tp)