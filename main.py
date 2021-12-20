import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from scipy import sparse
from scipy.ndimage import gaussian_filter

# Load csv data using pandas
df = pd.read_csv('data.csv')  # Format: avgMaxPlateau,avgRandOps,avgDuration

# Pivot table
table = pd.pivot_table(df, values='avgDuration',
                          index='avgMaxPlateau', columns='avgRandOps', aggfunc=np.mean)

# Set NaNs to 0
table = table.fillna(0)

# Smooth the data
table = gaussian_filter(table, sigma=2)

# Set title
plt.title('Max Plateau vs Random Operations -> Average Duration')

ax = sns.heatmap(table, cmap=sns.color_palette("viridis_r", as_cmap=True))
ax.invert_yaxis()

ax.set_xlabel('no. random operations', fontsize=10)
ax.set_ylabel('max plateau value', fontsize=10)

plt.show()

