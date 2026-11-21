import pandas as pd
import numpy as np
import joypy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 1. Recreate the data from your table
data = {
    'Subscale': ['Fluency', 'Fluency', 'Fluency', 'Trust', 'Trust', 'Trust', 'Alliance', 'Alliance', 'Alliance'],
    'Contrast': ['Successful Delivery', 'Control', 'EA', 'Successful Delivery', 'Control', 'EA', 'Successful Delivery', 'Control', 'EA'],
    'Mean': [3.804, 3.718, 3.595, 3.815, 3.510, 3.491, 4.030, 3.990, 3.990],
    'HDI_lower': [3.498, 3.419, 3.287, 3.531, 3.210, 3.182, 3.766, 3.715, 3.708],
    'HDI_upper': [4.086, 4.024, 3.889, 4.122, 3.817, 3.790, 4.313, 4.262, 4.258]
}
df = pd.DataFrame(data)

# 2. Approximate posterior samples (long format)
df['std_approx'] = (df['HDI_upper'] - df['HDI_lower']) / 3.92 # (2 * 1.96)
plot_data_list = []
np.random.seed(42) # for reproducibility
for i, row in df.iterrows():
    samples = np.random.normal(row['Mean'], row['std_approx'], 5000)
    plot_data_list.append(pd.DataFrame({
        'Subscale': row['Subscale'],
        'Contrast': row['Contrast'],
        'Value': samples
    }))
plot_df_long = pd.concat(plot_data_list)

# --- KEY CHANGE 1: Reshape data from 'long' to 'wide' format ---
# This is crucial for plotting overlapping distributions with joypy.
# We create columns for each distribution we want to plot on the same row.
plot_df_wide = plot_df_long.pivot_table(
    # A unique index for each of the 5000 samples per group
    index=[plot_df_long.groupby(['Subscale', 'Contrast']).cumcount(), 'Subscale'],
    columns='Contrast',
    values='Value'
).reset_index()


# 3. Define colors and styling
contrast_colors = ['#0072B2','#D55E00','#009E73']
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'

# Set global font sizes for all text elements
plt.rcParams.update({
    'font.size': 30,
    'figure.titlesize': 0,
    'axes.labelsize': 30,
    'xtick.labelsize': 30,
    'ytick.labelsize': 18, # The Subscale names on the y-axis
    'legend.fontsize': 25,
    'legend.title_fontsize': 30
})

# 4. Create the Ridgeline Plot using the WIDE dataframe
fig, axes = joypy.joyplot(
    plot_df_wide,
    # --- KEY CHANGE 2: How joypy is called ---
    by='Subscale',
    column=['Successful Delivery', 'Control', 'EA'], # Columns to plot on each row
    color=contrast_colors,
    legend=False,
    alpha=0.6,
    linewidth=1.2,
    linecolor='black',
    figsize=(12, 8),
    overlap=0.3 # Reduce overlap to see the distributions clearly
)

# 5. Customize axes and reference line
plt.xlabel("Mean Scores (Posterior Estimate)")
for ax in axes:
    ax.axvline(x=0, linestyle='--', color='black', alpha=0.8, lw=1.5)

# Create custom "handles" for the legend
legend_elements = [
    Patch(facecolor=contrast_colors[0], edgecolor='black', label='Successful Delivery'),
    Patch(facecolor=contrast_colors[1], edgecolor='black', label='Control'),
    Patch(facecolor=contrast_colors[2], edgecolor='black', label='EA')
]

# Add the legend to the FIGURE (fig), not the axes
# loc=2 is 'upper left'. bbox_to_anchor moves it relative to the whole chart.
fig.legend(handles=legend_elements, 
           title="", 
           loc='upper left', 
           bbox_to_anchor=(0.05, 1), 
           frameon=False,
           fontsize=25, 
           title_fontsize=30)

# --- FINAL POLISH ---
plt.tight_layout()
# plt.savefig('overlapping_ridgeline_plot.png', dpi=300, bbox_inches='tight')
plt.show()