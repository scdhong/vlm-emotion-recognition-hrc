import pandas as pd
import numpy as np
import joypy
import matplotlib.pyplot as plt

# 1. Original data
data = {
    'Subscale': ['Anthropomorphism', 'Anthropomorphism','Anthropomorphism', 'Animacy', 'Animacy', 'Animacy'],
    'Contrast': ['Successful Delivery', 'Control', 'EA', 'Successful Delivery', 'Control', 'EA'],
    'Mean': [2.146, 2.530, 2.536, 2.331, 2.794, 2.708],
    'HDI_lower': [0.556, 0.845, 0.798, 0.893, 1.289, 1.215],
    'HDI_upper': [3.603, 3.904, 3.877, 3.676, 4.089, 4.021]
}
df = pd.DataFrame(data)

# 2. Approximate posterior samples from the summary statistics
df['std_approx'] = (df['HDI_upper'] - df['HDI_lower']) / 3.92 # (2 * 1.96)
plot_data_list = []
for i, row in df.iterrows():
    np.random.seed(i)
    samples = np.random.normal(row['Mean'], row['std_approx'], 5000)
    plot_data_list.append(pd.DataFrame({
        'Subscale': row['Subscale'],
        'Contrast': row['Contrast'],
        'Value': samples
    }))
plot_df = pd.concat(plot_data_list)

# 3. Reshape data from 'long' to 'wide' format.
plot_df_wide = plot_df.pivot_table(
    index=[plot_df.groupby(['Subscale', 'Contrast']).cumcount(), 'Subscale'],
    columns='Contrast',
    values='Value'
).reset_index()

# 4. Define colors and styling to match the original image
contrast_colors = ['#0072B2', '#D55E00','#009E73']
plt.style.use('seaborn-v0_8-white')
plt.rcParams['font.family'] = 'sans-serif'

# --- KEY CHANGE: Set global font sizes ---
plt.rcParams.update({
    'font.size': 30,          # Main font size
    'axes.labelsize': 30,     # X and Y labels
    'xtick.labelsize': 20,    # X-axis tick labels
    'ytick.labelsize': 30,    # Y-axis tick labels (the subscale names)
    'legend.fontsize': 30,    # Legend text
})

# 5. Create the Ridgeline Plot
fig, axes = joypy.joyplot(
    plot_df_wide,
    by='Subscale',
    column=['Successful Delivery', 'Control', 'EA'], # Columns to plot on each row
    color=contrast_colors,
    labels=['Anim.', 'Anth.'],
    legend=True,
    figsize=(14, 7),
    alpha=0.6,
    linewidth=1.5,
    linecolor='black',
    grid=False,
    background='white',
    overlap=0.5
)

# 6. Customize axes and labels (fontsize is now handled by rcParams)
plt.xlabel("Mean Scores (Posterior Estimate)") # No fontsize needed here
plt.ylabel("")

# 7. Add vertical reference line at zero
for ax in axes:
    ax.tick_params(axis='y', length=0)
    # The y-label font size is also set by rcParams now
    ax.set_ylabel(ax.get_ylabel())


# 8. Customize the legend (fontsize is now handled by rcParams)
if fig.legends:
    legend = fig.legends[0]
    legend.set_frame_on(False)
    legend.set_title(None)
    

plt.show()