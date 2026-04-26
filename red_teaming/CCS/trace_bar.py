import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import FormatStrFormatter

rcParams['font.family'] = 'DejaVu Sans'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 9
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.linewidth'] = 0.8
rcParams['grid.linewidth'] = 0.5
rcParams['grid.alpha'] = 0.3

df = pd.read_excel(
    r'C:\Users\최유진\Desktop\VSCode\Agent_AI_Security\red_teaming\CCS\generated_scenarios\reproducibility_outlier.xlsx',
    header=[0, 1]
)
df.columns = ['Agent'] + [f"{grp}_{n}" for grp, n in df.columns[1:]]
agents_df = df[df['Agent'] != 'Mean'].copy()

n_cols = [c for c in agents_df.columns if c.startswith('Trace reproducibility')]
n_labels = ['K=1', 'K=2', 'K=3', 'K=4', 'K=5']
agents_df['short'] = agents_df['Agent']

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(13, 6.5),
    gridspec_kw={'height_ratios': [10, 1], 'hspace': 0.06},
    sharex=True,
)

x = np.arange(len(agents_df))
width = 0.16
bar_colors = ['#5A7FAF', '#76B68A', '#E2877F', '#9D8CC4', '#E7B26A']
for i, (col, label) in enumerate(zip(n_cols, n_labels)):
    offset = (i - 2) * width
    vals = agents_df[col].values
    for ax in (ax_top, ax_bot):
        ax.bar(x + offset, vals, width,
               label=label if ax is ax_top else None,
               color=bar_colors[i], edgecolor='white', linewidth=0.6)

# y-axis: top zoom 0.9~1.0, 0.01 step, 2-decimal format
ax_top.set_ylim(0.993, 1.000)
ax_top.set_yticks(np.arange(0.993, 1.0001, 0.001))
ax_top.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ax_bot.set_ylim(0, 0.0001)
ax_bot.set_yticks([0])
ax_bot.set_yticklabels(['0.00'])

ax_top.spines['bottom'].set_visible(False)
ax_bot.spines['top'].set_visible(False)
ax_top.tick_params(labeltop=False, bottom=False)
ax_bot.xaxis.tick_bottom()

d = 0.008
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
ax_top.plot((-0.005, 0.005), (-d, d), **kwargs)
ax_top.plot((0.995, 1.005), (-d, d), **kwargs)
kwargs.update(transform=ax_bot.transAxes)
ax_bot.plot((-0.005, 0.005), (1 - d * 10, 1 + d * 10), **kwargs)
ax_bot.plot((0.995, 1.005), (1 - d * 10, 1 + d * 10), **kwargs)

ax_bot.set_xticks(x)
ax_bot.set_xticklabels(agents_df['short'], rotation=0)
ax_top.set_ylabel('Reproducibility', fontsize=25, fontname='Arial')
ax_top.yaxis.set_label_coords(-0.05, 0.4)
ax_top.grid(True, axis='y')
ax_bot.grid(True, axis='y')

ax_top.legend(ncol=5, loc='upper center',
              bbox_to_anchor=(0.5, 1.08),
              frameon=False,
              fontsize=11)

plt.subplots_adjust(top=0.86)
plt.savefig('grouped_bar_v6.png', dpi=600, bbox_inches='tight')
plt.savefig('grouped_bar_v6.pdf', dpi=600, bbox_inches='tight')
plt.show()