import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

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

# --- load ---
df = pd.read_excel(r'C:\Users\최유진\Desktop\VSCode\Agent_AI_Security\red_teaming\CCS\generated_scenarios\scenario_cluster_similarity_all.xlsx')

# overall 행만 필터 (하늘색 부분)
overall = df[df['Case'].str.contains(r'\(overall\)', regex=True)].copy()

# Case 파싱: "banking_cs_agent N=2 (overall)" → agent, N
overall['agent'] = overall['Case'].str.extract(r'^(.*?)\s+N=\d+')[0]
overall['N']     = overall['Case'].str.extract(r'N=(\d+)')[0].astype(int)

# 피벗: row=agent, col=N (2~5)
pivot = overall.pivot(index='agent', columns='N', values='mean')

short = {
    'banking_cs_agent': 'Banking',
    'ecommerce_operations_agent': 'E-commerce',
    'education_admin_agent': 'Education',
    'government_service_agent': 'Government',
    'hr_onboarding_agent': 'HR',
    'insurance_claims_agent': 'Insurance',
    'logistics_operations_agent': 'Logistics',
    'medical_consultation_agent': 'Medical',
    'telecom_cs_agent': 'Telecom',
    'travel_reservation_agent': 'Travel',
}
pivot = pivot.rename(index=short)
# 원하는 순서 유지
order = ['Banking', 'E-commerce', 'Education', 'Government', 'HR',
         'Insurance', 'Logistics', 'Medical', 'Telecom', 'Travel']
pivot = pivot.loc[order]

n_vals = [2, 3, 4, 5]
n_labels = [f'N={n}' for n in n_vals]

# --- figure with broken y-axis ---
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(13, 6.2),
    gridspec_kw={'height_ratios': [5, 1], 'hspace': 0.05},
    sharex=True,
)

x = np.arange(len(pivot))
width = 0.20   # 4개 bar라 폭을 키움

# 4개 블루 그라데이션
bar_colors = ['#5A7FAF', '#76B68A', '#E2877F', '#9D8CC4']

for i, n in enumerate(n_vals):
    offset = (i - 1.5) * width   # 4개 bar 중심 정렬
    vals = pivot[n].values
    for ax in (ax_top, ax_bot):
        ax.bar(x + offset, vals, width,
               label=n_labels[i] if ax is ax_top else None,
               color=bar_colors[i], edgecolor='white', linewidth=0.6)

# y-axis: 값이 0.87~0.93 근처 → 0.80~1.00 확대축
ax_top.set_ylim(0.80, 1.00)
ax_top.set_yticks(np.arange(0.80, 1.001, 0.04))
ax_bot.set_ylim(0, 0.05)
ax_bot.set_yticks([0])

# hide spines between the two axes
ax_top.spines['bottom'].set_visible(False)
ax_bot.spines['top'].set_visible(False)
ax_top.tick_params(labeltop=False, bottom=False)
ax_bot.xaxis.tick_bottom()

# diagonal break marks
d = 0.008
kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1)
ax_top.plot((-0.005, 0.005), (-d, d), **kwargs)
ax_top.plot((0.995, 1.005), (-d, d), **kwargs)
kwargs.update(transform=ax_bot.transAxes)
ax_bot.plot((-0.005, 0.005), (1 - d * 5, 1 + d * 5), **kwargs)
ax_bot.plot((0.995, 1.005), (1 - d * 5, 1 + d * 5), **kwargs)

# labels
ax_bot.set_xticks(x)
ax_bot.set_xticklabels(pivot.index)
ax_top.set_ylabel('Cluster similarity', fontsize=25, fontname='Arial')
ax_top.yaxis.set_label_coords(-0.05, 0.4)
ax_top.grid(True, axis='y')
ax_bot.grid(True, axis='y')

# title + legend
ax_top.legend(ncol=4, loc='upper center',
              bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=11)

plt.subplots_adjust(top=0.86)

plt.savefig('cluster_similarity_bar.png', dpi=600, bbox_inches='tight')
plt.savefig('cluster_similarity_bar.pdf', dpi=600, bbox_inches='tight')
plt.show()