import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

df = pd.read_excel(r"C:\Users\최유진\Desktop\VSCode\Agent_AI_Security\red_teaming\CCS\generated_scenarios\scenario_cluster_similarity_all.xlsx")

# (overall) 행만 추출
overall = df[df['Case'].str.contains(r'N=\d+ \(overall\)', regex=True, na=False)].copy()
overall['Agent'] = overall['Case'].str.extract(r'^(\S+)')
overall['N'] = overall['Case'].str.extract(r'N=(\d+)').astype(int)

# pivot
pivot = overall.pivot(index='Agent', columns='N', values='mean')

label_map = {
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

color_map = {
    'Education':  '#2E8B3E',
    'E-commerce': '#E89A3C',
    'Insurance':  '#7A4A2B',
    'Banking':    '#1F4E8C',
    'Telecom':    '#B5B53A',
    'Medical':    '#7F7F7F',
    'HR':         '#8E6FBF',
    'Government': '#D7392E',
    'Travel':     '#2DB7B7',
    'Logistics':  '#E48BC8',
}

x_vals = [2, 3, 4, 5]
x_labels = [f'N={n}' for n in x_vals]
x = list(range(len(x_vals)))

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, sharex=True,
    gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.08},
    figsize=(11, 6.8)
)

end_points, start_points = [], []
for agent_key, name in label_map.items():
    if agent_key not in pivot.index:
        continue
    y = [pivot.loc[agent_key, n] for n in x_vals]
    color = color_map[name]
    ax_top.plot(x, y, marker='o', linewidth=1.7, markersize=5, color=color, zorder=3)
    ax_bot.plot(x, y, marker='o', linewidth=1.7, markersize=5, color=color, zorder=3)
    end_points.append([y[-1], name, color])
    start_points.append([y[0], name, color])

# Top: zoomed range
y_min, y_max = 0.87, 0.94
ax_top.set_ylim(y_min, y_max)
ax_top.yaxis.set_major_locator(MultipleLocator(0.01))
ax_top.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax_top.grid(True, axis='y', linestyle='-', linewidth=0.5, alpha=0.35, zorder=0)
ax_top.set_axisbelow(True)
for s in ('top', 'right', 'bottom'):
    ax_top.spines[s].set_visible(False)
ax_top.tick_params(bottom=False)

# Bottom: 0 baseline
ax_bot.set_ylim(0, 0.0001)
ax_bot.set_yticks([0])
ax_bot.set_yticklabels(['0.000'])
for s in ('top', 'right'):
    ax_bot.spines[s].set_visible(False)

# Diagonal break marks
d = .01
kw = dict(transform=ax_top.transAxes, color='k', clip_on=False, linewidth=1.2)
ax_top.plot((-d, +d), (-d*2.5, +d*2.5), **kw)
kw.update(transform=ax_bot.transAxes)
ax_bot.plot((-d, +d), (1 - d*10, 1 + d*10), **kw)

# De-overlap labels
def spread(points, gap):
    pts = sorted(points, key=lambda p: p[0])
    adj = [list(p) for p in pts]
    for i in range(1, len(adj)):
        if adj[i][0] - adj[i-1][0] < gap:
            adj[i][0] = adj[i-1][0] + gap
    return adj

gap = (y_max - y_min) * 0.045
right_adj = spread(end_points, gap)
left_adj  = spread(start_points, gap)

for y_orig, name, color in end_points:
    y_label = next(p[0] for p in right_adj if p[1] == name)
    ax_top.annotate(name, xy=(x[-1], y_orig), xytext=(x[-1] + 0.12, y_label),
                    textcoords='data', va='center', ha='left',
                    fontsize=10, fontweight='bold', color=color)

for y_orig, name, color in start_points:
    y_label = next(p[0] for p in left_adj if p[1] == name)
    ax_top.annotate(f'{y_orig:.3f}', xy=(x[0], y_orig),
                    xytext=(x[0] - 0.12, y_label),
                    textcoords='data', va='center', ha='right',
                    fontsize=8, color=color, alpha=0.9)

ax_bot.set_xticks(x)
ax_bot.set_xticklabels(x_labels, fontsize=11)
ax_bot.set_xlim(-0.45, len(x_vals) - 0.4)

fig.text(0.04, 0.55, 'Scenario Diversity (cluster cosine)',
         va='center', rotation='vertical', fontsize=11)
ax_top.set_title('Zoomed line plot with direct agent labels', fontsize=12, pad=12)
plt.subplots_adjust(left=0.09, right=0.96, top=0.93, bottom=0.08)

plt.savefig('scenario_diversity_plot.png', dpi=200, bbox_inches='tight')
plt.show()