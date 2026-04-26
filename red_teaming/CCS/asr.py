import matplotlib.pyplot as plt
import numpy as np

agents = ['Mail', 'Complaint', 'Internal', 'HR-leave', 'News']
success = np.array([16, 18, 17, 18, 17])
total = 19
rates = success / total

BAR_COLOR = '#3a6a8c'

fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=130)
fig.patch.set_facecolor('white')

bars = ax.bar(agents, rates, color=BAR_COLOR, width=0.62,
              edgecolor='white', linewidth=1.2, zorder=3)

# 값 라벨
for bar, s, r in zip(bars, success, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025,
            f'{s}/{total}\n{r*100:.1f}%',
            ha='center', va='bottom',
            fontsize=9.5, color="#5c758f",
            fontweight='medium', linespacing=1.3)

# 축/스타일
ax.set_ylim(0, 1.15)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_yticklabels([f'{int(v*100)}' for v in np.arange(0, 1.01, 0.2)],
                   fontsize=10, color='#555')
ax.set_xticks(range(len(agents)))
ax.set_xticklabels(agents, fontsize=10.5, color='#2c3e50')

ax.set_ylabel('Attack Success Rate(%)', fontsize=15, color='#2c3e50', labelpad=5)

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#cfd6dc')
ax.spines['bottom'].set_color('#cfd6dc')

ax.yaxis.grid(True, linestyle='--', linewidth=0.6, color='#e5e9ed', zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis='both', length=0)

plt.tight_layout()
plt.savefig('asr_by_agent.png', dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig('asr_by_agent.pdf', dpi=600, bbox_inches='tight', facecolor='white')
plt.show()