import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

sla = {'CRITICAL':350000,'PERFORMANCE':300000,'BUSINESS':20000000}
order = ['CRITICAL','PERFORMANCE','BUSINESS']
colors = {'CRITICAL':'#4a90d9','PERFORMANCE':'#7fb069','BUSINESS':'#e8743b'}

df = pd.read_csv('C/scen_c_full12_20260603_1445.csv')
df['ue_label'] = df['e2_node'].str[-1] + '/f' + df['f1ap'].astype(str)
df['target'] = df['slice_name'].map(sla)
df['violated'] = df['dl_brate_bps'] < df['target']
df['gap_pct'] = ((df['target']-df['dl_brate_bps'])/df['target']*100).clip(lower=0)

ue_order = []
for s in order:
    ue_order += sorted(df.loc[df['slice_name']==s,'ue_label'].unique())

slices = df.groupby('ue_label')['slice_name'].first().reindex(ue_order)
bar_colors = [colors[s] for s in slices]
handles = [Patch(color=colors[s], label=s) for s in order]
sep = [i for i in range(1,len(ue_order)) if slices.iloc[i]!=slices.iloc[i-1]]

# ---- Panel A: violation rate per UE ----
fig, ax = plt.subplots(figsize=(12,5))
rate = df.groupby('ue_label')['violated'].mean().reindex(ue_order)*100
ax.bar(range(len(ue_order)), rate.values, color=bar_colors)
ax.set_xticks(range(len(ue_order))); ax.set_xticklabels(ue_order, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Violation rate (%)')
ax.set_title('Scenario C — SLA Violation Rate per UE')
ax.legend(handles=handles)
for b in sep: ax.axvline(b-0.5, color='gray', linestyle=':', linewidth=0.8)
plt.tight_layout(); plt.savefig('C_violation_rate_per_ue.png', dpi=150)

# ---- Panel B: violation gap distribution per slice (boxplot) ----
fig, ax = plt.subplots(figsize=(7,5))
data = [df.loc[(df['slice_name']==s)&df['violated'],'gap_pct'] for s in order]
bp = ax.boxplot(data, labels=order, patch_artist=True, showmeans=True)
for patch,s in zip(bp['boxes'],order): patch.set_facecolor(colors[s])
ax.set_ylabel('Violation gap (%)')
ax.set_title('Scenario C — SLA Violation Gap by Slice (all violating samples)')
plt.tight_layout(); plt.savefig('C_violation_gap.png', dpi=150)

# ---- Panel C: DL + UL throughput vs SLA target per UE ----
fig, ax = plt.subplots(figsize=(14,6))
x = np.arange(len(ue_order))
dl_means = df.groupby('ue_label')['dl_brate_bps'].mean().reindex(ue_order)
ul_means = df.groupby('ue_label')['ul_brate_bps'].mean().reindex(ue_order)

ax.bar(x-0.2, dl_means.values, width=0.4, label='DL (dl_brate_bps)', color=bar_colors)
ax.bar(x+0.2, ul_means.values, width=0.4, label='UL (ul_brate_bps)', color=bar_colors, alpha=0.45, hatch='//')

for s in order:
    idx = [i for i,sl in enumerate(slices) if sl==s]
    ax.hlines(sla[s], min(idx)-0.4, max(idx)+0.4, color='red', linestyle='--', linewidth=1.5)
    ax.text(np.mean(idx), sla[s]*1.1, f'DL SLA {sla[s]:,}', ha='center', fontsize=8, color='red')

ax.set_xticks(x); ax.set_xticklabels(ue_order, rotation=45, ha='right', fontsize=8)
ax.set_yscale('log')
ax.set_ylabel('Average throughput (bps)')
ax.set_title('Scenario C — DL vs UL Throughput per UE (vs DL SLA Target)')

# legend: slices + DL/UL hatch meaning + SLA line
legend_handles = handles + [
    Patch(facecolor='gray', label='DL (solid)'),
    Patch(facecolor='gray', alpha=0.45, hatch='//', label='UL (hatched)'),
    plt.Line2D([0],[0], color='red', linestyle='--', label='DL SLA target'),
]
ax.legend(handles=legend_handles, fontsize=8)

# annotate the 3 zero-DL UEs
for label in ['1/f0','1/f1','4/f0']:
    if label in ue_order:
        i = ue_order.index(label)
        ax.annotate('DL=0', xy=(i-0.2, 1), xytext=(i-0.2, 5e6),
                     ha='center', fontsize=8, color='red',
                     arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

for b in sep: ax.axvline(b-0.5, color='gray', linestyle=':', linewidth=0.8)
plt.tight_layout(); plt.savefig('C_dl_ul_vs_sla_per_ue.png', dpi=150)

print("saved: C_violation_rate_per_ue.png, C_violation_gap.png, C_dl_ul_vs_sla_per_ue.png")