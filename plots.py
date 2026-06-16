import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

DATASETS = {
    'A_10MHz': 'A_10MHz/scen_A_20260526_1646.csv',
    'A_5MHz':  'A_5MHz/scenario_A_3physical_UEs_low_density_5MHz.csv',
    'B':       'B/scen_B_20260601_1110.csv',
    'C':       'C/scen_c_full12_20260603_1445.csv',
}

ORDER  = ['CRITICAL', 'PERFORMANCE', 'BUSINESS']
COLORS = {'CRITICAL': '#4a90d9', 'PERFORMANCE': '#7fb069', 'BUSINESS': '#e8743b'}
SLA    = {'CRITICAL': 350_000, 'PERFORMANCE': 300_000, 'BUSINESS': 20_000_000}
DEAD   = {'C': ['CRITICAL·1/f0', 'PERFORMANCE·1/f1', 'CRITICAL·4/f0'],
          'B': ['CRITICAL·virt_61441']}

def label_fn(name, df):
    if name == 'C':
        base = df['e2_node'].str[-1] + '/f' + df['f1ap'].astype(str)
    elif name == 'B':
        base = 'virt_' + df['rnti'].astype(str)
    else:
        base = 'f' + df['f1ap_id'].astype(str)
    return df['slice_name'] + '·' + base

def violated(row):
    s = row['slice_name']
    d = row['dl_brate_bps']
    if s == 'CRITICAL':    return d < 350_000
    if s == 'PERFORMANCE': return d < 300_000
    if s == 'BUSINESS':    return d < 20_000_000
    return False

def gap_pct(row):
    s = row['slice_name']
    d = row['dl_brate_bps']
    if s == 'CRITICAL':    return max(0, (350_000    - d) / 350_000    * 100)
    if s == 'PERFORMANCE': return max(0, (300_000    - d) / 300_000    * 100)
    if s == 'BUSINESS':    return max(0, (20_000_000 - d) / 20_000_000 * 100)
    return 0

for name, path in DATASETS.items():
    df = pd.read_csv(path)
    df['ue_label'] = label_fn(name, df)
    df['violated'] = df.apply(violated, axis=1)
    df['gap_pct']  = df.apply(gap_pct,  axis=1)

    ue_order = []
    for s in ORDER:
        ue_order += sorted(df.loc[df['slice_name'] == s, 'ue_label'].unique())

    slices     = df.groupby('ue_label')['slice_name'].first().reindex(ue_order)
    bar_colors = [COLORS[s] for s in slices]
    handles    = [Patch(color=COLORS[s], label=s) for s in ORDER]
    sep        = [i for i in range(1, len(ue_order)) if slices.iloc[i] != slices.iloc[i-1]]

    os.makedirs('plots', exist_ok=True)

    # violation rate
    fig, ax = plt.subplots(figsize=(max(8, len(ue_order) * 1.1), 5))
    rate = df.groupby('ue_label')['violated'].mean().reindex(ue_order) * 100
    bars = ax.bar(range(len(ue_order)), rate.values, color=bar_colors, width=0.6, zorder=3)
    for bar, val in zip(bars, rate.values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(ue_order)))
    ax.set_xticklabels(ue_order, rotation=40, ha='right', fontsize=8)
    ax.set_ylabel('SLA Violation Rate (%)', fontsize=10)
    ax.set_xlabel('UE', fontsize=10)
    ax.set_title(f'Scenario {name} : SLA Violation Rate per UE', fontsize=12, fontweight='bold', pad=12)
    ax.legend(handles=handles, framealpha=0.5)
    ax.set_ylim(0, 115)
    for b in sep: ax.axvline(b - 0.5, color='gray', linestyle=':', linewidth=1)
    plt.tight_layout()
    plt.savefig(f'plots/{name}_violation_rate.png', dpi=150)
    plt.close()

    # violation gap
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [df.loc[(df['slice_name'] == s) & df['violated'], 'gap_pct'] for s in ORDER]
    data = [d if len(d) > 0 else pd.Series([0]) for d in data]
    bp = ax.boxplot(data, labels=ORDER, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markersize=6),
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, s in zip(bp['boxes'], ORDER):
        patch.set_facecolor(COLORS[s]); patch.set_alpha(0.75)
    ax.set_ylabel('Violation Gap (%)', fontsize=10)
    ax.set_xlabel('Slice', fontsize=10)
    ax.set_title(f'Scenario {name} : SLA Violation Gap by Slice', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(f'plots/{name}_violation_gap.png', dpi=150)
    plt.close()

    # DL + UL vs SLA
    fig, ax = plt.subplots(figsize=(max(10, len(ue_order) * 1.2), 6))
    x        = np.arange(len(ue_order))
    dl_means = df.groupby('ue_label')['dl_brate_bps'].mean().reindex(ue_order)
    ul_means = df.groupby('ue_label')['ul_brate_bps'].mean().reindex(ue_order)
    ax.bar(x - 0.2, dl_means.values, width=0.38, color=bar_colors, zorder=3)
    ax.bar(x + 0.2, ul_means.values, width=0.38, color=bar_colors, alpha=0.4, hatch='//', zorder=3)
    for s in ORDER:
        idx = [i for i, sl in enumerate(slices) if sl == s]
        if idx:
            ax.hlines(SLA[s], min(idx) - 0.45, max(idx) + 0.45,
                      colors='red', linestyles='--', linewidth=1.5, zorder=4)
            ax.text(np.mean(idx), SLA[s] * 1.3, f'min {SLA[s]//1000}k',
                    ha='center', fontsize=8, color='red', fontweight='bold')
    for dead in DEAD.get(name, []):
        if dead in ue_order:
            i = ue_order.index(dead)
            ax.annotate('DL=0', xy=(i - 0.2, 2), xytext=(i - 0.2, 500_000),
                        ha='center', fontsize=8, color='red',
                        arrowprops=dict(arrowstyle='->', color='red', lw=1))
    ax.set_xticks(x)
    ax.set_xticklabels(ue_order, rotation=40, ha='right', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel('Average Throughput (bps)', fontsize=10)
    ax.set_xlabel('UE', fontsize=10)
    ax.set_title(f'Scenario {name} : DL vs UL Throughput per UE', fontsize=12, fontweight='bold', pad=12)
    legend_handles = handles + [
        Patch(facecolor='gray', label='DL (solid)'),
        Patch(facecolor='gray', alpha=0.4, hatch='//', label='UL (hatched)'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='SLA minimum'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, framealpha=0.5)
    for b in sep: ax.axvline(b - 0.5, color='gray', linestyle=':', linewidth=1)
    plt.tight_layout()
    plt.savefig(f'plots/{name}_dl_ul_vs_sla.png', dpi=150)
    plt.close()

    print(f'{name}: saved violation_rate, violation_gap, dl_ul_vs_sla')