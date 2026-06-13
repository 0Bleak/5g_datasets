import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

sla = {'CRITICAL':350000,'PERFORMANCE':300000,'BUSINESS':20000000}
order = ['CRITICAL','PERFORMANCE','BUSINESS']
colors = {'CRITICAL':'#4a90d9','PERFORMANCE':'#7fb069','BUSINESS':'#e8743b'}

def make_plots(df, ue_col, label_prefix, out_prefix):
    df = df.copy()
    df['ue_label'] = label_prefix + df[ue_col].astype(str)
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

    # Panel A: violation rate
    fig, ax = plt.subplots(figsize=(8,5))
    rate = df.groupby('ue_label')['violated'].mean().reindex(ue_order)*100
    ax.bar(range(len(ue_order)), rate.values, color=bar_colors)
    ax.set_xticks(range(len(ue_order))); ax.set_xticklabels(ue_order, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Violation rate (%)')
    ax.set_title(f'{out_prefix} — SLA Violation Rate per UE')
    ax.legend(handles=handles)
    for b in sep: ax.axvline(b-0.5, color='gray', linestyle=':', linewidth=0.8)
    plt.tight_layout(); plt.savefig(f'{out_prefix}_violation_rate_per_ue.png', dpi=150)

    # Panel B: violation gap boxplot
    fig, ax = plt.subplots(figsize=(7,5))
    data = [df.loc[(df['slice_name']==s)&df['violated'],'gap_pct'] for s in order]
    data = [d if len(d)>0 else pd.Series([0]) for d in data]
    bp = ax.boxplot(data, labels=order, patch_artist=True, showmeans=True)
    for patch,s in zip(bp['boxes'],order): patch.set_facecolor(colors[s])
    ax.set_ylabel('Violation gap (%)')
    ax.set_title(f'{out_prefix} — SLA Violation Gap by Slice')
    plt.tight_layout(); plt.savefig(f'{out_prefix}_violation_gap.png', dpi=150)

    # Panel C: DL+UL vs SLA per UE
    fig, ax = plt.subplots(figsize=(10,6))
    x = np.arange(len(ue_order))
    dl_means = df.groupby('ue_label')['dl_brate_bps'].mean().reindex(ue_order)
    ul_means = df.groupby('ue_label')['ul_brate_bps'].mean().reindex(ue_order)
    ax.bar(x-0.2, dl_means.values, width=0.4, label='DL', color=bar_colors)
    ax.bar(x+0.2, ul_means.values, width=0.4, label='UL', color=bar_colors, alpha=0.45, hatch='//')
    for s in order:
        idx = [i for i,sl in enumerate(slices) if sl==s]
        if idx:
            ax.hlines(sla[s], min(idx)-0.4, max(idx)+0.4, color='red', linestyle='--', linewidth=1.5)
            ax.text(np.mean(idx), sla[s]*1.1, f'DL SLA {sla[s]:,}', ha='center', fontsize=8, color='red')
    ax.set_xticks(x); ax.set_xticklabels(ue_order, rotation=45, ha='right', fontsize=8)
    ax.set_yscale('log')
    ax.set_ylabel('Average throughput (bps)')
    ax.set_title(f'{out_prefix} — DL vs UL Throughput per UE (vs DL SLA Target)')
    legend_handles = handles + [
        Patch(facecolor='gray', label='DL (solid)'),
        Patch(facecolor='gray', alpha=0.45, hatch='//', label='UL (hatched)'),
        plt.Line2D([0],[0], color='red', linestyle='--', label='DL SLA target'),
    ]
    ax.legend(handles=legend_handles, fontsize=8)
    for b in sep: ax.axvline(b-0.5, color='gray', linestyle=':', linewidth=0.8)
    plt.tight_layout(); plt.savefig(f'{out_prefix}_dl_ul_vs_sla_per_ue.png', dpi=150)

    print(f"saved 3 plots with prefix {out_prefix}")

# A_10MHz
dfA10 = pd.read_csv('A_10MHz/scen_A_20260526_1646.csv')
make_plots(dfA10, 'f1ap_id', 'f', 'A_10MHz')

# A_5MHz
dfA5 = pd.read_csv('A_5MHz/scenario_A_3physical_UEs_low_density_5MHz.csv')
make_plots(dfA5, 'f1ap_id', 'f', 'A_5MHz')

# B
dfB = pd.read_csv('B/scen_B_20260601_1110.csv')
dfB['type'] = dfB['rnti'].apply(lambda r: 'phy' if r<0xF001 else 'virt')
dfB['ue_id'] = dfB['type'] + '_' + dfB['rnti'].astype(str)
make_plots(dfB, 'ue_id', '', 'B')