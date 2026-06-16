import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

ORDER  = ['CRITICAL', 'PERFORMANCE', 'BUSINESS']
COLORS = {'CRITICAL': '#4a90d9', 'PERFORMANCE': '#7fb069', 'BUSINESS': '#e8743b'}

DATASETS = {
    'A_10MHz': 'A_10MHz/scen_A_20260526_1646.csv',
    'A_5MHz':  'A_5MHz/scenario_A_3physical_UEs_low_density_5MHz.csv',
    'B':       'B/scen_B_20260601_1110.csv',
    'C':       'C/scen_c_full12_20260603_1445.csv',
}

def label_fn(name, df):
    if name == 'C':
        base = df['e2_node'].str[-1] + '/f' + df['f1ap'].astype(str)
    elif name == 'B':
        base = 'virt_' + df['rnti'].astype(str)
    else:
        base = 'f' + df['f1ap_id'].astype(str)
    return df['slice_name'] + '·' + base

for name, path in DATASETS.items():
    df = pd.read_csv(path)
    df['t']        = df['timestamp'] - df['timestamp'].min()
    df['ue_label'] = label_fn(name, df)
    outdir         = f'plots/{name}'
    os.makedirs(outdir, exist_ok=True)

    # CQI distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in ORDER:
        sub = df.loc[df['slice_name'] == s, 'cqi']
        if len(sub):
            ax.hist(sub, bins=range(0, 17), alpha=0.65, label=s,
                    color=COLORS[s], edgecolor='white', linewidth=0.5)
    ax.set_xlabel('CQI', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'Scenario {name} : CQI Distribution by Slice', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlim(0, 16)
    ax.legend(framealpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{outdir}/{name}_cqi_dist.png', dpi=150)
    plt.close()

    # DL timeseries
    fig, ax = plt.subplots(figsize=(13, 5))
    for ue, g in df.groupby('ue_label'):
        s = g['slice_name'].iloc[0]
        g = g.sort_values('t')
        ax.plot(g['t'], g['dl_brate_bps'], color=COLORS[s],
                alpha=0.75, linewidth=0.9, label=ue)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('DL Throughput (bps)', fontsize=10)
    ax.set_title(f'Scenario {name} : DL Throughput Over Time', fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=7, ncol=3, framealpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{outdir}/{name}_dl_timeseries.png', dpi=150)
    plt.close()

    # UL timeseries
    fig, ax = plt.subplots(figsize=(13, 5))
    for ue, g in df.groupby('ue_label'):
        s = g['slice_name'].iloc[0]
        g = g.sort_values('t')
        ax.plot(g['t'], g['ul_brate_bps'], color=COLORS[s],
                alpha=0.75, linewidth=0.9, label=ue)
    ax.set_xlabel('Time (s)', fontsize=10)
    ax.set_ylabel('UL Throughput (bps)', fontsize=10)
    ax.set_title(f'Scenario {name} : UL Throughput Over Time', fontsize=12, fontweight='bold', pad=12)
    ax.legend(fontsize=7, ncol=3, framealpha=0.4)
    plt.tight_layout()
    plt.savefig(f'{outdir}/{name}_ul_timeseries.png', dpi=150)
    plt.close()

    # DL MCS distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in ORDER:
        sub = df.loc[df['slice_name'] == s, 'dl_mcs']
        if len(sub):
            ax.hist(sub, bins=range(0, 30), alpha=0.65, label=s,
                    color=COLORS[s], edgecolor='white', linewidth=0.5)
    ax.set_xlabel('DL MCS', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'Scenario {name} : DL MCS Distribution by Slice', fontsize=12, fontweight='bold', pad=12)
    ax.legend(framealpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{outdir}/{name}_dl_mcs_dist.png', dpi=150)
    plt.close()

    print(f'{name}: saved cqi_dist, dl_timeseries, ul_timeseries, dl_mcs_dist')