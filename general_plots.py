import pandas as pd
import matplotlib.pyplot as plt
import os

colors = {'CRITICAL':'#4a90d9','PERFORMANCE':'#7fb069','BUSINESS':'#e8743b'}
order = ['CRITICAL','PERFORMANCE','BUSINESS']

paths = {
    'A_10MHz': 'A_10MHz/scen_A_20260526_1646.csv',
    'A_5MHz':  'A_5MHz/scenario_A_3physical_UEs_low_density_5MHz.csv',
    'B':       'B/scen_B_20260601_1110.csv',
    'C':       'C/scen_c_full12_20260603_1445.csv',
}

def label_fn(name, df):
    if name == 'C':
        return df['e2_node'].str[-1] + '/f' + df['f1ap'].astype(str)
    elif name == 'B':
        return df['rnti'].apply(lambda r: 'phy_' if r < 0xF001 else 'virt_') + df['rnti'].astype(str)
    else:
        return 'f' + df['f1ap_id'].astype(str)

for name, path in paths.items():
    df = pd.read_csv(path)
    df['t'] = df['timestamp'] - df['timestamp'].min()
    df['ue_label'] = label_fn(name, df)
    outdir = f'plots/{name}'
    os.makedirs(outdir, exist_ok=True)

    # CQI distribution by slice
    fig, ax = plt.subplots(figsize=(8,5))
    for s in order:
        sub = df.loc[df['slice_name']==s,'cqi']
        if len(sub): ax.hist(sub, bins=range(0,17), alpha=0.6, label=s, color=colors[s])
    ax.set_xlabel('CQI'); ax.set_ylabel('Count'); ax.legend()
    ax.set_title(f'{name} — CQI Distribution by Slice')
    plt.tight_layout(); plt.savefig(f'{outdir}/{name}_cqi_dist.png', dpi=150); plt.close()

    # DL throughput over time per UE
    fig, ax = plt.subplots(figsize=(12,5))
    for ue,g in df.groupby('ue_label'):
        s = g['slice_name'].iloc[0]; g=g.sort_values('t')
        ax.plot(g['t'], g['dl_brate_bps'], color=colors[s], alpha=0.7, linewidth=0.8, label=f'{ue} ({s})')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('DL throughput (bps)')
    ax.set_title(f'{name} — DL Throughput Over Time')
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(f'{outdir}/{name}_dl_timeseries.png', dpi=150); plt.close()

    # UL throughput over time per UE
    fig, ax = plt.subplots(figsize=(12,5))
    for ue,g in df.groupby('ue_label'):
        s = g['slice_name'].iloc[0]; g=g.sort_values('t')
        ax.plot(g['t'], g['ul_brate_bps'], color=colors[s], alpha=0.7, linewidth=0.8, label=f'{ue} ({s})')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('UL throughput (bps)')
    ax.set_title(f'{name} — UL Throughput Over Time')
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(f'{outdir}/{name}_ul_timeseries.png', dpi=150); plt.close()

    # dl_mcs distribution by slice
    fig, ax = plt.subplots(figsize=(8,5))
    for s in order:
        sub = df.loc[df['slice_name']==s,'dl_mcs']
        if len(sub): ax.hist(sub, bins=range(0,30), alpha=0.6, label=s, color=colors[s])
    ax.set_xlabel('dl_mcs'); ax.set_ylabel('Count'); ax.legend()
    ax.set_title(f'{name} — DL MCS Distribution by Slice')
    plt.tight_layout(); plt.savefig(f'{outdir}/{name}_dl_mcs_dist.png', dpi=150); plt.close()

    print(f'{name}: saved cqi_dist, dl_timeseries, ul_timeseries, dl_mcs_dist')