# normality_checks.py
import pandas as pd
import numpy as np
from scipy.stats import shapiro, normaltest
import warnings
import sys

warnings.filterwarnings("ignore")

def col_letter_to_index(letter):
    letter = letter.upper().strip()
    idx = 0
    for ch in letter:
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1

def run_tests_on_series(series):
    """Return (N, shapiro_stat, shapiro_p, dag_stat, dag_p). If too few points, return NaNs."""
    data = pd.Series(series).dropna().astype(float)
    n = len(data)
    if n < 3:
        return n, np.nan, np.nan, np.nan, np.nan
    # Shapiro-Wilk (good for n<=5000)
    try:
        sw_stat, sw_p = shapiro(data)
    except Exception:
        sw_stat, sw_p = np.nan, np.nan
    # D'Agostino's K^2
    try:
        k2_stat, k2_p = normaltest(data)
    except Exception:
        k2_stat, k2_p = np.nan, np.nan
    return n, float(sw_stat), float(sw_p), float(k2_stat), float(k2_p)

def summarize_normality(rows):
    df = pd.DataFrame(rows)
    # mark Normal if both p-values > 0.05
    df['Normal?'] = df.apply(lambda r: 'Yes' if (pd.notna(r['Shapiro_p']) and pd.notna(r['DAgostino_p']) and r['Shapiro_p']>0.05 and r['DAgostino_p']>0.05) else 'No', axis=1)
    return df

def main(infile="Emotion recognition in human robot collaboration (Responses).xlsx",
         outfile="Normality_Results.xlsx"):

    # load
    df = pd.read_excel(infile, sheet_name=0, engine="openpyxl")
    print(f"Loaded {len(df)} rows from {infile}")

    # participant ID column (E)
    c_participant = col_letter_to_index('E')
    participant_ids = df.iloc[:, c_participant].astype(str).str.strip()

    # ---------- Column blocks (zero-based indices) ----------
    # Godspeed: pre(K:AH), control(AK:BH), part3(BT:CQ), part4(DC:DZ)
    pre_start, pre_end       = col_letter_to_index('K'),  col_letter_to_index('AH')
    control_start, control_end = col_letter_to_index('AK'), col_letter_to_index('BH')
    part3_start, part3_end   = col_letter_to_index('BT'),  col_letter_to_index('CQ')
    part4_start, part4_end   = col_letter_to_index('DC'),  col_letter_to_index('DZ')

    # HRC: control(BI:BO), part2(CR:CX), part3(EA:EG)
    hrc_control_start, hrc_control_end = col_letter_to_index('BI'), col_letter_to_index('BO')
    hrc_part2_start, hrc_part2_end     = col_letter_to_index('CR'), col_letter_to_index('CX')
    hrc_part3_start, hrc_part3_end     = col_letter_to_index('EA'), col_letter_to_index('EG')

    # ---------- Subscale slice maps (within each block) ----------
    godspeed_subscales = {
        'Anthropomorphism': (0, 4),
        'Animacy': (5, 9),
        'Likeability': (10, 15),
        'Intelligence': (16, 20),
        'Safety': (21, 23),
    }
    # HRC block is 7 columns per part (Fluency 0-1, Trust 2-3, WorkingAlliance 4-6)
    hrc_subscales = {
        'Fluency': (0, 1),
        'Trust': (2, 3),
        'WorkingAlliance': (4, 6),
    }

    # Helper to extract mean per subscale from a block (start..end inclusive)
    def extract_block_means(df_in, start_col, end_col, subscale_map):
        block = df_in.iloc[:, start_col:end_col+1].apply(pd.to_numeric, errors='coerce')
        out = {}
        for name, (s, e) in subscale_map.items():
            # slice columns within block
            out[name] = block.iloc[:, s:e+1].mean(axis=1, skipna=True)
        return pd.DataFrame(out)

    # ---------- Build Godspeed dataframes (pre, control, part3, part4) ----------
    pre_df = extract_block_means(df, pre_start, pre_end, godspeed_subscales)
    gs_ctrl_df = extract_block_means(df, control_start, control_end, godspeed_subscales)
    gs_p3_df = extract_block_means(df, part3_start, part3_end, godspeed_subscales)
    gs_p4_df = extract_block_means(df, part4_start, part4_end, godspeed_subscales)

    # For Godspeed, need to assign which part is NE/EA depending on participant ID ending A/B
    godspeed_rows = []
    for i, pid in enumerate(participant_ids):
        order_group = 'A' if pid.endswith('A') else ('B' if pid.endswith('B') else 'Unknown')
        # baseline (pre) for this participant
        for sub in godspeed_subscales.keys():
            # baseline test
            baseline_series = pre_df[sub].iloc[i]
            # prepare tests for baseline (store as Condition 'Pre')
            godspeed_rows.append({
                'Subscale': sub,
                'Condition': 'Pre',
                'N': None, 'Shapiro_stat': None, 'Shapiro_p': None,
                'DAgostino_stat': None, 'DAgostino_p': None,
            })

    # Rather than constructing per-row like above, we'll compute programmatically per subscale × condition
    gs_results = []
    for sub in godspeed_subscales.keys():
        # baseline
        series_pre = pre_df[sub]
        n, sw_s, sw_p, k2_s, k2_p = run_tests_on_series(series_pre)
        gs_results.append({
            'Subscale': sub, 'Condition': 'Pre', 'N': n,
            'Shapiro_stat': sw_s, 'Shapiro_p': sw_p,
            'DAgostino_stat': k2_s, 'DAgostino_p': k2_p
        })

        # for block-level conditions we need to map block to NE/EA depending on id suffix
        # We'll form arrays of values per participant for Control, NE, EA
        ctrl_vals = gs_ctrl_df[sub].copy()
        # part3 is block at BT:CQ but may be NE or EA
        p3_vals = gs_p3_df[sub].copy()
        p4_vals = gs_p4_df[sub].copy()

        # create Series for NE and EA by using participant id suffix logic
        ne_vals = []
        ea_vals = []
        for i, pid in enumerate(participant_ids):
            order_group = 'A' if pid.endswith('A') else ('B' if pid.endswith('B') else 'Unknown')
            if order_group == 'A':
                # part3 == NE, part4 == EA
                ne_vals.append(p3_vals.iloc[i])
                ea_vals.append(p4_vals.iloc[i])
            elif order_group == 'B':
                ne_vals.append(p4_vals.iloc[i])
                ea_vals.append(p3_vals.iloc[i])
            else:
                # unknown ordering -> put NaN
                ne_vals.append(np.nan)
                ea_vals.append(np.nan)

        ne_series = pd.Series(ne_vals)
        ea_series = pd.Series(ea_vals)

        # run tests for Control, NE, EA
        for cond_name, series in [('Control', ctrl_vals), ('NE', ne_series), ('EA', ea_series)]:
            n, sw_s, sw_p, k2_s, k2_p = run_tests_on_series(series)
            gs_results.append({
                'Subscale': sub, 'Condition': cond_name, 'N': n,
                'Shapiro_stat': sw_s, 'Shapiro_p': sw_p,
                'DAgostino_stat': k2_s, 'DAgostino_p': k2_p
            })

    # ---------- HRC: extract block means per participant and perform tests ----------
    hrc_ctrl_df = extract_block_means(df, hrc_control_start, hrc_control_end, hrc_subscales)
    hrc_p2_df   = extract_block_means(df, hrc_part2_start, hrc_part2_end, hrc_subscales)
    hrc_p3_df   = extract_block_means(df, hrc_part3_start, hrc_part3_end, hrc_subscales)

    hrc_results = []
    for sub in hrc_subscales.keys():
        # control
        ctrl_series = hrc_ctrl_df[sub]
        n, sw_s, sw_p, k2_s, k2_p = run_tests_on_series(ctrl_series)
        hrc_results.append({
            'Subscale': sub, 'Condition': 'Control', 'N': n,
            'Shapiro_stat': sw_s, 'Shapiro_p': sw_p,
            'DAgostino_stat': k2_s, 'DAgostino_p': k2_p
        })

        # build NE/EA per participant using counterbalance suffix
        p2_vals = hrc_p2_df[sub]
        p3_vals = hrc_p3_df[sub]

        ne_vals = []
        ea_vals = []
        for i, pid in enumerate(participant_ids):
            order_group = 'A' if pid.endswith('A') else ('B' if pid.endswith('B') else 'Unknown')
            if order_group == 'A':
                ne_vals.append(p2_vals.iloc[i])
                ea_vals.append(p3_vals.iloc[i])
            elif order_group == 'B':
                ne_vals.append(p3_vals.iloc[i])
                ea_vals.append(p2_vals.iloc[i])
            else:
                ne_vals.append(np.nan)
                ea_vals.append(np.nan)

        # tests
        for cond_name, series in [('NE', pd.Series(ne_vals)), ('EA', pd.Series(ea_vals))]:
            n, sw_s, sw_p, k2_s, k2_p = run_tests_on_series(series)
            hrc_results.append({
                'Subscale': sub, 'Condition': cond_name, 'N': n,
                'Shapiro_stat': sw_s, 'Shapiro_p': sw_p,
                'DAgostino_stat': k2_s, 'DAgostino_p': k2_p
            })

    # ---------- Summarize and flag normality ----------
    gs_df = pd.DataFrame(gs_results)
    gs_df['Normal?'] = gs_df.apply(lambda r: 'Yes' if (pd.notna(r['Shapiro_p']) and pd.notna(r['DAgostino_p']) and r['Shapiro_p']>0.05 and r['DAgostino_p']>0.05) else 'No', axis=1)

    hrc_df = pd.DataFrame(hrc_results)
    hrc_df['Normal?'] = hrc_df.apply(lambda r: 'Yes' if (pd.notna(r['Shapiro_p']) and pd.notna(r['DAgostino_p']) and r['Shapiro_p']>0.05 and r['DAgostino_p']>0.05) else 'No', axis=1)

    # reorder columns for readability
    cols = ['Subscale','Condition','N','Shapiro_stat','Shapiro_p','DAgostino_stat','DAgostino_p','Normal?']
    gs_df = gs_df[cols]
    hrc_df = hrc_df[cols]

    # ---------- Save to Excel ----------
    with pd.ExcelWriter(outfile, engine='openpyxl') as w:
        gs_df.to_excel(w, sheet_name='Godspeed', index=False)
        hrc_df.to_excel(w, sheet_name='HRC', index=False)

    print(f"Saved normality test results to {outfile}")
    print("\nGodspeed non-normal rows:")
    print(gs_df[gs_df['Normal?']=='No'][['Subscale','Condition','Shapiro_p','DAgostino_p']])
    print("\nHRC non-normal rows:")
    print(hrc_df[hrc_df['Normal?']=='No'][['Subscale','Condition','Shapiro_p','DAgostino_p']])

if __name__ == "__main__":
    infile = "Emotion recognition in human robot collaboration (Responses).xlsx"
    outfile = "Normality_Results.xlsx"
    if len(sys.argv) > 1:
        infile = sys.argv[1]
    if len(sys.argv) > 2:
        outfile = sys.argv[2]
    main(infile, outfile)
