import sys
import pandas as pd
import numpy as np
from itertools import combinations
from statsmodels.stats.multitest import multipletests
from scipy import stats

# ---------- USER INPUT ----------
if len(sys.argv) > 1:
    infile = sys.argv[1]
else:
    infile = "Emotion recognition in human robot collaboration (Responses).xlsx"  # change to your file path

outfile = "HRC_Friedman_results.xlsx"

# ---------- Helper: Excel column letter -> zero-based index ----------
def col_letter_to_index(letter):
    letter = letter.upper().strip()
    idx = 0
    for ch in letter:
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1

# ---------- Column positions ----------
c_participant = col_letter_to_index('E')  # Participant ID
control_start, control_end = col_letter_to_index('BI'), col_letter_to_index('BO')
part2_start, part2_end     = col_letter_to_index('CR'), col_letter_to_index('CX')
part3_start, part3_end     = col_letter_to_index('EA'), col_letter_to_index('EG')

# ---------- Subscale mapping ----------
subscale_slices = {
    'Fluency': (0, 1),
    'Trust': (2, 3),
    'WorkingAlliance': (4, 6)
}

# ---------- Load file ----------
if infile.lower().endswith('.csv'):
    df = pd.read_csv(infile, dtype=str)
else:
    df = pd.read_excel(infile, sheet_name=0, dtype=str, engine="openpyxl")

df = df.apply(pd.to_numeric, errors='ignore')
participant_ids = df.iloc[:, c_participant].astype(str).str.strip()

# ---------- Extract HRC subscale means ----------
def extract_block_means(df, start_col, end_col):
    block = df.iloc[:, start_col:end_col+1].copy()
    block = block.apply(pd.to_numeric, errors='coerce')
    results = {}
    for name, (s, e) in subscale_slices.items():
        results[name] = block.iloc[:, s:e+1].mean(axis=1, skipna=True)
    return pd.DataFrame(results)

control_df = extract_block_means(df, control_start, control_end)
part2_df   = extract_block_means(df, part2_start, part2_end)
part3_df   = extract_block_means(df, part3_start, part3_end)

# Attach participant IDs
for d in (control_df, part2_df, part3_df):
    d['ParticipantID'] = participant_ids.values

# ---------- Build long dataframe ----------
records = []
for pid, ctrl_row, p2_row, p3_row in zip(participant_ids,
                                         control_df.to_dict('records'),
                                         part2_df.to_dict('records'),
                                         part3_df.to_dict('records')):
    order_group = 'A' if pid.endswith('A') else ('B' if pid.endswith('B') else 'Unknown')
    if order_group == 'A':
        part2_cond, part3_cond = 'NE', 'EA'
    elif order_group == 'B':
        part2_cond, part3_cond = 'EA', 'NE'
    else:
        part2_cond, part3_cond = 'Part2', 'Part3'

    for cond, row in [('Control', ctrl_row), (part2_cond, p2_row), (part3_cond, p3_row)]:
        r = row.copy()
        r.update({
            'ParticipantID': pid,
            'Condition': cond,
            'OrderGroup': order_group,
        })
        records.append(r)

long_df = pd.DataFrame(records)
long_df['Condition'] = pd.Categorical(long_df['Condition'], categories=['Control', 'NE', 'EA'], ordered=False)
long_df['OrderGroup'] = pd.Categorical(long_df['OrderGroup'])

# ---------- Run Friedman test + Wilcoxon post-hoc ----------
writer = pd.ExcelWriter(outfile, engine='openpyxl')
long_df.to_excel(writer, sheet_name="LongData", index=False)

for sub in subscale_slices.keys():
    model_df = long_df[[sub, 'Condition', 'ParticipantID']].dropna()
    
    # Pivot to wide format for Friedman test
    wide = model_df.pivot(index='ParticipantID', columns='Condition', values=sub)
    wide = wide[['Control', 'NE', 'EA']]  # ensure order
    wide = wide.dropna()

    # ---------- Friedman test ----------
    friedman_stat, friedman_p = stats.friedmanchisquare(wide['Control'], wide['NE'], wide['EA'])
    friedman_df = pd.DataFrame({
        'Subscale': [sub],
        'Statistic': [friedman_stat],
        'p_value': [friedman_p]
    })
    friedman_df.to_excel(writer, sheet_name=f"{sub}_Friedman", index=False)

    # ---------- Post-hoc Wilcoxon tests with Holm correction ----------
    results = []
    conditions = ['Control', 'NE', 'EA']
    for a, b in combinations(conditions, 2):
        vals_a = wide[a]
        vals_b = wide[b]
        stat, p = stats.wilcoxon(vals_a, vals_b, zero_method='wilcox', correction=False)
        diff = vals_a.mean() - vals_b.mean()
        results.append((f"{a} vs {b}", diff, stat, p))

    # Holm correction
    pvals = [r[3] for r in results]
    reject, pvals_corr, _, _ = multipletests(pvals, method='holm')

    posthoc_df = pd.DataFrame({
        'Comparison': [r[0] for r in results],
        'Mean_Diff': [r[1] for r in results],
        'Wilcoxon_stat': [r[2] for r in results],
        'p_raw': [r[3] for r in results],
        'p_adj': pvals_corr,
        'Significant': reject
    })
    posthoc_df.to_excel(writer, sheet_name=f"{sub}_PostHoc", index=False)

writer.close()
print(f"\n✅ Done. Non-parametric HRC results saved to {outfile}")
