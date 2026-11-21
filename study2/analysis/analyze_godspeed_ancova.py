import sys
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
import itertools

# ---------- USER INPUT ----------
if len(sys.argv) > 1:
    infile = sys.argv[1]
else:
    infile = "Emotion recognition in human robot collaboration (Responses).xlsx"  # change to your file path

outfile = "ancova_results.xlsx"

# ---------- Helper: Excel column letter -> zero-based index ----------
def col_letter_to_index(letter):
    letter = letter.upper().strip()
    idx = 0
    for ch in letter:
        idx = idx * 26 + (ord(ch) - ord('A') + 1)
    return idx - 1

# ---------- Column positions ----------
c_participant = col_letter_to_index('E')
pre_start, pre_end = col_letter_to_index('K'), col_letter_to_index('AH')
control_start, control_end = col_letter_to_index('AK'), col_letter_to_index('BH')
part3_start, part3_end = col_letter_to_index('BT'), col_letter_to_index('CQ')
part4_start, part4_end = col_letter_to_index('DC'), col_letter_to_index('DZ')

# ---------- Load file ----------
if infile.lower().endswith('.csv'):
    df = pd.read_csv(infile, dtype=str)
else:
    df = pd.read_excel(infile, sheet_name=0, dtype=str, engine="openpyxl")

df = df.apply(pd.to_numeric, errors='ignore')
participant_ids = df.iloc[:, c_participant].astype(str).str.strip()

# ---------- Subscale mapping ----------
subscale_slices = {
    'Anthropomorphism': (0, 4),
    'Animacy': (5, 9),
    'Likeability': (10, 15),
    'Intelligence': (16, 20),
    'Safety': (21, 23),
}

def extract_block_means(df, start_col, end_col):
    block = df.iloc[:, start_col:end_col+1].copy()
    block = block.apply(pd.to_numeric, errors='coerce')
    results = {}
    for name, (s, e) in subscale_slices.items():
        results[name] = block.iloc[:, s:e+1].mean(axis=1, skipna=True)
    return pd.DataFrame(results)

# Compute subscale means for each part
pre_df = extract_block_means(df, pre_start, pre_end)
control_df = extract_block_means(df, control_start, control_end)
part3_df = extract_block_means(df, part3_start, part3_end)
part4_df = extract_block_means(df, part4_start, part4_end)

# Attach participant IDs
for d in (pre_df, control_df, part3_df, part4_df):
    d['ParticipantID'] = participant_ids.values

# ---------- Build long dataframe ----------
records = []
for pid, pre_row, ctrl_row, p3_row, p4_row in zip(participant_ids,
                                                 pre_df.to_dict('records'),
                                                 control_df.to_dict('records'),
                                                 part3_df.to_dict('records'),
                                                 part4_df.to_dict('records')):
    order_group = 'A' if pid.endswith('A') else ('B' if pid.endswith('B') else 'Unknown')
    if order_group == 'A':
        part3_cond, part4_cond = 'NE', 'EA'
    elif order_group == 'B':
        part3_cond, part4_cond = 'EA', 'NE'
    else:
        part3_cond, part4_cond = 'Part3', 'Part4'

    baseline = pre_row.copy()

    for cond, row in [('Control', ctrl_row), (part3_cond, p3_row), (part4_cond, p4_row)]:
        r = row.copy()
        r.update({
            'ParticipantID': pid,
            'Condition': cond,
            'OrderGroup': order_group,
        })
        for sub in subscale_slices.keys():
            r[f'Baseline_{sub}'] = baseline[sub]
        records.append(r)

long_df = pd.DataFrame(records)
long_df['Condition'] = pd.Categorical(long_df['Condition'], categories=['Control','NE','EA'], ordered=False)
long_df['OrderGroup'] = pd.Categorical(long_df['OrderGroup'])

# ---------- Run ANCOVA + Post-hoc, save results ----------
writer = pd.ExcelWriter(outfile, engine='openpyxl')
long_df.to_excel(writer, sheet_name="LongData", index=False)

for sub in subscale_slices.keys():
    baseline_col = 'Baseline_' + sub
    model_df = long_df[[sub, 'Condition', 'OrderGroup', baseline_col, 'ParticipantID']].dropna()
    formula = f'Q("{sub}") ~ C(Condition) * C(OrderGroup) + Q("{baseline_col}")'
    ols_model = smf.ols(formula=formula, data=model_df).fit()
    anova_tbl = anova_lm(ols_model, typ=2)

    # Save ANCOVA table
    anova_df = anova_tbl.reset_index().rename(columns={'index': 'Effect'})
    anova_df.to_excel(writer, sheet_name=f"{sub}_ANCOVA", index=False)

    # ---- Post-hoc contrasts ----
    conditions = ["Control", "NE", "EA"]
    results = []
    for a, b in itertools.combinations(conditions, 2):
        # Use t_test contrast
        contrast = ols_model.t_test([1 if f"C(Condition)[T.{a}]" in p else
                                     -1 if f"C(Condition)[T.{b}]" in p else 0
                                     for p in ols_model.params.index])
        results.append((f"{a} vs {b}",
                        float(contrast.effect),
                        float(contrast.sd),
                        float(contrast.tvalue),
                        float(contrast.pvalue)))

    # Holm correction
    pvals = [r[4] for r in results]
    reject, pvals_corr, _, _ = multipletests(pvals, method='holm')
    posthoc_df = pd.DataFrame({
        'Comparison': [r[0] for r in results],
        'Diff': [r[1] for r in results],
        'SE': [r[2] for r in results],
        't': [r[3] for r in results],
        'p_raw': [r[4] for r in results],
        'p_adj': pvals_corr,
        'Significant': reject
    })
    posthoc_df.to_excel(writer, sheet_name=f"{sub}_PostHoc", index=False)

writer.close()
print(f"\n✅ Done. Results saved to {outfile}")
