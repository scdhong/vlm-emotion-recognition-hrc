if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import pymc as pm
    import arviz as az

    def col_letter_to_index(letter):
        letter = letter.upper().strip()
        idx = 0
        for ch in letter:
            idx = idx * 26 + (ord(ch) - ord('A') + 1)
        return idx - 1

    # ---------- USER SETTINGS ----------
    infile = "Emotion recognition in human robot collaboration (Responses).xlsx"
    outfile = "bayesian_godspeed_results.xlsx"
    draws = 2000
    tune = 1000
    chains = 4
    # ---------- end settings ----------

    # Column indices
    c_participant = col_letter_to_index('E')
    pre_start, pre_end = col_letter_to_index('K'), col_letter_to_index('AH')
    control_start, control_end = col_letter_to_index('AK'), col_letter_to_index('BH')
    part3_start, part3_end = col_letter_to_index('BT'), col_letter_to_index('CQ')
    part4_start, part4_end = col_letter_to_index('DC'), col_letter_to_index('DZ')

    # Subscale slices
    subscale_slices = {
        'Anthropomorphism': (0, 4),
        'Animacy': (5, 9),
        'Likeability': (10, 15),
        'Intelligence': (16, 20),
        'Safety': (21, 23),
    }

    # Load data
    df = pd.read_excel(infile, sheet_name=0, engine="openpyxl").apply(pd.to_numeric, errors='coerce')
    participant_ids = df.iloc[:, c_participant].astype(str).str.strip()

    def extract_block_means(df, start, end):
        block = df.iloc[:, start:end+1].apply(pd.to_numeric, errors='coerce')
        out = {}
        for name, (s, e) in subscale_slices.items():
            out[name] = block.iloc[:, s:e+1].mean(axis=1)
        return pd.DataFrame(out)

    pre_df = extract_block_means(df, pre_start, pre_end)
    ctrl_df = extract_block_means(df, control_start, control_end)
    p3_df = extract_block_means(df, part3_start, part3_end)
    p4_df = extract_block_means(df, part4_start, part4_end)

    # Build records
    records = []
    for pid, pre, ctrl, p3, p4 in zip(participant_ids, pre_df.to_dict('records'),
                                      ctrl_df.to_dict('records'),
                                      p3_df.to_dict('records'),
                                      p4_df.to_dict('records')):
        order = 'A' if pid.endswith('A') else 'B'
        part3_cond, part4_cond = ('NE', 'EA') if order == 'A' else ('EA', 'NE')
        for cond, block in [('Control', ctrl), (part3_cond, p3), (part4_cond, p4)]:
            rec = dict(block)
            rec['Condition'] = cond
            rec['OrderGroup'] = order
            rec['ParticipantID'] = pid
            for sub in subscale_slices.keys():
                rec[f'Baseline_{sub}'] = pre[sub]
            records.append(rec)

    long_df = pd.DataFrame(records)
    long_df['Condition'] = pd.Categorical(long_df['Condition'], categories=['Control','NE','EA'])
    long_df['OrderGroup'] = pd.Categorical(long_df['OrderGroup'])

    out_rows = []

    for sub in subscale_slices.keys():
        print(f"Running Bayesian ANCOVA for {sub} ...")
        data = long_df[[sub, 'Condition', 'OrderGroup', f'Baseline_{sub}', 'ParticipantID']].dropna()
        y = data[sub].values
        subj = pd.Categorical(data['ParticipantID']).codes
        Nsubj = len(np.unique(subj))

        cond = pd.Categorical(data['Condition'], categories=['Control','NE','EA'])
        cond_NE = (cond == 'NE').astype(int)
        cond_EA = (cond == 'EA').astype(int)
        order = pd.Categorical(data['OrderGroup'], categories=['A','B'])
        order_B = (order == 'B').astype(int)
        baseline = data[f'Baseline_{sub}'].values

        with pm.Model() as model:
            mu_a = pm.Normal('mu_a', 0, 1)
            sigma_a = pm.Exponential('sigma_a', 1)
            a_subj = pm.Normal('a_subj', mu=mu_a, sigma=sigma_a, shape=Nsubj)

            beta_NE = pm.Normal('beta_NE', 0, 0.5)
            beta_EA = pm.Normal('beta_EA', 0, 0.5)
            beta_order = pm.Normal('beta_order', 0, 0.5)
            beta_base = pm.Normal('beta_base', 0, 0.5)
            sigma = pm.Exponential('sigma', 1)

            mu = a_subj[subj] + beta_NE*cond_NE + beta_EA*cond_EA + beta_order*order_B + beta_base*baseline
            pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

            trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.9,
                              progressbar=True, return_inferencedata=True)

        post = trace.posterior
        b_NE = post['beta_NE'].values.reshape(-1)
        b_EA = post['beta_EA'].values.reshape(-1)
        diff_EA_NE = b_EA - b_NE
        diff_EA_Control = b_EA
        diff_NE_Control = b_NE

        def summarize(samples):
            mean = samples.mean()
            hdi = az.hdi(samples, hdi_prob=0.95)
            p_gt0 = (samples > 0).mean()
            return mean, float(hdi[0]), float(hdi[1]), float(p_gt0)

        for cname, s in [("EA - NE", diff_EA_NE), ("EA - Control", diff_EA_Control), ("NE - Control", diff_NE_Control)]:
            mean, lo, hi, p = summarize(s)
            out_rows.append({'Subscale': sub, 'Contrast': cname,
                             'Mean': mean, '95% HDI Lower': lo, '95% HDI Upper': hi, 'P(>0)': p})

    pd.DataFrame(out_rows).to_excel(outfile, index=False)
    print(f"✅ Saved Bayesian Godspeed results to {outfile}")
