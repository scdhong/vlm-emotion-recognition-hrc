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
    outfile = "raw_bayesian_godspeed_results.xlsx"
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

        # =========================================================================
        # THIS IS THE SECTION THAT WAS CHANGED
        # =========================================================================
        post = trace.posterior
        
        # Extract posterior samples for all relevant parameters
        mu_a_samps = post['mu_a'].values.reshape(-1)
        b_NE_samps = post['beta_NE'].values.reshape(-1)
        b_EA_samps = post['beta_EA'].values.reshape(-1)
        b_base_samps = post['beta_base'].values.reshape(-1)

        # Calculate the mean of the baseline covariate to get adjusted means
        avg_baseline = baseline.mean()
        
        # Calculate the effect of the average baseline score
        adj_baseline_effect = b_base_samps * avg_baseline

        # Calculate the posterior distribution for the estimated mean of each condition
        # These are the means adjusted for an average baseline score.
        est_mean_Control = mu_a_samps + adj_baseline_effect
        est_mean_NE = mu_a_samps + b_NE_samps + adj_baseline_effect
        est_mean_EA = mu_a_samps + b_EA_samps + adj_baseline_effect

        def summarize(samples):
            """Calculates mean, 95% HDI, and probability of being > 0."""
            mean = samples.mean()
            hdi = az.hdi(samples, hdi_prob=0.95)
            p_gt0 = (samples > 0).mean()
            return mean, float(hdi[0]), float(hdi[1]), float(p_gt0)

        # Summarize the estimated mean for each condition
        for cond_name, samples in [("Control", est_mean_Control),
                                   ("NE", est_mean_NE),
                                   ("EA", est_mean_EA)]:
            mean, lo, hi, p = summarize(samples)
            out_rows.append({'Subscale': sub, 'Condition': cond_name,
                             'Mean': mean, '95% HDI Lower': lo, '95% HDI Upper': hi, 'P(>0)': p})
        # =========================================================================
        # END OF CHANGED SECTION
        # =========================================================================

    pd.DataFrame(out_rows).to_excel(outfile, index=False)
    print(f"✅ Saved Bayesian Godspeed results to {outfile}")