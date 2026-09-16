def calculate_map3(preds, targets):
    scores = []
    for pred_str, true_ans in zip(preds, targets):
        pl = pred_str.split()
        scores.append(1.0 / (pl.index(true_ans) + 1) if true_ans in pl else 0.0)
    return float(np.mean(scores))


val_scratch_probs, val_ce_probs, val_llm_probs = [], [], []
for _, row in val_df.iterrows():
    options_text = {o: row[o] for o in OPTIONS}
    ctx = combined_context(row["prompt"])
    val_scratch_probs.append(scratch_score_row(row["prompt"], options_text, ctx))
    val_ce_probs.append(ce_score_row(row["prompt"], options_text, ctx))
    val_llm_probs.append(llm_score_row_tuned(row["prompt"], options_text, ctx))

# Report each model solo, for reference, before blending
val_targets = val_df["answer"].tolist()
scratch_preds = [" ".join(OPTIONS[i] for i in np.argsort(p)[::-1][:3]) for p in val_scratch_probs]
scratch_map3 = calculate_map3(scratch_preds, val_targets)

ce_preds = [" ".join(OPTIONS[i] for i in np.argsort(p)[::-1][:3]) for p in val_ce_probs]
ce_map3 = calculate_map3(ce_preds, val_targets)

llm_preds = [" ".join(OPTIONS[i] for i in np.argsort(p)[::-1][:3]) for p in val_llm_probs]
llm_map3 = calculate_map3(llm_preds, val_targets)


print(f"{'from-scratch':>13s} solo MAP@3: {scratch_map3:.4f}")
print(f"{'cross-encoder':>13s} solo MAP@3: {ce_map3:.4f}")
print(f"{'LLM':>13s} solo MAP@3: {llm_map3:.4f}")

ensemble_preds_top1 = []

def preds_from_probs(scratch_list, ce_list, llm_list, w_scratch, w_ce, w_llm):
    preds = []
    for s_p, ce_p, llm_p in zip(scratch_list, ce_list, llm_list):
        blended = w_scratch * s_p + w_ce * ce_p + w_llm * llm_p
        ranked = np.argsort(blended)[::-1][:3]
        preds.append(" ".join(OPTIONS[i] for i in ranked))
    return preds

# Grid search over a weight simplex (step of 0.1: w_scratch + w_ce + w_llm == 1)
best_weights, best_map3 = (1/3, 1/3, 1/3), -1.0
step = 0.1
grid = np.round(np.arange(0.0, 1.0001, step), 2)
for w_s in grid:
    for w_c in grid:
        w_l = round(1.0 - w_s - w_c, 2)
        if w_l < 0 or w_l > 1.0:
            continue
        preds = preds_from_probs(val_scratch_probs, val_ce_probs, val_llm_probs, w_s, w_c, w_l)
        m = calculate_map3(preds, val_targets)
        if m > best_map3:
            best_map3, best_weights = m, (w_s, w_c, w_l)

best_w_scratch, best_w_ce, best_w_llm = best_weights
print(f"\nBest weights — scratch={best_w_scratch:.2f}, ce={best_w_ce:.2f}, llm={best_w_llm:.2f}")
print(f"Held-out val MAP@3 with best blend: {best_map3:.4f}")



#Finally GENERATING PREDICTIONS

test_preds = []
for _, row in test_df.iterrows():
    options_text = {o: row[o] for o in OPTIONS}
    ctx = combined_context(row["prompt"])
    s_p = scratch_score_row(row["prompt"], options_text, ctx)
    ce_p = ce_score_row(row["prompt"], options_text, ctx)
    llm_p = llm_score_row(row["prompt"], options_text, ctx)
    blended = best_w_scratch * s_p + best_w_ce * ce_p + best_w_llm * llm_p
    ranked = np.argsort(blended)[::-1][:3]
    test_preds.append(" ".join(OPTIONS[i] for i in ranked))

submission_df = pd.DataFrame({"ID": test_df["id"], "Prediction": test_preds})
submission_df.to_csv("submission.csv", index=False)
submission_df.head(10)
