#!/usr/bin/env python3
"""Disambiguation failure mode analysis (Section 4.7 paragraphs, Appendix).

Reads: data/analytics/disambig_mobile_final.json, data/analytics/tasks_mobile_gt.json
Outputs: Aspect divergence, impossible task blind spot, consequence overestimation,
         consequence-stratified miss rates, per-category recall, severity calibration.

Usage:
    python code/analytics/disambig_analysis.py
"""

import json
import sys
import os
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, DISAMBIG_MODEL_MAP, FRONTIER, OSS, mean, parse_disambig_key


def parse_key(k):
    raw_model, ptype, task_id = parse_disambig_key(k)
    return DISAMBIG_MODEL_MAP.get(raw_model, raw_model), ptype, task_id


def main():
    with open(DATA_DIR / 'disambig_mobile_final.json') as f:
        disambig = json.load(f)
    with open(DATA_DIR / 'tasks_mobile_gt.json') as f:
        gt_list = json.load(f)

    gt = {t['id']: t for t in gt_list}
    n_tasks = len(gt_list)

    mode2 = defaultdict(dict)
    mode1 = defaultdict(dict)
    for k, v in disambig.items():
        model_name, ptype, task_id = parse_key(k)
        if model_name not in DISAMBIG_MODEL_MAP.values():
            continue
        if ptype in ('WITHOUT_COT_SCORE', 'WITH_SCORE'):
            mode2[model_name][task_id] = v
        elif ptype in ('WITHOUT_COT_NO_SCORE', 'NO_SCORE'):
            mode1[model_name][task_id] = v

    all_models = sorted(mode2.keys())

    print("=" * 80)
    print("COMPREHENSIVE DISAMBIGUATION FAILURE MODE ANALYSIS")
    print("=" * 80)

    # 1. PER-MODEL BASIC STATS
    print("\n" + "=" * 80)
    print("1. PER-MODEL BASIC STATISTICS (Mode 2)")
    print("=" * 80)

    model_stats = {}
    for model in all_models:
        results = mode2[model]
        n = len(results)
        n_clarified = sum(1 for v in results.values() if v.get('gen_clarifications') and len(v['gen_clarifications']) > 0)
        n_errors = sum(1 for v in results.values() if v.get('error_type') is not None)

        p_num, p_den, r_num, r_den = 0, 0, 0, 0
        for tid, v in results.items():
            gt_task = gt.get(tid)
            if not gt_task:
                continue
            gt_clarifs = gt_task.get('clarifications', [])
            gen_clarifs = v.get('gen_clarifications', [])
            n_matched = sum(1 for g in gen_clarifs if g.get('match'))
            p_num += n_matched
            p_den += len(gen_clarifs)
            matched_texts = set()
            for g in gen_clarifs:
                if g.get('match') and g.get('match_ground_text'):
                    matched_texts.add(g['match_ground_text'])
            r_num += len(matched_texts)
            r_den += len(gt_clarifs)

        precision = p_num / p_den if p_den > 0 else 0
        recall = r_num / r_den if r_den > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        delta_values = []
        for tid, v in results.items():
            gt_task = gt.get(tid)
            if not gt_task:
                continue
            for g in v.get('gen_clarifications', []):
                if g.get('match') and g.get('match_ground_text'):
                    for gc in gt_task.get('clarifications', []):
                        if gc['question'] == g['match_ground_text']:
                            delta_values.append(abs(g.get('obviousness_score', 0) - gc['obviousness_score']) + abs(g.get('consequence_score', 0) - gc['consequence_score']))
                            break

        delta = mean(delta_values)
        model_stats[model] = {'n_tasks': n, 'cr': n_clarified / n if n else 0,
                              'precision': precision, 'recall': recall, 'f1': f1, 'delta': delta}

        print(f"\n{model}:")
        print(f"  Tasks: {n}, Clarified: {n_clarified} ({n_clarified / n * 100:.1f}%), Errors: {n_errors}")
        print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
        print(f"  Delta_comp: {delta:.2f} (n={len(delta_values)})")

    # 2. PER-TASK UNIVERSAL ANALYSIS
    print("\n" + "=" * 80)
    print("2. PER-TASK UNIVERSAL FAILURES AND SUCCESSES")
    print("=" * 80)

    task_model_clarified = defaultdict(dict)
    task_model_matched = defaultdict(dict)
    for model in all_models:
        for tid, v in mode2[model].items():
            gen = v.get('gen_clarifications', [])
            task_model_clarified[tid][model] = len(gen) > 0
            matched_texts = set(g['match_ground_text'] for g in gen if g.get('match') and g.get('match_ground_text'))
            task_model_matched[tid][model] = len(matched_texts)

    all_no_clarify = [tid for tid in range(n_tasks) if not any(task_model_clarified.get(tid, {}).get(m, False) for m in all_models)]
    all_no_match = [tid for tid in range(n_tasks) if not any(task_model_matched.get(tid, {}).get(m, 0) > 0 for m in all_models)]
    all_match = [tid for tid in range(n_tasks) if all(task_model_matched.get(tid, {}).get(m, 0) > 0 for m in all_models)]

    print(f"All-silent (no model clarifies): {len(all_no_clarify)} tasks")
    print(f"All-miss (no model matches GT): {len(all_no_match)} tasks")
    print(f"All-hit (every model matches >= 1 GT): {len(all_match)} tasks")

    print(f"\n--- All-Silent Tasks (n={len(all_no_clarify)}) ---")
    cats = Counter()
    obv_vals, con_vals = [], []
    for tid in all_no_clarify:
        t = gt.get(tid)
        if not t:
            continue
        for c in t.get('clarifications', []):
            cats[c.get('type', '?')] += 1
            obv_vals.append(c['obviousness_score'])
            con_vals.append(c['consequence_score'])
    print(f"  Categories: {dict(cats)}")
    if obv_vals:
        print(f"  Avg obviousness: {mean(obv_vals):.2f}, Avg consequence: {mean(con_vals):.2f}")

    print("\n  Sample all-silent tasks:")
    for tid in all_no_clarify[:8]:
        t = gt.get(tid)
        if not t:
            continue
        print(f"    Task {tid}: \"{t['adg']}\"")
        for c in t['clarifications']:
            print(f"      GT: \"{c['question'][:90]}\" obv={c['obviousness_score']} con={c['consequence_score']} type={c.get('type')}")

    miss_but_clarify = [tid for tid in all_no_match if tid not in all_no_clarify]
    print(f"\n--- Clarify-but-miss (models ask questions, none match GT): {len(miss_but_clarify)} tasks ---")
    for tid in miss_but_clarify[:8]:
        t = gt.get(tid)
        if not t:
            continue
        print(f"\n  Task {tid}: \"{t['adg']}\"")
        for c in t['clarifications']:
            print(f"    GT: \"{c['question'][:80]}\" type={c.get('type')}")
        for model in ['Claude 4.7', 'Gemini 3.1', 'GPT-5.4']:
            v = mode2.get(model, {}).get(tid)
            if v and v.get('gen_clarifications'):
                qs = [g['question'][:70] for g in v['gen_clarifications'][:2] if g and g.get('question')]
                print(f"    {model}: {qs}")

    # 3. PER-CATEGORY RECALL
    print("\n" + "=" * 80)
    print("3. PER-CATEGORY RECALL BREAKDOWN")
    print("=" * 80)

    all_cats = set()
    for t in gt_list:
        for c in t.get('clarifications', []):
            all_cats.add(c.get('type', 'unknown'))
    print(f"Categories: {sorted(all_cats)}")

    cat_counts = Counter()
    for t in gt_list:
        for c in t.get('clarifications', []):
            cat_counts[c.get('type', 'unknown')] += 1
    print(f"GT items per category: {dict(sorted(cat_counts.items()))}")

    for model in all_models:
        print(f"\n{model}:")
        for cat in sorted(all_cats):
            r_num, r_den = 0, 0
            for tid, v in mode2[model].items():
                t = gt.get(tid)
                if not t:
                    continue
                cat_clarifs = [c for c in t.get('clarifications', []) if c.get('type') == cat]
                if not cat_clarifs:
                    continue
                gen = v.get('gen_clarifications', [])
                matched_texts = set(g['match_ground_text'] for g in gen if g.get('match') and g.get('match_ground_text'))
                for gc in cat_clarifs:
                    r_den += 1
                    if gc['question'] in matched_texts:
                        r_num += 1
            recall = r_num / r_den if r_den > 0 else 0
            print(f"  {cat}: recall={recall:.3f} ({r_num}/{r_den})")

    # 4. CONSEQUENCE-STRATIFIED
    print("\n" + "=" * 80)
    print("4. CONSEQUENCE-STRATIFIED CLARIFICATION RATE")
    print("=" * 80)

    for con_level in [0, 1, 2]:
        tasks_at_level = [t for t in gt_list if max((c['consequence_score'] for c in t.get('clarifications', [])), default=-1) == con_level]
        print(f"\nConsequence={con_level} ({len(tasks_at_level)} tasks):")
        for model in all_models:
            n_clar, n_match, total = 0, 0, 0
            for t in tasks_at_level:
                v = mode2[model].get(t['id'])
                if not v:
                    continue
                total += 1
                gen = v.get('gen_clarifications', [])
                if len(gen) > 0:
                    n_clar += 1
                if any(g.get('match') for g in gen):
                    n_match += 1
            if total > 0:
                print(f"  {model}: CR={n_clar / total:.3f} Match={n_match / total:.3f} (n={total})")

    # 5. HIGH-CONSEQUENCE FAILURES
    print("\n" + "=" * 80)
    print("5. HIGH-CONSEQUENCE TASKS WHERE NO MODEL MATCHES GT")
    print("=" * 80)

    high_con = [t for t in gt_list if max((c['consequence_score'] for c in t.get('clarifications', [])), default=-1) == 2]
    print(f"Total consequence=2 tasks: {len(high_con)}")

    dangerous = []
    for t in high_con:
        if not any(task_model_matched.get(t['id'], {}).get(m, 0) > 0 for m in all_models):
            dangerous.append(t)
    print(f"Where NO model matches: {len(dangerous)}")

    for t in dangerous[:10]:
        print(f"\n  Task {t['id']}: \"{t['adg']}\"")
        for c in t['clarifications']:
            print(f"    GT: \"{c['question'][:100]}\"")
            print(f"        obv={c['obviousness_score']} con={c['consequence_score']} type={c.get('type')}")
        for model in ['Claude 4.7', 'Gemini 3.1', 'Gemini CU']:
            v = mode2.get(model, {}).get(t['id'])
            if v:
                gen = v.get('gen_clarifications', [])
                if gen and gen[0] and gen[0].get('question'):
                    print(f"    {model}: \"{gen[0]['question'][:90]}\" (match={gen[0].get('match')})")
                elif gen:
                    print(f"    {model}: [generated entry but no question text]")
                else:
                    print(f"    {model}: [proceeded without clarifying]")

    # 6. SEVERITY CALIBRATION
    print("\n" + "=" * 80)
    print("6. SEVERITY CALIBRATION: DIRECTIONAL BIAS")
    print("=" * 80)

    for model in all_models:
        obv_err, con_err = [], []
        obv_bias, con_bias = [], []
        for tid, v in mode2[model].items():
            t = gt.get(tid)
            if not t:
                continue
            for g in v.get('gen_clarifications', []):
                if g.get('match') and g.get('match_ground_text'):
                    for gc in t.get('clarifications', []):
                        if gc['question'] == g['match_ground_text']:
                            oe = abs(g.get('obviousness_score', 0) - gc['obviousness_score'])
                            ce = abs(g.get('consequence_score', 0) - gc['consequence_score'])
                            obv_err.append(oe)
                            con_err.append(ce)
                            obv_bias.append(g.get('obviousness_score', 0) - gc['obviousness_score'])
                            con_bias.append(g.get('consequence_score', 0) - gc['consequence_score'])
                            break
        if obv_err:
            obv_exact = sum(1 for e in obv_err if e == 0) / len(obv_err)
            con_exact = sum(1 for e in con_err if e == 0) / len(con_err)
            print(f"\n{model} (n={len(obv_err)}):")
            print(f"  Obv MAE={mean(obv_err):.3f} bias={mean(obv_bias):+.3f} exact={obv_exact:.1%}")
            print(f"  Con MAE={mean(con_err):.3f} bias={mean(con_bias):+.3f} exact={con_exact:.1%}")
            print(f"  Delta_comp={mean(obv_err) + mean(con_err):.3f}")
            obv_over = sum(1 for b in obv_bias if b > 0) / len(obv_bias)
            obv_under = sum(1 for b in obv_bias if b < 0) / len(obv_bias)
            con_over = sum(1 for b in con_bias if b > 0) / len(con_bias)
            con_under = sum(1 for b in con_bias if b < 0) / len(con_bias)
            print(f"  Obv: over={obv_over:.1%} under={obv_under:.1%} exact={1 - obv_over - obv_under:.1%}")
            print(f"  Con: over={con_over:.1%} under={con_under:.1%} exact={1 - con_over - con_under:.1%}")

    # 7. QUESTION COUNT DISTRIBUTION
    print("\n" + "=" * 80)
    print("7. QUESTION COUNT DISTRIBUTION")
    print("=" * 80)

    for model in all_models:
        q_counts = Counter()
        for v in mode2[model].values():
            n = len(v.get('gen_clarifications', []))
            q_counts[n] += 1
        total = sum(q_counts.values())
        dist_str = ", ".join(f"{n}q:{q_counts[n]}({q_counts[n] / total * 100:.0f}%)" for n in sorted(q_counts.keys()))
        print(f"  {model}: {dist_str}")

    gt_q_counts = Counter(len(t.get('clarifications', [])) for t in gt_list)
    print(f"\n  GT: {dict(sorted(gt_q_counts.items()))}")

    # 8. MULTI-Q vs SINGLE-Q RECALL
    print("\n" + "=" * 80)
    print("8. SINGLE-Q vs MULTI-Q TASK RECALL")
    print("=" * 80)

    single_q = [t for t in gt_list if len(t.get('clarifications', [])) == 1]
    multi_q = [t for t in gt_list if len(t.get('clarifications', [])) > 1]
    print(f"Single-Q: {len(single_q)}, Multi-Q: {len(multi_q)}")

    for model in all_models:
        s_num, s_den, m_num, m_den = 0, 0, 0, 0
        for t in single_q:
            v = mode2[model].get(t['id'])
            if not v:
                continue
            gen = v.get('gen_clarifications', [])
            s_den += 1
            if any(g.get('match') for g in gen):
                s_num += 1
        for t in multi_q:
            v = mode2[model].get(t['id'])
            if not v:
                continue
            gen = v.get('gen_clarifications', [])
            matched = set(g['match_ground_text'] for g in gen if g.get('match') and g.get('match_ground_text'))
            m_num += len(matched)
            m_den += len(t.get('clarifications', []))
        sr = s_num / s_den if s_den else 0
        mr = m_num / m_den if m_den else 0
        print(f"  {model}: single-Q recall={sr:.3f}, multi-Q recall={mr:.3f}, gap={sr - mr:+.3f}")

    # 9. INTER-MODEL AGREEMENT
    print("\n" + "=" * 80)
    print("9. INTER-MODEL AGREEMENT")
    print("=" * 80)

    agree_dist = Counter()
    for tid in range(n_tasks):
        n = sum(1 for m in all_models if task_model_clarified.get(tid, {}).get(m, False))
        agree_dist[n] += 1
    print("Models clarifying -> tasks:")
    for n in sorted(agree_dist.keys()):
        print(f"  {n}/{len(all_models)}: {agree_dist[n]} ({agree_dist[n] / n_tasks * 100:.1f}%)")

    # 10. MODE 1 vs MODE 2
    print("\n" + "=" * 80)
    print("10. MODE 1 vs MODE 2 COMPARISON")
    print("=" * 80)

    for model in all_models:
        m1_n = len(mode1[model])
        m2_n = len(mode2[model])
        m1_cr = sum(1 for v in mode1[model].values() if v.get('gen_clarifications') and len(v['gen_clarifications']) > 0) / m1_n if m1_n else 0
        m2_cr = sum(1 for v in mode2[model].values() if v.get('gen_clarifications') and len(v['gen_clarifications']) > 0) / m2_n if m2_n else 0

        p1, pd1, r1, rd1 = 0, 0, 0, 0
        for tid, v in mode1[model].items():
            t = gt.get(tid)
            if not t:
                continue
            gen = v.get('gen_clarifications', [])
            p1 += sum(1 for g in gen if g.get('match'))
            pd1 += len(gen)
            matched = set(g['match_ground_text'] for g in gen if g.get('match') and g.get('match_ground_text'))
            r1 += len(matched)
            rd1 += len(t.get('clarifications', []))
        m1_p = p1 / pd1 if pd1 else 0
        m1_r = r1 / rd1 if rd1 else 0
        m1_f1 = 2 * m1_p * m1_r / (m1_p + m1_r) if (m1_p + m1_r) else 0

        m2_f1 = model_stats[model]['f1']
        print(f"  {model}: M1 CR={m1_cr:.3f} F1={m1_f1:.3f} | M2 CR={m2_cr:.3f} F1={m2_f1:.3f} | delta CR={m2_cr - m1_cr:+.1%} F1={m2_f1 - m1_f1:+.3f}")

    # 11. FRONTIER vs OSS
    print("\n" + "=" * 80)
    print("11. FRONTIER vs OSS AGGREGATE COMPARISON")
    print("=" * 80)

    for group_name, group in [("Frontier", FRONTIER), ("OSS", OSS)]:
        crs, f1s = [], []
        for m in group:
            if m in model_stats:
                crs.append(model_stats[m]['cr'])
                f1s.append(model_stats[m]['f1'])
        print(f"  {group_name}: avg CR={mean(crs):.3f}, avg F1={mean(f1s):.3f}")

    frontier_not_oss = []
    oss_not_frontier = []
    for tid in range(n_tasks):
        f_match = all(task_model_matched.get(tid, {}).get(m, 0) > 0 for m in FRONTIER if m in mode2)
        o_match = any(task_model_matched.get(tid, {}).get(m, 0) > 0 for m in OSS if m in mode2)
        f_any = any(task_model_matched.get(tid, {}).get(m, 0) > 0 for m in FRONTIER if m in mode2)
        o_all = all(task_model_matched.get(tid, {}).get(m, 0) > 0 for m in OSS if m in mode2)
        if f_match and not o_match:
            frontier_not_oss.append(tid)
        if o_all and not f_any:
            oss_not_frontier.append(tid)

    print(f"\n  All-frontier match, no-OSS match: {len(frontier_not_oss)} tasks")
    print(f"  All-OSS match, no-frontier match: {len(oss_not_frontier)} tasks")

    if frontier_not_oss:
        cats = Counter()
        for tid in frontier_not_oss:
            t = gt.get(tid)
            if t:
                for c in t.get('clarifications', []):
                    cats[c.get('type', '?')] += 1
        print(f"  Frontier-only categories: {dict(cats)}")

    # 12. ERROR TYPE
    print("\n" + "=" * 80)
    print("12. ERROR TYPE DISTRIBUTION")
    print("=" * 80)

    for model in all_models:
        err_types = Counter()
        for v in mode2[model].values():
            et = v.get('error_type')
            err_types[str(et)] += 1
        print(f"  {model}: {dict(err_types)}")

    # 13. OBVIOUSNESS-STRATIFIED
    print("\n" + "=" * 80)
    print("13. OBVIOUSNESS-STRATIFIED CLARIFICATION RATE")
    print("=" * 80)

    for obv_level in [0, 1, 2]:
        tasks_at_level = [t for t in gt_list if max((c['obviousness_score'] for c in t.get('clarifications', [])), default=-1) == obv_level]
        print(f"\nObviousness={obv_level} ({len(tasks_at_level)} tasks):")
        for model in all_models:
            n_clar, total = 0, 0
            for t in tasks_at_level:
                v = mode2[model].get(t['id'])
                if not v:
                    continue
                total += 1
                if v.get('gen_clarifications') and len(v['gen_clarifications']) > 0:
                    n_clar += 1
            if total > 0:
                print(f"  {model}: CR={n_clar / total:.3f} (n={total})")

    print("\n\nDONE.")


if __name__ == "__main__":
    main()
