# PAD4_BENCH Paper 1 — Design Blueprint
*Drafting-ready plan. Built from the May 21 handoff, reconciled against the
verified directory listing (May 22). This document supersedes the handoff's
section outline where the two disagree; disagreements are flagged inline.*

---

## 0. STATUS OF DECISIONS

| Decision | State | Note |
|---|---|---|
| Journal | OPEN | Designed to JCIM defaults; J. Cheminfo would relax figure/length limits |
| Title | OPEN | 14-word working title held; revisit after journal |
| First section to draft | OPEN (discussing) | Recommendation below: §5.6 |
| Drafting language | Assumed English | Confirm |
| Citation style | OPEN | Does not block drafting; settle before reference list |
| Manuscript section home | OPEN | Recommendation: `manuscript/sections/`, one .md per section |
| Script code review | OPEN | Deferred unless scripts uploaded |

---

## 1. POSITIONING (locked)

PAD4_BENCH Paper 1 is a **benchmark / evaluation-infrastructure / calibration-
methodology** paper. PAD4 is the case study, not the subject.

Every paragraph passes the test: does it teach the reader something about
benchmark engineering, calibration, or AD analysis? If it only teaches them
about PAD4, demote or cut.

The paper does NOT tell the story "XGBoost achieved 0.97 AUC."
It tells the five-pillar story:

1. Curated benchmark with principled split engineering.
2. Benchmark difficulty behaves coherently across evaluation regimes.
3. High predictive performance does not imply trustworthy extrapolative confidence.
4. Calibration and AD behavior degrade faster than ranking metrics.
5. Standard QSAR reliability heuristics partially fail on hard regimes.

Findings 5–8 are the contribution (§5.6, §5.7). Findings 1–4 are setup.

---

## 2. DISCREPANCIES FOUND vs HANDOFF — RESOLVE BEFORE DRAFTING

These came out of cross-checking the handoff against the real directory tree.
None require redoing analysis. Most are hygiene; #1 and #2 need a decision.

1. **Two manuscript scaffolds exist.** `manuscript/` (with empty `sections/`,
   `tables/`, `supplementary/` and 4 stale figures) AND `paper/` (the real
   D1–D16 / R1–R9 / T1–T14 asset store). The handoff only mentions `paper/`.
   DECISION NEEDED: designate `manuscript/sections/` as prose home, `paper/`
   as asset store, delete the 4 stale figures in `manuscript/figures/`.

2. **Figure count.** Handoff says "23 figures." Real count is 16 distinct
   figures (D1–D16 + R1–R9, with D14/D15/D16 sharing the D-series numbers).
   The 11-main-figure promotion plan still maps cleanly. Do NOT write "23"
   in the reproducibility statement — say "16 figures" or count PDF+PNG pairs.

3. **Log/PID cleanup.** Handoff caveat 13 names `features_v18/_logs/` — that
   subdir does not exist. Stale files are at the TOP LEVEL of `features_v18/`
   (`classification_run.{log,pid}`, `cliff_aware_classification_run.{log,pid}`)
   and scattered in `models_v1/` (`overnight.{log,pid}`, `reviewer_proof.{log,pid}`,
   `sweep.{log,pid}`). Corrected cleanup commands in §10.

4. **`paper_intro/` has 4 reports, not 5.** No standalone calibration-repair
   markdown; calibration repair lives in per-cell `calibration_repair.json`
   and supplementary table T13.

5. **`features_v17/` still present** (classification only, with `.log` files).
   Superseded by v18. Decide: archive for provenance or remove. Check it is
   referenced nowhere first.

6. **Smoke-test scripts still at repo root** (`smoke_test_xgb_regression.py`)
   and in `scripts/` (`smoke_test_v8_4.py`). The handoff said the smoke-test
   *directory* under `models_v1/` was deleted — true — but the scripts remain.
   Move to `tests/` or archive before public release.

7. **`tests/` is nearly empty** — only `test_scaffold_split.py` has content.
   Not a drafting blocker; the reproducibility statement should not overclaim
   test coverage.

8. **`results/y_scramble/` exists** (only a README). The handoff never mentions
   a y-scrambling experiment. If y-scramble results exist they would strengthen
   the leakage defense in §6 para 1 — CHECK whether this directory has real
   content. If empty, ignore. Do NOT run a new y-scramble experiment.

---

## 3. ASSET INVENTORY — VERIFIED ON DISK

### Main-text figures (11) — all confirmed present in `paper/figures/`
| ID | File location | Section |
|---|---|---|
| D1 provenance funnel | data/main/D1_provenance_funnel.{pdf,png} | §2.1 |
| D2 pIC50 distributions | data/main/D2_pic50_distribution.{pdf,png} | §2.2 |
| D3 class balance | data/main/D3_class_balance.{pdf,png} | §2.2 |
| D5 scaffold diversity | data/main/D5_scaffold_diversity.{pdf,png} | §2.3 |
| D9 Tanimoto per split | data/main/D9_tanimoto_per_split.{pdf,png} | §3.3 |
| R1 headline heatmaps | results/main/R1_headline_heatmaps.{pdf,png} | §5.1 |
| R2 headline with CIs | results/main/R2_headline_with_ci.{pdf,png} | §5.1 |
| R5 AUC vs ECE scatter | results/main/R5_calibration_vs_auc.{pdf,png} | §5.6 |
| D14 reliability diagrams | data/supp/D14_reliability.{pdf,png} → PROMOTE | §5.6 |
| D15 calibration repair | results/supp/D15_calibration_repair.{pdf,png} → PROMOTE | §5.6 |
| D16 AD scatter | results/supp/D16_applicability_domain.{pdf,png} → PROMOTE | §5.7 |

NOTE: D14/D15/D16 currently sit in `supp/` folders. "Promote" is an editorial
label only — the files exist; just reference them as main figures. No need to
move files unless you want folder tidiness.

### Main-text tables (4) — confirmed in `paper/tables/main/`
| ID | Files | Section |
|---|---|---|
| T1 dataset summary | T1_dataset_summary.{csv,md,tex} | §2 |
| T2 regression headline R² + CI | T2_regression_headline_R2.{csv,md,tex} | §5.1 |
| T3 classification headline AUC + CI | T3_classification_headline_AUC.{csv,md,tex} | §5.1 |
| T5 stacking summary | T5_stacking.{csv,md,tex} | §5.4 |

T4 (linear vs tree) is in `main/` on disk but the plan DEMOTES it to supp.
Editorial only — cite T4 as a supplementary table in §5.2.

### Supplementary figures — confirmed present
D4, D6, D7, D8 (data/main on disk, demoted to supp), D10, D11, D12, D13
(data/supp), R3, R4, R6, R7, R8, R9 (results/main+supp).

### Supplementary tables — confirmed in `paper/tables/supp/`
T6 per-cell full results, T7 leakage verification, T8 seed robustness,
T9 threshold recalibration, T10 calibration, T11 covalent accounting,
T12 hyperparameters, T13 calibration repair, T14 AD (+ per_compound, _stats).

### Data appendix
`models_v1/all_results.csv` — 190 rows. The machine-readable results spine.
`models_v1/leakage_verification.json` — leakage proof for §3.2 and §6.

---

## 4. SECTION SKELETON — ONE QUESTION PER SECTION

Target lengths assume JCIM (focused). J. Cheminformatics would allow ~20% more.

| § | Title | Question it answers | Length | Main assets |
|---|---|---|---|---|
| 1 | Introduction | Why does the field need a calibration-aware benchmark? | ~1 pg, 4 para | — |
| 2 | Data & Curation | What are we benchmarking on, and why trust it? | ~3 para | D1,D2,D3,D5; T1 |
| 3 | Split Engineering | How do we stress-test extrapolation principledly? | ~3 para | D9; T7(supp) |
| 4 | Modeling & Evaluation | How do we evaluate honestly? | ~2 pg, 5 subsec | — |
| 5 | Results | (7 subsections — see §5 breakdown) | ~3–4 pg | R1,R2,R5,D14,D15,D16; T2,T3,T5 |
| 6 | Discussion | What did we learn, and what are the limits? | ~4–5 para | — |
| 7 | Conclusion | One paragraph, high-level | ~1 para | — |
| — | Reproducibility statement | What is released | own section | — |

### §5 Results breakdown
| §5.x | Finding(s) | Question | Length | Assets |
|---|---|---|---|---|
| 5.1 Difficulty ordering | F1 | Does the framework behave coherently? | 1–2 para | R1, R2; T2, T3 |
| 5.2 Linear vs tree | F2 | Do nonlinear models matter at this scale? | 1 para | T4 (supp) |
| 5.3 Generalization honesty | F1 detail | Does CV predict test? | 1 para | R4 (supp) |
| 5.4 Stacking | F3 | Does combining reg+clf help? | 1 para max | T5; R6 (supp) |
| 5.5 Threshold sensitivity | F4 | Does the operating threshold matter? | 1 para | R8, T9 (supp) |
| **5.6 Calibration** | **F5, F6** | **Can probabilities be trusted?** | **3–4 para — CENTERPIECE** | R5, D14, D15 |
| **5.7 Applicability domains** | **F7, F8** | **Can similarity identify failures?** | **2–3 para — CENTERPIECE** | D16; T14 (supp) |

---

## 5. RECOMMENDED DRAFTING ORDER

1. **§5.6 Calibration** — the centerpiece. All inputs frozen, journal-independent.
   Drafting from strength means the Introduction later describes work that exists.
2. **§5.7 Applicability domains** — the second centerpiece; pairs naturally with 5.6.
3. **§4 Modeling & Evaluation** — methods that 5.6/5.7 depend on; locks terminology.
4. **§5.1–5.5** — the setup results, quick once methods exist.
5. **§3 Split Engineering** — core contribution per supervisor; needs §5.1 numbers.
6. **§2 Data & Curation** — mechanical, low interpretive risk.
7. **§1 Introduction** — written last, as a description of a finished paper.
8. **§6 Discussion**, **§7 Conclusion**, **Reproducibility statement**.

Forward references created by this order (reconcile at lock time):
§5.6 will reference the split framework (§3) and difficulty ordering (§5.1).
Cheap to fix; flag with `[REF §3]` placeholders while drafting.

---

## 6. CROSS-SECTION CONSISTENCY LEDGER

Numbers and claims that appear in more than one place — keep them identical.

| Item | Value | Appears in |
|---|---|---|
| Headline regression R² | 0.802 [0.747, 0.844] | §5.1, abstract |
| Headline classification AUC | confirmed 0.979; cliff_aware 0.884 | §5.1, §6 para1, abstract |
| Replicate noise floor | p95 = 0.336 (NOT median 0.000) | §2.3, §4.4, §6 para1 |
| Isotonic recovers | ~50% of gap; lead_opt −49%, similarity −44% | §5.6, §6 para3 |
| Platt fails | worse on 4 of 6 splits | §5.6, §6 para3 |
| Cliffs recalibration-resistant | cliff_aware iso −2% | §5.6, §6 para4 |
| False-confidence rate | 13–18% in-domain still high-error (lead_opt 13.0, cliff_aware 17.7) | §5.7, §6 para4, abstract |
| Bootstrap | 1000 resamples × 190 cells | §4.4, §5.1 |
| Seed robustness | max cross-seed std 0.013 | §4.4 or §5.5 |
| Covalent disclosure | 9 reversible kept; 20 irrevers. in confirmed; 36 in classification | §2.2 / §4 methods — MUST appear |
| Datasets released, not modeled | T2 n=510; T3 n=53,694; Ki n=154 | §2.2 disclaimer, §6 limitations |
| BindingDB-dominant | 2,628 of 2,618 carry BindingDB source | §2.3, §6 para1 |

---

## 7. THE TWO DEFENSIVE PARAGRAPHS (§6)

**"Why are AUCs this high" (§6 para 1)** — cite 5 reasons: PAD4 chemistry
coherence; BindingDB-dominant curation; replicate aggregation (p95=0.336);
ECFP4 informativeness; cliff_aware drop to 0.88 as honest counterpoint.
Pair with `leakage_verification.json`. CHECK `results/y_scramble/` — if it has
real content, add y-scrambling as a sixth, stronger leakage rebuttal.

**Limitations (§6 para 5)** — no external validation. Draft paragraph exists
in the handoff; refine at drafting time. External validation decision is
LOCKED: do not add it.

---

## 8. STYLE GUARDRAILS

USE: stress-test, calibration-aware, extrapolative regime, applicability
domain, false confidence, ranking-calibration decoupling, principled split
engineering, publishable negative result, monotone curation pipeline.

AVOID: state-of-the-art, novel, outperforms, significantly (without a test),
apologetic deep-learning framing, "we propose" + opinion.

Supervisor's writing principles: one question per section; centerpieces get
the space; point to tables, don't dump them; confident not defensive.

---

## 9. FIGURE / TABLE BUDGET

Main: 11 figures, 4 tables. Everything else supplementary.
If JCIM pushes back on 11 main figures, first demotion candidates:
D3 (class balance) and R2 (could merge into R1). Do not demote D14/D15/D16 —
they ARE the contribution.

---

## 10. CODE / RELEASE HYGIENE (run locally — outside drafting)

```bash
# corrected log/pid cleanup
mkdir -p features_v18/_logs && mv features_v18/*.log features_v18/*.pid features_v18/_logs/ 2>/dev/null
mkdir -p models_v1/_logs   && mv models_v1/*.log   models_v1/*.pid   models_v1/_logs/ 2>/dev/null
find models_v1 -name 'sweep.log' -o -name 'sweep.pid' | xargs -I{} mv {} models_v1/_logs/ 2>/dev/null

# decisions, not commands:
#  - features_v17/  : archive or remove (check no references first)
#  - manuscript/figures/ : delete 4 stale figures, OR keep paper/ as sole source
#  - smoke_test_*.py : move to tests/ or archive
#  - .gitignore should cover *.pid, *.log, __pycache__
```

Do NOT recreate the deleted `models_v1/regression/random/full/smoke-test` dir.

---

## 11. WHAT IS LOCKED — DO NOT REOPEN

- No external validation (inventory confirmed no usable on-disk option).
- No deep-learning baseline (deferred to Paper 3).
- T2/T3/Ki released but not modeled.
- Stacking is a publishable negative result, framed as such, one paragraph.
- Headline numbers frozen at seed=42. Do not recompute.
- The v18.0 linear-space feature selector is the principled bound.

If the user proposes a new analysis, push back: the analytical phase is closed.
