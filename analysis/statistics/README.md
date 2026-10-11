# Reported statistics and figures

Effect sizes, confidence intervals, assumption tests, and Bayes factors as reported in the
paper, plus the figure scripts. Every value is computed from the archived data
(DOI 10.18150/IPGR44).

## Scripts

| Script | Purpose | Requires |
|--------|---------|----------|
| `recompute_stats.py` | Shared helpers, and reconstruction of the survey scale scores from the raw Qualtrics export into `$STUDY_DATA_ROOT/statistics_output/survey_scored.pkl` | Qualtrics export |
| `compute_survey.py` | Survey and performance between-group statistics → `survey_enhanced.csv` | `recompute_stats.py`, `survey_analysis.py` output |
| `compute_behavioral.py` | Cramér's *V* cluster bootstrap, correlation and ANOVA CIs → `cramersv_enhanced.csv`, `perf_correlates_ci.csv`, `sig_correlations_ci.csv`, `anova_eta2_ci.csv` (its conversation-level comparisons are superseded by `../conversations_analysis/behavior_corrected.py`) | `gpt_assessment_analysis.py` and `survey_analysis.py` output |
| `tost_figures_corrected.py` | Equivalence-test figures (main-text Fig. 6, Supplementary Figs. 14–15), with the difference test FDR-corrected across the ten measures; the classification table goes to `SURVEY_data/analysis_output/tost_bh_classification.csv` | `equivalence_testing_tost.py` output |
| `exploratory/` | Self-reported frustration item, task completion, feedback volume and deliverable provenance (Supplementary Notes 8 and 11); see `exploratory/README.md` | Qualtrics export, conversation logs, step-6 codes |
| `regenerate_manuscript_figures.py` | Supplementary Figs. 11–12 (task experience and pre-post attitudes on the full scale) | `recompute_stats.py` |

The helpers in `recompute_stats.py` cover noncentral-*t* CIs for Cohen's *d*, Fisher-*z* CIs
for correlations, partial-η² CIs, rank-biserial *r*, Hodges-Lehmann median differences, BCa
bootstrap, Cramér's *V*, and JZS Bayes factors.

## Shipped values

The `*_enhanced.csv` and `*_ci.csv` files in this folder are copies of the outputs of the scripts
above (which write to `$STUDY_DATA_ROOT/statistics_output/`) as used in the paper, included so the reported numbers can be checked without rerunning the
full pipeline.

| File | Contents |
|------|----------|
| `survey_enhanced.csv` | Per-measure descriptives, Welch *t*, Mann-Whitney *U*/*Z*, Cohen's *d* [CI], rank-biserial *r* [CI], Hodges-Lehmann [CI], Shapiro-Wilk, Levene, variance ratio, Bayes factors |
| `behavioral_corrected_tests.csv` | Conversation-level dialogue comparisons as reported (counts and lengths from the logs, strategy switches, share of messages per tone): descriptives, Mann-Whitney *U*/*Z*, rank-biserial *r* [CI], Hodges-Lehmann [CI], Cohen's *d* [CI], Welch *p* (written by `../conversations_analysis/behavior_corrected.py`) |
| `behavioral_chisquare_corrected.csv` | Message-level χ² tests with Cramér's *V* and conversation-level cluster bootstrap CIs (same script) |
| `recode/` | Re-codes of the dialogue corpus by four further language models (codes only, no message text), the batch state file, and the inter-coder agreement report (`../conversations_analysis/recode_batch.py`). The `dominant_emotion` column of the code files is the alphabetically first of a message's tones, not a ranked tone, and is not used; the report summarises tone per tone and by frustration |
| `cramersv_enhanced.csv` | Cramér's *V* with conversation-level cluster bootstrap CIs |
| `physio_participant_tests.csv` | Participant-level EDA and fEMG comparisons: means, SDs, Cohen's *d* [CI], Welch and Mann-Whitney *p*, BCa CIs for EDA peak and AUC differences and ratios, mixed-model and ANCOVA estimates, sensitivity analyses (written by `03_statistical_analysis/participant_level_stats.py`) |
| `physio_participant_clusters.csv` | Candidate clusters from the participant-level cluster-based permutation tests, with cluster mass, extent and *p* |
| `anova_eta2_ci.csv` | Mixed and one-way ANOVA partial η² with 90% CIs and Greenhouse-Geisser ε |
| `sig_correlations_ci.csv` | Significant correlations with Fisher-*z* 95% CIs and per-correlation *n* |
| `perf_correlates_ci.csv` | Performance correlates with Fisher-*z* 95% CIs |

Random seeds are fixed. BCa bootstrap intervals use 5,000 resamples and may vary in the last
reported digit.
