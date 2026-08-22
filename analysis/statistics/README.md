# Reported statistics and figures

Effect sizes, confidence intervals, assumption tests, and Bayes factors as reported in the
paper, plus the two figure scripts. Every value is computed from the archived data
(DOI 10.18150/IPGR44).

## Scripts

| Script | Purpose | Requires |
|--------|---------|----------|
| `recompute_stats.py` | Shared helpers, and reconstruction of the survey scale scores from the raw Qualtrics export into `$STUDY_DATA_ROOT/statistics_output/survey_scored.pkl` | Qualtrics export |
| `compute_survey.py` | Survey and performance between-group statistics → `survey_enhanced.csv` | `recompute_stats.py`, `survey_analysis.py` output |
| `compute_behavioral.py` | Behavioral Mann-Whitney effect sizes, Cramér's *V* cluster bootstrap, correlation and ANOVA CIs → `behavioral_enhanced.csv`, `cramersv_enhanced.csv`, `perf_correlates_ci.csv`, `sig_correlations_ci.csv`, `anova_eta2_ci.csv` | `gpt_assessment_analysis.py` and `survey_analysis.py` output |
| `compute_physio.py` | Participant-level EDA and fEMG bootstrap CIs → `physio_enhanced.csv` | `event_related_signal_analysis.py` output |
| `make_main_fig2.py` | Main-text Figure 2: communication strategy, emotional tone, strategy diversity, turning points | `gpt_assessment_analysis.py` output |
| `regenerate_manuscript_figures.py` | Supplementary figures: per-category strategy, emotion, engagement; task experience and pre-post attitudes on the full scale | `gpt_assessment_analysis.py` output and `recompute_stats.py` |

The helpers in `recompute_stats.py` cover noncentral-*t* CIs for Cohen's *d*, Fisher-*z* CIs
for correlations, partial-η² CIs, rank-biserial *r*, Hodges-Lehmann median differences, BCa
bootstrap, Cramér's *V*, and JZS Bayes factors.

## Shipped values

The `*_enhanced.csv` and `*_ci.csv` files in this folder are the outputs of the scripts above
as used in the paper, included so the reported numbers can be checked without rerunning the
full pipeline.

| File | Contents |
|------|----------|
| `survey_enhanced.csv` | Per-measure descriptives, Welch *t*, Mann-Whitney *U*/*Z*, Cohen's *d* [CI], rank-biserial *r* [CI], Hodges-Lehmann [CI], Shapiro-Wilk, Levene, variance ratio, Bayes factors |
| `behavioral_enhanced.csv` | Conversation-level behavioral comparisons with *Z*, rank-biserial *r* [CI], Hodges-Lehmann [CI], Cohen's *d* [CI] |
| `cramersv_enhanced.csv` | Cramér's *V* with conversation-level cluster bootstrap CIs |
| `physio_enhanced.csv` | EDA peak amplitude and AUC: participant-level differences and ratios with BCa CIs |
| `anova_eta2_ci.csv` | Mixed and one-way ANOVA partial η² with 90% CIs and Greenhouse-Geisser ε |
| `sig_correlations_ci.csv` | Significant correlations with Fisher-*z* 95% CIs and per-correlation *n* |
| `perf_correlates_ci.csv` | Performance correlates with Fisher-*z* 95% CIs |

Random seeds are fixed. BCa bootstrap intervals use 5,000 resamples and may vary in the last
reported digit.
