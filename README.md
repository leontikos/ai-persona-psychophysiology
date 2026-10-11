# The system's shadow — AI persona, stress, behavior, and work quality

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22060071.svg)](https://doi.org/10.5281/zenodo.22060071)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Chatbot application and analysis code for a laboratory experiment in which 58 participants
completed a four-part marketing task supervised by one of two AI chatbot personas built on
identical technology: a supportive **Servant Leader** or an antagonistic **Dark Triad** leader.
The two conditions differ only in the system prompt. Electrodermal activity and facial
electromyography were recorded continuously, dialogue was coded automatically with an LLM,
task deliverables were rated blind by expert judges, and self-report was collected at three
time points.

This repository contains everything needed to run the experiment and reproduce the reported
analyses: the persona prompts, the chatbot that delivered them, the psychophysiological
signal-processing pipeline, the LLM dialogue-coding pipeline, and the statistical analysis
and figure scripts.

**Data.** The study data are published separately at
[https://doi.org/10.18150/IPGR44](https://doi.org/10.18150/IPGR44). No participant data are
included in this repository.

## Repository layout

```
.
├── chatbot/                     Experiment delivery
│   ├── personas/                The two system prompts, verbatim as deployed
│   │   ├── servant_leader.txt       condition 1
│   │   └── dark_triad.txt           condition 2
│   ├── streamlit_app.py         Chat interface and conversation logger
│   ├── run_chatbot.bat          Windows launcher
│   └── requirements.txt
├── analysis/
│   ├── 01_preprocessing/        Signal quality assessment and cleaning
│   ├── 02_feature_extraction/   Conversation-event extraction and signal alignment
│   ├── 03_statistical_analysis/ Event-related analysis of EDA and fEMG
│   ├── conversations_analysis/  LLM dialogue coding and coded-data analysis
│   ├── manipulation_verification/  Chatbot-behavior and participant-perception checks
│   ├── survey_analysis/         Self-report, power, and equivalence analyses
│   ├── statistics/              Reported effect sizes, CIs, Bayes factors, figures
│   └── utils/                   Data preparation, conversation-log housekeeping and validation
├── docs/
│   ├── STUDY_PROTOCOL.md        Full protocol: design, measures, processing parameters
│   └── MANIPULATION_CHECKS.md   Manipulation-check design, implemented and prospective
├── requirements.txt             Analysis environment
├── .env.example
├── CITATION.cff
└── LICENSE
```

## Setup

Python 3.11.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Then edit `.env`:

```
STUDY_DATA_ROOT=/absolute/path/to/study_data
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

`STUDY_DATA_ROOT` is the only path you need to set. Every script derives its inputs and
outputs from it and loads it from `.env` automatically, so scripts run correctly from any
working directory. `OPENAI_API_KEY` is needed only by the chatbot, the dialogue coding (step 6)
and the cross-model re-coding (step 7c); `ANTHROPIC_API_KEY` only by the re-coding.

## Data layout

Place the three raw inputs under `$STUDY_DATA_ROOT` as follows. Every other folder is
created by the pipeline. Step 6 (language-model coding) is paid and not exactly reproducible, so
to reproduce the reported dialogue statistics also use `CONVERSATION_ANALYSES/` (the step-6 codes
in `successful_analyses/` and the step-7 tables) from the data archive; step 7b reads the archived
`conversations_dataframe.csv` for its row order, which fixes the bootstrap intervals.

To build this layout from the data archive, download all of its files into one folder (choose
"Original File Format" for the survey table) and run

```bash
python analysis/utils/prepare_repod_data.py path/to/downloads
```

It unpacks the archives into `$STUDY_DATA_ROOT` and writes `SURVEY_data/AI_boss_qualtrics_quality.csv`
from the archived survey table, `AI_boss_data_pretest_posttest`, whose expert-rater columns
`Judge_A`–`Judge_C` the scripts call `Judge1`–`Judge3`.

```
$STUDY_DATA_ROOT/
├── psychophysiology_signal/          INPUT   BIOPAC recordings
│                                             {participant_id}_main.acq
│                                             {participant_id}_baseline.acq
├── conversation_logs/                INPUT   conversation_YYYYMMDD_HHMMSS.json
├── SURVEY_data/
│   ├── AI_boss_qualtrics_quality.csv INPUT   Qualtrics export
│   └── analysis_output/              step 8
├── signal_quality_analysis/          step 1
├── cleaned_signals/                  steps 3-4
├── CONVERSATION_ANALYSES/            steps 6-7c
├── event_related_analysis/           step 5 (descriptive, epoch level)
├── event_related_participant_level/  step 5b (reported results)
├── PERCEPTION_ANALYSIS/              step 9
├── statistics_output/                steps 5b, 10
└── figures/                          steps 5b, 7b, 10
```

## Running the experiment

```bash
cd chatbot
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The operator enters a participant ID and a condition number — `1` for Servant Leader, `2` for
Dark Triad. The interface shows only the numbers, so the condition label is not visible to the
participant. Streamlit branding is suppressed and both conditions open with the same neutral
greeting, "How may I help you?".

Model, temperature (0.0), context window (64,000 tokens, oldest exchanges truncated first),
interface, and streaming are identical across conditions. The only difference is which file in
`personas/` is loaded as the system prompt. Each conversation is written to
`conversation_logs/conversation_<timestamp>.json` with the participant ID, condition, model,
and per-message role, content, start and end timestamps, and token count.

On Windows, `run_chatbot.bat` activates a conda environment named `chatbot` and starts the app.

## Analysis pipeline

Run the steps in order. Steps 1–5 (physiology), 6–7 (dialogue), and 8 (self-report) are
independent of one another; steps 9–10 depend on the earlier ones.

### Conversation-log preparation

Run before dialogue coding if any session was interrupted and produced more than one log file:

```bash
python analysis/utils/merge_duplicate_conversations.py     # merge split sessions
python analysis/utils/validate_group_assignments.py        # log condition vs. Qualtrics group
```

### Physiology

| # | Script                                                                                  | Reads                                         | Writes                                                                                                                                         |
| - | --------------------------------------------------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | `analysis/01_preprocessing/signal_quality_assessment.py`                              | `psychophysiology_signal/`                  | `signal_quality_analysis/` — per-channel quality scores, HTML reports, `summaries/progressive_results.json`                               |
| 2 | `analysis/01_preprocessing/signal_quality_summary_generator.py`                       | `signal_quality_analysis/`                  | Interactive HTML dashboards across participants and channels                                                                                   |
| 3 | `analysis/01_preprocessing/signal_cleaner.py`                                         | `psychophysiology_signal/`, quality results | `cleaned_signals/` — filtered signals as `.npy` plus metadata and cleaning reports                                                        |
| 4 | `analysis/02_feature_extraction/conversation_event_extractor_after_signal_cleaner.py` | `cleaned_signals/`, `conversation_logs/`  | Per-participant conversation-event, experimental-info, and event-summary JSON alongside the cleaned signals                                    |
| 5 | `analysis/03_statistical_analysis/event_related_signal_analysis.py`                   | `cleaned_signals/` and the event files      | `event_related_analysis/` — epoch-level grand averages and descriptive comparisons |
| 5b | `analysis/03_statistical_analysis/participant_level_extract.py`, then `participant_level_stats.py`, then `participant_level_figures.py` | `psychophysiology_signal/`, `conversation_logs/`, experimental-info files from step 4 | `event_related_participant_level/` — per-participant epochs, participant-level statistics; `figures/` — main-text Fig. 4 and Supplementary Figs. 9–10 |

Step 5b produces every event-related result reported in the paper; `analysis/03_statistical_analysis/audit_synchronisation.py` audits the alignment of the recordings and the conversation logs (Supplementary Methods) and writes `statistics_output/synchronisation_audit.csv`. It expects one
conversation log per participant, so run the conversation-log preparation above first. It reads
the raw recordings: EDA stays in absolute microsiemens and is low-pass filtered once (4th-order
zero-phase Butterworth, 5 Hz); fEMG is band-pass filtered once (4th-order zero-phase
Butterworth, 20–500 Hz) and summarised as a 50 ms RMS envelope. No software notch filter is
applied, because the recordings already carry AcqKnowledge's real-time line-frequency comb
filter, and no sample-level artifact correction is applied; outlier fEMG epochs (mean envelope
more than 5 MAD above the participant's median) are excluded, and every fEMG result is also
computed with all epochs. Epochs span −2 to +18 s for EDA and −2 to +13 s for fEMG around
the onset of each chatbot reply, each referenced to its 2 s pre-onset mean. All tests use the
participant as the unit: participant-averaged waveforms, BCa bootstrap CIs, cluster-based
permutation tests with condition labels permuted across participants (10,000 permutations),
and a linear mixed model of single-epoch EDA peaks with a participant random intercept. Step 5
pools epochs across participants and is kept for its descriptive outputs only. Full parameters
are in [docs/STUDY_PROTOCOL.md](docs/STUDY_PROTOCOL.md).

Step 4 applies a two-hour timezone correction to the BIOPAC timestamps to align them with the
conversation logs. Verify this offset against your own recording setup before running it.

Two further scripts cover the same data by other routes and are not part of the reported
pipeline: `01_preprocessing/biopac_data_quality_check.py` produces exploratory per-participant
plots with conversation events overlaid, and `03_statistical_analysis/biopac_data_analysis.py`
is a standalone end-to-end pipeline including PCA and clustering.

Optional: `analysis/utils/delete_csv_files_from_cleaned_files.py` deletes the CSV copies of
cleaned signals, keeping the `.npy` files, to reclaim disk space. It has a dry-run mode.

### Dialogue coding

| # | Script                                                            | Reads                      | Writes                                                                                                                                                    |
| - | ----------------------------------------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 6 | `analysis/conversations_analysis/analysis_pipeline_with_gpt.py` | `conversation_logs/`     | `CONVERSATION_ANALYSES/successful_analyses/` — structured codes per message                                                                            |
| 7 | `analysis/conversations_analysis/gpt_assessment_analysis.py`    | `CONVERSATION_ANALYSES/` | `participant_messages_dataframe.csv`, `chatbot_messages_dataframe.csv`, `interactions_dataframe.csv`, `conversations_dataframe.csv`, plus figures |
| 7b | `analysis/conversations_analysis/behavior_corrected.py`        | `conversation_logs/`, `CONVERSATION_ANALYSES/` | `CONVERSATION_ANALYSES/behavior_corrected/` — per-message and per-conversation tables and all reported dialogue statistics; `figures/` — main-text Figs. 2–3 and Supplementary Figs. 1–8 |
| 7c | `analysis/conversations_analysis/recode_batch.py` (`submit`, `status`, `collect`, `report`) | `conversation_logs/`, `CONVERSATION_ANALYSES/` | `analysis/statistics/recode/` — re-codes by four further language models and the inter-coder agreement report |

Step 6 calls the OpenAI API (`gpt-4.1-2025-04-14`) with structured outputs, processing each
conversation in chunks of 16 messages. Participant messages are coded for communication
strategy, emotional tone, and task engagement; chatbot messages for leadership tactics, tone,
response type, and guidance style. The full taxonomy and the coding prompts are in the script
and documented in [docs/STUDY_PROTOCOL.md](docs/STUDY_PROTOCOL.md). This step incurs API cost
and is the only nondeterministic stage of the pipeline.

Step 7b produces every dialogue result reported in the paper. It counts messages and characters
from the conversation logs, takes the condition from the logged chatbot version, and derives
strategy diversity (Shannon entropy over the 17 strategy subtypes), strategy switches between
consecutive participant messages, and the share of messages carrying each emotional tone from
the step-6 codes. The conversation-level table written by step 7 also holds the coding model's
own summary estimates (message counts and lengths, a list of key turning points, the inferred
condition); these are not used in any reported analysis.

Step 7c re-codes the corpus with four further models through the Anthropic and OpenAI batch APIs
(`ANTHROPIC_API_KEY` and `OPENAI_API_KEY` in `.env`); its `submit` stage reads
`CONVERSATION_ANALYSES/extracted_participants.csv`, written by
`analysis/utils/participants_names_check.py`. Its `report` stage needs no API key and
reproduces the agreement statistics and the cross-coder effect sizes from the codes shipped in
`analysis/statistics/recode/`.

`analysis/utils/identify_missing_conversations.py` reports conversations whose condition could
not be inferred.

### Manipulation verification and self-report

| # | Script                                                                            | Reads                                         | Writes                                                                                                                                                                         |
| - | --------------------------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 8 | `analysis/survey_analysis/survey_analysis.py`                                   | `SURVEY_data/AI_boss_qualtrics_quality.csv` | `SURVEY_data/analysis_output/` — reliability, descriptives, baseline equivalence, condition and pre-post comparisons, ANOVAs, correlations, moderation, performance and ICC |
| 8 | `analysis/survey_analysis/sensitivity_power_analysis.py`                        | same                                          | Minimum detectable effect size for the achieved sample                                                                                                                         |
| 8 | `analysis/survey_analysis/equivalence_testing_tost.py`                          | same                                          | TOST equivalence tests against the SESOI and MDES bounds                                                                                                                       |
| 9 | `analysis/manipulation_verification/chatbot_manipulation_verification.py`       | `CONVERSATION_ANALYSES/`                    | Tactic, tone, and response-type distributions by condition with tests and figures                                                                                              |
| 9 | `analysis/manipulation_verification/posthoc_participant_perception_analysis.py` | Qualtrics export                              | `PERCEPTION_ANALYSIS/` — participant-perception manipulation check                                                                                                          |

### Reported statistics and figures (step 10)

Run from inside `analysis/statistics/`. The scripts write to `$STUDY_DATA_ROOT/statistics_output/`
(figures to `$STUDY_DATA_ROOT/figures/`); copies of the outputs used in the paper are shipped in
this folder (`*_enhanced.csv`, `*_ci.csv`, `behavioral_*.csv`, `recode/`).

```bash
cd analysis/statistics
python recompute_stats.py                  # rebuilds statistics_output/survey_scored.pkl
python compute_survey.py                   # needs step 8
python compute_behavioral.py               # needs steps 7 and 8
python regenerate_manuscript_figures.py    # needs recompute_stats.py; Supplementary Figs. 11-12
python tost_figures_corrected.py           # needs step 8; main-text Fig. 6, Supplementary Figs. 14-15
cd exploratory                             # Supplementary Notes 8 and 11
python frustration_item.py && python task_completion.py && python feedback_volume.py && python deliverable_provenance.py
python open_ended_agreement.py             # needs the open-ended coding files (see exploratory/README.md)
```

`recompute_stats.py` reconstructs the survey scale scores from the raw Qualtrics export and
holds the shared helpers: noncentral-*t* CIs for Cohen's *d*, Fisher-*z* CIs, partial-η² CIs,
rank-biserial *r*, Hodges-Lehmann estimates, BCa bootstrap, Cramér's *V*, and JZS Bayes
factors. The other scripts in this folder consume it. The physiological statistics come from
step 5b; their aggregate outputs are shipped here as `physio_participant_tests.csv` and
`physio_participant_clusters.csv`. Random seeds are fixed; BCa bootstrap intervals
(5,000 resamples) may vary in the last reported digit.

The figure scripts write PNGs to `$STUDY_DATA_ROOT/figures/`.

## Reproducibility notes

- `requirements.txt` pins the versions used to produce the reported results (Python 3.11,
  scipy 1.17.1, numpy 2.3.5, pandas 3.0.3, pingouin 0.6.1, statsmodels 0.14.6).
  The archived step-8 outputs were written with an earlier pingouin release, which named some
  output columns differently (e.g. `p-unc` rather than `p_unc`, `ICC2k` rather than `ICC(A,k)`);
  the scripts accept both naming schemes and reproduce the same values.
- The persona prompts in `chatbot/personas/` are byte-identical to those deployed during data
  collection. Editing them changes the manipulation.
- The chatbot ran at temperature 0.0, but GPT-4o responses are not guaranteed to be
  bit-reproducible; expect close but not identical dialogue on a rerun.
- Step 6 (LLM dialogue coding) is the one stage whose output can shift between runs. Its codes
  and the dataframes derived from them in the published analysis are part of the data archive
  (`CONVERSATION_ANALYSES.zip`).
- All statistics scripts set explicit seeds; the participant-level physiology uses seed 20261006.
- Version 1.1.0 replaced the reported physiological analysis with the participant-level
  pipeline (step 5b). Version 1.0.0 derived those results from epoch-pooled tests on signals
  that had been sample-wise clipped and filtered twice; it remains available under its own DOI.
- Version 1.2.0 computes every reported dialogue statistic from the conversation logs and the
  message-level codes (step 7b) instead of the coding model's own summary fields, adds the
  cross-model re-coding (step 7c) and the exploratory self-report and performance analyses
  (`analysis/statistics/exploratory/`), and classifies equivalence-test results with an
  FDR-corrected difference test.

## Citation

Ciechanowski, L., Kovbasiuk, A., Sowa, K., Triantoro, T., Przegalińska, A., & Freeman, R.
*The system's shadow: AI persona shapes behavior and human–AI output more than satisfaction ratings.*

Code: [https://github.com/leontikos/ai-persona-psychophysiology](https://github.com/leontikos/ai-persona-psychophysiology)

Code DOI (all versions): [https://doi.org/10.5281/zenodo.22060071](https://doi.org/10.5281/zenodo.22060071)

Data: [https://doi.org/10.18150/IPGR44](https://doi.org/10.18150/IPGR44)

## Funding

Polish National Science Centre (NCN) OPUS 21 grant no. 2021/41/B/HS4/03664, with additional
support from NCN SONATA 21 grant no. 2025/59/D/HS4/02851. The study protocol was approved by
the Research Ethics Committee (Komisja Etyki Badań) of Kozminski University (opinion of
8 December 2022).

## License

Apache License 2.0. See [LICENSE](LICENSE).
