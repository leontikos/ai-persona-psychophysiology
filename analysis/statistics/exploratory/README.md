# Exploratory self-report and performance analyses

Scripts behind Supplementary Note 8 (self-reported frustration, open-ended feedback) and Supplementary Note 11
(task completion, chatbot-drafted deliverable text, feedback volume). They read the Qualtrics
export, the conversation logs and the archived dialogue codes, and write CSV outputs to
`$EXPLORATORY_OUT` (default `$STUDY_DATA_ROOT/SURVEY_data/analysis_output/exploratory`).

Run in this order (the provenance script reads the completion output):

```bash
python frustration_item.py
python task_completion.py
python feedback_volume.py
python deliverable_provenance.py
python open_ended_agreement.py
```

`open_ended_agreement.py` compares the human and language-model coding of the open-ended feedback
and tabulates mentions of the chatbot's treatment by condition. It reads three files from
`SURVEY_data/analysis_output/`: `open_ended_coding_ai.csv` (the model's codes, keyed by random codes),
`open_ended_coding_key.csv` (code to participant) and, optionally, `open_ended_coding_for_verification.csv`
(the same responses coded 0/1 by a human coder blind to condition); these files are deposited with the
study data. The
codebook is `docs/open_ended_codebook.md` in the code repository. The model codes were produced by a
language model (Claude Opus 5.5) applying that codebook, blind to condition, in two independent passes.
Without human codes, the script reports the model codes (as in the paper).

`common.py` holds the statistical helpers (copied from `../recompute_stats.py`, logic unchanged)
and the scale scoring. Seeds: `numpy.random.default_rng(20260704)`, 5,000 bootstrap resamples.
