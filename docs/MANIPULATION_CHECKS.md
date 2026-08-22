# Manipulation checks

How the persona manipulation was verified in this study, and what a replication could add.
The design, measures, and coding taxonomy these checks build on are documented in
[STUDY_PROTOCOL.md](STUDY_PROTOCOL.md).

## Implemented checks

The study collected no direct perception scales at the time of data collection, so the
manipulation was verified retrospectively along three converging lines.

### 1. Chatbot message coding

Chatbot messages, not only participant messages, were coded with the leadership-tactic
taxonomy, allowing a direct test of whether each condition behaved as designed:

- **Dark Triad condition** — expected predominance of intimidation, harsh criticism,
  manipulation, dismissiveness, blame-shifting, and grandiosity.
- **Servant Leader condition** — expected predominance of encouragement, empowerment,
  constructive feedback, active listening, collaboration, and support.

Analysis: proportions of each tactic type per condition, chi-square and Mann-Whitney tests on
the tactic distributions, effect sizes quantifying the separation between conditions.

Implemented in `analysis/manipulation_verification/chatbot_manipulation_verification.py`,
which also tests message tone and response type.

### 2. Post-hoc participant perception

Post-task self-report measures already in the battery serve as indirect perception checks:

| Measure | Relevance |
|---------|-----------|
| Anthropomorphism (Lifelike, Humanlike, Natural, Conscious) | May differ by condition |
| AI collaboration post-task scale (21 items) | Expected to show condition differences |
| Open-ended feedback | Qualitative verification of differential experience |

Implemented in `analysis/manipulation_verification/posthoc_participant_perception_analysis.py`.

### 3. Behavioral signature verification

The coded dialogue itself provides indirect verification: participant responses diverged
sharply by condition — more frustration, more resistance, and more jailbreaking attempts under
the Dark Triad persona, more confident tone under the Servant Leader persona. These are
responses consistent with having experienced genuinely different leadership styles.

Produced by `analysis/conversations_analysis/gpt_assessment_analysis.py`.

## Prospective checks for replications

Measures worth administering directly rather than reconstructing after the fact.

### Direct perception scales, administered immediately post-interaction

**Leadership style** (1–7):

- The chatbot was supportive
- The chatbot was critical
- The chatbot empowered me to do my best work
- The chatbot was dismissive of my contributions
- The chatbot seemed to care about my success

**Dark Triad perception** (adapted from the Short Dark Triad):

- The chatbot seemed manipulative
- The chatbot seemed arrogant
- The chatbot lacked empathy

**Servant leadership perception** (adapted from SL-7):

- The chatbot put my needs first
- The chatbot helped me develop my skills
- The chatbot created a sense of community

### Warmth and competence

The Stereotype Content Model dimensions: warmth (friendly, warm, sincere, trustworthy) and
competence (capable, skilled, competent, efficient). Expected pattern: Servant Leader high on
both; Dark Triad low on warmth, variable on competence.

### Emotional climate

- How stressful was interacting with the chatbot? (1–7)
- How psychologically safe did you feel? (1–7)
- How comfortable were you expressing disagreement? (1–7)

### Blind condition classification

Independent raters read de-identified transcripts, classify each conversation's condition, and
rate the chatbot's leadership style. Classification accuracy above roughly 75–80% confirms the
manipulation was clearly implemented.
