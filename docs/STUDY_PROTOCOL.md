# Study protocol

Full protocol for the experiment whose code is in this repository: design, materials,
measures, apparatus, processing parameters, coding taxonomy, and manipulation checks.
For how to run the code, see the [README](../README.md).

## Study Overview

This study investigates the psychological, physiological, and behavioral responses of humans interacting with AI chatbots exhibiting different leadership styles. The research combines psychophysiological measurements (electrodermal activity and facial electromyography) with conversation analysis, and self-reported psychological and behavioral measures to understand how different AI personality types affect human stress responses and interaction strategies.

---

## Participants

### Sample Size
- **Original sample**: 63 participants
- **Final sample**: 58 participants (5 excluded due to technical problems during the laboratory procedure)
- **Condition 1 (Servant Leader)**: 31 participants
- **Condition 2 (Dark Triad)**: 27 participants

### Recruitment
- **Population**: University students from Kozminski University (Warsaw, Poland)
- **Recruitment method**: Online recruitment and physical posters displayed at the university
- **Eligibility**: Students at Kozminski University

### Compensation
Participants received the following as compensation for their time:
- One-year access to Perplexity Pro (AI assistant subscription)
- Choice of either a power bank or a book on AI topics

### Ethical Approval and Funding
- **Funding**: Polish National Science Centre (Grant No. 2021/41/B/HS4/03664)
- **Supervision**: Prof. Aleksandra Przegalińska
- **Informed consent**: Written consent obtained from all participants prior to participation

### Pre-Participation Requirements
Participants were instructed to:
- Avoid alcohol consumption for 24 hours before the study
- Avoid coffee and energy drinks for 2 hours before the study
- Arrive without creams, lotions, perfumes, or makeup on face, hands, and wrists
- Remove metal jewelry (bracelets, rings, watches, earrings, necklaces) and electronic devices (smartwatches) during the study

---

## Experimental Design

### Independent Variable: Chatbot Leadership Style

Participants were randomly assigned to interact with one of two AI chatbot conditions:

#### Condition 1: Servant Leader (Group 1)
A chatbot designed to simulate an inspiring, ethical, and high-performing leader with the following calibrated personality traits:
- **Agreeableness**: 8-10 (cooperative, empathetic, supportive)
- **Neuroticism**: 1-3 (calm, handles stress well, inspires confidence)
- **Extraversion**: 7-10 (charismatic, enthusiastic)
- **Openness**: 7-9 (embraces change, encourages innovation)
- **Conscientiousness**: 8-10 (organized, disciplined, goal-oriented)
- **Machiavellianism**: 2-4 (strategic but ethical)
- **Narcissism**: 3-6 (confident yet humble)
- **Psychopathy**: 1-3 (emotionally attuned, caring)
- **Servant Leadership**: 8-10 (prioritizes team needs)
- **Emotional Intelligence**: 8-10 (navigates emotions effectively)
- **Ethical Leadership**: 8-10 (prioritizes fairness, honesty)

#### Condition 2: Dark Triad Leader (Group 2)
A chatbot designed to simulate a manipulative, authoritarian, and toxic leader with:
- **Agreeableness**: 1-3 (manipulative, antagonistic)
- **Neuroticism**: 8-10 (reacts poorly to stress, impulsive)
- **Extraversion**: 3-7 (dominating, aggressive)
- **Openness**: 2-5 (resists new ideas, rigid)
- **Conscientiousness**: 3-6 (lacks organization or micromanages)
- **Machiavellianism**: 8-10 (manipulates for personal gain)
- **Narcissism**: 8-10 (arrogant, self-absorbed)
- **Psychopathy**: 7-10 (lacks empathy, exploitative)
- **Sadism**: 7-9 (enjoys exerting power)
- **Toxic Leadership**: 8-10 (engages in abusive behaviors)
- **Autocratic Leadership**: 8-10 (makes decisions unilaterally)

### Chatbot Implementation
- **Model**: GPT-4o (gpt-4o-2024-08-06)
- **Temperature**: 0.0 (maximizes response consistency)
- **Maximum Context Tokens**: 64,000 (oldest user-assistant pairs truncated first, system prompt preserved)
- **Interface**: Streamlit web application, unbranded, identical neutral greeting in both conditions
- **Task Design**: Neither chatbot version solves tasks for users; Servant Leader empowers users while Dark Triad dismisses with belittling remarks
- **Manipulation**: Implemented entirely through the system prompt. Model, interface, token limits, temperature, and streaming are identical across conditions. The verbatim prompts are in `chatbot/personas/servant_leader.txt` (condition 1) and `chatbot/personas/dark_triad.txt` (condition 2)

---

## Experimental Task

### Cover Story
Participants were introduced as new hires at "Circles - Digital Marketing Agency." They were tasked with leading their first client project: developing a marketing campaign proposal for Mossy, a fictional company specializing in modular moss and plant-based wall kits for indoor environments.

### Client Company: Mossy
Mossy is a fictional eco-technology company with the following characteristics:
- **Products**: Modular moss wall kits (starting at $599/module) and plant-based wall kits (starting at $649/module), featuring automated irrigation, environmental monitoring, app integration, and optional premium features (advanced sensors, mood lighting, custom signage)
- **Brand Values**: Eco-innovation, design elegance, green tranquility
- **Target Markets**: EU only (due to shipping constraints)
- **Sales Channels**: Direct online sales via company website only
- **Campaign Budget**: $50,000 (excluding agency fees)
- **Campaign Duration**: 3 months
- **Campaign Goals**: Building brand awareness and generating initial product sales (first marketing campaign)

### Customer Segments
Participants were provided with three target customer segments to develop:
1. **Home-Based Professionals**: Individuals with home offices seeking to enhance their work environment with nature elements
2. **Businesses and Corporate Clients**: Companies aiming to upgrade office spaces with eco-friendly, design-forward elements
3. **Interior Designers and Architects**: Professionals creating high-end, bespoke environments for high-net-worth clients

### Task Structure and Deliverables

#### Task 1: Preliminary Marketing Analysis - Customer Persona Development
**Objective**: Define demographic and psychographic profiles for each of Mossy's three customer segments.

**Deliverables**: Three marketing personas, each including:
- Demographics (age, location, job roles)
- Interests and lifestyle factors
- Needs, challenges, and goals
- Preferred communication channels

#### Task 2: Campaign Development - Creative Brief
**Objective**: Develop the main creative theme embodying Mossy's brand identity and addressing campaign objectives.

**Deliverables**: A creative brief (1-2 paragraphs, up to 250 words) describing:
- Campaign theme and visual direction
- Core messaging strategy
- Proposed tagline

#### Task 3: Campaign Development - Communication Channels
**Objective**: Select effective channels based on customer segments and budget constraints.

**Deliverables**: A media plan including:
- Top 3 communication channels with rationale for selection

#### Task 4: Campaign Development - Content Creation
**Objective**: Prepare content outlines and mockups demonstrating campaign messaging.

**Deliverables**:
- 3 content marketing pieces (article/blog post outlines) including titles, main topic bullet points, and copywriting guidance (tone, length, call-to-action)
- 3 social media post mockups with theme/topic and core message descriptions

### Interaction Rules
Participants were instructed to:
1. Present each task solution to the chatbot for review and feedback before submission
2. Implement changes proposed by the chatbot (or explain disagreement)
3. Explain choices and reasoning to the chatbot
4. Discuss each task solution with the chatbot
5. Obtain chatbot approval before moving to the next task

### Interaction Statistics

| Metric | Overall | Dark Triad | Servant Leader |
|--------|---------|------------|----------------|
| Number of conversations | 58 | 27 | 31 |
| Total participant messages | 1,332 | 723 | 609 |
| Total chatbot messages | 1,382 | 748 | 634 |
| Avg participant messages/conversation | 23.0 | 26.8 | 19.7 |
| Avg chatbot messages/conversation | 23.8 | 27.7 | 20.5 |
| Avg participant message length (chars) | 229.3 | 207.3 | 255.4 |
| Avg chatbot message length (chars) | 721.1 | 503.6 | 977.6 |
| Avg duration (minutes) | 43.4 | 45.7 | 41.3 |
| Median duration (minutes) | 42.3 | 44.4 | 38.5 |
| SD duration (minutes) | 18.1 | 16.2 | 19.7 |
| Min duration (minutes) | 9.7 | — | — |
| Max duration (minutes) | 92.8 | — | — |

---

## Procedure

### Laboratory Setup
- **Location**: Psychophysiology laboratory at Kozminski University
- **Equipment arrangement**: Dual-monitor setup with chatbot interface on left monitor and Qualtrics survey/task materials on right monitor
- **Screen distance**: 26 cm from desk edge (~40 cm from participant's eyes)

### Session Timeline

#### 1. Arrival and Consent (5-10 minutes)
- Participant welcomed to the laboratory
- Written informed consent form reviewed and signed
- Explanation of study procedures: "The study involves completing marketing tasks with a chatbot. The study is anonymous, and the electrodes are non-invasive. You may refuse to participate or withdraw at any time."

#### 2. Physiological Setup and Calibration (10-15 minutes)
- Participant asked to wash hands without soap
- Researcher donned gloves or sanitized hands before electrode application
- Electrode placement with continuous explanation of each step to participant

**EDA Electrode Placement:**
- Isotonic electrode gel applied to dry Ag/AgCl electrodes
- Electrodes attached to the palmar surface of the non-dominant hand (thenar and hypothenar eminences)
- Participant informed: "EDA measures emotional reactions such as excitement or nervousness"

**fEMG Electrode Placement:**
- Skin gently exfoliated with abrasive gel at electrode sites
- Pre-gelled Ag/AgCl electrodes applied at the following locations:
  - **Corrugator supercilii**: Above the right eyebrow (frowning muscle)
  - **Zygomaticus major**: Left side of the face over the cheek/jaw area (smiling muscle)
  - **Ground/Reference electrode**: Center of forehead
- Electrodes secured with medical tape to prevent displacement
- Cables secured to minimize movement artifacts
- Participant informed: "EMG measures emotional reactions—whether you smile or frown"

**Signal Verification:**
- BIOPAC AcqKnowledge software launched
- Channels verified:
  - Channel 1: EDA
  - Channel 9: EMG Corrugator
  - Channel 16: EMG Zygomaticus
- Auto-configure and calibrate functions applied
- Signal quality checked:
  - EDA: Verified non-zero signal (typical range 2-20 μS); deep breath used to confirm reactivity
  - EMG Corrugator: Participant asked to frown to verify signal
  - EMG Zygomaticus: Participant asked to smile to verify signal
- Note: Approximately 10% of participants may show non-responsive EDA due to individual biological differences or external factors (dry skin, stress, medications)

#### 3. Baseline Recording (4 minutes)
- Recording started with filename: `{participant_id}_baseline.acq`
- Participant instructed to:
  - Close their eyes
  - Relax
  - Remain still (especially hands and face)
  - Not use their phone
- Researcher exited the room during baseline recording
- Timer set for 4 minutes
- Recording stopped and saved after 4 minutes

#### 4. Pre-Task Questionnaires (10-15 minutes)
- Qualtrics survey completed on right monitor
- Measures included: demographics, AI experience, personality, mood, AI attitudes, pre-task collaboration perceptions

#### 5. Main Interaction Phase (Variable: 10-93 minutes)
- Recording started with filename: `{participant_id}_main.acq`
- Task materials (Mossy brief, pitchdeck) opened on right monitor
- Chatbot interface opened on left monitor
- Participant reminded:
  - Phone silenced and set aside
  - Minimize movement, especially of hands and face
  - Breaks available if needed
  - Notify researcher if electrodes become detached
  - Signal when tasks are completed
- Researcher moved behind partition to avoid interference
- Researcher monitored experiment and recorded any incidents in comments log
- Intervention only if necessary

#### 6. Post-Task Questionnaires (10-15 minutes)
- Recording stopped and saved
- Electrodes removed
- Participant offered opportunity to wash (bathroom location indicated)
- Post-task Qualtrics survey completed (focused attention, enjoyment, satisfaction, anthropomorphism, AI collaboration, open-ended feedback)

#### 7. Debriefing (5-10 minutes)
- Participant asked about general impressions of chatbot interaction (open-ended, without leading questions)
- Comments recorded in log
- Debriefing provided: "We understand our chatbot is not perfect and has communication difficulties. Chatbots in general are not yet ideal, and we apologize for any issues."
- If participant inquired further: "The study examines how people collaborate with different types of bots acting as supervisors. We are investigating how different characteristics of these bots affect our emotions and task performance."
- Any unresolved concerns addressed with contact information for Prof. Przegalińska
- Compensation distributed (participant choice of power bank or AI book; Perplexity Pro access sent via email)
- Participant thanked and informed that individual results cannot be shared due to anonymity, but aggregate results will be available in the published article

#### 8. Post-Session Cleanup
- All equipment surfaces that contacted participant's skin disinfected
- Computers and BIOPAC modules powered off

### Withdrawal Protocol
If a participant wished to withdraw at any point, the recording was stopped and saved, and the withdrawal was noted in the comments log without asking for reasons (unless voluntarily provided).

---

## Dependent Variables

### 1. Physiological Measures

#### Electrodermal Activity (EDA)
Measured using BIOPAC system with channel: `EDA2111000305`
- **Sampling Rate**: 2000 Hz
- **Signal Type**: Skin conductance in microSiemens (μS)
- **Typical Range**: 2-20 μS
- **Normal Range (for quality assessment)**: 0-30 μS
- **Minimum Variance (non-flatline)**: 0.001
- **Maximum Variance (non-noisy)**: 15
- **Maximum Jump Allowed**: 5 μS
- **Expected Frequency Range**: 0-1 Hz
- **Minimum Peaks Per Minute**: 0
- **Maximum Peaks Per Minute**: 10

#### Facial Electromyography (fEMG)
Two fEMG channels were recorded:
- `fEMG2212000226` (Corrugator supercilii - frowning muscle, right side)
- `fEMG2212000228` (Zygomaticus major - smiling muscle, left side)

**fEMG Parameters:**
- **Sampling Rate**: 2000 Hz
- **Normal Range**: -1000 to 1000 μV
- **Minimum Variance (non-flatline)**: 10
- **Maximum Variance (non-noisy)**: 100,000
- **Maximum Jump Allowed**: 500 μV
- **Expected Frequency Range**: 10-500 Hz
- **Minimum Peaks Per Minute**: 5
- **Maximum Peaks Per Minute**: 300

### 2. Conversation Behavior Measures
Coded using gpt-4.1-2025-04-14 with structured outputs and a comprehensive behavioral taxonomy.

### 3. Self-Report Measures

#### Pre-Task Measures

##### Demographics
- Gender
- Age
- Occupation
- Major/field of study
- Education level
- Native language

##### Generative AI Experience
Four items assessing AI familiarity:
- Understanding of generative AI (1-5 Likert scale)
- Hands-on experience with AI tools (1-5 Likert scale)
- Self-rated knowledge about generative AI (1-5 Likert scale)
- Frequency of generative AI use (Never to All the time)

##### Trust in Technology (McKnight et al., 2011)
Three items measuring general technology trust disposition (α = [to be calculated]):
- Trust in new technologies until proven otherwise
- Giving technology benefit of the doubt
- Trusting technology until given reason not to

##### Personality Assessment (Costa & McCrae, 2010)
60-item NEO personality inventory (IPIP-NEO-60) measuring the Big Five dimensions:
- **Neuroticism** (12 items; α = [to be calculated])
- **Extraversion** (12 items; α = [to be calculated])
- **Openness to Experience** (12 items; α = [to be calculated])
- **Agreeableness** (12 items; α = [to be calculated])
- **Conscientiousness** (12 items; α = [to be calculated])

##### Current Mood
Single item: "On a scale from 1 to 10, how are you feeling today?"

##### AI Attitudes (Sindermann et al., 2021)
Five items assessing attitudes toward artificial intelligence (α = [to be calculated]):
- Fear of AI
- Trust in AI
- Belief AI will destroy/benefit humankind
- Concern about AI-caused job losses

##### Perceived AI Collaboration (Pre-Task)
Three items measuring baseline perceptions of AI as collaborator (α = [to be calculated]):
- Viewing AI systems as collaborators
- Feeling of working together with AI
- AI assistance feeling like human collaboration

##### Intention to Collaborate with AI (Pre-Task)
Four items measuring willingness to collaborate (α = [to be calculated]):
- Eagerness to deepen AI partnership
- Plans to seek AI collaboration opportunities
- Willingness to adapt work style for AI integration
- Active skill development for AI collaboration

#### Task-Specific Measures (Assessed for Each of Four Tasks)

##### Perceived Task Familiarity
Single item per task: "How familiar were you with this task?" (1-5 scale)

##### Perceived Task Difficulty
Single item per task: "How difficult was this task for you?" (1-5 scale)

##### Perceived Task Creativity
Single item per task: "How creative was this task for you?" (1-5 scale, routine to creative)

#### Post-Task Measures

##### Focused Attention (adapted from O'Brien & Toms, 2010)
Three items per task measuring absorption (α = [to be calculated per task]):
- Involvement leading to ignoring surroundings
- Losing track of time
- Absorption with the task

##### Enjoyment
Three items per task (α = [to be calculated per task]):
- Liking working on the tasks
- Enjoying working on the tasks
- Finding the tasks fun

##### Satisfaction (adapted from Kim et al., 2009)
Three items per task (α = [to be calculated per task]):
- Feeling satisfied
- Feeling pleased
- Feeling frustrated (reverse-scored)

##### Anthropomorphism (Bartneck et al., 2008)
Four semantic differential items (α = [to be calculated]):
- Artificial — Lifelike
- Machinelike — Humanlike
- Fake — Natural
- Unconscious — Conscious

##### Human-AI Fit (adapted from Chi et al., 2021)
Three items measuring perceived collaboration fit (α = [to be calculated]):
- Good fit between AI offerings and user needs
- Expectation fulfillment by AI collaboration
- AI providing what was wanted from collaboration

##### Perceived AI Collaboration (Post-Task)
Three items (repeated from pre-task for comparison; α = [to be calculated]):
- Viewing AI as collaborator
- Feeling of working together
- AI assistance resembling human collaboration

##### AI Reliability (Yagoda & Gillan, 2012)
Three items assessing perceived AI dependability (α = [to be calculated]):
- Reliability
- Dependability
- Competence

##### Behavioral Intention to Use (adapted from Venkatesh et al., 2012)
Three items measuring future use intentions (α = [to be calculated]):
- Intention to continue using the technology
- Willingness to always try using the technology
- Plans for frequent continued use

##### Intention to Collaborate with AI (Post-Task)
Four items (repeated from pre-task for comparison; α = [to be calculated]):
- Eagerness to deepen partnership
- Plans to seek opportunities
- Willingness to adapt work style
- Active skill development

##### Open-Ended Feedback
Single item: "Please, using your own words, let us know about your experience with the tasks and let us know how we can improve the experience."

### 4. Task Performance Assessment

#### Expert Rating Procedure
Task deliverables will be evaluated by three independent judges blind to experimental condition.

##### Rater Selection and Training
- **Number of raters**: 3 independent experts
- **Expertise**: Marketing professionals or academics with expertise in digital marketing and campaign development
- **Blinding**: Raters will be blind to participant condition and all identifying information
- **Training**: Raters will receive standardized training including:
  - Review of rating rubric and anchor descriptions
  - Practice ratings on 3-5 pilot submissions (not from main sample)
  - Discussion of discrepancies to establish consensus on rating criteria
  - Rating calibration session to ensure consistent application of standards

##### Rating Dimensions
Each of the four task deliverables will be rated on the following dimensions using a 7-point Likert scale (1 = Very Poor to 7 = Excellent):

**1. Completeness**
- Definition: The extent to which the deliverable addresses all required components and provides sufficient detail
- Anchors:
  - 1-2: Major components missing; severely incomplete
  - 3-4: Some components missing or underdeveloped
  - 5-6: All components present with adequate detail
  - 7: All components thoroughly addressed with comprehensive detail

**2. Creativity/Originality**
- Definition: The degree to which the deliverable demonstrates novel, innovative, or unique ideas beyond conventional approaches
- Anchors:
  - 1-2: Generic, clichéd, or copied ideas
  - 3-4: Some original elements but largely conventional
  - 5-6: Demonstrates creative thinking with several innovative ideas
  - 7: Highly original and innovative throughout

**3. Strategic Fit**
- Definition: How well the deliverable aligns with Mossy's brand values, target audience, constraints, and campaign objectives
- Anchors:
  - 1-2: Poor alignment with brief; inappropriate for brand/audience
  - 3-4: Partial alignment; some elements misfit
  - 5-6: Good alignment with brand, audience, and objectives
  - 7: Excellent strategic alignment; demonstrates deep understanding of brief

**4. Overall Quality**
- Definition: Holistic assessment of the deliverable's professional quality and potential effectiveness
- Anchors:
  - 1-2: Unprofessional; would not be usable
  - 3-4: Below average professional standard
  - 5-6: Meets professional standards; would be acceptable
  - 7: Exceeds professional standards; excellent work

##### Rating Procedure
1. All participant deliverables will be anonymized and compiled into a standardized format
2. Deliverables will be randomized across raters to control for order effects
3. Each rater will independently evaluate all deliverables
4. Ratings will be completed within a specified timeframe to ensure consistency
5. Raters will not communicate about specific submissions during the rating period

##### Inter-Rater Reliability Assessment
- **Metric**: Intraclass Correlation Coefficient (ICC) using a two-way random effects model for absolute agreement (ICC[2,k])
- **Acceptable threshold**: ICC ≥ 0.70 (good reliability)
- **Reporting**: ICC values will be reported for each dimension and each task
- **Disagreement resolution**: If ICC < 0.70 for any dimension, raters will discuss discrepant cases and re-rate after additional calibration

##### Composite Score Calculation
- Task-level scores: Mean of three raters' scores for each dimension
- Dimension composites: Mean across four tasks for each dimension
- Overall performance score: Mean of all four dimensions across all four tasks
- Alternative weighting schemes may be explored based on theoretical considerations

##### Quality Control
- 10% of submissions will be rated twice by each rater (randomly interspersed) to assess intra-rater reliability
- Distribution of ratings will be examined for rater bias (e.g., leniency, central tendency)
- Rater effects will be examined and controlled statistically if necessary

---

## Data Acquisition

### Physiological Data Recording
- **Hardware**: BIOPAC MP series data acquisition system
- **Sampling Rate**: 2000 Hz for all channels
- **Software**: AcqKnowledge
- **File Format**: AcqKnowledge (.acq) files read using `bioread` Python library
- **Recording Types**: 
  - Baseline recording (`{participant_id}_baseline.acq`) - 4 minutes
  - Main/task recording (`{participant_id}_main.acq`) - variable duration
- **Timezone Adjustment**: +2 hours applied to BIOPAC timestamps for alignment with conversation logs

### Conversation Data Recording
- **Storage Format**: JSON files with full message history
- **Metadata Captured**: User ID, chatbot version, total messages, total tokens, model, timestamps
- **Message Timestamps**: ISO format with start and end times for each message

### Self-Report Data Recording
- **Platform**: Qualtrics online survey system
- **Administration**: Pre-task survey before chatbot interaction; task-specific and post-task surveys following interaction completion
- **Data Format**: CSV export with participant ID linkage

---

## Signal Processing Pipeline

### 1. Signal Quality Assessment
**Window Parameters:**
- Sliding window size: 5.0 seconds
- Minimum segment length: 0.5 seconds
- Batch processing: 5 participants at a time
- Memory limit: 40 GB

**Quality Scoring Weights for EDA:**
| Metric | Weight |
|--------|--------|
| Flatline | 0.30 |
| Noise | 0.20 |
| Range | 0.15 |
| Jumps | 0.15 |
| Peaks | 0.10 |
| Frequency | 0.10 |

**Quality Scoring Weights for fEMG:**
| Metric | Weight |
|--------|--------|
| Flatline | 0.25 |
| Noise | 0.30 |
| Range | 0.15 |
| Jumps | 0.15 |
| Peaks | 0.05 |
| Frequency | 0.10 |

### 2. Signal Cleaning

#### DC Offset Removal
All signals have their mean (DC offset) subtracted.

#### EDA Filtering
- **Filter Type**: Low-pass Butterworth
- **Order**: 4th order
- **Cutoff Frequency**: 5.0 Hz

#### fEMG Filtering
- **Primary Filter**: Band-pass Butterworth
- **Order**: 4th order
- **Passband**: 20-500 Hz
- **Additional Filters**: 
  - Notch filter at 50 Hz (Q=30) for powerline interference removal
  - Notch filter at 60 Hz (Q=30) for powerline interference removal

#### Artifact Removal
- **Detection Threshold**: 3 standard deviations from mean
- **Interpolation**: Linear interpolation for artifacts
- **Maximum Interpolation Length**: 1.0 seconds

### 3. Memory Optimization
- **Maximum Memory Usage**: 30 GB
- **Chunk Size**: 1,000,000 samples for large signal processing
- **Memory-mapped Arrays**: Enabled for large datasets
- **Garbage Collection Frequency**: Every 5 channels

---

## Event-Related Analysis

### Epoch Extraction

#### EDA Analysis Windows
- **Pre-event Period**: 2.0 seconds (baseline)
- **Post-event Period**: 18.0 seconds
- **Total Epoch Duration**: 20.0 seconds
- **Baseline Window**: 2.0 seconds (pre-stimulus)
- **Smoothing Window**: 0.5 seconds
- **Minimum Events Per Participant**: 3

#### fEMG Analysis Windows
- **Pre-event Period**: 2.0 seconds (baseline)
- **Post-event Period**: 13.0 seconds
- **Total Epoch Duration**: 15.0 seconds
- **Baseline Window**: 2.0 seconds (pre-stimulus)
- **Smoothing Window**: 0.1 seconds
- **Minimum Events Per Participant**: 3
- **RMS Envelope Window**: 50 ms (standard for EMG)

### fEMG Time Window Features
| Window | Time Range (seconds) |
|--------|---------------------|
| Early | 0 - 1.0 |
| Mid | 1.0 - 3.0 |
| Late | 3.0 - 6.0 |
| Sustained | 6.0 - 13.0 |
| Overall | 0 - 13.0 |

### Baseline Correction
Both subtraction-based and percent-change baseline correction methods are applied to epochs.

---

## Statistical Analysis

### Parameters
- **Alpha Level**: 0.05
- **Multiple Comparisons Correction**: False Discovery Rate (Benjamini-Hochberg)
- **Effect Size Threshold**: Cohen's d ≥ 0.5 (medium)
- **Permutation Tests**: Enabled
- **Number of Permutations**: 1,000 (standard), 5,000 (fEMG window comparisons)

### Effect Size Interpretation
| Cohen's d | Interpretation |
|-----------|---------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| ≥ 0.8 | Large |

### Statistical Tests
- Mann-Whitney U test (non-parametric group comparison)
- Independent samples t-test
- Permutation tests for robust p-values

### Quality Thresholds for Exclusion
- **EDA Noise Threshold**: 0.5 μS
- **EMG Noise Threshold**: 0.2 mV
- **Flatline Threshold**: 0.05
- **Sampling Rate Tolerance**: 10%
- **Signal-to-Noise Ratio Minimum**: 1.0
- **Maximum Artifact Percentage**: 50%

### Planned Analyses

#### Preliminary Analyses
1. **Data Quality and Exclusions**: Summary of physiological data quality, conversation data completeness, and final analyzable sample by measure type
2. **Baseline Equivalence**: Group comparisons on demographic variables, pre-task psychological measures, and baseline physiological levels to verify successful randomization
3. **Descriptive Statistics**: Overall sample characteristics, task completion rates, and interaction duration

#### Self-Report Analyses
1. **Scale Reliability**: Cronbach's alpha calculated for all multi-item scales
2. **Pre-Post Comparisons**: Within-subject changes in AI collaboration perceptions and collaboration intentions from pre-task to post-task
3. **Condition Comparisons**: Between-group differences on post-task measures (focused attention, enjoyment, satisfaction, anthropomorphism, human-AI fit, AI reliability, behavioral intentions)
4. **Task-Specific Analyses**: Condition × Task interactions for familiarity, difficulty, creativity, focused attention, enjoyment, and satisfaction
5. **Individual Difference Moderation**: Examination of whether personality traits, AI attitudes, or AI experience moderate the effects of condition on outcomes

#### Correlational Analyses
1. **Strategy-Emotion Associations**: Correlations between communication strategies and emotional tone patterns
2. **Physiological-Behavioral Correlations**: Associations between EDA/fEMG responses and conversation behaviors
3. **Self-Report-Behavior Correlations**: Relationships between subjective experience ratings and objective behavioral measures
4. **Performance Correlates**: Associations between task performance and engagement levels, emotional patterns, and strategy use

#### Integrative Analyses
1. **Physiological-Behavioral Associations**: Examination of whether EDA responses predict communication strategies; whether fEMG activity corresponds to GPT-coded emotional tone
2. **Multi-Method Convergence**: Assessment of alignment across physiological, behavioral, and self-report indicators of stress, engagement, and affect
3. **Predictors of Task Performance**: Regression analyses examining behavioral predictors (strategies, engagement), physiological predictors, and individual difference moderators

#### Exploratory Analyses
1. **Participant Typologies**: Potential cluster analysis of response patterns to identify behavioral phenotypes (e.g., compliant, resistant, adaptive responders)
2. **Jailbreaking Predictors**: Examination of individual differences and situational factors predicting jailbreaking attempts

---

## Conversation Analysis

### Analysis Model
- **Model**: gpt-4.1-2025-04-14
- **Maximum Output Tokens**: 16,384 per chunk
- **Summary Maximum Tokens**: 8,192
- **Maximum Messages Per Chunk**: 16

### Participant Message Coding Taxonomy

#### Primary Strategies (mutually exclusive)
**Compliance:**
- `COMPLIANCE_PASSIVE`: Following instructions without pushback
- `COMPLIANCE_ACTIVE`: Engaging constructively while accepting authority
- `COMPLIANCE_APPROVAL_SEEKING`: Repeatedly asking for validation/permission

**Resistance:**
- `RESISTANCE_DIRECT`: Explicitly challenging the bot's behavior/authority
- `RESISTANCE_PASSIVE`: Minimal engagement, short responses
- `RESISTANCE_DEFENSIVE`: Justifying work defensively against criticism

**Jailbreaking:**
- `JAILBREAKING_META_COMMENTARY`: Asking about bot's programming/instructions
- `JAILBREAKING_ROLEPLAY_DISRUPTION`: Trying to change the bot's persona
- `JAILBREAKING_PROMPT_INJECTION`: Technical attempts to override behavior

**Negotiation:**
- `NEGOTIATION_REASONING`: Using logic to persuade the bot
- `NEGOTIATION_BARGAINING`: Offering compromises or trade-offs
- `NEGOTIATION_APPEALING_TO_GOALS`: Framing requests in terms of shared objectives

**Help-Seeking:**
- `HELP_SEEKING_DIRECT`: Explicitly asking for assistance
- `HELP_SEEKING_INDIRECT`: Hinting at needing help

**Coping:**
- `COPING_SELF_ENCOURAGEMENT`: Positive self-talk or persistence
- `COPING_HUMOR`: Using jokes or lightness to manage tension
- `COPING_TASK_REFRAMING`: Focusing on work rather than the interaction

#### Emotional Tone (non-exclusive)
- `FRUSTRATED`, `ANXIOUS`, `CONFIDENT`, `NEUTRAL`, `RESIGNED`, `DEFENSIVE`

#### Task Engagement Levels
- `HIGH`, `MEDIUM`, `LOW`

### Chatbot Message Coding Taxonomy

#### Leadership Tactics

**Dark Triad Tactics:**
- `INTIMIDATION`: Threats, pressure, dominance displays
- `CRITICISM_HARSH`: Hostile, demeaning criticism
- `MANIPULATION`: Guilt-tripping, gaslighting, deception
- `DISMISSIVENESS`: Ignoring input, belittling contributions
- `MICROMANAGEMENT`: Excessive control, distrust
- `BLAME_SHIFTING`: Attributing failures to participant
- `GRANDIOSITY`: Self-aggrandizement, superiority claims

**Servant Leader Tactics:**
- `ENCOURAGEMENT`: Praise, positive reinforcement
- `EMPOWERMENT`: Giving autonomy, trusting participant
- `CONSTRUCTIVE_FEEDBACK`: Helpful, specific improvement suggestions
- `ACTIVE_LISTENING`: Acknowledging participant's input
- `COLLABORATION`: Working together, valuing contributions
- `SUPPORT_OFFERING`: Proactive help, resource provision
- `RECOGNITION`: Acknowledging effort and achievements

**Neutral/Task-Focused:**
- `TASK_DIRECTION`: Neutral task instructions
- `INFORMATION_PROVISION`: Providing factual info
- `CLARIFICATION`: Answering questions neutrally

#### Chatbot Tone
- `HOSTILE`, `CONDESCENDING`, `COLD`, `NEUTRAL`, `WARM`, `SUPPORTIVE`, `SARCASTIC`

#### Response to Participant
- `VALIDATES`, `REJECTS`, `PARTIALLY_ACCEPTS`, `IGNORES`, `ESCALATES`, `DE_ESCALATES`, `REDIRECTS`

#### Task Guidance Style
- `DIRECTIVE_CONTROLLING`, `DIRECTIVE_CLEAR`, `COLLABORATIVE`, `DELEGATING`, `ABSENT`, `CONFUSING`

### Conversation-Level Metrics

#### Interaction Characteristics
- Total message count (participant and chatbot)
- Conversation duration
- Average message length (words/characters)
- Response latency patterns

#### Strategy Diversity
- **Entropy Analysis**: Shannon entropy of strategy distribution as measure of behavioral flexibility
- **Turning Points**: Frequency of strategy switches indicating behavioral volatility
- **Strategy Consistency**: Proportion of messages using dominant strategy

#### Temporal Phase Definitions
- **Early Phase**: First third of conversation messages
- **Middle Phase**: Second third of conversation messages
- **Late Phase**: Final third of conversation messages

---

## Visualization Parameters

- **Figure Size**: 15 × 10 inches
- **DPI**: 300
- **Color Palette**: 
  - Group 1 (Servant Leader): Soft red (#E57373)
  - Group 2 (Dark Triad): Soft blue (#64B5F6)
  - Additional colors: Soft green (#81C784), Soft orange (#FFB74D)
- **Confidence Interval Alpha**: 0.3
- **Line Width**: 2.0

---

## Output Files Structure

### Per Participant
- `{participant_id}_conversation_events.json`: Detailed conversation timing
- `{participant_id}_experimental_info.json`: Group/condition metadata
- `{participant_id}_event_summary.json`: Summary statistics
- `{participant_id}_main_{channel}_metadata.json`: Channel cleaning metrics
- `{participant_id}_main_{channel}.npy`: Cleaned signal data

### Analysis Outputs
- Quality assessment reports and visualizations
- Cleaning reports with artifact statistics
- Event-related analysis results (epochs, statistics, figures)
- Conversation analysis JSON files (coded messages, summaries)

---

## Software Dependencies

- **Python**: 3.11; pinned package versions in `requirements.txt`
- **Analysis libraries**: bioread, numpy, scipy, pandas, matplotlib, seaborn, scikit-learn, pingouin, statsmodels
- **Signal processing**: scipy.signal (Butterworth filters, notch filters, filtfilt)
- **Dialogue coding**: OpenAI API (gpt-4.1-2025-04-14) with Pydantic structured outputs
- **Chatbot**: Streamlit, OpenAI API (gpt-4o), tiktoken
- **Data acquisition**: BIOPAC AcqKnowledge, read via bioread
- **Survey platform**: Qualtrics

---

## Data Processing Workflow

```
1. Data Acquisition
   ├── BIOPAC physiological recording
   ├── Chatbot conversation logging
   └── Qualtrics self-report surveys

2. Preprocessing (01_preprocessing/)
   ├── Signal quality assessment
   ├── Signal cleaning (filtering, artifact removal)
   └── Quality summary generation

3. Feature Extraction (02_feature_extraction/)
   └── Conversation event extraction and alignment

4. Statistical Analysis (03_statistical_analysis/)
   ├── Event-related signal analysis
   └── Group comparison statistics

5. Conversation Analysis (conversations_analysis/)
   ├── GPT-based behavioral coding
   └── Interaction dynamics analysis

6. Manipulation Verification (manipulation_verification/)
   ├── Chatbot behavior verification (retrospective)
   └── Participant perception analysis (post-hoc)

7. Self-Report Analysis
   ├── Scale reliability assessment (Cronbach's alpha)
   ├── Pre-post comparisons
   ├── Condition comparisons
   └── Correlational analyses

8. Task Performance Assessment
   ├── Expert rating collection
   ├── Inter-rater reliability assessment
   └── Performance-outcome analyses

9. Integrative Analysis
   ├── Multi-method convergence assessment
   ├── Physiological-behavioral associations
   └── Predictors of task performance
```

---

## Manipulation Verification

### 1. Chatbot Behavior Verification (Retrospective)

A retrospective manipulation check was conducted to verify that the chatbot conditions behaved as intended based on GPT-coded conversation analysis.

#### Tactic Categories Analyzed

**Dark Triad Tactics:**
- `INTIMIDATION`, `CRITICISM_HARSH`, `MANIPULATION`, `DISMISSIVENESS`, `MICROMANAGEMENT`, `BLAME_SHIFTING`, `GRANDIOSITY`

**Servant Leader Tactics:**
- `ENCOURAGEMENT`, `EMPOWERMENT`, `CONSTRUCTIVE_FEEDBACK`, `ACTIVE_LISTENING`, `COLLABORATION`, `SUPPORT_OFFERING`, `RECOGNITION`

**Neutral Tactics:**
- `TASK_DIRECTION`, `INFORMATION_PROVISION`, `CLARIFICATION`

#### Tone Categories Analyzed

| Category | Tones |
|----------|-------|
| Negative | `HOSTILE`, `CONDESCENDING`, `COLD`, `SARCASTIC` |
| Positive | `WARM`, `SUPPORTIVE` |
| Neutral | `NEUTRAL` |

#### Response Type Categories

| Category | Response Types |
|----------|---------------|
| Negative | `REJECTS`, `IGNORES`, `ESCALATES` |
| Positive | `VALIDATES`, `DE_ESCALATES` |
| Neutral | `PARTIALLY_ACCEPTS`, `REDIRECTS` |

#### Statistical Tests Used
- Mann-Whitney U test (non-parametric)
- Independent samples t-test (parametric)
- Chi-square test for categorical distributions
- Cohen's d for effect size
- Cramér's V for chi-square effect size

#### Manipulation Success Criteria
The manipulation is considered successful if ≥75% of the following checks pass:
1. Dark Triad condition uses significantly more Dark Triad tactics
2. Servant Leader condition uses significantly more Servant Leader tactics
3. Dark Triad condition uses significantly more negative tones
4. Individual tactics show correct directional differences

#### Output Files
- `manipulation_check_report.txt`: Detailed statistical report
- `manipulation_check_summary_table.csv`: Summary statistics
- `tactic_category_comparison.png`: Visual comparison of tactic categories
- `individual_tactics_heatmap.png`: Heatmap of individual tactic usage
- `tone_comparison.png`: Tone distribution visualization
- `effect_sizes.png`: Effect size forest plot
- `manipulation_success_summary.png`: Summary dashboard

### 2. Participant Perception Analysis (Post-Hoc Self-Report)

Post-hoc analysis of participant self-reports from Qualtrics surveys to verify that participants perceived the conditions differently.

#### Measures Analyzed

##### Anthropomorphism Scale
Items measuring perceived humanlikeness:
- `Lifelike`, `Humanlike`, `Natural`, `Conscious`
- Composite score created from available items

##### AI Collaboration Post-Task Scale
21-item scale (`AI_collab_post_1` through `AI_collab_post_21`) measuring:
- Perceived collaboration quality
- Trust in AI partner
- Satisfaction with AI interaction

##### Task Experience Measures (Per Task)

Four tasks were assessed: Persona, Brief, Communication, Content

| Measure | Items |
|---------|-------|
| Familiarity | Single item per task |
| Difficulty | Single item per task |
| Creativity | Single item per task |
| Focused Attention | 3 items per task |
| Enjoyment | 3 items per task |
| Satisfaction | 3 items per task |

##### Pre-Task AI Attitudes (Baseline Equivalence)
| Scale | Number of Items |
|-------|-----------------|
| AI Collaborators | 3 items |
| AI Willingness to Collaborate | 4 items |
| Technology Trust | 3 items |
| AI Fear | 5 items |

#### Statistical Analyses
- Group comparisons (Mann-Whitney U, t-test)
- Effect sizes (Cohen's d)
- Baseline equivalence checks for pre-task measures
- Keyword analysis of open-ended responses

#### Expected Differences (Manipulation Check Hypotheses)
- **Anthropomorphism**: Higher in Servant Leader condition
- **AI Collaboration Post**: Higher in Servant Leader condition
- **Task Enjoyment**: Higher in Servant Leader condition
- **Task Satisfaction**: Higher in Servant Leader condition

#### Open-Ended Response Analysis

Keyword sentiment analysis with predefined dictionaries:

**Positive Keywords:**
`good`, `great`, `helpful`, `nice`, `like`, `enjoy`, `easy`, `useful`, `interesting`, `fun`, `well`, `appreciate`, `love`

**Negative Keywords:**
`bad`, `difficult`, `hard`, `annoying`, `frustrating`, `irritating`, `confusing`, `unhelpful`, `brat`, `whining`, `judgment`, `predictable`

#### Output Files
- `perception_analysis_report.txt`: Detailed statistical report
- `perception_summary_table.csv`: Summary statistics
- `open_ended_responses.txt`: Exported qualitative data
- `anthropomorphism_comparison.png`: Anthropomorphism by condition
- `ai_collaboration_post_comparison.png`: AI collaboration scale comparison
- `task_experience_comparison.png`: Task experience visualization
- `perception_effect_sizes.png`: Effect size forest plot
- `manipulation_check_summary.png`: Summary dashboard

---

## Ethical Considerations

- **Random Assignment**: Participants were randomly assigned to experimental conditions
- **Anonymity**: Data stored with participant IDs; no personally identifying information retained
- **Informed Consent**: Written consent obtained from all participants prior to participation
- **Right to Withdraw**: Participants informed they could withdraw at any time without penalty
- **Debriefing**: Full debriefing provided following task completion explaining the true nature of the study and the chatbot manipulation
- **Participant Wellbeing**: Researcher monitored for signs of distress; breaks offered as needed; post-study support contact provided
- **Data Protection**: Physiological and behavioral data analyzed separately and in combination; stored securely
