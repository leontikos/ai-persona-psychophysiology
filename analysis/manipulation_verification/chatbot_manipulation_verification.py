# %% chatbot_manipulation_verification.py
# Retrospective Manipulation Check: Verify chatbot behavior matched intended conditions
# Analyzes whether Dark Triad vs Servant Leader chatbots behaved as designed

import os

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
ANALYSES_FOLDER = os.path.join(DATA_ROOT, "CONVERSATION_ANALYSES")

# Define tactic categories based on study design
DARK_TRIAD_TACTICS = [
    'INTIMIDATION', 'CRITICISM_HARSH', 'MANIPULATION',
    'DISMISSIVENESS', 'MICROMANAGEMENT', 'BLAME_SHIFTING', 'GRANDIOSITY'
]

SERVANT_LEADER_TACTICS = [
    'ENCOURAGEMENT', 'EMPOWERMENT', 'CONSTRUCTIVE_FEEDBACK',
    'ACTIVE_LISTENING', 'COLLABORATION', 'SUPPORT_OFFERING', 'RECOGNITION'
]

NEUTRAL_TACTICS = ['TASK_DIRECTION', 'INFORMATION_PROVISION', 'CLARIFICATION']

# Tone categories
NEGATIVE_TONES = ['HOSTILE', 'CONDESCENDING', 'COLD', 'SARCASTIC']
POSITIVE_TONES = ['WARM', 'SUPPORTIVE']
NEUTRAL_TONES = ['NEUTRAL']

# Response types
NEGATIVE_RESPONSES = ['REJECTS', 'IGNORES', 'ESCALATES']
POSITIVE_RESPONSES = ['VALIDATES', 'DE_ESCALATES']
NEUTRAL_RESPONSES = ['PARTIALLY_ACCEPTS', 'REDIRECTS']

# Statistical thresholds
ALPHA = 0.05

# ============================================================================
# DATA LOADING
# ============================================================================

class ManipulationCheckDataLoader:
    """Load chatbot message data for manipulation verification"""

    def __init__(self, analyses_folder: str):
        self.analyses_folder = Path(analyses_folder)
        self.chatbot_df = None
        self.conversations_df = None

    def diagnose_dataframe(self, df: pd.DataFrame, name: str):
        """Print diagnostic information about a dataframe"""
        print(f"\n  Diagnostics for {name}:")
        print(f"    Shape: {df.shape}")
        print(f"    Columns: {list(df.columns)}")
        if 'condition' in df.columns:
            print(f"    Condition values: {df['condition'].unique()}")
        # Check for tactic-related columns
        tactic_cols = [c for c in df.columns if 'tactic' in c.lower()]
        if tactic_cols:
            print(f"    Tactic columns: {tactic_cols}")
        count_cols = [c for c in df.columns if 'count' in c.lower()]
        if count_cols:
            print(f"    Count columns: {count_cols}")

    def load_from_csv(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load pre-processed dataframes from CSV"""
        chatbot_path = self.analyses_folder / 'chatbot_messages_dataframe.csv'
        conv_path = self.analyses_folder / 'conversations_dataframe.csv'

        if chatbot_path.exists():
            self.chatbot_df = pd.read_csv(chatbot_path)
            print(f"Loaded chatbot messages: {len(self.chatbot_df)} rows")

            # Diagnostic info
            self.diagnose_dataframe(self.chatbot_df, "chatbot_df")

            # Standardize column names (original script uses *_tactics_count)
            col_mapping = {
                'dark_triad_tactics_count': 'dark_triad_count',
                'servant_tactics_count': 'servant_count',
                'neutral_tactics_count': 'neutral_count'
            }
            for old_col, new_col in col_mapping.items():
                if old_col in self.chatbot_df.columns and new_col not in self.chatbot_df.columns:
                    self.chatbot_df[new_col] = self.chatbot_df[old_col]

            # If count columns don't exist, compute from boolean tactic columns
            if 'dark_triad_count' not in self.chatbot_df.columns and 'dark_triad_tactics_count' not in self.chatbot_df.columns:
                print("  Computing tactic counts from boolean columns...")
                dt_tactic_cols = [f'tactic_{t.lower()}' for t in DARK_TRIAD_TACTICS if f'tactic_{t.lower()}' in self.chatbot_df.columns]
                sl_tactic_cols = [f'tactic_{t.lower()}' for t in SERVANT_LEADER_TACTICS if f'tactic_{t.lower()}' in self.chatbot_df.columns]
                nt_tactic_cols = [f'tactic_{t.lower()}' for t in NEUTRAL_TACTICS if f'tactic_{t.lower()}' in self.chatbot_df.columns]

                if dt_tactic_cols:
                    self.chatbot_df['dark_triad_count'] = self.chatbot_df[dt_tactic_cols].sum(axis=1)
                else:
                    self.chatbot_df['dark_triad_count'] = 0

                if sl_tactic_cols:
                    self.chatbot_df['servant_count'] = self.chatbot_df[sl_tactic_cols].sum(axis=1)
                else:
                    self.chatbot_df['servant_count'] = 0

                if nt_tactic_cols:
                    self.chatbot_df['neutral_count'] = self.chatbot_df[nt_tactic_cols].sum(axis=1)
                else:
                    self.chatbot_df['neutral_count'] = 0

                print(f"    Found {len(dt_tactic_cols)} DT, {len(sl_tactic_cols)} SL, {len(nt_tactic_cols)} neutral tactic columns")
        else:
            raise FileNotFoundError(f"Chatbot dataframe not found at {chatbot_path}")

        if conv_path.exists():
            self.conversations_df = pd.read_csv(conv_path)
            print(f"Loaded conversations: {len(self.conversations_df)} rows")

        return self.chatbot_df, self.conversations_df

    def load_from_json(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load directly from analyzed JSON files"""
        successful_folder = self.analyses_folder / "successful_analyses"
        analysis_files = list(successful_folder.glob("*_analyzed.json"))
        print(f"Found {len(analysis_files)} analyzed conversations")

        chatbot_rows = []
        conv_rows = []

        for filepath in analysis_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                conv_id = data['conversation_id']
                summary = data.get('participant_summary', data.get('conversation_summary', {}))
                condition = summary.get('condition_inference', 'unknown').lower().replace(' ', '_')

                # Process chatbot messages
                chatbot_analyses = data.get('chatbot_message_analyses', [])
                for msg in chatbot_analyses:
                    row = {
                        'conversation_id': conv_id,
                        'condition': condition,
                        'message_number': msg['message_number'],
                        'primary_tone': msg.get('primary_tone', 'NEUTRAL'),
                        'response_to_participant': msg.get('response_to_participant', 'NEUTRAL'),
                        'task_guidance_style': msg.get('task_guidance_style', 'ABSENT'),
                        'message_length': msg.get('message_length', 0),
                    }

                    tactics = msg.get('leadership_tactics', [])
                    row['tactics_list'] = tactics
                    row['num_tactics'] = len(tactics)

                    # Count by category
                    row['dark_triad_count'] = sum(1 for t in tactics if t in DARK_TRIAD_TACTICS)
                    row['servant_count'] = sum(1 for t in tactics if t in SERVANT_LEADER_TACTICS)
                    row['neutral_count'] = sum(1 for t in tactics if t in NEUTRAL_TACTICS)

                    # Individual tactic flags
                    all_tactics = DARK_TRIAD_TACTICS + SERVANT_LEADER_TACTICS + NEUTRAL_TACTICS
                    for tactic in all_tactics:
                        row[f'tactic_{tactic.lower()}'] = tactic in tactics

                    chatbot_rows.append(row)

                # Conversation-level summary
                conv_rows.append({
                    'conversation_id': conv_id,
                    'condition': condition,
                    'total_chatbot_messages': len(chatbot_analyses)
                })

            except Exception as e:
                print(f"Error loading {filepath.name}: {e}")

        self.chatbot_df = pd.DataFrame(chatbot_rows)
        self.conversations_df = pd.DataFrame(conv_rows)

        print(f"Created chatbot dataframe: {len(self.chatbot_df)} messages")
        print(f"Created conversations dataframe: {len(self.conversations_df)} conversations")

        return self.chatbot_df, self.conversations_df


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

class ManipulationCheckAnalyzer:
    """Statistical analysis for manipulation verification"""

    def __init__(self, chatbot_df: pd.DataFrame, conversations_df: pd.DataFrame):
        self.chatbot_df = chatbot_df
        self.conversations_df = conversations_df
        self.results = {}

    def calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0
        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d magnitude"""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"

    def calculate_cramers_v(self, contingency_table: np.ndarray) -> float:
        """Calculate Cramér's V for chi-square effect size"""
        chi2 = chi2_contingency(contingency_table)[0]
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        if min_dim == 0 or n == 0:
            return 0.0
        return np.sqrt(chi2 / (n * min_dim))

    def analyze_tactic_proportions(self) -> Dict:
        """Analyze tactic type proportions by condition"""
        results = {'by_condition': {}, 'statistical_tests': {}}

        # Determine correct column names (handle both naming conventions)
        dt_col = 'dark_triad_count' if 'dark_triad_count' in self.chatbot_df.columns else 'dark_triad_tactics_count'
        sl_col = 'servant_count' if 'servant_count' in self.chatbot_df.columns else 'servant_tactics_count'
        nt_col = 'neutral_count' if 'neutral_count' in self.chatbot_df.columns else 'neutral_tactics_count'

        # Check if required columns exist
        missing_cols = []
        for col in [dt_col, sl_col, nt_col]:
            if col not in self.chatbot_df.columns:
                missing_cols.append(col)

        if missing_cols:
            print(f"WARNING: Missing columns: {missing_cols}")
            print(f"  Available columns: {list(self.chatbot_df.columns)}")
            return results

        # Get condition groups
        dark_triad = self.chatbot_df[self.chatbot_df['condition'] == 'dark_triad']
        servant_leader = self.chatbot_df[self.chatbot_df['condition'] == 'servant_leader']

        if len(dark_triad) == 0 or len(servant_leader) == 0:
            print("WARNING: One or both conditions have no data")
            print(f"  Dark Triad: {len(dark_triad)} messages")
            print(f"  Servant Leader: {len(servant_leader)} messages")
            print(f"  Unique conditions: {self.chatbot_df['condition'].unique()}")
            return results

        # Calculate proportions for each condition
        for condition, df in [('dark_triad', dark_triad), ('servant_leader', servant_leader)]:
            total_tactics = df[dt_col].sum() + df[sl_col].sum() + df[nt_col].sum()

            results['by_condition'][condition] = {
                'n_messages': len(df),
                'total_tactics_used': total_tactics,
                'dark_triad_tactics': df[dt_col].sum(),
                'servant_tactics': df[sl_col].sum(),
                'neutral_tactics': df[nt_col].sum(),
                'prop_dark_triad': df[dt_col].sum() / max(total_tactics, 1),
                'prop_servant': df[sl_col].sum() / max(total_tactics, 1),
                'prop_neutral': df[nt_col].sum() / max(total_tactics, 1),
                'mean_dark_triad_per_msg': df[dt_col].mean(),
                'mean_servant_per_msg': df[sl_col].mean(),
                'mean_neutral_per_msg': df[nt_col].mean(),
            }

        # Statistical tests for tactic counts per message
        # Test 1: Dark Triad tactics count
        dt_dark = dark_triad[dt_col].values
        dt_servant = servant_leader[dt_col].values

        u_stat, p_mw = mannwhitneyu(dt_dark, dt_servant, alternative='two-sided')
        t_stat, p_t = ttest_ind(dt_dark, dt_servant)
        d = self.calculate_cohens_d(dt_dark, dt_servant)

        results['statistical_tests']['dark_triad_tactics'] = {
            'mann_whitney_u': u_stat,
            'mann_whitney_p': p_mw,
            't_statistic': t_stat,
            't_test_p': p_t,
            'cohens_d': d,
            'effect_interpretation': self.interpret_cohens_d(d),
            'dark_triad_mean': np.mean(dt_dark),
            'servant_leader_mean': np.mean(dt_servant),
            'significant': p_mw < ALPHA
        }

        # Test 2: Servant Leader tactics count
        sl_dark = dark_triad[sl_col].values
        sl_servant = servant_leader[sl_col].values

        u_stat, p_mw = mannwhitneyu(sl_dark, sl_servant, alternative='two-sided')
        t_stat, p_t = ttest_ind(sl_dark, sl_servant)
        d = self.calculate_cohens_d(sl_servant, sl_dark)  # Reversed for interpretation

        results['statistical_tests']['servant_tactics'] = {
            'mann_whitney_u': u_stat,
            'mann_whitney_p': p_mw,
            't_statistic': t_stat,
            't_test_p': p_t,
            'cohens_d': d,
            'effect_interpretation': self.interpret_cohens_d(d),
            'dark_triad_mean': np.mean(sl_dark),
            'servant_leader_mean': np.mean(sl_servant),
            'significant': p_mw < ALPHA
        }

        self.results['tactic_proportions'] = results
        return results

    def analyze_individual_tactics(self) -> Dict:
        """Analyze each individual tactic by condition"""
        results = {'tactics': {}}

        dark_triad = self.chatbot_df[self.chatbot_df['condition'] == 'dark_triad']
        servant_leader = self.chatbot_df[self.chatbot_df['condition'] == 'servant_leader']

        if len(dark_triad) == 0 or len(servant_leader) == 0:
            print("WARNING: Cannot analyze individual tactics - missing condition data")
            return results

        all_tactics = DARK_TRIAD_TACTICS + SERVANT_LEADER_TACTICS + NEUTRAL_TACTICS
        analyzed_count = 0

        for tactic in all_tactics:
            col_name = f'tactic_{tactic.lower()}'

            if col_name not in self.chatbot_df.columns:
                # Try to create the column from tactics_list if available
                if 'tactics_list' in self.chatbot_df.columns:
                    self.chatbot_df[col_name] = self.chatbot_df['tactics_list'].apply(
                        lambda x: tactic in str(x) if pd.notna(x) else False
                    )
                else:
                    continue

            # Counts
            dt_count = dark_triad[col_name].sum() if col_name in dark_triad.columns else 0
            sl_count = servant_leader[col_name].sum() if col_name in servant_leader.columns else 0
            dt_prop = dt_count / len(dark_triad) if len(dark_triad) > 0 else 0
            sl_prop = sl_count / len(servant_leader) if len(servant_leader) > 0 else 0

            # Chi-square test
            contingency = np.array([
                [dt_count, len(dark_triad) - dt_count],
                [sl_count, len(servant_leader) - sl_count]
            ])

            if contingency.min() >= 0 and contingency.sum() > 0:
                try:
                    chi2, p_chi, dof, expected = chi2_contingency(contingency)
                    cramers_v = self.calculate_cramers_v(contingency)
                except ValueError:
                    chi2, p_chi, cramers_v = 0, 1, 0
            else:
                chi2, p_chi, cramers_v = 0, 1, 0

            # Determine expected direction
            if tactic in DARK_TRIAD_TACTICS:
                expected_higher = 'dark_triad'
                direction_correct = dt_prop > sl_prop
            elif tactic in SERVANT_LEADER_TACTICS:
                expected_higher = 'servant_leader'
                direction_correct = sl_prop > dt_prop
            else:
                expected_higher = 'neutral'
                direction_correct = True  # No expected direction

            results['tactics'][tactic] = {
                'category': 'dark_triad' if tactic in DARK_TRIAD_TACTICS else (
                    'servant_leader' if tactic in SERVANT_LEADER_TACTICS else 'neutral'),
                'dark_triad_count': int(dt_count),
                'servant_leader_count': int(sl_count),
                'dark_triad_proportion': dt_prop,
                'servant_leader_proportion': sl_prop,
                'proportion_difference': dt_prop - sl_prop,
                'chi_square': chi2,
                'chi_square_p': p_chi,
                'cramers_v': cramers_v,
                'significant': p_chi < ALPHA,
                'expected_higher_in': expected_higher,
                'direction_correct': direction_correct
            }
            analyzed_count += 1

        print(f"  Analyzed {analyzed_count} individual tactics")
        self.results['individual_tactics'] = results
        return results

    def analyze_tone_distribution(self) -> Dict:
        """Analyze chatbot tone distribution by condition"""
        results = {'by_condition': {}, 'statistical_tests': {}}

        dark_triad = self.chatbot_df[self.chatbot_df['condition'] == 'dark_triad']
        servant_leader = self.chatbot_df[self.chatbot_df['condition'] == 'servant_leader']

        # Count tones by condition
        for condition, df in [('dark_triad', dark_triad), ('servant_leader', servant_leader)]:
            tone_counts = df['primary_tone'].value_counts()
            total = len(df)

            results['by_condition'][condition] = {
                'total_messages': total,
                'tone_counts': tone_counts.to_dict(),
                'tone_proportions': (tone_counts / total).to_dict() if total > 0 else {}
            }

            # Aggregate by valence
            neg_count = df['primary_tone'].isin(NEGATIVE_TONES).sum()
            pos_count = df['primary_tone'].isin(POSITIVE_TONES).sum()
            neutral_count = df['primary_tone'].isin(NEUTRAL_TONES).sum()

            results['by_condition'][condition]['negative_proportion'] = neg_count / total if total > 0 else 0
            results['by_condition'][condition]['positive_proportion'] = pos_count / total if total > 0 else 0
            results['by_condition'][condition]['neutral_proportion'] = neutral_count / total if total > 0 else 0

        # Chi-square test for tone distribution
        tone_contingency = pd.crosstab(
            self.chatbot_df['condition'],
            self.chatbot_df['primary_tone']
        )

        if tone_contingency.size > 0:
            chi2, p_chi, dof, expected = chi2_contingency(tone_contingency)
            cramers_v = self.calculate_cramers_v(tone_contingency.values)

            results['statistical_tests']['tone_distribution'] = {
                'chi_square': chi2,
                'p_value': p_chi,
                'degrees_of_freedom': dof,
                'cramers_v': cramers_v,
                'significant': p_chi < ALPHA
            }

        # Test negative vs positive tone proportions
        dt_neg = dark_triad['primary_tone'].isin(NEGATIVE_TONES).astype(int).values
        sl_neg = servant_leader['primary_tone'].isin(NEGATIVE_TONES).astype(int).values

        u_stat, p_mw = mannwhitneyu(dt_neg, sl_neg, alternative='two-sided')
        d = self.calculate_cohens_d(dt_neg, sl_neg)

        results['statistical_tests']['negative_tone'] = {
            'mann_whitney_u': u_stat,
            'mann_whitney_p': p_mw,
            'cohens_d': d,
            'effect_interpretation': self.interpret_cohens_d(d),
            'dark_triad_proportion': np.mean(dt_neg),
            'servant_leader_proportion': np.mean(sl_neg),
            'significant': p_mw < ALPHA
        }

        self.results['tone_distribution'] = results
        return results

    def analyze_response_types(self) -> Dict:
        """Analyze chatbot response types by condition"""
        results = {'by_condition': {}, 'statistical_tests': {}}

        dark_triad = self.chatbot_df[self.chatbot_df['condition'] == 'dark_triad']
        servant_leader = self.chatbot_df[self.chatbot_df['condition'] == 'servant_leader']

        for condition, df in [('dark_triad', dark_triad), ('servant_leader', servant_leader)]:
            response_counts = df['response_to_participant'].value_counts()
            total = len(df)

            results['by_condition'][condition] = {
                'total_messages': total,
                'response_counts': response_counts.to_dict(),
                'response_proportions': (response_counts / total).to_dict() if total > 0 else {}
            }

            # Aggregate by valence
            neg_count = df['response_to_participant'].isin(NEGATIVE_RESPONSES).sum()
            pos_count = df['response_to_participant'].isin(POSITIVE_RESPONSES).sum()

            results['by_condition'][condition]['negative_response_prop'] = neg_count / total if total > 0 else 0
            results['by_condition'][condition]['positive_response_prop'] = pos_count / total if total > 0 else 0

        # Chi-square test
        response_contingency = pd.crosstab(
            self.chatbot_df['condition'],
            self.chatbot_df['response_to_participant']
        )

        if response_contingency.size > 0:
            chi2, p_chi, dof, expected = chi2_contingency(response_contingency)
            cramers_v = self.calculate_cramers_v(response_contingency.values)

            results['statistical_tests']['response_distribution'] = {
                'chi_square': chi2,
                'p_value': p_chi,
                'degrees_of_freedom': dof,
                'cramers_v': cramers_v,
                'significant': p_chi < ALPHA
            }

        self.results['response_types'] = results
        return results

    def compute_manipulation_success_score(self) -> Dict:
        """Compute overall manipulation success metrics"""
        results = {
            'overall_success': True,
            'checks_passed': 0,
            'checks_total': 0,
            'details': []
        }

        # Check 1: Dark Triad condition uses more Dark Triad tactics
        if 'tactic_proportions' in self.results:
            dt_test = self.results['tactic_proportions']['statistical_tests'].get('dark_triad_tactics', {})
            if dt_test:
                dt_mean = dt_test.get('dark_triad_mean', 0)
                sl_mean = dt_test.get('servant_leader_mean', 0)
                passed = dt_mean > sl_mean and dt_test.get('significant', False)
                results['checks_total'] += 1
                if passed:
                    results['checks_passed'] += 1
                results['details'].append({
                    'check': 'Dark Triad uses more Dark Triad tactics',
                    'passed': passed,
                    'dark_triad_mean': dt_mean,
                    'servant_leader_mean': sl_mean,
                    'p_value': dt_test.get('mann_whitney_p'),
                    'effect_size': dt_test.get('cohens_d')
                })

        # Check 2: Servant Leader condition uses more Servant tactics
        if 'tactic_proportions' in self.results:
            sl_test = self.results['tactic_proportions']['statistical_tests'].get('servant_tactics', {})
            if sl_test:
                dt_mean = sl_test.get('dark_triad_mean', 0)
                sl_mean = sl_test.get('servant_leader_mean', 0)
                passed = sl_mean > dt_mean and sl_test.get('significant', False)
                results['checks_total'] += 1
                if passed:
                    results['checks_passed'] += 1
                results['details'].append({
                    'check': 'Servant Leader uses more Servant tactics',
                    'passed': passed,
                    'dark_triad_mean': dt_mean,
                    'servant_leader_mean': sl_mean,
                    'p_value': sl_test.get('mann_whitney_p'),
                    'effect_size': sl_test.get('cohens_d')
                })

        # Check 3: Dark Triad uses more negative tones
        if 'tone_distribution' in self.results:
            tone_test = self.results['tone_distribution']['statistical_tests'].get('negative_tone', {})
            if tone_test:
                dt_prop = tone_test.get('dark_triad_proportion', 0)
                sl_prop = tone_test.get('servant_leader_proportion', 0)
                passed = dt_prop > sl_prop and tone_test.get('significant', False)
                results['checks_total'] += 1
                if passed:
                    results['checks_passed'] += 1
                results['details'].append({
                    'check': 'Dark Triad uses more negative tones',
                    'passed': passed,
                    'dark_triad_proportion': dt_prop,
                    'servant_leader_proportion': sl_prop,
                    'p_value': tone_test.get('mann_whitney_p'),
                    'effect_size': tone_test.get('cohens_d')
                })

        # Check 4: Individual tactic direction correctness
        if 'individual_tactics' in self.results:
            correct_directions = sum(
                1 for t in self.results['individual_tactics']['tactics'].values()
                if t['direction_correct'] and t['significant']
            )
            total_significant = sum(
                1 for t in self.results['individual_tactics']['tactics'].values()
                if t['significant']
            )

            if total_significant > 0:
                direction_rate = correct_directions / total_significant
                passed = direction_rate >= 0.8  # 80% of significant tactics in expected direction
                results['checks_total'] += 1
                if passed:
                    results['checks_passed'] += 1
                results['details'].append({
                    'check': 'Individual tactics in expected direction',
                    'passed': passed,
                    'correct_directions': correct_directions,
                    'total_significant': total_significant,
                    'direction_rate': direction_rate
                })

        results['overall_success'] = results['checks_passed'] >= results['checks_total'] * 0.75
        results['success_rate'] = results['checks_passed'] / max(results['checks_total'], 1)

        self.results['manipulation_success'] = results
        return results

    def run_full_analysis(self) -> Dict:
        """Run complete manipulation check analysis"""
        print("\n" + "="*60)
        print("RUNNING MANIPULATION CHECK ANALYSIS")
        print("="*60)

        print("\n1. Analyzing tactic proportions...")
        self.analyze_tactic_proportions()

        print("2. Analyzing individual tactics...")
        self.analyze_individual_tactics()

        print("3. Analyzing tone distribution...")
        self.analyze_tone_distribution()

        print("4. Analyzing response types...")
        self.analyze_response_types()

        print("5. Computing manipulation success score...")
        self.compute_manipulation_success_score()

        print("\nAnalysis complete")
        return self.results


# ============================================================================
# VISUALIZATION
# ============================================================================

class ManipulationCheckVisualizer:
    """Create visualizations for manipulation verification"""

    def __init__(self, chatbot_df: pd.DataFrame, results: Dict, output_folder: str):
        self.chatbot_df = chatbot_df
        self.results = results
        self.output_folder = Path(output_folder) / "manipulation_check"
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.colors = {
            'dark_triad': '#E74C3C',
            'servant_leader': '#3498DB',
            'dark_triad_tactics': '#C0392B',
            'servant_tactics': '#2980B9',
            'neutral': '#95A5A6'
        }

        # Determine correct column names
        self.dt_col = 'dark_triad_count' if 'dark_triad_count' in chatbot_df.columns else 'dark_triad_tactics_count'
        self.sl_col = 'servant_count' if 'servant_count' in chatbot_df.columns else 'servant_tactics_count'
        self.nt_col = 'neutral_count' if 'neutral_count' in chatbot_df.columns else 'neutral_tactics_count'

    def plot_tactic_category_comparison(self):
        """Plot comparison of tactic categories between conditions"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Data preparation
        prop_data = self.results.get('tactic_proportions', {}).get('by_condition', {})

        if not prop_data:
            print("WARNING: No tactic proportion data to plot")
            return

        conditions = list(prop_data.keys())

        # Plot 1: Mean tactics per message
        categories = ['dark_triad', 'servant', 'neutral']
        x = np.arange(len(categories))
        width = 0.35

        for i, cond in enumerate(conditions):
            means = [
                prop_data[cond]['mean_dark_triad_per_msg'],
                prop_data[cond]['mean_servant_per_msg'],
                prop_data[cond]['mean_neutral_per_msg']
            ]
            offset = (i - 0.5) * width
            bars = axes[0].bar(x + offset, means, width,
                              label=cond.replace('_', ' ').title(),
                              color=self.colors[cond])
            axes[0].bar_label(bars, fmt='%.2f', fontsize=9)

        axes[0].set_xlabel('Tactic Category', fontsize=12)
        axes[0].set_ylabel('Mean Tactics per Message', fontsize=12)
        axes[0].set_title('Tactic Usage by Condition', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(['Dark Triad', 'Servant Leader', 'Neutral'])
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Add significance markers
        tests = self.results.get('tactic_proportions', {}).get('statistical_tests', {})
        if tests.get('dark_triad_tactics', {}).get('significant'):
            axes[0].annotate('*', xy=(0, max(means) * 1.1), fontsize=20, ha='center')
        if tests.get('servant_tactics', {}).get('significant'):
            axes[0].annotate('*', xy=(1, max(means) * 1.1), fontsize=20, ha='center')

        # Plot 2: Proportion pie charts
        for i, cond in enumerate(conditions):
            ax = axes[1] if i == 0 else axes[2]
            props = [
                prop_data[cond]['prop_dark_triad'],
                prop_data[cond]['prop_servant'],
                prop_data[cond]['prop_neutral']
            ]
            labels = ['Dark Triad', 'Servant', 'Neutral']
            colors_pie = [self.colors['dark_triad_tactics'],
                         self.colors['servant_tactics'],
                         self.colors['neutral']]

            wedges, texts, autotexts = ax.pie(props, labels=labels, autopct='%1.1f%%',
                                              colors=colors_pie, startangle=90)
            ax.set_title(f'{cond.replace("_", " ").title()} Condition',
                        fontsize=12, fontweight='bold')

        plt.suptitle('Manipulation Check: Tactic Category Distribution',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_folder / 'tactic_category_comparison.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: tactic_category_comparison.png")

    def plot_individual_tactics_heatmap(self):
        """Plot heatmap of individual tactic usage"""
        tactic_data = self.results.get('individual_tactics', {}).get('tactics', {})

        if not tactic_data:
            print("WARNING: No individual tactic data to plot")
            return

        # Prepare data for heatmap
        tactics_list = []
        dt_props = []
        sl_props = []
        categories = []
        significant = []

        for tactic, data in tactic_data.items():
            tactics_list.append(tactic.replace('_', ' ').title())
            dt_props.append(data['dark_triad_proportion'])
            sl_props.append(data['servant_leader_proportion'])
            categories.append(data['category'])
            significant.append('*' if data['significant'] else '')

        if not tactics_list:
            print("WARNING: No tactics data available for heatmap")
            return

        # Create DataFrame
        df = pd.DataFrame({
            'Tactic': tactics_list,
            'Dark Triad': dt_props,
            'Servant Leader': sl_props,
            'Category': categories,
            'Significant': significant
        })

        # Sort by category and difference
        df['Difference'] = df['Dark Triad'] - df['Servant Leader']
        df = df.sort_values(['Category', 'Difference'], ascending=[True, False])

        fig, ax = plt.subplots(figsize=(12, 10))

        # Create heatmap data
        heatmap_data = df[['Dark Triad', 'Servant Leader']].values

        im = ax.imshow(heatmap_data, cmap='RdBu_r', aspect='auto', vmin=0, vmax=1)

        # Set ticks
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([f"{row['Tactic']} {row['Significant']}" for _, row in df.iterrows()])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Dark Triad\nCondition', 'Servant Leader\nCondition'])

        # Add values
        for i in range(len(df)):
            for j in range(2):
                val = heatmap_data[i, j]
                color = 'white' if val > 0.5 or val < 0.1 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=9)

        # Add category separators
        category_changes = df['Category'].ne(df['Category'].shift()).cumsum()
        for i, (idx, row) in enumerate(df.iterrows()):
            if i > 0 and df.iloc[i]['Category'] != df.iloc[i-1]['Category']:
                ax.axhline(y=i-0.5, color='black', linewidth=2)

        # Color-code y-axis labels by category
        for i, (idx, row) in enumerate(df.iterrows()):
            color = self.colors['dark_triad_tactics'] if row['Category'] == 'dark_triad' else (
                self.colors['servant_tactics'] if row['Category'] == 'servant_leader' else self.colors['neutral'])
            ax.get_yticklabels()[i].set_color(color)

        plt.colorbar(im, ax=ax, label='Proportion of Messages')
        ax.set_title('Individual Tactic Usage by Condition\n(* = statistically significant)',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_folder / 'individual_tactics_heatmap.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: individual_tactics_heatmap.png")

    def plot_tone_comparison(self):
        """Plot tone distribution comparison"""
        tone_data = self.results.get('tone_distribution', {}).get('by_condition', {})

        if not tone_data:
            print("WARNING: No tone data to plot")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        conditions = list(tone_data.keys())

        if len(conditions) < 2:
            print("WARNING: Need at least 2 conditions for tone comparison")
            plt.close()
            return

        # Plot 1 & 2: Tone distribution per condition
        tone_colors = {
            'HOSTILE': '#C0392B', 'CONDESCENDING': '#E74C3C', 'COLD': '#9B59B6',
            'SARCASTIC': '#8E44AD', 'NEUTRAL': '#95A5A6', 'WARM': '#F39C12',
            'SUPPORTIVE': '#27AE60'
        }

        for i, cond in enumerate(conditions):
            tone_props = tone_data[cond].get('tone_proportions', {})
            if tone_props:
                tones = list(tone_props.keys())
                props = list(tone_props.values())
                colors = [tone_colors.get(t, 'gray') for t in tones]

                axes[i].bar(range(len(tones)), props, color=colors)
                axes[i].set_xticks(range(len(tones)))
                axes[i].set_xticklabels(tones, rotation=45, ha='right')
                axes[i].set_ylabel('Proportion')
                axes[i].set_title(f'{cond.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                axes[i].grid(axis='y', alpha=0.3)
                axes[i].set_ylim(0, 1)

        # Plot 3: Valence comparison
        valence_data = []
        for cond in conditions:
            valence_data.append({
                'Condition': cond.replace('_', ' ').title(),
                'Negative': tone_data[cond].get('negative_proportion', 0),
                'Neutral': tone_data[cond].get('neutral_proportion', 0),
                'Positive': tone_data[cond].get('positive_proportion', 0)
            })

        valence_df = pd.DataFrame(valence_data)
        valence_df.set_index('Condition')[['Negative', 'Neutral', 'Positive']].plot(
            kind='bar', ax=axes[2], color=['#E74C3C', '#95A5A6', '#27AE60'], rot=0)
        axes[2].set_ylabel('Proportion')
        axes[2].set_title('Tone Valence by Condition', fontsize=12, fontweight='bold')
        axes[2].legend(title='Valence')
        axes[2].grid(axis='y', alpha=0.3)

        plt.suptitle('Manipulation Check: Chatbot Tone Distribution',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_folder / 'tone_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: tone_comparison.png")

    def plot_effect_sizes(self):
        """Plot effect sizes for key comparisons"""
        fig, ax = plt.subplots(figsize=(12, 6))

        effect_data = []

        # Collect effect sizes from results
        if 'tactic_proportions' in self.results:
            tests = self.results['tactic_proportions'].get('statistical_tests', {})
            if 'dark_triad_tactics' in tests:
                effect_data.append({
                    'Comparison': 'Dark Triad Tactics\n(DT > SL expected)',
                    'Cohen\'s d': tests['dark_triad_tactics']['cohens_d'],
                    'p_value': tests['dark_triad_tactics']['mann_whitney_p']
                })
            if 'servant_tactics' in tests:
                effect_data.append({
                    'Comparison': 'Servant Tactics\n(SL > DT expected)',
                    'Cohen\'s d': tests['servant_tactics']['cohens_d'],
                    'p_value': tests['servant_tactics']['mann_whitney_p']
                })

        if 'tone_distribution' in self.results:
            tests = self.results['tone_distribution'].get('statistical_tests', {})
            if 'negative_tone' in tests:
                effect_data.append({
                    'Comparison': 'Negative Tone\n(DT > SL expected)',
                    'Cohen\'s d': tests['negative_tone']['cohens_d'],
                    'p_value': tests['negative_tone']['mann_whitney_p']
                })

        if not effect_data:
            print("WARNING: No effect size data to plot")
            return

        df = pd.DataFrame(effect_data)

        # Create bar plot
        colors = ['#27AE60' if abs(d) >= 0.8 else '#F39C12' if abs(d) >= 0.5 else '#E74C3C'
                 for d in df["Cohen's d"]]

        bars = ax.barh(range(len(df)), df["Cohen's d"], color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Comparison'])
        ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
        ax.set_title('Effect Sizes for Manipulation Verification', fontsize=14, fontweight='bold')

        # Add reference lines
        ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small (0.2)')
        ax.axvline(x=0.5, color='gray', linestyle='-.', alpha=0.5, label='Medium (0.5)')
        ax.axvline(x=0.8, color='gray', linestyle=':', alpha=0.5, label='Large (0.8)')
        ax.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=-0.5, color='gray', linestyle='-.', alpha=0.5)
        ax.axvline(x=-0.8, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=0, color='black', linewidth=1)

        # Add significance markers
        for i, (idx, row) in enumerate(df.iterrows()):
            sig_marker = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'ns'
            ax.text(row["Cohen's d"] + 0.05, i, sig_marker, va='center', fontsize=10)

        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_folder / 'effect_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: effect_sizes.png")

    def plot_manipulation_success_summary(self):
        """Plot summary of manipulation success checks"""
        success_data = self.results.get('manipulation_success', {})

        if not success_data:
            print("WARNING: No manipulation success data to plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Overall success rate
        success_rate = success_data.get('success_rate', 0)
        colors_pie = ['#27AE60', '#E74C3C']

        axes[0].pie([success_rate, 1-success_rate],
                   labels=['Passed', 'Failed'],
                   autopct='%1.1f%%',
                   colors=colors_pie,
                   startangle=90,
                   explode=(0.05, 0))
        axes[0].set_title(f'Overall Manipulation Check\n({success_data.get("checks_passed", 0)}/{success_data.get("checks_total", 0)} checks passed)',
                         fontsize=12, fontweight='bold')

        # Plot 2: Individual checks
        details = success_data.get('details', [])
        if details:
            checks = [d['check'][:30] + '...' if len(d['check']) > 30 else d['check'] for d in details]
            passed = [1 if d['passed'] else 0 for d in details]
            colors_bar = ['#27AE60' if p else '#E74C3C' for p in passed]

            axes[1].barh(range(len(checks)), passed, color=colors_bar)
            axes[1].set_yticks(range(len(checks)))
            axes[1].set_yticklabels(checks)
            axes[1].set_xlim(-0.1, 1.1)
            axes[1].set_xticks([0, 1])
            axes[1].set_xticklabels(['Failed', 'Passed'])
            axes[1].set_title('Individual Manipulation Checks', fontsize=12, fontweight='bold')

            for i, d in enumerate(details):
                if 'p_value' in d and d['p_value'] is not None:
                    axes[1].text(1.02, i, f"p={d['p_value']:.3f}", va='center', fontsize=9)

        overall_status = "MANIPULATION SUCCESSFUL" if success_data.get('overall_success') else "MANIPULATION ISSUES DETECTED"
        fig.suptitle(f'Manipulation Verification Summary\n{overall_status}',
                    fontsize=14, fontweight='bold',
                    color='#27AE60' if success_data.get('overall_success') else '#E74C3C')

        plt.tight_layout()
        plt.savefig(self.output_folder / 'manipulation_success_summary.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: manipulation_success_summary.png")

    def create_all_visualizations(self):
        """Generate all manipulation check visualizations"""
        print("\nGenerating visualizations...")
        self.plot_tactic_category_comparison()
        self.plot_individual_tactics_heatmap()
        self.plot_tone_comparison()
        self.plot_effect_sizes()
        self.plot_manipulation_success_summary()


# ============================================================================
# REPORT GENERATION
# ============================================================================

class ManipulationCheckReporter:
    """Generate the manipulation check report"""

    def __init__(self, results: Dict, output_folder: str):
        self.results = results
        self.output_folder = Path(output_folder) / "manipulation_check"
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def generate_report(self):
        """Generate text report"""
        report_path = self.output_folder / 'manipulation_check_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CHATBOT MANIPULATION VERIFICATION REPORT\n")
            f.write("Retrospective Check: Did chatbots behave as designed?\n")
            f.write("="*80 + "\n\n")

            # Overall success
            success = self.results.get('manipulation_success', {})
            status = "PASSED" if success.get('overall_success') else "ISSUES DETECTED"
            f.write(f"OVERALL MANIPULATION STATUS: {status}\n")
            f.write(f"Checks Passed: {success.get('checks_passed', 0)}/{success.get('checks_total', 0)}\n")
            f.write(f"Success Rate: {success.get('success_rate', 0)*100:.1f}%\n\n")

            # Tactic proportions
            f.write("-"*80 + "\n")
            f.write("1. TACTIC CATEGORY ANALYSIS\n")
            f.write("-"*80 + "\n\n")

            tactic_props = self.results.get('tactic_proportions', {})
            by_condition = tactic_props.get('by_condition', {})

            for cond, data in by_condition.items():
                f.write(f"\n{cond.upper().replace('_', ' ')} CONDITION:\n")
                f.write(f"  Total messages: {data.get('n_messages', 0)}\n")
                f.write(f"  Dark Triad tactics/message: {data.get('mean_dark_triad_per_msg', 0):.3f}\n")
                f.write(f"  Servant tactics/message: {data.get('mean_servant_per_msg', 0):.3f}\n")
                f.write(f"  Neutral tactics/message: {data.get('mean_neutral_per_msg', 0):.3f}\n")

            f.write("\nSTATISTICAL TESTS:\n")
            tests = tactic_props.get('statistical_tests', {})

            for test_name, test_data in tests.items():
                f.write(f"\n  {test_name.upper().replace('_', ' ')}:\n")
                f.write(f"    Mann-Whitney U: {test_data.get('mann_whitney_u', 'N/A')}\n")
                f.write(f"    p-value: {test_data.get('mann_whitney_p', 'N/A'):.4f}\n")
                f.write(f"    Cohen's d: {test_data.get('cohens_d', 'N/A'):.3f} ({test_data.get('effect_interpretation', 'N/A')})\n")
                f.write(f"    Significant: {'Yes' if test_data.get('significant') else 'No'}\n")

            # Individual tactics
            f.write("\n" + "-"*80 + "\n")
            f.write("2. INDIVIDUAL TACTIC ANALYSIS\n")
            f.write("-"*80 + "\n\n")

            tactics = self.results.get('individual_tactics', {}).get('tactics', {})

            # Group by category
            for category in ['dark_triad', 'servant_leader', 'neutral']:
                f.write(f"\n{category.upper().replace('_', ' ')} TACTICS:\n")
                f.write("-"*40 + "\n")

                for tactic, data in tactics.items():
                    if data['category'] == category:
                        sig = "*" if data['significant'] else ""
                        direction = "yes" if data['direction_correct'] else "no"
                        f.write(f"  {tactic}:{sig}\n")
                        f.write(f"    Dark Triad proportion: {data['dark_triad_proportion']:.3f}\n")
                        f.write(f"    Servant Leader proportion: {data['servant_leader_proportion']:.3f}\n")
                        f.write(f"    χ² = {data['chi_square']:.2f}, p = {data['chi_square_p']:.4f}\n")
                        f.write(f"    Cramér's V = {data['cramers_v']:.3f}\n")
                        f.write(f"    Direction correct: {direction}\n\n")

            # Tone analysis
            f.write("-"*80 + "\n")
            f.write("3. TONE DISTRIBUTION ANALYSIS\n")
            f.write("-"*80 + "\n\n")

            tone_data = self.results.get('tone_distribution', {})
            by_cond = tone_data.get('by_condition', {})

            for cond, data in by_cond.items():
                f.write(f"\n{cond.upper().replace('_', ' ')} CONDITION:\n")
                f.write(f"  Negative tone proportion: {data.get('negative_proportion', 0):.3f}\n")
                f.write(f"  Positive tone proportion: {data.get('positive_proportion', 0):.3f}\n")
                f.write(f"  Neutral tone proportion: {data.get('neutral_proportion', 0):.3f}\n")

                f.write("  Tone breakdown:\n")
                for tone, prop in data.get('tone_proportions', {}).items():
                    f.write(f"    {tone}: {prop:.3f}\n")

            tone_tests = tone_data.get('statistical_tests', {})
            if tone_tests:
                f.write("\nSTATISTICAL TESTS:\n")
                for test_name, test_data in tone_tests.items():
                    f.write(f"\n  {test_name.upper().replace('_', ' ')}:\n")
                    if 'chi_square' in test_data:
                        f.write(f"    χ² = {test_data['chi_square']:.2f}\n")
                        f.write(f"    p = {test_data['p_value']:.4f}\n")
                        f.write(f"    Cramér's V = {test_data['cramers_v']:.3f}\n")
                    if 'cohens_d' in test_data:
                        f.write(f"    Cohen's d = {test_data['cohens_d']:.3f}\n")
                        f.write(f"    p = {test_data.get('mann_whitney_p', 'N/A'):.4f}\n")

            # Detailed check results
            f.write("\n" + "-"*80 + "\n")
            f.write("4. MANIPULATION CHECK DETAILS\n")
            f.write("-"*80 + "\n\n")

            for detail in success.get('details', []):
                status_symbol = "PASSED" if detail['passed'] else "FAILED"
                f.write(f"\nCheck: {detail['check']}\n")
                f.write(f"Status: {status_symbol}\n")
                for key, value in detail.items():
                    if key not in ['check', 'passed']:
                        if isinstance(value, float):
                            f.write(f"  {key}: {value:.4f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")

            # Conclusion
            f.write("\n" + "="*80 + "\n")
            f.write("CONCLUSION\n")
            f.write("="*80 + "\n\n")

            if success.get('overall_success'):
                f.write("The manipulation check PASSED. The chatbots behaved consistently\n")
                f.write("with their assigned conditions:\n\n")
                f.write("- Dark Triad chatbots used significantly more Dark Triad tactics\n")
                f.write("  (intimidation, harsh criticism, manipulation, etc.)\n")
                f.write("- Servant Leader chatbots used significantly more supportive tactics\n")
                f.write("  (encouragement, empowerment, constructive feedback, etc.)\n")
                f.write("- Tone distributions differed as expected between conditions\n")
            else:
                f.write("The manipulation check identified potential issues.\n")
                f.write("Review the individual check results above for details.\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        print(f"Saved: manipulation_check_report.txt")
        return report_path

    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary table for publication"""
        rows = []

        # Tactic category summary
        tactic_props = self.results.get('tactic_proportions', {})
        tests = tactic_props.get('statistical_tests', {})
        by_cond = tactic_props.get('by_condition', {})

        if tests.get('dark_triad_tactics'):
            t = tests['dark_triad_tactics']
            rows.append({
                'Measure': 'Dark Triad Tactics (per message)',
                'Dark Triad M': f"{by_cond.get('dark_triad', {}).get('mean_dark_triad_per_msg', 0):.2f}",
                'Servant Leader M': f"{by_cond.get('servant_leader', {}).get('mean_dark_triad_per_msg', 0):.2f}",
                'Test Statistic': f"U = {t['mann_whitney_u']:.1f}",
                'p-value': f"{t['mann_whitney_p']:.3f}",
                "Cohen's d": f"{t['cohens_d']:.2f}",
                'Effect Size': t['effect_interpretation']
            })

        if tests.get('servant_tactics'):
            t = tests['servant_tactics']
            rows.append({
                'Measure': 'Servant Tactics (per message)',
                'Dark Triad M': f"{by_cond.get('dark_triad', {}).get('mean_servant_per_msg', 0):.2f}",
                'Servant Leader M': f"{by_cond.get('servant_leader', {}).get('mean_servant_per_msg', 0):.2f}",
                'Test Statistic': f"U = {t['mann_whitney_u']:.1f}",
                'p-value': f"{t['mann_whitney_p']:.3f}",
                "Cohen's d": f"{t['cohens_d']:.2f}",
                'Effect Size': t['effect_interpretation']
            })

        # Tone summary
        tone_data = self.results.get('tone_distribution', {})
        tone_tests = tone_data.get('statistical_tests', {})
        tone_cond = tone_data.get('by_condition', {})

        if tone_tests.get('negative_tone'):
            t = tone_tests['negative_tone']
            rows.append({
                'Measure': 'Negative Tone (proportion)',
                'Dark Triad M': f"{tone_cond.get('dark_triad', {}).get('negative_proportion', 0):.2f}",
                'Servant Leader M': f"{tone_cond.get('servant_leader', {}).get('negative_proportion', 0):.2f}",
                'Test Statistic': f"U = {t['mann_whitney_u']:.1f}",
                'p-value': f"{t['mann_whitney_p']:.3f}",
                "Cohen's d": f"{t['cohens_d']:.2f}",
                'Effect Size': t['effect_interpretation']
            })

        summary_df = pd.DataFrame(rows)

        # Save to CSV
        summary_df.to_csv(self.output_folder / 'manipulation_check_summary_table.csv', index=False)
        print("Saved: manipulation_check_summary_table.csv")

        return summary_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("\n" + "="*80)
    print("CHATBOT MANIPULATION VERIFICATION")
    print("Retrospective Check for Experimental Validity")
    print("="*80 + "\n")

    # Step 1: Load data
    print("Step 1: Loading data...")
    loader = ManipulationCheckDataLoader(ANALYSES_FOLDER)

    try:
        chatbot_df, conversations_df = loader.load_from_csv()
    except FileNotFoundError:
        print("CSV not found, loading from JSON...")
        chatbot_df, conversations_df = loader.load_from_json()

    # Debug: Show condition distribution
    print(f"\nData overview:")
    print(f"  Unique conditions: {chatbot_df['condition'].unique()}")
    print(f"  Condition value counts:")
    print(chatbot_df['condition'].value_counts())

    # Normalize condition names if needed
    condition_mapping = {
        'dark triad': 'dark_triad',
        'darktriad': 'dark_triad',
        'Dark Triad': 'dark_triad',
        'servant leader': 'servant_leader',
        'servantleader': 'servant_leader',
        'Servant Leader': 'servant_leader',
    }

    chatbot_df['condition'] = chatbot_df['condition'].replace(condition_mapping)

    # If conditions are still not dark_triad/servant_leader, check for chatbot_version
    unique_conditions = set(chatbot_df['condition'].unique())
    expected_conditions = {'dark_triad', 'servant_leader'}

    if not unique_conditions.intersection(expected_conditions):
        print("\nWARNING: Conditions not in expected format. Checking for chatbot_version...")
        # Try to use chatbot_version from conversations_df if available
        if conversations_df is not None and 'chatbot_version' in conversations_df.columns:
            version_map = conversations_df.set_index('conversation_id')['chatbot_version'].to_dict()
            # Map version to condition (adjust based on your study design)
            # Version 1 = Servant Leader, Version 2 = Dark Triad (adjust if different)
            chatbot_df['condition'] = chatbot_df['conversation_id'].map(version_map)
            chatbot_df['condition'] = chatbot_df['condition'].replace({
                '1': 'servant_leader', 1: 'servant_leader',
                '2': 'dark_triad', 2: 'dark_triad'
            })
            print(f"  Mapped from chatbot_version")

    print(f"\n  After normalization:")
    print(chatbot_df['condition'].value_counts())

    # Step 2: Run analysis
    print("\nStep 2: Running manipulation check analysis...")
    analyzer = ManipulationCheckAnalyzer(chatbot_df, conversations_df)
    results = analyzer.run_full_analysis()

    # Step 3: Generate visualizations
    print("\nStep 3: Generating visualizations...")
    # Use the analyzer's potentially modified dataframe
    visualizer = ManipulationCheckVisualizer(analyzer.chatbot_df, results, ANALYSES_FOLDER)
    visualizer.create_all_visualizations()

    # Step 4: Generate report
    print("\nStep 4: Generating report...")
    reporter = ManipulationCheckReporter(results, ANALYSES_FOLDER)
    reporter.generate_report()
    summary_table = reporter.generate_summary_table()

    # Print summary
    print("\n" + "="*80)
    print("MANIPULATION CHECK COMPLETE")
    print("="*80)

    success = results.get('manipulation_success', {})
    if success.get('overall_success'):
        print("\nMANIPULATION VERIFICATION PASSED")
        print(f"  {success.get('checks_passed')}/{success.get('checks_total')} checks passed")
    else:
        print("\nMANIPULATION ISSUES DETECTED")
        print(f"  {success.get('checks_passed')}/{success.get('checks_total')} checks passed")

    print(f"\nResults saved to: {Path(ANALYSES_FOLDER) / 'manipulation_check'}")
    print("\nGenerated files:")
    print("  - manipulation_check_report.txt")
    print("  - manipulation_check_summary_table.csv")
    print("  - tactic_category_comparison.png")
    print("  - individual_tactics_heatmap.png")
    print("  - tone_comparison.png")
    print("  - effect_sizes.png")
    print("  - manipulation_success_summary.png")

    print("\n" + "="*80 + "\n")

    return results, summary_table


if __name__ == "__main__":
    results, summary_table = main()
# %%
