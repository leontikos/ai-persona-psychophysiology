# %% posthoc_participant_perception_analysis.py
# Post-Hoc Participant Perception Analysis for Manipulation Verification
# Analyzes whether participants perceived the chatbot conditions differently

import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency, spearmanr, pearsonr
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
QUALTRICS_FILE = os.path.join(DATA_ROOT, "SURVEY_data", "AI_boss_qualtrics_quality.csv")
OUTPUT_FOLDER = os.path.join(DATA_ROOT, "PERCEPTION_ANALYSIS")

# Condition mapping (adjust if your study uses different coding)
CONDITION_MAP = {
    1: 'servant_leader',
    2: 'dark_triad'
}

# Statistical thresholds
ALPHA = 0.05

# Define measure groupings based on the CSV structure
ANTHROPOMORPHISM_ITEMS = ['Lifelike', 'Humanlike', 'Natural', 'Conscious']

# Post-task AI collaboration subscales (matching survey_analysis.py)
AI_COLLAB_POST_SUBSCALES = {
    'AI_Collaboration_post': [f'AI_collab_post_{i}' for i in range(1, 4)],
    'AI_Reliability': [f'AI_collab_post_{i}' for i in range(4, 7)],
    'Behavioral_Intention': [f'AI_collab_post_{i}' for i in range(7, 10)],
    'AI_Willingness_post': [f'AI_collab_post_{i}' for i in range(10, 14)],
    'AI_Fear_post': ['AI_collab_post_14', 'AI_collab_post_15', 'AI_collab_post_16',
                     'AI_collab_post_17', 'AI_collab_post_18'],
    'Human_AI_Fit': [f'AI_collab_post_{i}' for i in range(19, 22)],
}
# Items requiring reverse scoring within AI_Fear_post
AI_FEAR_POST_REVERSE = ['AI_collab_post_15', 'AI_collab_post_17']

# Task experience measures (per task)
TASK_MEASURES = {
    'Persona': {
        'familiarity': 'Familarity_Persona',
        'difficulty': 'Difficulty_Persona',
        'creativity': 'Creativity_Persona',
        'focused_attention': ['Focused_Attention_Persona_1', 'Focused_Attention_Persona_2', 'Focused_Attention_Persona_3'],
        'enjoyment': ['Enjoyment_Persona_1', 'Enjoyment_Persona_2', 'Enjoyment_Persona_3'],
        'satisfaction': ['Satisfaction_Persona_1', 'Satisfaction_Persona_2', 'Satisfaction_Persona_3_rek']
    },
    'Brief': {
        'familiarity': 'Familiarity_Brief',
        'difficulty': 'Difficulty_Brief ',  # Note: trailing space in original
        'creativity': 'Creativity_Brief',
        'focused_attention': ['Focused_Attention_Brief_1', 'Focused_Attention_Brief_2', 'Focused_Attention_Brief_3'],
        'enjoyment': ['Enjoyment_Brief_1', 'Enjoyment_Brief_2', 'Enjoyment_Brief_3'],
        'satisfaction': ['Satisfaction_Brief_1', 'Satisfaction_Brief_2', 'Satisfaction_Brief_3_rek']
    },
    'Communication': {
        'familiarity': 'Familiarity_Communication',
        'difficulty': 'Difficulty_Communication',
        'creativity': 'Creativity_Communication',
        'focused_attention': ['Focused_Attention_Communication_1', 'Focused_Attention_Communication_2', 'Focused_Attention_Communication_3'],
        'enjoyment': ['Enjoyment_Communication_1', 'Enjoyment_Communication_2', 'Enjoyment_Communication_3'],
        'satisfaction': ['Satisfaction_Communication_1', 'Satisfaction_Communication_2', 'Satisfaction_Communication_3_rek']
    },
    'Content': {
        'familiarity': 'Familarity_Content',
        'difficulty': 'Difficulty_Content',
        'creativity': 'Creativity_Content',
        'focused_attention': ['Focused_Attention_Content_1', 'Focused_Attention_Content_2', 'Focused_Attention_Content_3'],
        'enjoyment': ['Enjoyment_Content_1', 'Enjoyment_Content_2', 'Enjoyment_Content_3'],
        'satisfaction': ['Satisfaction_Content_1', 'Satisfaction_Content_2', 'Satisfaction_Content_3_rek']
    }
}

# Pre-task AI attitudes (potential covariates)
PRE_TASK_MEASURES = {
    'AI_collaborators': ['AI_collaborators_1', 'AI_collaborators_2', 'AI_collaborators_3'],
    'AI_willingness': ['AI_willingness_to_collaborate_1', 'AI_willingness_to_collaborate_2',
                       'AI_willingness_to_collaborate_3', 'AI_willingness_to_collaborate_4'],
    'Technology_Trust': ['Technology_Trust_1', 'Technology_Trust_2', 'Technology_Trust_3'],
    'AI_fear': ['AI_fear_1', 'AI_fear_2', 'AI_fear_3', 'AI_fear_4', 'AI_fear_5']
}

# AI Fear pre-task items requiring reverse scoring (items 2 and 4)
AI_FEAR_PRE_REVERSE = ['AI_fear_2', 'AI_fear_4']


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class PerceptionDataLoader:
    """Load and preprocess Qualtrics perception data"""

    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        self.df = None
        self.condition_col = 'Group'

    def load_data(self) -> pd.DataFrame:
        """Load the Qualtrics CSV file"""
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.df)} participants")
        print(f"  Columns: {len(self.df.columns)}")

        # Show condition distribution
        if self.condition_col in self.df.columns:
            print(f"\n  Condition distribution:")
            print(self.df[self.condition_col].value_counts())

        return self.df

    def preprocess(self) -> pd.DataFrame:
        """Preprocess the data"""
        if self.df is None:
            self.load_data()

        # Map condition codes to labels
        if self.condition_col in self.df.columns:
            self.df['condition'] = self.df[self.condition_col].map(CONDITION_MAP)
            print(f"\n  Mapped conditions:")
            print(self.df['condition'].value_counts())

        # Create composite scores
        self._create_composite_scores()

        return self.df

    def _create_composite_scores(self):
        """Create composite scores for multi-item measures"""

        # Anthropomorphism composite
        available_anthro = [col for col in ANTHROPOMORPHISM_ITEMS if col in self.df.columns]
        if available_anthro:
            self.df['anthropomorphism_composite'] = self.df[available_anthro].mean(axis=1)
            print(f"  Created anthropomorphism composite from {len(available_anthro)} items")

        # AI collaboration post-task subscales
        for subscale_name, items in AI_COLLAB_POST_SUBSCALES.items():
            if subscale_name == 'AI_Fear_post':
                # Reverse-score items 15 and 17 before averaging
                scored_items = []
                for item in items:
                    if item in AI_FEAR_POST_REVERSE and item in self.df.columns:
                        rev_col = f'{item}_R'
                        self.df[rev_col] = 6 - self.df[item]  # 5-point scale
                        scored_items.append(rev_col)
                    elif item in self.df.columns:
                        scored_items.append(item)
                if scored_items:
                    self.df[subscale_name] = self.df[scored_items].mean(axis=1)
                    print(f"  Created {subscale_name} from {len(scored_items)} items (with reverse scoring)")
            else:
                available = [col for col in items if col in self.df.columns]
                if available:
                    self.df[subscale_name] = self.df[available].mean(axis=1)
                    print(f"  Created {subscale_name} from {len(available)} items")

        # Task-specific composites
        for task_name, measures in TASK_MEASURES.items():
            # Focused attention composite
            fa_cols = measures.get('focused_attention', [])
            available_fa = [col for col in fa_cols if col in self.df.columns]
            if available_fa:
                self.df[f'{task_name.lower()}_focused_attention'] = self.df[available_fa].mean(axis=1)

            # Enjoyment composite
            enj_cols = measures.get('enjoyment', [])
            available_enj = [col for col in enj_cols if col in self.df.columns]
            if available_enj:
                self.df[f'{task_name.lower()}_enjoyment'] = self.df[available_enj].mean(axis=1)

            # Satisfaction composite
            sat_cols = measures.get('satisfaction', [])
            available_sat = [col for col in sat_cols if col in self.df.columns]
            if available_sat:
                self.df[f'{task_name.lower()}_satisfaction'] = self.df[available_sat].mean(axis=1)

        # Overall task experience composites (across all tasks)
        fa_composites = [col for col in self.df.columns if '_focused_attention' in col]
        if fa_composites:
            self.df['overall_focused_attention'] = self.df[fa_composites].mean(axis=1)

        enj_composites = [col for col in self.df.columns if '_enjoyment' in col]
        if enj_composites:
            self.df['overall_enjoyment'] = self.df[enj_composites].mean(axis=1)

        sat_composites = [col for col in self.df.columns if '_satisfaction' in col]
        if sat_composites:
            self.df['overall_satisfaction'] = self.df[sat_composites].mean(axis=1)

        # Pre-task composites
        for measure_name, cols in PRE_TASK_MEASURES.items():
            if measure_name == 'AI_fear':
                # Reverse-score items 2 and 4 before averaging
                scored_cols = []
                for col in cols:
                    if col in AI_FEAR_PRE_REVERSE and col in self.df.columns:
                        rev_col = f'{col}_R'
                        self.df[rev_col] = 6 - self.df[col]  # 5-point scale
                        scored_cols.append(rev_col)
                    elif col in self.df.columns:
                        scored_cols.append(col)
                if scored_cols:
                    self.df[f'{measure_name.lower()}_composite'] = self.df[scored_cols].mean(axis=1)
                    print(f"  Created {measure_name} composite with reverse scoring for items 2 & 4")
            else:
                available_cols = [col for col in cols if col in self.df.columns]
                if available_cols:
                    self.df[f'{measure_name.lower()}_composite'] = self.df[available_cols].mean(axis=1)


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

class PerceptionStatistics:
    """Statistical analysis for perception-based manipulation check"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}

    def calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2:
            return 0.0
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

    def compare_groups(self, variable: str, label: str = None) -> Dict:
        """Compare a variable between conditions"""
        if variable not in self.df.columns:
            return None

        label = label or variable

        # Get condition groups
        servant = self.df[self.df['condition'] == 'servant_leader'][variable].dropna()
        dark = self.df[self.df['condition'] == 'dark_triad'][variable].dropna()

        if len(servant) < 2 or len(dark) < 2:
            return None

        # Descriptive statistics
        result = {
            'variable': label,
            'servant_leader_n': len(servant),
            'servant_leader_mean': servant.mean(),
            'servant_leader_std': servant.std(),
            'servant_leader_median': servant.median(),
            'dark_triad_n': len(dark),
            'dark_triad_mean': dark.mean(),
            'dark_triad_std': dark.std(),
            'dark_triad_median': dark.median(),
            'mean_difference': servant.mean() - dark.mean()
        }

        # Statistical tests
        # Mann-Whitney U (non-parametric)
        u_stat, p_mw = mannwhitneyu(servant, dark, alternative='two-sided')
        result['mann_whitney_u'] = u_stat
        result['mann_whitney_p'] = p_mw

        # Independent t-test (parametric)
        t_stat, p_t = ttest_ind(servant, dark)
        result['t_statistic'] = t_stat
        result['t_test_p'] = p_t

        # Effect size
        d = self.calculate_cohens_d(servant.values, dark.values)
        result['cohens_d'] = d
        result['effect_interpretation'] = self.interpret_cohens_d(d)
        result['significant'] = p_mw < ALPHA

        return result

    def analyze_anthropomorphism(self) -> Dict:
        """Analyze anthropomorphism items by condition"""
        results = {'items': {}, 'composite': None}

        # Individual items
        for item in ANTHROPOMORPHISM_ITEMS:
            if item in self.df.columns:
                result = self.compare_groups(item, item)
                if result:
                    results['items'][item] = result

        # Composite score
        if 'anthropomorphism_composite' in self.df.columns:
            results['composite'] = self.compare_groups('anthropomorphism_composite', 'Anthropomorphism (Composite)')

        self.results['anthropomorphism'] = results
        return results

    def analyze_ai_collaboration_post(self) -> Dict:
        """Analyze AI collaboration post-task subscales by condition"""
        results = {'subscales': {}}

        subscale_labels = {
            'AI_Collaboration_post': 'Perceived AI Collaboration',
            'AI_Reliability': 'AI Reliability',
            'Behavioral_Intention': 'Behavioral Intention',
            'AI_Willingness_post': 'Intention to Collaborate (post)',
            'AI_Fear_post': 'AI Fear/Attitudes (post)',
            'Human_AI_Fit': 'Human-AI Fit',
        }

        for subscale_name in AI_COLLAB_POST_SUBSCALES:
            if subscale_name in self.df.columns:
                label = subscale_labels.get(subscale_name, subscale_name)
                result = self.compare_groups(subscale_name, label)
                if result:
                    results['subscales'][subscale_name] = result

        self.results['ai_collab_post'] = results
        return results

    def analyze_task_experience(self) -> Dict:
        """Analyze task experience measures by condition"""
        results = {}

        for task_name in TASK_MEASURES.keys():
            task_results = {}

            # Focused attention
            fa_col = f'{task_name.lower()}_focused_attention'
            if fa_col in self.df.columns:
                task_results['focused_attention'] = self.compare_groups(fa_col, f'{task_name} Focused Attention')

            # Enjoyment
            enj_col = f'{task_name.lower()}_enjoyment'
            if enj_col in self.df.columns:
                task_results['enjoyment'] = self.compare_groups(enj_col, f'{task_name} Enjoyment')

            # Satisfaction
            sat_col = f'{task_name.lower()}_satisfaction'
            if sat_col in self.df.columns:
                task_results['satisfaction'] = self.compare_groups(sat_col, f'{task_name} Satisfaction')

            # Difficulty
            measures = TASK_MEASURES[task_name]
            diff_col = measures.get('difficulty', '')
            if diff_col and diff_col in self.df.columns:
                task_results['difficulty'] = self.compare_groups(diff_col, f'{task_name} Difficulty')

            results[task_name] = task_results

        # Overall composites
        results['overall'] = {}
        for measure in ['focused_attention', 'enjoyment', 'satisfaction']:
            col = f'overall_{measure}'
            if col in self.df.columns:
                results['overall'][measure] = self.compare_groups(col, f'Overall {measure.replace("_", " ").title()}')

        self.results['task_experience'] = results
        return results

    def analyze_pre_task_measures(self) -> Dict:
        """Analyze pre-task measures to check for baseline equivalence"""
        results = {}

        for measure_name in PRE_TASK_MEASURES.keys():
            composite_col = f'{measure_name.lower()}_composite'
            if composite_col in self.df.columns:
                results[measure_name] = self.compare_groups(composite_col, measure_name.replace('_', ' ').title())

        self.results['pre_task'] = results
        return results

    def compute_manipulation_check_summary(self) -> Dict:
        """Compute summary of manipulation check results"""
        summary = {
            'total_measures_tested': 0,
            'significant_differences': 0,
            'expected_direction': 0,
            'details': []
        }

        # Key measures for manipulation check (where we expect differences)
        key_measures = [
            ('anthropomorphism', 'composite', 'Anthropomorphism', 'higher_in_servant'),
            ('ai_collab_post', 'subscales', 'AI Collaboration Post Subscales', 'higher_in_servant'),
            ('task_experience', 'overall', 'Overall Enjoyment', 'higher_in_servant'),
            ('task_experience', 'overall', 'Overall Satisfaction', 'higher_in_servant'),
        ]

        # Check anthropomorphism composite
        anthro = self.results.get('anthropomorphism', {}).get('composite')
        if anthro:
            summary['total_measures_tested'] += 1
            expected_direction = anthro['mean_difference'] > 0  # Higher in servant leader
            if anthro['significant']:
                summary['significant_differences'] += 1
                if expected_direction:
                    summary['expected_direction'] += 1
            summary['details'].append({
                'measure': 'Anthropomorphism',
                'servant_mean': anthro['servant_leader_mean'],
                'dark_triad_mean': anthro['dark_triad_mean'],
                'p_value': anthro['mann_whitney_p'],
                'cohens_d': anthro['cohens_d'],
                'significant': anthro['significant'],
                'expected_direction': expected_direction
            })

        # Check AI collaboration post subscales
        collab_subscales = self.results.get('ai_collab_post', {}).get('subscales', {})
        for subscale_name, result in collab_subscales.items():
            summary['total_measures_tested'] += 1
            expected_direction = result['mean_difference'] > 0  # Higher in servant leader
            if result['significant']:
                summary['significant_differences'] += 1
                if expected_direction:
                    summary['expected_direction'] += 1
            summary['details'].append({
                'measure': result['variable'],
                'servant_mean': result['servant_leader_mean'],
                'dark_triad_mean': result['dark_triad_mean'],
                'p_value': result['mann_whitney_p'],
                'cohens_d': result['cohens_d'],
                'significant': result['significant'],
                'expected_direction': expected_direction
            })

        # Check overall task experience
        task_exp = self.results.get('task_experience', {}).get('overall', {})
        for measure in ['enjoyment', 'satisfaction']:
            result = task_exp.get(measure)
            if result:
                summary['total_measures_tested'] += 1
                expected_direction = result['mean_difference'] > 0  # Higher in servant leader
                if result['significant']:
                    summary['significant_differences'] += 1
                    if expected_direction:
                        summary['expected_direction'] += 1
                summary['details'].append({
                    'measure': f'Overall {measure.title()}',
                    'servant_mean': result['servant_leader_mean'],
                    'dark_triad_mean': result['dark_triad_mean'],
                    'p_value': result['mann_whitney_p'],
                    'cohens_d': result['cohens_d'],
                    'significant': result['significant'],
                    'expected_direction': expected_direction
                })

        # Check individual anthropomorphism items
        anthro_items = self.results.get('anthropomorphism', {}).get('items', {})
        for item, result in anthro_items.items():
            summary['total_measures_tested'] += 1
            expected_direction = result['mean_difference'] > 0
            if result['significant']:
                summary['significant_differences'] += 1
                if expected_direction:
                    summary['expected_direction'] += 1
            summary['details'].append({
                'measure': item,
                'servant_mean': result['servant_leader_mean'],
                'dark_triad_mean': result['dark_triad_mean'],
                'p_value': result['mann_whitney_p'],
                'cohens_d': result['cohens_d'],
                'significant': result['significant'],
                'expected_direction': expected_direction
            })

        self.results['manipulation_summary'] = summary
        return summary

    def run_full_analysis(self) -> Dict:
        """Run complete perception analysis"""
        print("\n" + "="*60)
        print("RUNNING PERCEPTION-BASED MANIPULATION CHECK")
        print("="*60)

        print("\n1. Analyzing anthropomorphism items...")
        self.analyze_anthropomorphism()

        print("2. Analyzing AI collaboration post-task scale...")
        self.analyze_ai_collaboration_post()

        print("3. Analyzing task experience measures...")
        self.analyze_task_experience()

        print("4. Analyzing pre-task measures (baseline equivalence)...")
        self.analyze_pre_task_measures()

        print("5. Computing manipulation check summary...")
        self.compute_manipulation_check_summary()

        print("\nAnalysis complete")
        return self.results


# ============================================================================
# OPEN-ENDED RESPONSE ANALYSIS
# ============================================================================

class OpenEndedAnalyzer:
    """Analyze open-ended feedback responses"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}

    def extract_responses(self) -> Dict:
        """Extract open-ended responses by condition"""
        if 'Open_ended' not in self.df.columns:
            print("WARNING: No 'Open_ended' column found")
            return {}

        results = {
            'servant_leader': [],
            'dark_triad': [],
            'word_counts': {}
        }

        for condition in ['servant_leader', 'dark_triad']:
            responses = self.df[self.df['condition'] == condition]['Open_ended'].dropna()
            results[condition] = responses.tolist()
            results['word_counts'][condition] = {
                'n_responses': len(responses),
                'mean_length': responses.str.len().mean() if len(responses) > 0 else 0,
                'mean_word_count': responses.str.split().str.len().mean() if len(responses) > 0 else 0
            }

        self.results = results
        return results

    def keyword_analysis(self) -> Dict:
        """Simple keyword analysis for sentiment indicators"""
        if 'Open_ended' not in self.df.columns:
            return {}

        # Define sentiment keywords
        positive_keywords = ['good', 'great', 'helpful', 'nice', 'like', 'enjoy', 'easy',
                           'useful', 'interesting', 'fun', 'well', 'appreciate', 'love']
        negative_keywords = ['bad', 'difficult', 'hard', 'annoying', 'frustrating', 'irritating',
                           'confusing', 'unhelpful', 'brat', 'whining', 'judgment', 'predictable']

        results = {'servant_leader': {}, 'dark_triad': {}}

        for condition in ['servant_leader', 'dark_triad']:
            responses = self.df[self.df['condition'] == condition]['Open_ended'].dropna()
            all_text = ' '.join(responses.str.lower())

            pos_count = sum(all_text.count(kw) for kw in positive_keywords)
            neg_count = sum(all_text.count(kw) for kw in negative_keywords)

            results[condition] = {
                'positive_keywords': pos_count,
                'negative_keywords': neg_count,
                'sentiment_ratio': pos_count / max(neg_count, 1),
                'n_responses': len(responses)
            }

        return results


# ============================================================================
# VISUALIZATION
# ============================================================================

class PerceptionVisualizer:
    """Create visualizations for perception analysis"""

    def __init__(self, df: pd.DataFrame, results: Dict, output_folder: str):
        self.df = df
        self.results = results
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.colors = {
            'servant_leader': '#3498DB',
            'dark_triad': '#E74C3C'
        }

    def plot_anthropomorphism(self):
        """Plot anthropomorphism items comparison"""
        anthro_results = self.results.get('anthropomorphism', {})

        if not anthro_results.get('items'):
            print("WARNING: No anthropomorphism data to plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Individual items
        items = list(anthro_results['items'].keys())
        servant_means = [anthro_results['items'][item]['servant_leader_mean'] for item in items]
        dark_means = [anthro_results['items'][item]['dark_triad_mean'] for item in items]
        servant_stds = [anthro_results['items'][item]['servant_leader_std'] for item in items]
        dark_stds = [anthro_results['items'][item]['dark_triad_std'] for item in items]
        significant = [anthro_results['items'][item]['significant'] for item in items]

        x = np.arange(len(items))
        width = 0.35

        bars1 = axes[0].bar(x - width/2, servant_means, width, yerr=servant_stds,
                           label='Servant Leader', color=self.colors['servant_leader'],
                           capsize=3, alpha=0.8)
        bars2 = axes[0].bar(x + width/2, dark_means, width, yerr=dark_stds,
                           label='Dark Triad', color=self.colors['dark_triad'],
                           capsize=3, alpha=0.8)

        # Add significance markers
        for i, sig in enumerate(significant):
            if sig:
                max_height = max(servant_means[i] + servant_stds[i], dark_means[i] + dark_stds[i])
                axes[0].text(i, max_height + 0.3, '*', ha='center', fontsize=16, fontweight='bold')

        axes[0].set_xlabel('Item', fontsize=12)
        axes[0].set_ylabel('Mean Rating', fontsize=12)
        axes[0].set_title('Anthropomorphism Items by Condition', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(items, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Plot 2: Composite with distribution
        if 'anthropomorphism_composite' in self.df.columns:
            for condition in ['servant_leader', 'dark_triad']:
                data = self.df[self.df['condition'] == condition]['anthropomorphism_composite'].dropna()
                axes[1].hist(data, bins=15, alpha=0.6, label=condition.replace('_', ' ').title(),
                            color=self.colors[condition], density=True)
                axes[1].axvline(data.mean(), color=self.colors[condition], linestyle='--', linewidth=2)

            axes[1].set_xlabel('Anthropomorphism Composite Score', fontsize=12)
            axes[1].set_ylabel('Density', fontsize=12)
            axes[1].set_title('Anthropomorphism Composite Distribution', fontsize=14, fontweight='bold')
            axes[1].legend()
            axes[1].grid(axis='y', alpha=0.3)

            # Add statistics annotation
            composite = anthro_results.get('composite', {})
            if composite:
                stats_text = f"p = {composite.get('mann_whitney_p', 'N/A'):.3f}\nd = {composite.get('cohens_d', 'N/A'):.2f}"
                axes[1].text(0.95, 0.95, stats_text, transform=axes[1].transAxes,
                            ha='right', va='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(self.output_folder / 'anthropomorphism_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: anthropomorphism_comparison.png")

    def plot_ai_collaboration_post(self):
        """Plot AI collaboration post-task subscales comparison"""
        collab_results = self.results.get('ai_collab_post', {})
        subscales = collab_results.get('subscales', {})

        if not subscales:
            print("WARNING: No AI collaboration subscale data to plot")
            return

        fig, ax = plt.subplots(figsize=(14, 6))

        names = list(subscales.keys())
        labels = [subscales[n]['variable'] for n in names]
        servant_means = [subscales[n]['servant_leader_mean'] for n in names]
        dark_means = [subscales[n]['dark_triad_mean'] for n in names]
        servant_stds = [subscales[n]['servant_leader_std'] for n in names]
        dark_stds = [subscales[n]['dark_triad_std'] for n in names]
        significant = [subscales[n]['significant'] for n in names]

        x = np.arange(len(names))
        width = 0.35

        ax.bar(x - width/2, servant_means, width, yerr=servant_stds,
               label='Servant Leader', color=self.colors['servant_leader'],
               capsize=3, alpha=0.8)
        ax.bar(x + width/2, dark_means, width, yerr=dark_stds,
               label='Dark Triad', color=self.colors['dark_triad'],
               capsize=3, alpha=0.8)

        for i, sig in enumerate(significant):
            if sig:
                max_height = max(servant_means[i] + servant_stds[i], dark_means[i] + dark_stds[i])
                ax.text(i, max_height + 0.15, '*', ha='center', fontsize=16, fontweight='bold')

        ax.set_xlabel('Subscale', fontsize=12)
        ax.set_ylabel('Mean Rating', fontsize=12)
        ax.set_title('AI Collaboration Post-Task Subscales by Condition', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_folder / 'ai_collaboration_post_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: ai_collaboration_post_comparison.png")

    def plot_task_experience(self):
        """Plot task experience measures comparison"""
        task_results = self.results.get('task_experience', {})

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        # Plot for each task
        for idx, task_name in enumerate(['Persona', 'Brief', 'Communication', 'Content']):
            task_data = task_results.get(task_name, {})

            measures = ['focused_attention', 'enjoyment', 'satisfaction']
            measure_labels = ['Focused\nAttention', 'Enjoyment', 'Satisfaction']

            servant_means = []
            dark_means = []
            servant_stds = []
            dark_stds = []

            for measure in measures:
                result = task_data.get(measure, {})
                if result:
                    servant_means.append(result.get('servant_leader_mean', 0))
                    dark_means.append(result.get('dark_triad_mean', 0))
                    servant_stds.append(result.get('servant_leader_std', 0))
                    dark_stds.append(result.get('dark_triad_std', 0))
                else:
                    servant_means.append(0)
                    dark_means.append(0)
                    servant_stds.append(0)
                    dark_stds.append(0)

            x = np.arange(len(measures))
            width = 0.35

            axes[idx].bar(x - width/2, servant_means, width, yerr=servant_stds,
                         label='Servant Leader', color=self.colors['servant_leader'],
                         capsize=3, alpha=0.8)
            axes[idx].bar(x + width/2, dark_means, width, yerr=dark_stds,
                         label='Dark Triad', color=self.colors['dark_triad'],
                         capsize=3, alpha=0.8)

            axes[idx].set_xlabel('Measure', fontsize=11)
            axes[idx].set_ylabel('Mean Rating', fontsize=11)
            axes[idx].set_title(f'Task: {task_name}', fontsize=12, fontweight='bold')
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(measure_labels)
            axes[idx].legend(fontsize=9)
            axes[idx].grid(axis='y', alpha=0.3)

        plt.suptitle('Task Experience by Condition', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_folder / 'task_experience_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: task_experience_comparison.png")

    def plot_effect_sizes(self):
        """Plot effect sizes for key comparisons"""
        fig, ax = plt.subplots(figsize=(12, 8))

        effect_data = []

        # Collect effect sizes from anthropomorphism
        anthro = self.results.get('anthropomorphism', {})
        if anthro.get('composite'):
            effect_data.append({
                'Measure': 'Anthropomorphism\n(Composite)',
                "Cohen's d": anthro['composite']['cohens_d'],
                'p_value': anthro['composite']['mann_whitney_p'],
                'category': 'Anthropomorphism'
            })
        for item, data in anthro.get('items', {}).items():
            effect_data.append({
                'Measure': item,
                "Cohen's d": data['cohens_d'],
                'p_value': data['mann_whitney_p'],
                'category': 'Anthropomorphism'
            })

        # AI collaboration post subscales
        collab = self.results.get('ai_collab_post', {})
        for subscale_name, data in collab.get('subscales', {}).items():
            effect_data.append({
                'Measure': data['variable'].replace(' ', '\n'),
                "Cohen's d": data['cohens_d'],
                'p_value': data['mann_whitney_p'],
                'category': 'AI Collaboration'
            })

        # Task experience
        task_exp = self.results.get('task_experience', {}).get('overall', {})
        for measure, data in task_exp.items():
            if data:
                effect_data.append({
                    'Measure': f'Overall\n{measure.replace("_", " ").title()}',
                    "Cohen's d": data['cohens_d'],
                    'p_value': data['mann_whitney_p'],
                    'category': 'Task Experience'
                })

        if not effect_data:
            print("WARNING: No effect size data to plot")
            return

        df_effects = pd.DataFrame(effect_data)

        # Sort by effect size
        df_effects = df_effects.sort_values("Cohen's d", ascending=True)

        # Color by significance and direction
        colors = []
        for _, row in df_effects.iterrows():
            if row['p_value'] < ALPHA:
                colors.append('#27AE60' if row["Cohen's d"] > 0 else '#E74C3C')
            else:
                colors.append('#95A5A6')

        bars = ax.barh(range(len(df_effects)), df_effects["Cohen's d"], color=colors)
        ax.set_yticks(range(len(df_effects)))
        ax.set_yticklabels(df_effects['Measure'])
        ax.set_xlabel("Cohen's d (Positive = Higher in Servant Leader)", fontsize=12)
        ax.set_title('Effect Sizes: Servant Leader vs Dark Triad\n(Green = Significant & Higher in SL, Red = Significant & Higher in DT, Gray = Not Significant)',
                    fontsize=12, fontweight='bold')

        # Add reference lines
        ax.axvline(x=0, color='black', linewidth=1)
        ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='-.', alpha=0.5)
        ax.axvline(x=0.8, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=-0.5, color='gray', linestyle='-.', alpha=0.5)
        ax.axvline(x=-0.8, color='gray', linestyle=':', alpha=0.5)

        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_folder / 'perception_effect_sizes.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: perception_effect_sizes.png")

    def plot_manipulation_summary(self):
        """Plot manipulation check summary"""
        summary = self.results.get('manipulation_summary', {})

        if not summary.get('details'):
            print("WARNING: No manipulation summary data to plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Overall summary
        labels = ['Significant\nDifferences', 'Expected\nDirection', 'Total\nMeasures']
        values = [
            summary.get('significant_differences', 0),
            summary.get('expected_direction', 0),
            summary.get('total_measures_tested', 0)
        ]

        colors_bar = ['#27AE60' if i < 2 else '#3498DB' for i in range(3)]
        axes[0].bar(labels, values, color=colors_bar, alpha=0.8)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Manipulation Check Summary', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # Add percentage annotations
        if summary.get('total_measures_tested', 0) > 0:
            sig_pct = summary['significant_differences'] / summary['total_measures_tested'] * 100
            dir_pct = summary['expected_direction'] / max(summary['significant_differences'], 1) * 100
            axes[0].text(0, values[0] + 0.5, f'{sig_pct:.0f}%', ha='center', fontsize=10)
            axes[0].text(1, values[1] + 0.5, f'{dir_pct:.0f}%\nof sig.', ha='center', fontsize=10)

        # Plot 2: Detailed results
        details = summary.get('details', [])
        if details:
            measures = [d['measure'][:15] + '...' if len(d['measure']) > 15 else d['measure'] for d in details]
            p_values = [d['p_value'] for d in details]
            effect_sizes = [d['cohens_d'] for d in details]
            significant = [d['significant'] for d in details]

            # Color by significance
            colors_scatter = ['#27AE60' if s else '#E74C3C' for s in significant]

            axes[1].scatter(p_values, effect_sizes, c=colors_scatter, s=100, alpha=0.7)

            # Add measure labels
            for i, measure in enumerate(measures):
                axes[1].annotate(measure, (p_values[i], effect_sizes[i]),
                               fontsize=8, ha='left', va='bottom')

            axes[1].axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
            axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label="d = 0.5 (medium)")
            axes[1].set_xlabel('p-value', fontsize=12)
            axes[1].set_ylabel("Cohen's d", fontsize=12)
            axes[1].set_title('Effect Size vs p-value', fontsize=14, fontweight='bold')
            axes[1].legend(fontsize=9)
            axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_folder / 'manipulation_check_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: manipulation_check_summary.png")

    def create_all_visualizations(self):
        """Generate all visualizations"""
        print("\nGenerating visualizations...")
        self.plot_anthropomorphism()
        self.plot_ai_collaboration_post()
        self.plot_task_experience()
        self.plot_effect_sizes()
        self.plot_manipulation_summary()


# ============================================================================
# REPORT GENERATION
# ============================================================================

class PerceptionReporter:
    """Generate the perception analysis report"""

    def __init__(self, df: pd.DataFrame, results: Dict, output_folder: str):
        self.df = df
        self.results = results
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def generate_report(self):
        """Generate text report"""
        report_path = self.output_folder / 'perception_analysis_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("POST-HOC PARTICIPANT PERCEPTION ANALYSIS REPORT\n")
            f.write("Manipulation Check: Did participants perceive conditions differently?\n")
            f.write("="*80 + "\n\n")

            # Sample sizes
            f.write("SAMPLE INFORMATION\n")
            f.write("-"*80 + "\n")
            for condition in ['servant_leader', 'dark_triad']:
                n = len(self.df[self.df['condition'] == condition])
                f.write(f"  {condition.replace('_', ' ').title()}: n = {n}\n")
            f.write("\n")

            # Manipulation check summary
            summary = self.results.get('manipulation_summary', {})
            f.write("MANIPULATION CHECK SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(f"Total measures tested: {summary.get('total_measures_tested', 0)}\n")
            f.write(f"Significant differences: {summary.get('significant_differences', 0)}\n")
            f.write(f"In expected direction: {summary.get('expected_direction', 0)}\n\n")

            f.write("Detailed Results:\n")
            for detail in summary.get('details', []):
                sig_marker = "***" if detail['p_value'] < 0.001 else "**" if detail['p_value'] < 0.01 else "*" if detail['p_value'] < 0.05 else ""
                direction = "SL > DT" if detail['servant_mean'] > detail['dark_triad_mean'] else "DT > SL"
                f.write(f"\n  {detail['measure']}:{sig_marker}\n")
                f.write(f"    Servant Leader M = {detail['servant_mean']:.2f}\n")
                f.write(f"    Dark Triad M = {detail['dark_triad_mean']:.2f}\n")
                f.write(f"    Direction: {direction}\n")
                f.write(f"    p = {detail['p_value']:.4f}, d = {detail['cohens_d']:.2f}\n")

            # Anthropomorphism
            f.write("\n" + "-"*80 + "\n")
            f.write("ANTHROPOMORPHISM ANALYSIS\n")
            f.write("-"*80 + "\n")

            anthro = self.results.get('anthropomorphism', {})
            if anthro.get('composite'):
                comp = anthro['composite']
                f.write(f"\nComposite Score:\n")
                f.write(f"  Servant Leader: M = {comp['servant_leader_mean']:.2f} (SD = {comp['servant_leader_std']:.2f})\n")
                f.write(f"  Dark Triad: M = {comp['dark_triad_mean']:.2f} (SD = {comp['dark_triad_std']:.2f})\n")
                f.write(f"  Mann-Whitney U = {comp['mann_whitney_u']:.1f}, p = {comp['mann_whitney_p']:.4f}\n")
                f.write(f"  Cohen's d = {comp['cohens_d']:.2f} ({comp['effect_interpretation']})\n")

            f.write("\nIndividual Items:\n")
            for item, data in anthro.get('items', {}).items():
                sig = "*" if data['significant'] else ""
                f.write(f"  {item}:{sig} SL={data['servant_leader_mean']:.2f}, DT={data['dark_triad_mean']:.2f}, ")
                f.write(f"p={data['mann_whitney_p']:.3f}, d={data['cohens_d']:.2f}\n")

            # AI Collaboration Post
            f.write("\n" + "-"*80 + "\n")
            f.write("AI COLLABORATION POST-TASK SUBSCALES ANALYSIS\n")
            f.write("-"*80 + "\n")

            collab = self.results.get('ai_collab_post', {})
            for subscale_name, data in collab.get('subscales', {}).items():
                sig = "*" if data['significant'] else ""
                f.write(f"\n  {data['variable']}:{sig}\n")
                f.write(f"    Servant Leader: M = {data['servant_leader_mean']:.2f} (SD = {data['servant_leader_std']:.2f})\n")
                f.write(f"    Dark Triad: M = {data['dark_triad_mean']:.2f} (SD = {data['dark_triad_std']:.2f})\n")
                f.write(f"    Mann-Whitney U = {data['mann_whitney_u']:.1f}, p = {data['mann_whitney_p']:.4f}\n")
                f.write(f"    Cohen's d = {data['cohens_d']:.2f} ({data['effect_interpretation']})\n")

            # Task experience
            f.write("\n" + "-"*80 + "\n")
            f.write("TASK EXPERIENCE ANALYSIS\n")
            f.write("-"*80 + "\n")

            task_exp = self.results.get('task_experience', {})
            overall = task_exp.get('overall', {})

            f.write("\nOverall Composites:\n")
            for measure, data in overall.items():
                if data:
                    sig = "*" if data['significant'] else ""
                    f.write(f"  {measure.replace('_', ' ').title()}:{sig}\n")
                    f.write(f"    SL: M={data['servant_leader_mean']:.2f} (SD={data['servant_leader_std']:.2f})\n")
                    f.write(f"    DT: M={data['dark_triad_mean']:.2f} (SD={data['dark_triad_std']:.2f})\n")
                    f.write(f"    p={data['mann_whitney_p']:.4f}, d={data['cohens_d']:.2f}\n\n")

            # Pre-task measures (baseline equivalence)
            f.write("\n" + "-"*80 + "\n")
            f.write("BASELINE EQUIVALENCE CHECK (Pre-Task Measures)\n")
            f.write("-"*80 + "\n")

            pre_task = self.results.get('pre_task', {})
            for measure, data in pre_task.items():
                if data:
                    sig = "*" if data['significant'] else ""
                    equiv = "EQUIVALENT" if not data['significant'] else "DIFFERENCE DETECTED"
                    f.write(f"\n  {measure.replace('_', ' ').title()}:{sig} [{equiv}]\n")
                    f.write(f"    SL: M={data['servant_leader_mean']:.2f}, DT: M={data['dark_triad_mean']:.2f}\n")
                    f.write(f"    p={data['mann_whitney_p']:.4f}, d={data['cohens_d']:.2f}\n")

            # Conclusion
            f.write("\n" + "="*80 + "\n")
            f.write("CONCLUSION\n")
            f.write("="*80 + "\n\n")

            sig_count = summary.get('significant_differences', 0)
            total = summary.get('total_measures_tested', 0)
            expected = summary.get('expected_direction', 0)

            if sig_count > 0 and expected == sig_count:
                f.write("The perception-based manipulation check suggests that participants\n")
                f.write("perceived the chatbot conditions differently in the expected direction.\n")
                f.write(f"  - {sig_count}/{total} measures showed significant differences\n")
                f.write(f"  - All significant differences were in the expected direction\n")
                f.write("  - This supports the validity of the experimental manipulation.\n")
            elif sig_count > 0:
                f.write("The perception-based manipulation check shows mixed results.\n")
                f.write(f"  - {sig_count}/{total} measures showed significant differences\n")
                f.write(f"  - {expected}/{sig_count} were in the expected direction\n")
            else:
                f.write("The perception-based manipulation check did not reveal significant\n")
                f.write("differences between conditions on the measured variables.\n")
                f.write("This may warrant further investigation.\n")

            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        print(f"Saved: perception_analysis_report.txt")
        return report_path

    def generate_summary_table(self) -> pd.DataFrame:
        """Generate summary table for publication"""
        rows = []

        # Anthropomorphism
        anthro = self.results.get('anthropomorphism', {})
        if anthro.get('composite'):
            c = anthro['composite']
            rows.append({
                'Measure': 'Anthropomorphism (Composite)',
                'Servant Leader M (SD)': f"{c['servant_leader_mean']:.2f} ({c['servant_leader_std']:.2f})",
                'Dark Triad M (SD)': f"{c['dark_triad_mean']:.2f} ({c['dark_triad_std']:.2f})",
                'U': f"{c['mann_whitney_u']:.1f}",
                'p': f"{c['mann_whitney_p']:.3f}",
                "Cohen's d": f"{c['cohens_d']:.2f}",
                'Effect': c['effect_interpretation']
            })

        for item, data in anthro.get('items', {}).items():
            rows.append({
                'Measure': item,
                'Servant Leader M (SD)': f"{data['servant_leader_mean']:.2f} ({data['servant_leader_std']:.2f})",
                'Dark Triad M (SD)': f"{data['dark_triad_mean']:.2f} ({data['dark_triad_std']:.2f})",
                'U': f"{data['mann_whitney_u']:.1f}",
                'p': f"{data['mann_whitney_p']:.3f}",
                "Cohen's d": f"{data['cohens_d']:.2f}",
                'Effect': data['effect_interpretation']
            })

        # AI Collaboration Post subscales
        collab = self.results.get('ai_collab_post', {})
        for subscale_name, data in collab.get('subscales', {}).items():
            rows.append({
                'Measure': data['variable'],
                'Servant Leader M (SD)': f"{data['servant_leader_mean']:.2f} ({data['servant_leader_std']:.2f})",
                'Dark Triad M (SD)': f"{data['dark_triad_mean']:.2f} ({data['dark_triad_std']:.2f})",
                'U': f"{data['mann_whitney_u']:.1f}",
                'p': f"{data['mann_whitney_p']:.3f}",
                "Cohen's d": f"{data['cohens_d']:.2f}",
                'Effect': data['effect_interpretation']
            })

        # Task experience overall
        task_exp = self.results.get('task_experience', {}).get('overall', {})
        for measure, data in task_exp.items():
            if data:
                rows.append({
                    'Measure': f'Overall {measure.replace("_", " ").title()}',
                    'Servant Leader M (SD)': f"{data['servant_leader_mean']:.2f} ({data['servant_leader_std']:.2f})",
                    'Dark Triad M (SD)': f"{data['dark_triad_mean']:.2f} ({data['dark_triad_std']:.2f})",
                    'U': f"{data['mann_whitney_u']:.1f}",
                    'p': f"{data['mann_whitney_p']:.3f}",
                    "Cohen's d": f"{data['cohens_d']:.2f}",
                    'Effect': data['effect_interpretation']
                })

        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(self.output_folder / 'perception_summary_table.csv', index=False)
        print("Saved: perception_summary_table.csv")

        return summary_df

    def export_open_ended(self, open_ended_results: Dict):
        """Export open-ended responses for qualitative analysis"""
        export_path = self.output_folder / 'open_ended_responses.txt'

        with open(export_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("OPEN-ENDED RESPONSES BY CONDITION\n")
            f.write("="*80 + "\n\n")

            for condition in ['servant_leader', 'dark_triad']:
                f.write(f"\n{'='*40}\n")
                f.write(f"{condition.upper().replace('_', ' ')} CONDITION\n")
                f.write(f"{'='*40}\n\n")

                responses = open_ended_results.get(condition, [])
                word_counts = open_ended_results.get('word_counts', {}).get(condition, {})

                f.write(f"Number of responses: {word_counts.get('n_responses', len(responses))}\n")
                f.write(f"Mean length: {word_counts.get('mean_length', 0):.0f} characters\n")
                f.write(f"Mean word count: {word_counts.get('mean_word_count', 0):.0f} words\n\n")

                for i, response in enumerate(responses, 1):
                    f.write(f"--- Response {i} ---\n")
                    f.write(f"{response}\n\n")

        print(f"Saved: open_ended_responses.txt")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("\n" + "="*80)
    print("POST-HOC PARTICIPANT PERCEPTION ANALYSIS")
    print("Manipulation Check Based on Participant Self-Reports")
    print("="*80 + "\n")

    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    loader = PerceptionDataLoader(QUALTRICS_FILE)
    df = loader.preprocess()

    # Step 2: Run statistical analysis
    print("\nStep 2: Running statistical analysis...")
    stats_analyzer = PerceptionStatistics(df)
    results = stats_analyzer.run_full_analysis()

    # Step 3: Analyze open-ended responses
    print("\nStep 3: Analyzing open-ended responses...")
    open_ended_analyzer = OpenEndedAnalyzer(df)
    open_ended_results = open_ended_analyzer.extract_responses()
    keyword_results = open_ended_analyzer.keyword_analysis()

    print(f"  Keyword analysis:")
    for condition, data in keyword_results.items():
        print(f"    {condition}: positive={data.get('positive_keywords', 0)}, negative={data.get('negative_keywords', 0)}")

    # Step 4: Generate visualizations
    print("\nStep 4: Generating visualizations...")
    visualizer = PerceptionVisualizer(df, results, OUTPUT_FOLDER)
    visualizer.create_all_visualizations()

    # Step 5: Generate reports
    print("\nStep 5: Generating reports...")
    reporter = PerceptionReporter(df, results, OUTPUT_FOLDER)
    reporter.generate_report()
    summary_table = reporter.generate_summary_table()
    reporter.export_open_ended(open_ended_results)

    # Print summary
    print("\n" + "="*80)
    print("PERCEPTION ANALYSIS COMPLETE")
    print("="*80)

    summary = results.get('manipulation_summary', {})
    print(f"\nKey Findings:")
    print(f"  Measures tested: {summary.get('total_measures_tested', 0)}")
    print(f"  Significant differences: {summary.get('significant_differences', 0)}")
    print(f"  In expected direction: {summary.get('expected_direction', 0)}")

    print(f"\nResults saved to: {OUTPUT_FOLDER}")
    print("\nGenerated files:")
    print("  - perception_analysis_report.txt")
    print("  - perception_summary_table.csv")
    print("  - open_ended_responses.txt")
    print("  - anthropomorphism_comparison.png")
    print("  - ai_collaboration_post_comparison.png")
    print("  - task_experience_comparison.png")
    print("  - perception_effect_sizes.png")
    print("  - manipulation_check_summary.png")

    print("\n" + "="*80 + "\n")

    return df, results, summary_table


if __name__ == "__main__":
    df, results, summary_table = main()
# %%
