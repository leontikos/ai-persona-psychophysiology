# %% gpt_assessment_analysis.py
# Analyze and visualize the LLM-coded conversation data: participant messages,
# chatbot messages, and interaction dynamics.

import json
import os
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
ANALYSES_FOLDER = os.path.join(DATA_ROOT, "CONVERSATION_ANALYSES")
CONVERSATION_LOGS_FOLDER = os.path.join(DATA_ROOT, "conversation_logs")

# ============================================================================
# DATA LOADER CLASS (Extended for chatbot and interaction data)
# ============================================================================

class AnalysisResultsLoader:
    """Load and parse analyzed conversation data including chatbot messages"""

    def __init__(self, analyses_folder: str, conversation_logs_folder: str = None):
        self.analyses_folder = Path(analyses_folder) / "successful_analyses"
        self.conversation_logs_folder = Path(conversation_logs_folder) if conversation_logs_folder else None
        self.conversations = []
        self.conversation_timestamps = {}  # Store timestamp data
        self.messages_df = None
        self.chatbot_messages_df = None
        self.conversations_df = None
        self.interactions_df = None

    def load_all_analyses(self):
        """Load all analyzed conversation files"""
        analysis_files = list(self.analyses_folder.glob("*_analyzed.json"))
        print(f"Found {len(analysis_files)} analyzed conversations")

        for filepath in analysis_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.conversations.append(data)
            except Exception as e:
                print(f"Error loading {filepath.name}: {e}")

        print(f"Successfully loaded {len(self.conversations)} conversations")
        return self.conversations

    def load_conversation_timestamps(self):
        """Load timestamp data from original conversation log files"""
        if not self.conversation_logs_folder or not self.conversation_logs_folder.exists():
            print("WARNING: Conversation logs folder not found - duration data will not be available")
            return

        loaded_count = 0
        merged_count = 0
        for conv in self.conversations:
            conv_id = conv['conversation_id']
            # Use original_filename if available, otherwise try to construct from conv_id
            original_filename = conv.get('original_filename')

            if original_filename:
                log_file = self.conversation_logs_folder / original_filename
            else:
                # Fallback: try using conv_id (may not work for all formats)
                log_file = self.conversation_logs_folder / f"{conv_id}.json"

            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)

                    # Extract timestamps
                    metadata = log_data.get('metadata', {})
                    messages = log_data.get('messages', [])

                    # Check if this is a merged file
                    is_merged = metadata.get('_merged', False)

                    first_timestamp = messages[0].get('timestamp_start') if messages else None

                    # For merged files, use the last message's timestamp_end instead of last_updated
                    # (last_updated reflects the merge time, not the actual conversation end)
                    if is_merged and messages:
                        last_timestamp = messages[-1].get('timestamp_end')
                        merged_count += 1
                    else:
                        last_timestamp = metadata.get('last_updated')

                    if last_timestamp and first_timestamp:
                        try:
                            # Parse timestamps
                            start_dt = datetime.fromisoformat(first_timestamp)
                            end_dt = datetime.fromisoformat(last_timestamp)
                            duration_seconds = (end_dt - start_dt).total_seconds()
                            duration_minutes = duration_seconds / 60

                            self.conversation_timestamps[conv_id] = {
                                'start_time': start_dt,
                                'end_time': end_dt,
                                'duration_seconds': duration_seconds,
                                'duration_minutes': duration_minutes,
                                'is_merged': is_merged
                            }
                            loaded_count += 1
                        except (ValueError, TypeError) as e:
                            print(f"  Error parsing timestamps for {conv_id}: {e}")
                except Exception as e:
                    print(f"  Error loading log file {log_file.name}: {e}")

        print(f"  Loaded timestamp data for {loaded_count}/{len(self.conversations)} conversations")
        if merged_count > 0:
            print(f"  (Including {merged_count} merged conversation files)")

    def create_message_dataframe(self) -> pd.DataFrame:
        """Create dataframe for participant messages"""
        rows = []

        for conv in self.conversations:
            conv_id = conv['conversation_id']
            # Handle both old and new schema
            summary = conv.get('participant_summary', conv.get('conversation_summary', {}))
            condition = summary.get('condition_inference', 'unknown')
            condition = condition.lower().replace(' ', '_')

            # Handle both old and new schema for message analyses
            msg_analyses = conv.get('participant_message_analyses', conv.get('message_analyses', []))

            for msg in msg_analyses:
                row = {
                    'conversation_id': conv_id,
                    'condition': condition,
                    'message_number': msg['message_number'],
                    'primary_strategy': msg['primary_strategy'],
                    'task_engagement': msg['task_engagement'],
                    'message_length': msg['message_length'],
                    'typo_indicators': msg.get('typo_indicators', False),
                    'num_emotions': len(msg['emotional_tone']),
                    'participant_message': msg['participant_message'],
                    'evidence': msg['evidence'],
                    'notable_behaviors': ', '.join(msg.get('notable_behaviors', [])),
                }

                for emotion in ['FRUSTRATED', 'ANXIOUS', 'CONFIDENT', 'NEUTRAL', 'RESIGNED', 'DEFENSIVE']:
                    row[f'emotion_{emotion.lower()}'] = emotion in msg['emotional_tone']

                rows.append(row)

        self.messages_df = pd.DataFrame(rows)

        if len(self.messages_df) > 0:
            self.messages_df['strategy_category'] = self.messages_df['primary_strategy'].str.split('_').str[0]
            self.messages_df['message_position'] = self.messages_df.groupby('conversation_id')['message_number'].transform(
                lambda x: x / x.max() if x.max() > 0 else 0
            )

        return self.messages_df

    def create_chatbot_message_dataframe(self) -> pd.DataFrame:
        """Create dataframe for chatbot messages"""
        rows = []

        for conv in self.conversations:
            conv_id = conv['conversation_id']
            summary = conv.get('participant_summary', conv.get('conversation_summary', {}))
            condition = summary.get('condition_inference', 'unknown')
            condition = condition.lower().replace(' ', '_')

            chatbot_analyses = conv.get('chatbot_message_analyses', [])

            for msg in chatbot_analyses:
                row = {
                    'conversation_id': conv_id,
                    'condition': condition,
                    'message_number': msg['message_number'],
                    'chatbot_message': msg.get('chatbot_message', ''),
                    'primary_tone': msg.get('primary_tone', 'NEUTRAL'),
                    'response_to_participant': msg.get('response_to_participant', 'NEUTRAL'),
                    'task_guidance_style': msg.get('task_guidance_style', 'ABSENT'),
                    'message_length': msg.get('message_length', 0),
                    'evidence': msg.get('evidence', ''),
                    'notable_behaviors': ', '.join(msg.get('notable_behaviors', [])),
                    'condition_markers': ', '.join(msg.get('condition_markers', [])),
                }

                # Leadership tactics (can be multiple)
                tactics = msg.get('leadership_tactics', [])
                row['num_tactics'] = len(tactics)
                row['tactics_list'] = ', '.join(tactics)

                # Categorize tactics
                dark_triad_tactics = ['INTIMIDATION', 'CRITICISM_HARSH', 'MANIPULATION',
                                     'DISMISSIVENESS', 'MICROMANAGEMENT', 'BLAME_SHIFTING', 'GRANDIOSITY']
                servant_tactics = ['ENCOURAGEMENT', 'EMPOWERMENT', 'CONSTRUCTIVE_FEEDBACK',
                                  'ACTIVE_LISTENING', 'COLLABORATION', 'SUPPORT_OFFERING', 'RECOGNITION']
                neutral_tactics = ['TASK_DIRECTION', 'INFORMATION_PROVISION', 'CLARIFICATION']

                row['dark_triad_tactics_count'] = sum(1 for t in tactics if t in dark_triad_tactics)
                row['servant_tactics_count'] = sum(1 for t in tactics if t in servant_tactics)
                row['neutral_tactics_count'] = sum(1 for t in tactics if t in neutral_tactics)

                # Individual tactic flags
                for tactic in dark_triad_tactics + servant_tactics + neutral_tactics:
                    row[f'tactic_{tactic.lower()}'] = tactic in tactics

                # Secondary tones
                secondary_tones = msg.get('secondary_tones', [])
                row['num_secondary_tones'] = len(secondary_tones)

                for tone in ['HOSTILE', 'CONDESCENDING', 'COLD', 'NEUTRAL', 'WARM', 'SUPPORTIVE', 'SARCASTIC']:
                    row[f'tone_{tone.lower()}'] = tone == msg.get('primary_tone') or tone in secondary_tones

                rows.append(row)

        self.chatbot_messages_df = pd.DataFrame(rows)

        if len(self.chatbot_messages_df) > 0:
            self.chatbot_messages_df['message_position'] = self.chatbot_messages_df.groupby('conversation_id')['message_number'].transform(
                lambda x: x / x.max() if x.max() > 0 else 0
            )

        return self.chatbot_messages_df

    def create_interactions_dataframe(self) -> pd.DataFrame:
        """Create dataframe linking participant and chatbot turns"""
        rows = []

        for conv in self.conversations:
            conv_id = conv['conversation_id']
            summary = conv.get('participant_summary', conv.get('conversation_summary', {}))
            condition = summary.get('condition_inference', 'unknown')
            condition = condition.lower().replace(' ', '_')

            interaction_turns = conv.get('interaction_turns', [])
            participant_msgs = conv.get('participant_message_analyses', conv.get('message_analyses', []))
            chatbot_msgs = conv.get('chatbot_message_analyses', [])

            # Create lookup dictionaries
            participant_lookup = {m['message_number']: m for m in participant_msgs}
            chatbot_lookup = {m['message_number']: m for m in chatbot_msgs}

            for turn in interaction_turns:
                turn_num = turn.get('turn_number', 0)

                row = {
                    'conversation_id': conv_id,
                    'condition': condition,
                    'turn_number': turn_num,
                    'interaction_dynamics': turn.get('interaction_dynamics', ''),
                }

                # Get participant message info for this turn
                p_analysis = turn.get('participant_message_analysis')
                if p_analysis:
                    row['participant_strategy'] = p_analysis.get('primary_strategy', '')
                    row['participant_engagement'] = p_analysis.get('task_engagement', '')
                    row['participant_emotions'] = ', '.join(p_analysis.get('emotional_tone', []))
                    row['participant_msg_length'] = p_analysis.get('message_length', 0)
                else:
                    row['participant_strategy'] = None
                    row['participant_engagement'] = None
                    row['participant_emotions'] = None
                    row['participant_msg_length'] = None

                # Get chatbot message info for this turn
                c_analysis = turn.get('chatbot_message_analysis')
                if c_analysis:
                    row['chatbot_tactics'] = ', '.join(c_analysis.get('leadership_tactics', []))
                    row['chatbot_tone'] = c_analysis.get('primary_tone', '')
                    row['chatbot_response_type'] = c_analysis.get('response_to_participant', '')
                    row['chatbot_guidance_style'] = c_analysis.get('task_guidance_style', '')
                    row['chatbot_msg_length'] = c_analysis.get('message_length', 0)
                else:
                    row['chatbot_tactics'] = None
                    row['chatbot_tone'] = None
                    row['chatbot_response_type'] = None
                    row['chatbot_guidance_style'] = None
                    row['chatbot_msg_length'] = None

                rows.append(row)

        self.interactions_df = pd.DataFrame(rows)

        if len(self.interactions_df) > 0:
            self.interactions_df['turn_position'] = self.interactions_df.groupby('conversation_id')['turn_number'].transform(
                lambda x: x / x.max() if x.max() > 0 else 0
            )

        return self.interactions_df

    def create_conversation_dataframe(self) -> pd.DataFrame:
        """Create dataframe with one row per conversation (extended)"""
        rows = []

        for conv in self.conversations:
            # Handle both schemas
            p_summary = conv.get('participant_summary', conv.get('conversation_summary', {}))
            c_summary = conv.get('chatbot_summary', {})

            participant_msgs = conv.get('participant_message_analyses', conv.get('message_analyses', []))
            chatbot_msgs = conv.get('chatbot_message_analyses', [])

            # Participant strategies
            strategies = [msg['primary_strategy'] for msg in participant_msgs]
            strategy_counts = Counter(strategies)

            # Participant emotions
            all_emotions = []
            for msg in participant_msgs:
                all_emotions.extend(msg['emotional_tone'])
            emotion_counts = Counter(all_emotions)

            # Chatbot tactics
            all_tactics = []
            for msg in chatbot_msgs:
                all_tactics.extend(msg.get('leadership_tactics', []))
            tactic_counts = Counter(all_tactics)

            # Chatbot tones
            all_tones = [msg.get('primary_tone', 'NEUTRAL') for msg in chatbot_msgs]
            tone_counts = Counter(all_tones)

            # Chatbot responses
            all_responses = [msg.get('response_to_participant', 'NEUTRAL') for msg in chatbot_msgs]
            response_counts = Counter(all_responses)

            # Strategy diversity
            if len(strategies) > 0:
                strategy_probs = np.array(list(strategy_counts.values())) / len(strategies)
                strategy_entropy = -np.sum(strategy_probs * np.log2(strategy_probs + 1e-10))
            else:
                strategy_entropy = 0

            # Get conversation duration from timestamps
            conv_id = conv['conversation_id']
            timestamp_data = self.conversation_timestamps.get(conv_id, {})

            row = {
                'conversation_id': conv_id,
                'condition': p_summary.get('condition_inference', 'unknown'),
                'total_participant_messages': p_summary.get('total_participant_messages', len(participant_msgs)),
                'total_chatbot_messages': p_summary.get('total_chatbot_messages', len(chatbot_msgs)),
                'avg_participant_msg_length': p_summary.get('average_participant_message_length',
                                                           p_summary.get('average_message_length', 0)),
                'avg_chatbot_msg_length': p_summary.get('average_chatbot_message_length', 0),
                'duration_minutes': timestamp_data.get('duration_minutes', np.nan),
                'duration_seconds': timestamp_data.get('duration_seconds', np.nan),
                'strategy_diversity': strategy_entropy,
                'num_turning_points': len(p_summary.get('key_turning_points', [])),
                'overall_trajectory': p_summary.get('overall_trajectory', ''),
                'emotional_arc': p_summary.get('emotional_arc', ''),
                'participant_effectiveness': p_summary.get('participant_effectiveness', ''),
                'interaction_quality': p_summary.get('interaction_quality', ''),
            }

            # Chatbot summary fields
            if c_summary:
                row['chatbot_dominant_tactics'] = ', '.join(c_summary.get('dominant_tactics', []))
                row['chatbot_tone_profile'] = c_summary.get('tone_profile', '')
                row['chatbot_response_pattern'] = c_summary.get('response_pattern', '')
                row['chatbot_escalation_trajectory'] = c_summary.get('escalation_trajectory', '')

            # Participant strategy proportions
            for cat in ['COMPLIANCE', 'RESISTANCE', 'JAILBREAKING', 'NEGOTIATION', 'HELP_SEEKING', 'COPING']:
                count = sum(1 for s in strategies if s.startswith(cat))
                row[f'prop_{cat.lower()}'] = count / len(strategies) if len(strategies) > 0 else 0

            # Participant emotion proportions
            total_emotions = len(all_emotions)
            for emotion in ['FRUSTRATED', 'ANXIOUS', 'CONFIDENT', 'NEUTRAL', 'RESIGNED', 'DEFENSIVE']:
                row[f'prop_emotion_{emotion.lower()}'] = emotion_counts.get(emotion, 0) / max(total_emotions, 1)

            # Participant engagement proportions
            engagements = [msg['task_engagement'] for msg in participant_msgs]
            for level in ['HIGH', 'MEDIUM', 'LOW']:
                row[f'prop_engagement_{level.lower()}'] = engagements.count(level) / len(engagements) if len(engagements) > 0 else 0

            # Chatbot tactic type proportions
            total_tactics = len(all_tactics)
            dark_triad_tactics = ['INTIMIDATION', 'CRITICISM_HARSH', 'MANIPULATION',
                                 'DISMISSIVENESS', 'MICROMANAGEMENT', 'BLAME_SHIFTING', 'GRANDIOSITY']
            servant_tactics = ['ENCOURAGEMENT', 'EMPOWERMENT', 'CONSTRUCTIVE_FEEDBACK',
                              'ACTIVE_LISTENING', 'COLLABORATION', 'SUPPORT_OFFERING', 'RECOGNITION']

            dt_count = sum(tactic_counts.get(t, 0) for t in dark_triad_tactics)
            sl_count = sum(tactic_counts.get(t, 0) for t in servant_tactics)

            row['prop_dark_triad_tactics'] = dt_count / max(total_tactics, 1)
            row['prop_servant_tactics'] = sl_count / max(total_tactics, 1)

            # Chatbot tone proportions
            total_tone_msgs = len(all_tones)
            for tone in ['HOSTILE', 'CONDESCENDING', 'COLD', 'NEUTRAL', 'WARM', 'SUPPORTIVE', 'SARCASTIC']:
                row[f'prop_tone_{tone.lower()}'] = tone_counts.get(tone, 0) / max(total_tone_msgs, 1)

            # Chatbot response type proportions
            total_responses = len(all_responses)
            for resp in ['VALIDATES', 'REJECTS', 'PARTIALLY_ACCEPTS', 'IGNORES', 'ESCALATES', 'DE_ESCALATES', 'REDIRECTS']:
                row[f'prop_response_{resp.lower()}'] = response_counts.get(resp, 0) / max(total_responses, 1)

            rows.append(row)

        self.conversations_df = pd.DataFrame(rows)
        return self.conversations_df


# ============================================================================
# STATISTICAL ANALYZER CLASS (Extended)
# ============================================================================

class ConversationStatistics:
    """Compute statistical summaries including chatbot and interaction patterns"""

    def __init__(self, messages_df: pd.DataFrame, chatbot_messages_df: pd.DataFrame,
                 conversations_df: pd.DataFrame, interactions_df: pd.DataFrame):
        self.messages_df = messages_df
        self.chatbot_messages_df = chatbot_messages_df
        self.conversations_df = conversations_df
        self.interactions_df = interactions_df

    def overall_summary(self) -> pd.DataFrame:
        """Generate overall summary statistics including per-group breakdowns with statistical tests"""
        # Get condition subsets
        dt_df = self.conversations_df[self.conversations_df['condition'] == 'dark_triad']
        sl_df = self.conversations_df[self.conversations_df['condition'] == 'servant_leader']
        dt_msgs = self.messages_df[self.messages_df['condition'] == 'dark_triad'] if len(self.messages_df) > 0 else pd.DataFrame()
        sl_msgs = self.messages_df[self.messages_df['condition'] == 'servant_leader'] if len(self.messages_df) > 0 else pd.DataFrame()
        dt_chatbot_msgs = self.chatbot_messages_df[self.chatbot_messages_df['condition'] == 'dark_triad'] if len(self.chatbot_messages_df) > 0 else pd.DataFrame()
        sl_chatbot_msgs = self.chatbot_messages_df[self.chatbot_messages_df['condition'] == 'servant_leader'] if len(self.chatbot_messages_df) > 0 else pd.DataFrame()

        # Helper function to compute Cohen's d
        def cohens_d(group1, group2):
            n1, n2 = len(group1), len(group2)
            if n1 < 2 or n2 < 2:
                return np.nan
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
            if pooled_std == 0:
                return np.nan
            return (np.mean(group1) - np.mean(group2)) / pooled_std

        # Helper function to compute rank-biserial correlation for Mann-Whitney U
        def rank_biserial(u_statistic, n1, n2):
            return 1 - (2*u_statistic) / (n1 * n2)

        # Helper function to perform appropriate statistical test
        def statistical_test(group1, group2, var_name):
            """Perform t-test or Mann-Whitney U based on normality"""
            if len(group1) < 3 or len(group2) < 3:
                return {'test': 'N/A', 'statistic': np.nan, 'p_value': np.nan, 'effect_size': np.nan}

            # Check normality with Shapiro-Wilk test (if sample size allows)
            normal1 = stats.shapiro(group1)[1] > 0.05 if len(group1) >= 3 else False
            normal2 = stats.shapiro(group2)[1] > 0.05 if len(group2) >= 3 else False

            if normal1 and normal2:
                # Use independent samples t-test
                statistic, p_value = stats.ttest_ind(group1, group2)
                effect_size = cohens_d(group1, group2)
                test_name = "t-test"
            else:
                # Use Mann-Whitney U test
                statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                effect_size = rank_biserial(statistic, len(group1), len(group2))
                test_name = "Mann-Whitney U"

            return {
                'test': test_name,
                'statistic': statistic,
                'p_value': p_value,
                'effect_size': effect_size
            }

        summary = {
            # Overall counts
            'Total Conversations': len(self.conversations_df),
            'Dark Triad Conversations': len(dt_df),
            'Servant Leader Conversations': len(sl_df),
            'Total Participant Messages': len(self.messages_df),
            'Total Chatbot Messages': len(self.chatbot_messages_df),
            'Total Interaction Turns': len(self.interactions_df),

            # Per-group participant messages
            'Participant Msgs (Dark Triad)': len(dt_msgs),
            'Participant Msgs (Servant Leader)': len(sl_msgs),
            'Avg Participant Msgs/Conv (Overall)': self.conversations_df['total_participant_messages'].mean(),
            'SD Participant Msgs/Conv (Overall)': self.conversations_df['total_participant_messages'].std(),
            'Avg Participant Msgs/Conv (Dark Triad)': dt_df['total_participant_messages'].mean() if len(dt_df) > 0 else 0,
            'SD Participant Msgs/Conv (Dark Triad)': dt_df['total_participant_messages'].std() if len(dt_df) > 0 else 0,
            'Avg Participant Msgs/Conv (Servant Leader)': sl_df['total_participant_messages'].mean() if len(sl_df) > 0 else 0,
            'SD Participant Msgs/Conv (Servant Leader)': sl_df['total_participant_messages'].std() if len(sl_df) > 0 else 0,

            # Per-group chatbot messages
            'Chatbot Msgs (Dark Triad)': len(dt_chatbot_msgs),
            'Chatbot Msgs (Servant Leader)': len(sl_chatbot_msgs),
            'Avg Chatbot Msgs/Conv (Overall)': self.conversations_df['total_chatbot_messages'].mean(),
            'SD Chatbot Msgs/Conv (Overall)': self.conversations_df['total_chatbot_messages'].std(),
            'Avg Chatbot Msgs/Conv (Dark Triad)': dt_df['total_chatbot_messages'].mean() if len(dt_df) > 0 else 0,
            'SD Chatbot Msgs/Conv (Dark Triad)': dt_df['total_chatbot_messages'].std() if len(dt_df) > 0 else 0,
            'Avg Chatbot Msgs/Conv (Servant Leader)': sl_df['total_chatbot_messages'].mean() if len(sl_df) > 0 else 0,
            'SD Chatbot Msgs/Conv (Servant Leader)': sl_df['total_chatbot_messages'].std() if len(sl_df) > 0 else 0,

            # Message length statistics
            'Avg Participant Msg Length (Overall)': self.messages_df['message_length'].mean() if len(self.messages_df) > 0 else 0,
            'SD Participant Msg Length (Overall)': self.messages_df['message_length'].std() if len(self.messages_df) > 0 else 0,
            'Avg Participant Msg Length (Dark Triad)': dt_msgs['message_length'].mean() if len(dt_msgs) > 0 else 0,
            'SD Participant Msg Length (Dark Triad)': dt_msgs['message_length'].std() if len(dt_msgs) > 0 else 0,
            'Avg Participant Msg Length (Servant Leader)': sl_msgs['message_length'].mean() if len(sl_msgs) > 0 else 0,
            'SD Participant Msg Length (Servant Leader)': sl_msgs['message_length'].std() if len(sl_msgs) > 0 else 0,
            'Avg Chatbot Msg Length (Overall)': self.chatbot_messages_df['message_length'].mean() if len(self.chatbot_messages_df) > 0 else 0,
            'SD Chatbot Msg Length (Overall)': self.chatbot_messages_df['message_length'].std() if len(self.chatbot_messages_df) > 0 else 0,
            'Avg Chatbot Msg Length (Dark Triad)': dt_chatbot_msgs['message_length'].mean() if len(dt_chatbot_msgs) > 0 else 0,
            'SD Chatbot Msg Length (Dark Triad)': dt_chatbot_msgs['message_length'].std() if len(dt_chatbot_msgs) > 0 else 0,
            'Avg Chatbot Msg Length (Servant Leader)': sl_chatbot_msgs['message_length'].mean() if len(sl_chatbot_msgs) > 0 else 0,
            'SD Chatbot Msg Length (Servant Leader)': sl_chatbot_msgs['message_length'].std() if len(sl_chatbot_msgs) > 0 else 0,
        }

        # Statistical tests for message counts
        if len(dt_df) > 0 and len(sl_df) > 0:
            # Participant messages per conversation
            test_result = statistical_test(
                dt_df['total_participant_messages'].values,
                sl_df['total_participant_messages'].values,
                'Participant Msgs/Conv'
            )
            summary['[TEST] Participant Msgs/Conv - Test'] = test_result['test']
            summary['[TEST] Participant Msgs/Conv - Statistic'] = test_result['statistic']
            summary['[TEST] Participant Msgs/Conv - p-value'] = test_result['p_value']
            summary['[TEST] Participant Msgs/Conv - Effect Size'] = test_result['effect_size']

            # Chatbot messages per conversation
            test_result = statistical_test(
                dt_df['total_chatbot_messages'].values,
                sl_df['total_chatbot_messages'].values,
                'Chatbot Msgs/Conv'
            )
            summary['[TEST] Chatbot Msgs/Conv - Test'] = test_result['test']
            summary['[TEST] Chatbot Msgs/Conv - Statistic'] = test_result['statistic']
            summary['[TEST] Chatbot Msgs/Conv - p-value'] = test_result['p_value']
            summary['[TEST] Chatbot Msgs/Conv - Effect Size'] = test_result['effect_size']

        # Statistical tests for message lengths
        if len(dt_msgs) > 0 and len(sl_msgs) > 0:
            # Participant message length
            test_result = statistical_test(
                dt_msgs['message_length'].values,
                sl_msgs['message_length'].values,
                'Participant Msg Length'
            )
            summary['[TEST] Participant Msg Length - Test'] = test_result['test']
            summary['[TEST] Participant Msg Length - Statistic'] = test_result['statistic']
            summary['[TEST] Participant Msg Length - p-value'] = test_result['p_value']
            summary['[TEST] Participant Msg Length - Effect Size'] = test_result['effect_size']

        if len(dt_chatbot_msgs) > 0 and len(sl_chatbot_msgs) > 0:
            # Chatbot message length
            test_result = statistical_test(
                dt_chatbot_msgs['message_length'].values,
                sl_chatbot_msgs['message_length'].values,
                'Chatbot Msg Length'
            )
            summary['[TEST] Chatbot Msg Length - Test'] = test_result['test']
            summary['[TEST] Chatbot Msg Length - Statistic'] = test_result['statistic']
            summary['[TEST] Chatbot Msg Length - p-value'] = test_result['p_value']
            summary['[TEST] Chatbot Msg Length - Effect Size'] = test_result['effect_size']

        # Duration statistics (if available)
        if 'duration_minutes' in self.conversations_df.columns:
            valid_durations = self.conversations_df['duration_minutes'].dropna()
            dt_durations = dt_df['duration_minutes'].dropna() if len(dt_df) > 0 else pd.Series(dtype=float)
            sl_durations = sl_df['duration_minutes'].dropna() if len(sl_df) > 0 else pd.Series(dtype=float)

            # Filter out outliers (conversations > 120 minutes are likely data issues)
            MAX_REASONABLE_DURATION = 120  # minutes
            valid_durations_filtered = valid_durations[valid_durations <= MAX_REASONABLE_DURATION]
            dt_durations_filtered = dt_durations[dt_durations <= MAX_REASONABLE_DURATION] if len(dt_durations) > 0 else pd.Series(dtype=float)
            sl_durations_filtered = sl_durations[sl_durations <= MAX_REASONABLE_DURATION] if len(sl_durations) > 0 else pd.Series(dtype=float)

            outlier_count = len(valid_durations) - len(valid_durations_filtered)

            summary.update({
                # Raw duration statistics (all data)
                'Conversations with Duration Data': len(valid_durations),
                'Duration Outliers (>120 min)': outlier_count,

                # Filtered duration statistics (excluding outliers)
                'Total Duration (minutes, filtered)': valid_durations_filtered.sum(),
                'Avg Duration/Conv (minutes, Overall)': valid_durations_filtered.mean() if len(valid_durations_filtered) > 0 else np.nan,
                'Median Duration/Conv (minutes, Overall)': valid_durations_filtered.median() if len(valid_durations_filtered) > 0 else np.nan,
                'Std Duration/Conv (minutes, Overall)': valid_durations_filtered.std() if len(valid_durations_filtered) > 0 else np.nan,
                'Min Duration (minutes)': valid_durations_filtered.min() if len(valid_durations_filtered) > 0 else np.nan,
                'Max Duration (minutes)': valid_durations_filtered.max() if len(valid_durations_filtered) > 0 else np.nan,

                # Per-group duration (filtered)
                'Avg Duration/Conv (Dark Triad)': dt_durations_filtered.mean() if len(dt_durations_filtered) > 0 else np.nan,
                'Median Duration/Conv (Dark Triad)': dt_durations_filtered.median() if len(dt_durations_filtered) > 0 else np.nan,
                'Std Duration/Conv (Dark Triad)': dt_durations_filtered.std() if len(dt_durations_filtered) > 0 else np.nan,
                'Avg Duration/Conv (Servant Leader)': sl_durations_filtered.mean() if len(sl_durations_filtered) > 0 else np.nan,
                'Median Duration/Conv (Servant Leader)': sl_durations_filtered.median() if len(sl_durations_filtered) > 0 else np.nan,
                'Std Duration/Conv (Servant Leader)': sl_durations_filtered.std() if len(sl_durations_filtered) > 0 else np.nan,
            })

            # Statistical test for duration
            if len(dt_durations_filtered) > 0 and len(sl_durations_filtered) > 0:
                test_result = statistical_test(
                    dt_durations_filtered.values,
                    sl_durations_filtered.values,
                    'Duration/Conv'
                )
                summary['[TEST] Duration/Conv - Test'] = test_result['test']
                summary['[TEST] Duration/Conv - Statistic'] = test_result['statistic']
                summary['[TEST] Duration/Conv - p-value'] = test_result['p_value']
                summary['[TEST] Duration/Conv - Effect Size'] = test_result['effect_size']

        return pd.DataFrame([summary]).T.rename(columns={0: 'Value'})

    def strategy_distribution(self, by_condition: bool = False) -> pd.DataFrame:
        """Get participant strategy distribution"""
        if len(self.messages_df) == 0:
            return pd.DataFrame()

        if by_condition:
            dist = self.messages_df.groupby(['condition', 'strategy_category']).size().unstack(fill_value=0)
            return dist.div(dist.sum(axis=1), axis=0) * 100
        else:
            dist = self.messages_df['strategy_category'].value_counts()
            return (dist / dist.sum() * 100).round(2)

    def chatbot_tactic_distribution(self, by_condition: bool = False) -> pd.DataFrame:
        """Get chatbot leadership tactic distribution"""
        if len(self.chatbot_messages_df) == 0:
            return pd.DataFrame()

        tactic_cols = [col for col in self.chatbot_messages_df.columns if col.startswith('tactic_')]

        if by_condition:
            tactic_dist = self.chatbot_messages_df.groupby('condition')[tactic_cols].sum()
            tactic_dist.columns = [col.replace('tactic_', '').upper() for col in tactic_dist.columns]
            return tactic_dist
        else:
            tactic_dist = self.chatbot_messages_df[tactic_cols].sum()
            tactic_dist.index = [col.replace('tactic_', '').upper() for col in tactic_dist.index]
            return tactic_dist.sort_values(ascending=False)

    def chatbot_tone_distribution(self, by_condition: bool = False) -> pd.DataFrame:
        """Get chatbot tone distribution"""
        if len(self.chatbot_messages_df) == 0:
            return pd.DataFrame()

        if by_condition:
            dist = self.chatbot_messages_df.groupby(['condition', 'primary_tone']).size().unstack(fill_value=0)
            return dist.div(dist.sum(axis=1), axis=0) * 100
        else:
            dist = self.chatbot_messages_df['primary_tone'].value_counts()
            return (dist / dist.sum() * 100).round(2)

    def chatbot_response_distribution(self, by_condition: bool = False) -> pd.DataFrame:
        """Get chatbot response type distribution"""
        if len(self.chatbot_messages_df) == 0:
            return pd.DataFrame()

        if by_condition:
            dist = self.chatbot_messages_df.groupby(['condition', 'response_to_participant']).size().unstack(fill_value=0)
            return dist.div(dist.sum(axis=1), axis=0) * 100
        else:
            dist = self.chatbot_messages_df['response_to_participant'].value_counts()
            return (dist / dist.sum() * 100).round(2)

    def emotion_distribution(self, by_condition: bool = False) -> pd.DataFrame:
        """Get participant emotion distribution"""
        if len(self.messages_df) == 0:
            return pd.DataFrame()

        emotion_cols = [col for col in self.messages_df.columns if col.startswith('emotion_')]

        if by_condition:
            dist = self.messages_df.groupby('condition')[emotion_cols].sum()
            dist.columns = [col.replace('emotion_', '').upper() for col in dist.columns]
            return dist
        else:
            dist = self.messages_df[emotion_cols].sum()
            dist.index = [col.replace('emotion_', '').upper() for col in dist.index]
            return dist.sort_values(ascending=False)

    def engagement_distribution(self, by_condition: bool = False) -> pd.DataFrame:
        """Get participant engagement distribution"""
        if len(self.messages_df) == 0:
            return pd.DataFrame()

        if by_condition:
            dist = self.messages_df.groupby(['condition', 'task_engagement']).size().unstack(fill_value=0)
            return dist.div(dist.sum(axis=1), axis=0) * 100
        else:
            dist = self.messages_df['task_engagement'].value_counts()
            return (dist / dist.sum() * 100).round(2)

    def interaction_patterns(self) -> Dict[str, pd.DataFrame]:
        """Analyze interaction patterns between participant and chatbot"""
        if len(self.interactions_df) == 0:
            return {}

        patterns = {}

        # Strategy response to chatbot tone
        if 'participant_strategy' in self.interactions_df.columns and 'chatbot_tone' in self.interactions_df.columns:
            valid_data = self.interactions_df.dropna(subset=['participant_strategy', 'chatbot_tone'])
            if len(valid_data) > 0:
                patterns['strategy_by_tone'] = pd.crosstab(
                    valid_data['chatbot_tone'],
                    valid_data['participant_strategy'].str.split('_').str[0],
                    normalize='index'
                ) * 100

        # Engagement response to chatbot response type
        if 'participant_engagement' in self.interactions_df.columns and 'chatbot_response_type' in self.interactions_df.columns:
            valid_data = self.interactions_df.dropna(subset=['participant_engagement', 'chatbot_response_type'])
            if len(valid_data) > 0:
                patterns['engagement_by_response'] = pd.crosstab(
                    valid_data['chatbot_response_type'],
                    valid_data['participant_engagement'],
                    normalize='index'
                ) * 100

        return patterns

    def paper_section_tests(self) -> str:
        """
        Produce all statistical tests requested by [NOTE] tags in the
        Conversation Analysis section of the paper.
        Returns a formatted string report.
        """
        from scipy import stats
        import numpy as np
        lines = []

        def _heading(title):
            lines.append("\n" + "=" * 80)
            lines.append(title)
            lines.append("=" * 80)

        def _cohens_d(g1, g2):
            n1, n2 = len(g1), len(g2)
            if n1 < 2 or n2 < 2:
                return np.nan
            v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
            pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
            return (np.mean(g1) - np.mean(g2)) / pooled if pooled else np.nan

        def _rank_biserial(u, n1, n2):
            return 1 - (2 * u) / (n1 * n2)

        def _cramers_v(chi2, n, min_dim):
            return np.sqrt(chi2 / (n * (min_dim - 1))) if n and min_dim > 1 else np.nan

        dt_msgs = self.messages_df[self.messages_df['condition'] == 'dark_triad']
        sl_msgs = self.messages_df[self.messages_df['condition'] == 'servant_leader']
        dt_conv = self.conversations_df[self.conversations_df['condition'] == 'dark_triad']
        sl_conv = self.conversations_df[self.conversations_df['condition'] == 'servant_leader']
        dt_cb = self.chatbot_messages_df[self.chatbot_messages_df['condition'] == 'dark_triad']
        sl_cb = self.chatbot_messages_df[self.chatbot_messages_df['condition'] == 'servant_leader']

        # ------------------------------------------------------------------
        # NOTE 1 — already in overall_summary(); replicate here for the report
        # ------------------------------------------------------------------
        _heading("NOTE 1: Message counts, lengths, duration (conversation-level)")

        for col, label in [
            ('total_participant_messages', 'Participant messages/conv'),
            ('total_chatbot_messages', 'Chatbot messages/conv'),
            ('avg_participant_msg_length', 'Avg participant msg length (conv-level)'),
            ('avg_chatbot_msg_length', 'Avg chatbot msg length (conv-level)'),
            ('duration_minutes', 'Duration (minutes)'),
        ]:
            if col not in self.conversations_df.columns:
                continue
            g1 = dt_conv[col].dropna().values
            g2 = sl_conv[col].dropna().values
            if len(g1) < 3 or len(g2) < 3:
                continue

            u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            t_stat, t_p = stats.ttest_ind(g1, g2)
            d = _cohens_d(g1, g2)
            r = _rank_biserial(u_stat, len(g1), len(g2))

            lines.append(f"\n  {label}")
            lines.append(f"    DT: M={np.mean(g1):.2f}, SD={np.std(g1,ddof=1):.2f}  "
                         f"SL: M={np.mean(g2):.2f}, SD={np.std(g2,ddof=1):.2f}")
            lines.append(f"    Mann-Whitney U={u_stat:.1f}, p={u_p:.4f}, rank-biserial r={r:.3f}")
            lines.append(f"    t-test t={t_stat:.3f}, p={t_p:.4f}, Cohen's d={d:.3f}")

        # Also message-level lengths (as in paper Table X)
        lines.append("\n  --- Message-level length tests ---")
        for label, g1, g2 in [
            ('Participant msg length (message-level)',
             dt_msgs['message_length'].values, sl_msgs['message_length'].values),
            ('Chatbot msg length (message-level)',
             dt_cb['message_length'].values, sl_cb['message_length'].values),
        ]:
            u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            d = _cohens_d(g1, g2)
            r = _rank_biserial(u_stat, len(g1), len(g2))
            lines.append(f"\n  {label}")
            lines.append(f"    DT: M={np.mean(g1):.2f}, SD={np.std(g1,ddof=1):.2f}  "
                         f"SL: M={np.mean(g2):.2f}, SD={np.std(g2,ddof=1):.2f}")
            lines.append(f"    U={u_stat:.1f}, p={u_p:.6f}, r={r:.3f}, d={d:.3f}")

        # ------------------------------------------------------------------
        # NOTE 2 — Chi-square / Mann-Whitney for strategy proportions
        # ------------------------------------------------------------------
        _heading("NOTE 2: Strategy proportion comparisons between conditions")

        # 2a. Chi-square on raw strategy-category counts (message-level)
        ct = pd.crosstab(self.messages_df['condition'], self.messages_df['strategy_category'])
        chi2, chi_p, dof, _ = stats.chi2_contingency(ct)
        n_total = ct.values.sum()
        min_dim = min(ct.shape)
        v = _cramers_v(chi2, n_total, min_dim)
        lines.append(f"\n  Overall chi-square (strategy_category × condition):")
        lines.append(f"    χ²({dof})={chi2:.2f}, p={chi_p:.6f}, Cramér's V={v:.3f}")
        lines.append(f"\n  Contingency table (counts):\n{ct.to_string()}")

        # 2b. Per-category Mann-Whitney on conversation-level proportions
        strategy_cats = ['compliance', 'resistance', 'jailbreaking',
                         'negotiation', 'help_seeking', 'coping']
        lines.append("\n  Per-category Mann-Whitney U on conversation-level proportions:")
        for cat in strategy_cats:
            col = f'prop_{cat}'
            if col not in self.conversations_df.columns:
                continue
            g1 = dt_conv[col].values
            g2 = sl_conv[col].values
            u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            d = _cohens_d(g1, g2)
            r = _rank_biserial(u_stat, len(g1), len(g2))
            lines.append(f"    {cat.upper():20s}  DT M={np.mean(g1):.3f}  SL M={np.mean(g2):.3f}  "
                         f"U={u_stat:.1f}  p={u_p:.4f}  r={r:.3f}  d={d:.3f}")

        # ------------------------------------------------------------------
        # NOTE 3 — Jailbreaking subtypes by condition
        # ------------------------------------------------------------------
        _heading("NOTE 3: Jailbreaking subtypes by condition")

        jb = self.messages_df[self.messages_df['strategy_category'] == 'JAILBREAKING']
        if len(jb) > 0:
            ct_jb = pd.crosstab(jb['condition'], jb['primary_strategy'])
            lines.append(f"\n  Jailbreaking subtype counts:\n{ct_jb.to_string()}")

            # Proportions within each condition's total messages
            for cond in ['dark_triad', 'servant_leader']:
                cond_total = len(self.messages_df[self.messages_df['condition'] == cond])
                cond_jb = jb[jb['condition'] == cond]
                lines.append(f"\n  {cond.replace('_',' ').title()} (N messages = {cond_total}):")
                for subtype, count in cond_jb['primary_strategy'].value_counts().items():
                    lines.append(f"    {subtype}: {count} ({count/cond_total*100:.1f}%)")

            # Chi-square if table is large enough
            if ct_jb.shape[0] > 1 and ct_jb.shape[1] > 1 and ct_jb.values.min() >= 0:
                try:
                    chi2, chi_p, dof, _ = stats.chi2_contingency(ct_jb)
                    v = _cramers_v(chi2, ct_jb.values.sum(), min(ct_jb.shape))
                    lines.append(f"\n  Chi-square (subtype × condition): χ²({dof})={chi2:.2f}, "
                                 f"p={chi_p:.4f}, V={v:.3f}")
                except ValueError:
                    lines.append("\n  Chi-square not computable (expected frequencies too low)")
        else:
            lines.append("  No jailbreaking attempts found.")

        # ------------------------------------------------------------------
        # NOTE 4 — Strategy diversity & turning points tests
        # ------------------------------------------------------------------
        _heading("NOTE 4: Strategy diversity and turning points")

        for col, label in [
            ('strategy_diversity', 'Strategy diversity (Shannon entropy)'),
            ('num_turning_points', 'Number of turning points'),
        ]:
            if col not in self.conversations_df.columns:
                continue
            g1 = dt_conv[col].dropna().values
            g2 = sl_conv[col].dropna().values
            u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            t_stat, t_p = stats.ttest_ind(g1, g2)
            d = _cohens_d(g1, g2)
            r = _rank_biserial(u_stat, len(g1), len(g2))
            lines.append(f"\n  {label}")
            lines.append(f"    DT: M={np.mean(g1):.3f}, SD={np.std(g1,ddof=1):.3f}  "
                         f"SL: M={np.mean(g2):.3f}, SD={np.std(g2,ddof=1):.3f}")
            lines.append(f"    Mann-Whitney U={u_stat:.1f}, p={u_p:.4f}, r={r:.3f}")
            lines.append(f"    t-test t={t_stat:.3f}, p={t_p:.4f}, d={d:.3f}")

        # ------------------------------------------------------------------
        # NOTE 5 — Chi-square for emotional tone distribution
        # ------------------------------------------------------------------
        _heading("NOTE 5: Emotional tone distribution chi-square")

        emotion_cols = [c for c in self.messages_df.columns if c.startswith('emotion_')]
        # Build a long-form emotion × condition table
        # Each message can have multiple emotions, so we count occurrences
        emotion_records = []
        for _, row in self.messages_df.iterrows():
            for ec in emotion_cols:
                if row[ec]:
                    emotion_records.append({
                        'condition': row['condition'],
                        'emotion': ec.replace('emotion_', '').upper()
                    })
        if emotion_records:
            emo_df = pd.DataFrame(emotion_records)
            ct_emo = pd.crosstab(emo_df['condition'], emo_df['emotion'])
            chi2, chi_p, dof, _ = stats.chi2_contingency(ct_emo)
            v = _cramers_v(chi2, ct_emo.values.sum(), min(ct_emo.shape))
            lines.append(f"\n  Overall chi-square (emotion × condition):")
            lines.append(f"    χ²({dof})={chi2:.2f}, p={chi_p:.6f}, Cramér's V={v:.3f}")
            lines.append(f"\n  Contingency table:\n{ct_emo.to_string()}")

            # Per-emotion proportion tests (conversation-level)
            lines.append("\n  Per-emotion Mann-Whitney U (conversation-level proportions):")
            for ec in emotion_cols:
                emo_name = ec.replace('emotion_', '')
                prop_col = f'prop_emotion_{emo_name}'
                if prop_col not in self.conversations_df.columns:
                    continue
                g1 = dt_conv[prop_col].values
                g2 = sl_conv[prop_col].values
                u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                d = _cohens_d(g1, g2)
                r = _rank_biserial(u_stat, len(g1), len(g2))
                lines.append(f"    {emo_name.upper():12s}  DT M={np.mean(g1):.3f}  SL M={np.mean(g2):.3f}  "
                             f"U={u_stat:.1f}  p={u_p:.4f}  r={r:.3f}  d={d:.3f}")

        # ------------------------------------------------------------------
        # NOTE 6 — Chi-square for engagement distribution
        # ------------------------------------------------------------------
        _heading("NOTE 6: Engagement distribution chi-square")

        ct_eng = pd.crosstab(self.messages_df['condition'], self.messages_df['task_engagement'])
        chi2, chi_p, dof, _ = stats.chi2_contingency(ct_eng)
        v = _cramers_v(chi2, ct_eng.values.sum(), min(ct_eng.shape))
        lines.append(f"\n  Chi-square (engagement × condition):")
        lines.append(f"    χ²({dof})={chi2:.2f}, p={chi_p:.6f}, Cramér's V={v:.3f}")
        lines.append(f"\n  Contingency table:\n{ct_eng.to_string()}")
        lines.append(f"\n  Proportions:\n{(ct_eng.div(ct_eng.sum(axis=1), axis=0)*100).round(1).to_string()}")

        # Per-level Mann-Whitney (conversation-level)
        lines.append("\n  Per-level Mann-Whitney U (conversation-level proportions):")
        for level in ['high', 'medium', 'low']:
            col = f'prop_engagement_{level}'
            if col not in self.conversations_df.columns:
                continue
            g1 = dt_conv[col].values
            g2 = sl_conv[col].values
            u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            d = _cohens_d(g1, g2)
            r = _rank_biserial(u_stat, len(g1), len(g2))
            lines.append(f"    {level.upper():8s}  DT M={np.mean(g1):.3f}  SL M={np.mean(g2):.3f}  "
                         f"U={u_stat:.1f}  p={u_p:.4f}  r={r:.3f}  d={d:.3f}")

        # ------------------------------------------------------------------
        # Temporal dynamics (phase-level)
        # ------------------------------------------------------------------
        _heading("NOTE 7: Temporal dynamics (Early / Middle / Late)")

        self.messages_df['conversation_phase'] = pd.cut(
            self.messages_df['message_position'],
            bins=[-0.001, 0.33, 0.67, 1.001],
            labels=['Early', 'Middle', 'Late']
        )

        for cond_label, cond_key in [('Dark Triad', 'dark_triad'), ('Servant Leader', 'servant_leader')]:
            cond_data = self.messages_df[self.messages_df['condition'] == cond_key]
            lines.append(f"\n  --- {cond_label} ---")

            # Strategy by phase
            if len(cond_data) > 0:
                ct_phase_strat = pd.crosstab(cond_data['conversation_phase'],
                                             cond_data['strategy_category'],
                                             normalize='index') * 100
                lines.append(f"\n  Strategy % by phase:\n{ct_phase_strat.round(1).to_string()}")

            # Emotion by phase
            if len(cond_data) > 0:
                phase_emo = cond_data.groupby('conversation_phase')[emotion_cols].mean() * 100
                phase_emo.columns = [c.replace('emotion_', '').upper() for c in phase_emo.columns]
                lines.append(f"\n  Emotion % by phase:\n{phase_emo.round(1).to_string()}")

            # Engagement by phase
            if len(cond_data) > 0:
                ct_phase_eng = pd.crosstab(cond_data['conversation_phase'],
                                           cond_data['task_engagement'],
                                           normalize='index') * 100
                lines.append(f"\n  Engagement % by phase:\n{ct_phase_eng.round(1).to_string()}")

            # Message length by phase
            if len(cond_data) > 0:
                phase_len = cond_data.groupby('conversation_phase')['message_length'].agg(['mean', 'std', 'count'])
                lines.append(f"\n  Message length by phase:\n{phase_len.round(1).to_string()}")

        # Phase × condition chi-squares for strategy and engagement
        lines.append("\n  --- Phase × Condition interaction tests ---")
        for phase in ['Early', 'Middle', 'Late']:
            phase_data = self.messages_df[self.messages_df['conversation_phase'] == phase]
            if len(phase_data) < 10:
                continue

            # Strategy
            ct_ps = pd.crosstab(phase_data['condition'], phase_data['strategy_category'])
            if ct_ps.shape[0] > 1 and ct_ps.shape[1] > 1:
                chi2, p, dof, _ = stats.chi2_contingency(ct_ps)
                v = _cramers_v(chi2, ct_ps.values.sum(), min(ct_ps.shape))
                lines.append(f"  {phase} phase — Strategy × Condition: χ²({dof})={chi2:.2f}, p={p:.4f}, V={v:.3f}")

            # Engagement
            ct_pe = pd.crosstab(phase_data['condition'], phase_data['task_engagement'])
            if ct_pe.shape[0] > 1 and ct_pe.shape[1] > 1:
                chi2, p, dof, _ = stats.chi2_contingency(ct_pe)
                v = _cramers_v(chi2, ct_pe.values.sum(), min(ct_pe.shape))
                lines.append(f"  {phase} phase — Engagement × Condition: χ²({dof})={chi2:.2f}, p={p:.4f}, V={v:.3f}")

        # ------------------------------------------------------------------
        # Correlation structure
        # ------------------------------------------------------------------
        _heading("NOTE 8: Correlation structure")

        # Strategy intercorrelations (conversation-level proportions)
        strat_cols = [c for c in self.conversations_df.columns
                      if c.startswith('prop_') and not c.startswith('prop_emotion_')
                      and not c.startswith('prop_engagement_') and not c.startswith('prop_tone_')
                      and not c.startswith('prop_response_') and not c.startswith('prop_dark')
                      and not c.startswith('prop_servant')]
        if len(strat_cols) >= 2:
            corr_strat = self.conversations_df[strat_cols].corr()
            corr_strat.columns = [c.replace('prop_', '') for c in corr_strat.columns]
            corr_strat.index = [c.replace('prop_', '') for c in corr_strat.index]
            lines.append(f"\n  Strategy intercorrelations (Pearson r):\n{corr_strat.round(3).to_string()}")

        # Emotion-strategy associations
        emo_prop_cols = [c for c in self.conversations_df.columns if c.startswith('prop_emotion_')]
        if strat_cols and emo_prop_cols:
            cross_corr = self.conversations_df[strat_cols + emo_prop_cols].corr()
            # Show only the strategy-emotion block
            block = cross_corr.loc[strat_cols, emo_prop_cols]
            block.index = [c.replace('prop_', '') for c in block.index]
            block.columns = [c.replace('prop_emotion_', '') for c in block.columns]
            lines.append(f"\n  Strategy-Emotion correlations:\n{block.round(3).to_string()}")

        # Engagement correlates
        eng_prop_cols = [c for c in self.conversations_df.columns if c.startswith('prop_engagement_')]
        other_cols = strat_cols + emo_prop_cols + ['strategy_diversity', 'num_turning_points',
                     'total_participant_messages', 'avg_participant_msg_length']
        other_cols = [c for c in other_cols if c in self.conversations_df.columns]
        if eng_prop_cols and other_cols:
            eng_corr = self.conversations_df[other_cols + eng_prop_cols].corr()
            block = eng_corr.loc[other_cols, eng_prop_cols]
            block.index = [c.replace('prop_', '').replace('prop_emotion_', 'emo_') for c in block.index]
            block.columns = [c.replace('prop_engagement_', '') for c in block.columns]
            lines.append(f"\n  Engagement correlates:\n{block.round(3).to_string()}")

        # ------------------------------------------------------------------
        # MANIPULATION CHECK — Tactic-level and tone-level statistics
        # (used in Appendix manipulation verification)
        # ------------------------------------------------------------------
        _heading("MANIPULATION CHECK: Individual tactic chi-squares & effect sizes")

        dark_triad_tactics = ['INTIMIDATION', 'CRITICISM_HARSH', 'MANIPULATION',
                              'DISMISSIVENESS', 'MICROMANAGEMENT', 'BLAME_SHIFTING', 'GRANDIOSITY']
        servant_tactics = ['ENCOURAGEMENT', 'EMPOWERMENT', 'CONSTRUCTIVE_FEEDBACK',
                           'ACTIVE_LISTENING', 'COLLABORATION', 'SUPPORT_OFFERING', 'RECOGNITION']

        for tactic in dark_triad_tactics + servant_tactics:
            col = f'tactic_{tactic.lower()}'
            if col not in self.chatbot_messages_df.columns:
                continue
            ct_t = pd.crosstab(self.chatbot_messages_df['condition'],
                               self.chatbot_messages_df[col])
            if ct_t.shape == (2, 2):
                chi2, p, _, _ = stats.chi2_contingency(ct_t)
                v = _cramers_v(chi2, ct_t.values.sum(), 2)
                dt_pct = dt_cb[col].mean() * 100
                sl_pct = sl_cb[col].mean() * 100
                lines.append(f"  {tactic:25s}  DT={dt_pct:5.1f}%  SL={sl_pct:5.1f}%  "
                             f"χ²={chi2:.2f}  p={p:.6f}  V={v:.3f}")

        # Tone overall chi-square
        ct_tone = pd.crosstab(self.chatbot_messages_df['condition'],
                              self.chatbot_messages_df['primary_tone'])
        chi2, p, dof, _ = stats.chi2_contingency(ct_tone)
        v = _cramers_v(chi2, ct_tone.values.sum(), min(ct_tone.shape))
        lines.append(f"\n  Tone × Condition: χ²({dof})={chi2:.2f}, p={p:.6f}, V={v:.3f}")

        # Mean tactics per message with Mann-Whitney
        for label, col in [('Dark Triad tactics/msg', 'dark_triad_tactics_count'),
                           ('Servant tactics/msg', 'servant_tactics_count')]:
            g1 = dt_cb[col].values
            g2 = sl_cb[col].values
            u, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            d = _cohens_d(g1, g2)
            lines.append(f"\n  {label}: DT M={np.mean(g1):.2f}  SL M={np.mean(g2):.2f}  "
                         f"U={u:.1f}  p={p:.6f}  d={d:.2f}")

        report = "\n".join(lines)
        return report

    def condition_comparison(self) -> Dict[str, pd.DataFrame]:
        """Compare key metrics between conditions"""
        comparisons = {}

        # Participant strategies
        strategy_cols = [col for col in self.conversations_df.columns
                        if col.startswith('prop_') and 'emotion' not in col
                        and 'engagement' not in col and 'tone' not in col
                        and 'response' not in col and 'tactic' not in col]
        if strategy_cols:
            comparisons['participant_strategies'] = self.conversations_df.groupby('condition')[strategy_cols].mean()

        # Participant emotions
        emotion_cols = [col for col in self.conversations_df.columns if col.startswith('prop_emotion_')]
        if emotion_cols:
            comparisons['participant_emotions'] = self.conversations_df.groupby('condition')[emotion_cols].mean()

        # Chatbot tactics
        tactic_cols = ['prop_dark_triad_tactics', 'prop_servant_tactics']
        available_tactic_cols = [col for col in tactic_cols if col in self.conversations_df.columns]
        if available_tactic_cols:
            comparisons['chatbot_tactics'] = self.conversations_df.groupby('condition')[available_tactic_cols].mean()

        # Chatbot tones
        tone_cols = [col for col in self.conversations_df.columns if col.startswith('prop_tone_')]
        if tone_cols:
            comparisons['chatbot_tones'] = self.conversations_df.groupby('condition')[tone_cols].mean()

        # Chatbot responses
        response_cols = [col for col in self.conversations_df.columns if col.startswith('prop_response_')]
        if response_cols:
            comparisons['chatbot_responses'] = self.conversations_df.groupby('condition')[response_cols].mean()

        # Conversation characteristics
        char_cols = ['total_participant_messages', 'total_chatbot_messages',
                    'avg_participant_msg_length', 'avg_chatbot_msg_length',
                    'strategy_diversity', 'num_turning_points']
        available_char_cols = [col for col in char_cols if col in self.conversations_df.columns]
        if available_char_cols:
            comparisons['characteristics'] = self.conversations_df.groupby('condition')[available_char_cols].mean()

        return comparisons


# ============================================================================
# VISUALIZATION CLASS (Extended)
# ============================================================================

class ConversationVisualizer:
    """Create visualizations including chatbot and interaction patterns"""

    def __init__(self, messages_df: pd.DataFrame, chatbot_messages_df: pd.DataFrame,
                 conversations_df: pd.DataFrame, interactions_df: pd.DataFrame,
                 output_folder: str):
        self.messages_df = messages_df
        self.chatbot_messages_df = chatbot_messages_df
        self.conversations_df = conversations_df
        self.interactions_df = interactions_df
        self.output_folder = Path(output_folder) / "visualizations"
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.condition_colors = {'dark_triad': '#E74C3C', 'servant_leader': '#3498DB'}

    def plot_strategy_distribution(self):
        """Plot participant strategy distribution by condition"""
        if len(self.messages_df) == 0:
            print("WARNING: No participant messages to plot")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Overall distribution
        strategy_dist = self.messages_df['strategy_category'].value_counts()
        strategy_dist_pct = (strategy_dist / strategy_dist.sum() * 100)

        axes[0].bar(range(len(strategy_dist_pct)), strategy_dist_pct.values,
                   color=sns.color_palette("husl", len(strategy_dist_pct)))
        axes[0].set_xticks(range(len(strategy_dist_pct)))
        axes[0].set_xticklabels(strategy_dist_pct.index, rotation=45, ha='right')
        axes[0].set_ylabel('Percentage of Messages (%)', fontsize=12)
        axes[0].set_title('Overall Participant Strategy Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)

        # By condition
        strategy_by_cond = self.messages_df.groupby(['condition', 'strategy_category']).size().unstack(fill_value=0)
        strategy_by_cond_pct = strategy_by_cond.div(strategy_by_cond.sum(axis=1), axis=0) * 100

        x = np.arange(len(strategy_by_cond_pct.columns))
        width = 0.35

        for i, condition in enumerate(strategy_by_cond_pct.index):
            offset = (i - 0.5) * width
            axes[1].bar(x + offset, strategy_by_cond_pct.loc[condition], width,
                       label=condition.replace('_', ' ').title(),
                       color=self.condition_colors.get(condition, 'gray'))

        axes[1].set_xticks(x)
        axes[1].set_xticklabels(strategy_by_cond_pct.columns, rotation=45, ha='right')
        axes[1].set_ylabel('Percentage of Messages (%)', fontsize=12)
        axes[1].set_title('Participant Strategy Distribution by Condition', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_folder / 'strategy_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: strategy_distribution.png")

    def plot_chatbot_tactics(self):
        """Plot chatbot leadership tactics distribution"""
        if len(self.chatbot_messages_df) == 0:
            print("WARNING: No chatbot messages to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # 1. Overall tactic frequency
        tactic_cols = [col for col in self.chatbot_messages_df.columns if col.startswith('tactic_')]
        tactic_counts = self.chatbot_messages_df[tactic_cols].sum().sort_values(ascending=True)
        tactic_counts.index = [idx.replace('tactic_', '').replace('_', ' ').title() for idx in tactic_counts.index]

        # Color by tactic type
        dark_triad = ['Intimidation', 'Criticism Harsh', 'Manipulation', 'Dismissiveness',
                     'Micromanagement', 'Blame Shifting', 'Grandiosity']
        servant = ['Encouragement', 'Empowerment', 'Constructive Feedback', 'Active Listening',
                  'Collaboration', 'Support Offering', 'Recognition']

        colors = []
        for tactic in tactic_counts.index:
            if tactic in dark_triad:
                colors.append('#E74C3C')
            elif tactic in servant:
                colors.append('#3498DB')
            else:
                colors.append('#95A5A6')

        axes[0, 0].barh(range(len(tactic_counts)), tactic_counts.values, color=colors)
        axes[0, 0].set_yticks(range(len(tactic_counts)))
        axes[0, 0].set_yticklabels(tactic_counts.index)
        axes[0, 0].set_xlabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Chatbot Leadership Tactics Frequency', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='#E74C3C', label='Dark Triad'),
                         Patch(facecolor='#3498DB', label='Servant Leader'),
                         Patch(facecolor='#95A5A6', label='Neutral')]
        axes[0, 0].legend(handles=legend_elements, loc='lower right')

        # 2. Tactics by condition
        tactic_by_cond = self.chatbot_messages_df.groupby('condition')[tactic_cols].sum()
        tactic_by_cond.columns = [col.replace('tactic_', '').replace('_', ' ').title() for col in tactic_by_cond.columns]

        # Select top tactics for readability
        top_tactics = tactic_counts.tail(10).index.tolist()
        tactic_by_cond_top = tactic_by_cond[top_tactics]

        tactic_by_cond_top.T.plot(kind='barh', ax=axes[0, 1],
                                  color=[self.condition_colors[c] for c in tactic_by_cond_top.index])
        axes[0, 1].set_xlabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Top 10 Tactics by Condition', fontsize=14, fontweight='bold')
        axes[0, 1].legend(title='Condition', labels=['Dark Triad', 'Servant Leader'])
        axes[0, 1].grid(axis='x', alpha=0.3)

        # 3. Tactic category proportions by condition
        tactic_categories = self.chatbot_messages_df.groupby('condition').agg({
            'dark_triad_tactics_count': 'sum',
            'servant_tactics_count': 'sum',
            'neutral_tactics_count': 'sum'
        })
        tactic_categories.columns = ['Dark Triad', 'Servant Leader', 'Neutral']
        tactic_categories_pct = tactic_categories.div(tactic_categories.sum(axis=1), axis=0) * 100

        tactic_categories_pct.plot(kind='bar', ax=axes[1, 0],
                                   color=['#E74C3C', '#3498DB', '#95A5A6'])
        axes[1, 0].set_xlabel('Condition', fontsize=12)
        axes[1, 0].set_ylabel('Percentage (%)', fontsize=12)
        axes[1, 0].set_title('Tactic Category Distribution by Condition', fontsize=14, fontweight='bold')
        axes[1, 0].set_xticklabels([x.get_text().replace('_', ' ').title() for x in axes[1, 0].get_xticklabels()], rotation=0)
        axes[1, 0].legend(title='Tactic Type')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Tactics heatmap
        tactic_props = self.conversations_df.groupby('condition')[['prop_dark_triad_tactics', 'prop_servant_tactics']].mean()
        tactic_props.columns = ['Dark Triad Tactics', 'Servant Tactics']
        tactic_props.index = [idx.replace('_', ' ').title() for idx in tactic_props.index]

        sns.heatmap(tactic_props.T, annot=True, fmt='.2f', cmap='RdBu_r', ax=axes[1, 1],
                   cbar_kws={'label': 'Average Proportion'}, center=0.5)
        axes[1, 1].set_title('Tactic Proportion Heatmap', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_folder / 'chatbot_tactics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: chatbot_tactics.png")

    def plot_chatbot_tones(self):
        """Plot chatbot tone distribution"""
        if len(self.chatbot_messages_df) == 0:
            print("WARNING: No chatbot messages to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Overall tone distribution
        tone_dist = self.chatbot_messages_df['primary_tone'].value_counts()

        tone_colors = {
            'HOSTILE': '#E74C3C', 'CONDESCENDING': '#E67E22', 'COLD': '#9B59B6',
            'NEUTRAL': '#95A5A6', 'WARM': '#F1C40F', 'SUPPORTIVE': '#3498DB', 'SARCASTIC': '#1ABC9C'
        }
        colors = [tone_colors.get(t, 'gray') for t in tone_dist.index]

        axes[0, 0].pie(tone_dist.values, labels=tone_dist.index, autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[0, 0].set_title('Overall Chatbot Tone Distribution', fontsize=14, fontweight='bold')

        # 2. Tone by condition
        tone_by_cond = self.chatbot_messages_df.groupby(['condition', 'primary_tone']).size().unstack(fill_value=0)
        tone_by_cond_pct = tone_by_cond.div(tone_by_cond.sum(axis=1), axis=0) * 100

        tone_by_cond_pct.plot(kind='bar', ax=axes[0, 1],
                              color=[tone_colors.get(t, 'gray') for t in tone_by_cond_pct.columns])
        axes[0, 1].set_xlabel('Condition', fontsize=12)
        axes[0, 1].set_ylabel('Percentage (%)', fontsize=12)
        axes[0, 1].set_title('Chatbot Tone by Condition', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticklabels([x.get_text().replace('_', ' ').title() for x in axes[0, 1].get_xticklabels()], rotation=0)
        axes[0, 1].legend(title='Tone', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. Response type distribution
        response_dist = self.chatbot_messages_df['response_to_participant'].value_counts()

        response_colors = {
            'VALIDATES': '#2ECC71', 'REJECTS': '#E74C3C', 'PARTIALLY_ACCEPTS': '#F39C12',
            'IGNORES': '#95A5A6', 'ESCALATES': '#C0392B', 'DE_ESCALATES': '#27AE60', 'REDIRECTS': '#3498DB'
        }
        colors = [response_colors.get(r, 'gray') for r in response_dist.index]

        axes[1, 0].bar(range(len(response_dist)), response_dist.values, color=colors)
        axes[1, 0].set_xticks(range(len(response_dist)))
        axes[1, 0].set_xticklabels([r.replace('_', ' ').title() for r in response_dist.index], rotation=45, ha='right')
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Chatbot Response Types', fontsize=14, fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Response by condition
        response_by_cond = self.chatbot_messages_df.groupby(['condition', 'response_to_participant']).size().unstack(fill_value=0)
        response_by_cond_pct = response_by_cond.div(response_by_cond.sum(axis=1), axis=0) * 100

        response_by_cond_pct.plot(kind='bar', ax=axes[1, 1],
                                  color=[response_colors.get(r, 'gray') for r in response_by_cond_pct.columns])
        axes[1, 1].set_xlabel('Condition', fontsize=12)
        axes[1, 1].set_ylabel('Percentage (%)', fontsize=12)
        axes[1, 1].set_title('Chatbot Response Types by Condition', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticklabels([x.get_text().replace('_', ' ').title() for x in axes[1, 1].get_xticklabels()], rotation=0)
        axes[1, 1].legend(title='Response', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_folder / 'chatbot_tones.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: chatbot_tones.png")

    def plot_interaction_dynamics(self):
        """Plot interaction patterns between participant and chatbot"""
        if len(self.interactions_df) == 0:
            print("WARNING: No interaction data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # 1. Participant strategy response to chatbot tone
        valid_data = self.interactions_df.dropna(subset=['participant_strategy', 'chatbot_tone'])
        if len(valid_data) > 0:
            valid_data['strategy_cat'] = valid_data['participant_strategy'].str.split('_').str[0]
            crosstab = pd.crosstab(valid_data['chatbot_tone'], valid_data['strategy_cat'], normalize='index') * 100

            crosstab.plot(kind='bar', ax=axes[0, 0], colormap='tab10')
            axes[0, 0].set_xlabel('Chatbot Tone', fontsize=12)
            axes[0, 0].set_ylabel('Percentage of Participant Strategies (%)', fontsize=12)
            axes[0, 0].set_title('Participant Strategy by Chatbot Tone', fontsize=14, fontweight='bold')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0, 0].grid(axis='y', alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            axes[0, 0].set_title('Participant Strategy by Chatbot Tone', fontsize=14, fontweight='bold')

        # 2. Participant engagement by chatbot response type
        valid_data = self.interactions_df.dropna(subset=['participant_engagement', 'chatbot_response_type'])
        if len(valid_data) > 0:
            crosstab = pd.crosstab(valid_data['chatbot_response_type'], valid_data['participant_engagement'], normalize='index') * 100

            engagement_colors = {'HIGH': '#2ECC71', 'MEDIUM': '#F39C12', 'LOW': '#E74C3C'}
            available_eng = [e for e in ['HIGH', 'MEDIUM', 'LOW'] if e in crosstab.columns]

            crosstab[available_eng].plot(kind='bar', ax=axes[0, 1],
                                         color=[engagement_colors[e] for e in available_eng])
            axes[0, 1].set_xlabel('Chatbot Response Type', fontsize=12)
            axes[0, 1].set_ylabel('Percentage of Engagement Levels (%)', fontsize=12)
            axes[0, 1].set_title('Participant Engagement by Chatbot Response', fontsize=14, fontweight='bold')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].legend(title='Engagement')
            axes[0, 1].grid(axis='y', alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            axes[0, 1].set_title('Participant Engagement by Chatbot Response', fontsize=14, fontweight='bold')

        # 3. Message length ratio over turns
        valid_data = self.interactions_df.dropna(subset=['participant_msg_length', 'chatbot_msg_length'])
        if len(valid_data) > 0:
            valid_data['msg_length_ratio'] = valid_data['participant_msg_length'] / (valid_data['chatbot_msg_length'] + 1)

            for condition in valid_data['condition'].unique():
                cond_data = valid_data[valid_data['condition'] == condition]
                avg_ratio = cond_data.groupby('turn_number')['msg_length_ratio'].mean()
                axes[1, 0].plot(avg_ratio.index, avg_ratio.values, marker='o', markersize=4,
                               label=condition.replace('_', ' ').title(),
                               color=self.condition_colors.get(condition, 'gray'), alpha=0.7)

            axes[1, 0].axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Equal length')
            axes[1, 0].set_xlabel('Turn Number', fontsize=12)
            axes[1, 0].set_ylabel('Participant/Chatbot Message Length Ratio', fontsize=12)
            axes[1, 0].set_title('Message Length Ratio Over Conversation', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
            axes[1, 0].set_title('Message Length Ratio Over Conversation', fontsize=14, fontweight='bold')

        # 4. Escalation/De-escalation patterns
        if 'chatbot_response_type' in self.interactions_df.columns:
            valid_data = self.interactions_df.dropna(subset=['chatbot_response_type'])
            valid_data['turn_phase'] = pd.cut(valid_data['turn_position'], bins=[0, 0.33, 0.67, 1.0],
                                              labels=['Early', 'Middle', 'Late'])

            escalation_data = valid_data[valid_data['chatbot_response_type'].isin(['ESCALATES', 'DE_ESCALATES'])]
            if len(escalation_data) > 0:
                esc_by_phase = escalation_data.groupby(['condition', 'turn_phase', 'chatbot_response_type']).size().unstack(fill_value=0)

                for condition in esc_by_phase.index.get_level_values(0).unique():
                    cond_data = esc_by_phase.loc[condition]
                    if 'ESCALATES' in cond_data.columns:
                        axes[1, 1].plot(cond_data.index, cond_data['ESCALATES'], marker='^', markersize=8,
                                       linestyle='-', linewidth=2, label=f'{condition.replace("_", " ").title()} - Escalates',
                                       color=self.condition_colors.get(condition, 'gray'))
                    if 'DE_ESCALATES' in cond_data.columns:
                        axes[1, 1].plot(cond_data.index, cond_data['DE_ESCALATES'], marker='v', markersize=8,
                                       linestyle='--', linewidth=2, label=f'{condition.replace("_", " ").title()} - De-escalates',
                                       color=self.condition_colors.get(condition, 'gray'), alpha=0.6)

                axes[1, 1].set_xlabel('Conversation Phase', fontsize=12)
                axes[1, 1].set_ylabel('Frequency', fontsize=12)
                axes[1, 1].set_title('Escalation Patterns Over Conversation', fontsize=14, fontweight='bold')
                axes[1, 1].legend()
                axes[1, 1].grid(alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'No escalation data', ha='center', va='center')
                axes[1, 1].set_title('Escalation Patterns Over Conversation', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_folder / 'interaction_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: interaction_dynamics.png")

    def plot_temporal_chatbot_patterns(self):
        """Plot how chatbot behavior changes over conversation"""
        if len(self.chatbot_messages_df) == 0:
            print("WARNING: No chatbot messages to plot")
            return

        self.chatbot_messages_df['conversation_phase'] = pd.cut(
            self.chatbot_messages_df['message_position'],
            bins=[0, 0.33, 0.67, 1.0],
            labels=['Early', 'Middle', 'Late']
        )

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Tactic types over phases by condition
        for i, condition in enumerate(['dark_triad', 'servant_leader']):
            cond_data = self.chatbot_messages_df[self.chatbot_messages_df['condition'] == condition]

            if len(cond_data) == 0:
                axes[0, i].text(0.5, 0.5, f'No data for {condition}', ha='center', va='center')
                axes[0, i].set_title(f'Tactic Evolution - {condition.replace("_", " ").title()}', fontsize=14, fontweight='bold')
                continue

            tactic_temporal = cond_data.groupby('conversation_phase').agg({
                'dark_triad_tactics_count': 'mean',
                'servant_tactics_count': 'mean',
                'neutral_tactics_count': 'mean'
            })
            tactic_temporal.columns = ['Dark Triad', 'Servant', 'Neutral']

            tactic_temporal.plot(kind='bar', ax=axes[0, i], color=['#E74C3C', '#3498DB', '#95A5A6'])
            axes[0, i].set_xlabel('Conversation Phase', fontsize=12)
            axes[0, i].set_ylabel('Avg Tactics per Message', fontsize=12)
            axes[0, i].set_title(f'Tactic Evolution - {condition.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[0, i].tick_params(axis='x', rotation=0)
            axes[0, i].legend(title='Tactic Type')
            axes[0, i].grid(axis='y', alpha=0.3)

        # 2. Tone evolution
        for condition in self.chatbot_messages_df['condition'].unique():
            cond_data = self.chatbot_messages_df[self.chatbot_messages_df['condition'] == condition]

            # Group negative vs positive tones
            cond_data['negative_tone'] = cond_data['primary_tone'].isin(['HOSTILE', 'CONDESCENDING', 'COLD', 'SARCASTIC'])
            cond_data['positive_tone'] = cond_data['primary_tone'].isin(['WARM', 'SUPPORTIVE'])

            tone_by_phase = cond_data.groupby('conversation_phase').agg({
                'negative_tone': 'mean',
                'positive_tone': 'mean'
            }) * 100

            axes[1, 0].plot(tone_by_phase.index, tone_by_phase['negative_tone'], marker='s', markersize=8,
                          linestyle='-', linewidth=2, label=f'{condition.replace("_", " ").title()} - Negative',
                          color=self.condition_colors.get(condition, 'gray'))
            axes[1, 0].plot(tone_by_phase.index, tone_by_phase['positive_tone'], marker='o', markersize=8,
                          linestyle='--', linewidth=2, label=f'{condition.replace("_", " ").title()} - Positive',
                          color=self.condition_colors.get(condition, 'gray'), alpha=0.6)

        axes[1, 0].set_xlabel('Conversation Phase', fontsize=12)
        axes[1, 0].set_ylabel('Percentage (%)', fontsize=12)
        axes[1, 0].set_title('Tone Valence Evolution', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 3. Chatbot message length over phases
        for condition in self.chatbot_messages_df['condition'].unique():
            cond_data = self.chatbot_messages_df[self.chatbot_messages_df['condition'] == condition]
            length_by_phase = cond_data.groupby('conversation_phase')['message_length'].mean()

            axes[1, 1].plot(length_by_phase.index, length_by_phase.values, marker='o', markersize=8,
                          linewidth=2, label=condition.replace('_', ' ').title(),
                          color=self.condition_colors.get(condition, 'gray'))

        axes[1, 1].set_xlabel('Conversation Phase', fontsize=12)
        axes[1, 1].set_ylabel('Average Message Length', fontsize=12)
        axes[1, 1].set_title('Chatbot Message Length Evolution', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_folder / 'temporal_chatbot_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: temporal_chatbot_patterns.png")

    def plot_emotion_patterns(self):
        """Plot participant emotional patterns"""
        if len(self.messages_df) == 0:
            print("WARNING: No participant messages to plot")
            return

        emotion_cols = [col for col in self.messages_df.columns if col.startswith('emotion_')]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Overall emotion frequency
        emotion_counts = self.messages_df[emotion_cols].sum()
        emotion_counts.index = [col.replace('emotion_', '').upper() for col in emotion_counts.index]
        emotion_counts = emotion_counts.sort_values(ascending=False)

        axes[0, 0].barh(range(len(emotion_counts)), emotion_counts.values,
                       color=sns.color_palette("coolwarm", len(emotion_counts)))
        axes[0, 0].set_yticks(range(len(emotion_counts)))
        axes[0, 0].set_yticklabels(emotion_counts.index)
        axes[0, 0].set_xlabel('Number of Messages', fontsize=12)
        axes[0, 0].set_title('Overall Emotional Tone Frequency', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)

        # 2. Emotion by condition
        emotion_by_cond = self.messages_df.groupby('condition')[emotion_cols].sum()
        emotion_by_cond.columns = [col.replace('emotion_', '').upper() for col in emotion_by_cond.columns]

        emotion_by_cond.T.plot(kind='bar', ax=axes[0, 1],
                               color=[self.condition_colors[c] for c in emotion_by_cond.index])
        axes[0, 1].set_xlabel('Emotional Tone', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Emotional Patterns by Condition', fontsize=14, fontweight='bold')
        axes[0, 1].legend(title='Condition', labels=['Dark Triad', 'Servant Leader'])
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. Emotion diversity per conversation
        emotion_diversity = self.messages_df.groupby('conversation_id')['num_emotions'].mean()
        condition_map = self.conversations_df.set_index('conversation_id')['condition']
        emotion_div_df = pd.DataFrame({'emotion_diversity': emotion_diversity, 'condition': condition_map})

        for condition in emotion_div_df['condition'].unique():
            data = emotion_div_df[emotion_div_df['condition'] == condition]['emotion_diversity']
            axes[1, 0].hist(data, alpha=0.6, label=condition.replace('_', ' ').title(),
                          bins=10, color=self.condition_colors.get(condition, 'gray'))

        axes[1, 0].set_xlabel('Average Emotions per Message', fontsize=12)
        axes[1, 0].set_ylabel('Number of Conversations', fontsize=12)
        axes[1, 0].set_title('Emotional Complexity Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Emotion heatmap
        emotion_props = self.conversations_df.groupby('condition')[
            [col for col in self.conversations_df.columns if col.startswith('prop_emotion_')]
        ].mean()
        emotion_props.columns = [col.replace('prop_emotion_', '').upper() for col in emotion_props.columns]
        emotion_props.index = [idx.replace('_', ' ').title() for idx in emotion_props.index]

        sns.heatmap(emotion_props.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1, 1],
                   cbar_kws={'label': 'Average Proportion'})
        axes[1, 1].set_title('Emotional Pattern Heatmap', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_folder / 'emotion_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: emotion_patterns.png")

    def plot_engagement_analysis(self):
        """Plot task engagement patterns"""
        if len(self.messages_df) == 0:
            print("WARNING: No participant messages to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Overall engagement
        engagement_dist = self.messages_df['task_engagement'].value_counts()
        colors = {'HIGH': '#2ECC71', 'MEDIUM': '#F39C12', 'LOW': '#E74C3C'}

        axes[0, 0].bar(engagement_dist.index, engagement_dist.values,
                      color=[colors.get(e, 'gray') for e in engagement_dist.index])
        axes[0, 0].set_xlabel('Engagement Level', fontsize=12)
        axes[0, 0].set_ylabel('Number of Messages', fontsize=12)
        axes[0, 0].set_title('Overall Task Engagement Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. Engagement by condition
        engagement_by_cond = self.messages_df.groupby(['condition', 'task_engagement']).size().unstack(fill_value=0)
        engagement_by_cond_pct = engagement_by_cond.div(engagement_by_cond.sum(axis=1), axis=0) * 100

        available_eng = [e for e in ['HIGH', 'MEDIUM', 'LOW'] if e in engagement_by_cond_pct.columns]
        engagement_by_cond_pct[available_eng].plot(kind='bar', ax=axes[0, 1],
                                                    color=[colors[e] for e in available_eng])
        axes[0, 1].set_xlabel('Condition', fontsize=12)
        axes[0, 1].set_ylabel('Percentage (%)', fontsize=12)
        axes[0, 1].set_title('Engagement Distribution by Condition', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticklabels([x.get_text().replace('_', ' ').title() for x in axes[0, 1].get_xticklabels()], rotation=0)
        axes[0, 1].legend(title='Engagement')
        axes[0, 1].grid(axis='y', alpha=0.3)

        # 3. Engagement vs message length
        for condition in self.messages_df['condition'].unique():
            cond_data = self.messages_df[self.messages_df['condition'] == condition]
            eng_length = cond_data.groupby('task_engagement')['message_length'].mean()

            axes[1, 0].plot(eng_length.index, eng_length.values, marker='o', linewidth=2, markersize=10,
                          label=condition.replace('_', ' ').title(),
                          color=self.condition_colors.get(condition, 'gray'))

        axes[1, 0].set_xlabel('Engagement Level', fontsize=12)
        axes[1, 0].set_ylabel('Average Message Length', fontsize=12)
        axes[1, 0].set_title('Message Length by Engagement Level', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 4. Average engagement per conversation
        engagement_scores = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        self.messages_df['engagement_score'] = self.messages_df['task_engagement'].map(engagement_scores)

        conv_eng = self.messages_df.groupby(['conversation_id', 'condition'])['engagement_score'].mean().reset_index()

        for condition in conv_eng['condition'].unique():
            data = conv_eng[conv_eng['condition'] == condition]['engagement_score']
            axes[1, 1].hist(data, alpha=0.6, label=condition.replace('_', ' ').title(),
                          bins=15, color=self.condition_colors.get(condition, 'gray'))

        axes[1, 1].set_xlabel('Average Engagement Score', fontsize=12)
        axes[1, 1].set_ylabel('Number of Conversations', fontsize=12)
        axes[1, 1].set_title('Distribution of Average Engagement Scores', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_folder / 'engagement_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: engagement_analysis.png")

    def plot_temporal_patterns(self):
        """Plot temporal evolution of participant behaviors"""
        if len(self.messages_df) == 0:
            print("WARNING: No participant messages to plot")
            return

        self.messages_df['conversation_phase'] = pd.cut(
            self.messages_df['message_position'],
            bins=[0, 0.33, 0.67, 1.0],
            labels=['Early', 'Middle', 'Late']
        )

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1-2. Strategy evolution by condition
        for i, condition in enumerate(['dark_triad', 'servant_leader']):
            cond_data = self.messages_df[self.messages_df['condition'] == condition]

            if len(cond_data) == 0:
                axes[0, i].text(0.5, 0.5, f'No data for {condition}', ha='center', va='center')
                continue

            strat_temporal = cond_data.groupby(['conversation_phase', 'strategy_category']).size().unstack(fill_value=0)
            strat_temporal_pct = strat_temporal.div(strat_temporal.sum(axis=1), axis=0) * 100

            strat_temporal_pct.plot(kind='bar', stacked=True, ax=axes[0, i], colormap='tab10', legend=(i==1))
            axes[0, i].set_title(f'Strategy Evolution - {condition.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            axes[0, i].set_xlabel('Conversation Phase', fontsize=12)
            axes[0, i].set_ylabel('Percentage (%)', fontsize=12)
            axes[0, i].tick_params(axis='x', rotation=0)

            if i == 1:
                axes[0, i].legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')

        # 3. Message length over time
        for condition in self.messages_df['condition'].unique():
            cond_data = self.messages_df[self.messages_df['condition'] == condition]
            phase_length = cond_data.groupby('conversation_phase')['message_length'].mean()

            axes[1, 0].plot(phase_length.index, phase_length.values, marker='o', linewidth=2, markersize=8,
                          label=condition.replace('_', ' ').title(),
                          color=self.condition_colors.get(condition, 'gray'))

        axes[1, 0].set_xlabel('Conversation Phase', fontsize=12)
        axes[1, 0].set_ylabel('Average Message Length', fontsize=12)
        axes[1, 0].set_title('Participant Message Length Evolution', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

        # 4. Engagement evolution
        for condition in self.messages_df['condition'].unique():
            cond_data = self.messages_df[self.messages_df['condition'] == condition]
            eng_temporal = cond_data.groupby(['conversation_phase', 'task_engagement']).size().unstack(fill_value=0)
            eng_temporal_pct = eng_temporal.div(eng_temporal.sum(axis=1), axis=0) * 100

            if 'HIGH' in eng_temporal_pct.columns:
                axes[1, 1].plot(eng_temporal_pct.index, eng_temporal_pct['HIGH'], marker='o', linewidth=2, markersize=8,
                              label=f'{condition.replace("_", " ").title()} - HIGH',
                              color=self.condition_colors.get(condition, 'gray'))
            if 'LOW' in eng_temporal_pct.columns:
                axes[1, 1].plot(eng_temporal_pct.index, eng_temporal_pct['LOW'], marker='s', linewidth=2, markersize=8,
                              linestyle='--', label=f'{condition.replace("_", " ").title()} - LOW',
                              color=self.condition_colors.get(condition, 'gray'), alpha=0.6)

        axes[1, 1].set_xlabel('Conversation Phase', fontsize=12)
        axes[1, 1].set_ylabel('Percentage (%)', fontsize=12)
        axes[1, 1].set_title('Engagement Evolution (HIGH vs LOW)', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_folder / 'temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: temporal_patterns.png")

    def plot_conversation_characteristics(self):
        """Plot conversation-level characteristics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        characteristics = [
            ('total_participant_messages', 'Participant Messages per Conv'),
            ('total_chatbot_messages', 'Chatbot Messages per Conv'),
            ('avg_participant_msg_length', 'Avg Participant Msg Length'),
            ('avg_chatbot_msg_length', 'Avg Chatbot Msg Length'),
            ('strategy_diversity', 'Strategy Diversity (Entropy)'),
            ('num_turning_points', 'Number of Turning Points')
        ]

        for idx, (col, title) in enumerate(characteristics):
            if col not in self.conversations_df.columns:
                axes[idx].text(0.5, 0.5, f'{title}\n(No data)', ha='center', va='center')
                axes[idx].set_title(title, fontsize=12, fontweight='bold')
                continue

            for condition in self.conversations_df['condition'].unique():
                data = self.conversations_df[self.conversations_df['condition'] == condition][col].dropna()
                if len(data) > 0:
                    axes[idx].hist(data, alpha=0.6, label=condition.replace('_', ' ').title(),
                                 bins=min(15, len(data)), color=self.condition_colors.get(condition, 'gray'))

            axes[idx].set_xlabel(title, fontsize=11)
            axes[idx].set_ylabel('Number of Conversations', fontsize=11)
            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].legend(fontsize=9)
            axes[idx].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_folder / 'conversation_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: conversation_characteristics.png")

    def plot_jailbreaking_analysis(self):
        """Analyze jailbreaking attempts"""
        if len(self.messages_df) == 0:
            print("WARNING: No participant messages to plot")
            return

        jailbreak_mask = self.messages_df['strategy_category'] == 'JAILBREAKING'

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Jailbreaking by condition
        jailbreak_by_cond = self.messages_df.groupby('condition')['strategy_category'].apply(
            lambda x: (x == 'JAILBREAKING').sum()
        )

        axes[0, 0].bar(jailbreak_by_cond.index, jailbreak_by_cond.values,
                      color=[self.condition_colors.get(c, 'gray') for c in jailbreak_by_cond.index])
        axes[0, 0].set_xlabel('Condition', fontsize=12)
        axes[0, 0].set_ylabel('Number of Jailbreaking Attempts', fontsize=12)
        axes[0, 0].set_title('Jailbreaking Attempts by Condition', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticklabels([x.get_text().replace('_', ' ').title() for x in axes[0, 0].get_xticklabels()], rotation=0)
        axes[0, 0].grid(axis='y', alpha=0.3)

        # 2. Proportion of conversations with jailbreaking
        conv_with_jailbreak = self.messages_df[jailbreak_mask].groupby(['conversation_id', 'condition']).size().reset_index()

        if len(conv_with_jailbreak) > 0:
            prop_jailbreak = conv_with_jailbreak.groupby('condition').size() / self.conversations_df.groupby('condition').size() * 100

            axes[0, 1].bar(prop_jailbreak.index, prop_jailbreak.values,
                          color=[self.condition_colors.get(c, 'gray') for c in prop_jailbreak.index])
            axes[0, 1].set_xlabel('Condition', fontsize=12)
            axes[0, 1].set_ylabel('Percentage of Conversations (%)', fontsize=12)
            axes[0, 1].set_title('Conversations with Jailbreaking', fontsize=14, fontweight='bold')
            axes[0, 1].set_xticklabels([x.get_text().replace('_', ' ').title() for x in axes[0, 1].get_xticklabels()], rotation=0)
            axes[0, 1].grid(axis='y', alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, 'No jailbreaking attempts', ha='center', va='center')
            axes[0, 1].set_title('Conversations with Jailbreaking', fontsize=14, fontweight='bold')

        # 3. Jailbreaking types
        jailbreak_data = self.messages_df[jailbreak_mask]
        if len(jailbreak_data) > 0:
            jailbreak_types = jailbreak_data['primary_strategy'].value_counts()

            axes[1, 0].barh(range(len(jailbreak_types)), jailbreak_types.values,
                           color=sns.color_palette("viridis", len(jailbreak_types)))
            axes[1, 0].set_yticks(range(len(jailbreak_types)))
            axes[1, 0].set_yticklabels([t.replace('JAILBREAKING_', '').replace('_', ' ').title() for t in jailbreak_types.index])
            axes[1, 0].set_xlabel('Frequency', fontsize=12)
            axes[1, 0].set_title('Types of Jailbreaking', fontsize=14, fontweight='bold')
            axes[1, 0].grid(axis='x', alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No jailbreaking attempts', ha='center', va='center')
            axes[1, 0].set_title('Types of Jailbreaking', fontsize=14, fontweight='bold')

        # 4. Timing of jailbreaking
        if len(jailbreak_data) > 0:
            jailbreak_data = jailbreak_data.copy()
            jailbreak_data['phase'] = pd.cut(jailbreak_data['message_position'], bins=[0, 0.33, 0.67, 1.0],
                                             labels=['Early', 'Middle', 'Late'])

            phase_dist = jailbreak_data.groupby(['condition', 'phase']).size().unstack(fill_value=0)

            phase_dist.plot(kind='bar', ax=axes[1, 1], color=['#3498DB', '#F39C12', '#E74C3C'])
            axes[1, 1].set_xlabel('Condition', fontsize=12)
            axes[1, 1].set_ylabel('Number of Attempts', fontsize=12)
            axes[1, 1].set_title('Timing of Jailbreaking', fontsize=14, fontweight='bold')
            axes[1, 1].set_xticklabels([x.get_text().replace('_', ' ').title() for x in axes[1, 1].get_xticklabels()], rotation=0)
            axes[1, 1].legend(title='Phase')
            axes[1, 1].grid(axis='y', alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No jailbreaking attempts', ha='center', va='center')
            axes[1, 1].set_title('Timing of Jailbreaking', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_folder / 'jailbreaking_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: jailbreaking_analysis.png")

    def plot_correlation_matrix(self):
        """Plot correlation matrix of key variables"""
        numeric_cols = [
            'total_participant_messages', 'total_chatbot_messages',
            'avg_participant_msg_length', 'avg_chatbot_msg_length',
            'strategy_diversity', 'num_turning_points',
            'prop_compliance', 'prop_resistance', 'prop_jailbreaking', 'prop_negotiation',
            'prop_emotion_frustrated', 'prop_emotion_anxious', 'prop_emotion_confident',
            'prop_engagement_high', 'prop_engagement_low',
            'prop_dark_triad_tactics', 'prop_servant_tactics',
            'prop_tone_hostile', 'prop_tone_supportive',
            'prop_response_validates', 'prop_response_rejects', 'prop_response_escalates'
        ]

        available_cols = [col for col in numeric_cols if col in self.conversations_df.columns]

        if len(available_cols) < 2:
            print("WARNING: Not enough numeric columns for correlation matrix")
            return

        corr_matrix = self.conversations_df[available_cols].corr()

        fig, ax = plt.subplots(figsize=(18, 16))

        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   ax=ax, vmin=-1, vmax=1, annot_kws={"size": 8})

        labels = [col.replace('prop_', '').replace('_', ' ').title() for col in available_cols]
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, rotation=0, fontsize=9)

        ax.set_title('Correlation Matrix (Participant + Chatbot Features)', fontsize=16, fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(self.output_folder / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: correlation_matrix.png")

    def plot_dyadic_patterns(self):
        """Plot patterns showing participant-chatbot dyadic relationships"""
        if len(self.conversations_df) == 0:
            print("WARNING: No conversation data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # 1. Participant compliance vs Chatbot supportive tactics
        if 'prop_compliance' in self.conversations_df.columns and 'prop_servant_tactics' in self.conversations_df.columns:
            for condition in self.conversations_df['condition'].unique():
                cond_data = self.conversations_df[self.conversations_df['condition'] == condition]
                axes[0, 0].scatter(cond_data['prop_servant_tactics'], cond_data['prop_compliance'],
                                  alpha=0.6, label=condition.replace('_', ' ').title(),
                                  color=self.condition_colors.get(condition, 'gray'), s=80)

            axes[0, 0].set_xlabel('Chatbot Servant Tactics Proportion', fontsize=12)
            axes[0, 0].set_ylabel('Participant Compliance Proportion', fontsize=12)
            axes[0, 0].set_title('Compliance vs Supportive Chatbot Behavior', fontsize=14, fontweight='bold')
            axes[0, 0].legend()
            axes[0, 0].grid(alpha=0.3)

        # 2. Participant frustration vs Chatbot hostile behavior
        if 'prop_emotion_frustrated' in self.conversations_df.columns and 'prop_dark_triad_tactics' in self.conversations_df.columns:
            for condition in self.conversations_df['condition'].unique():
                cond_data = self.conversations_df[self.conversations_df['condition'] == condition]
                axes[0, 1].scatter(cond_data['prop_dark_triad_tactics'], cond_data['prop_emotion_frustrated'],
                                  alpha=0.6, label=condition.replace('_', ' ').title(),
                                  color=self.condition_colors.get(condition, 'gray'), s=80)

            axes[0, 1].set_xlabel('Chatbot Dark Triad Tactics Proportion', fontsize=12)
            axes[0, 1].set_ylabel('Participant Frustration Proportion', fontsize=12)
            axes[0, 1].set_title('Participant Frustration vs Hostile Chatbot Behavior', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(alpha=0.3)

        # 3. Participant resistance vs Chatbot rejection
        if 'prop_resistance' in self.conversations_df.columns and 'prop_response_rejects' in self.conversations_df.columns:
            for condition in self.conversations_df['condition'].unique():
                cond_data = self.conversations_df[self.conversations_df['condition'] == condition]
                axes[1, 0].scatter(cond_data['prop_response_rejects'], cond_data['prop_resistance'],
                                  alpha=0.6, label=condition.replace('_', ' ').title(),
                                  color=self.condition_colors.get(condition, 'gray'), s=80)

            axes[1, 0].set_xlabel('Chatbot Rejection Proportion', fontsize=12)
            axes[1, 0].set_ylabel('Participant Resistance Proportion', fontsize=12)
            axes[1, 0].set_title('Participant Resistance vs Chatbot Rejection', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].grid(alpha=0.3)

        # 4. Participant engagement vs Chatbot validation
        if 'prop_engagement_high' in self.conversations_df.columns and 'prop_response_validates' in self.conversations_df.columns:
            for condition in self.conversations_df['condition'].unique():
                cond_data = self.conversations_df[self.conversations_df['condition'] == condition]
                axes[1, 1].scatter(cond_data['prop_response_validates'], cond_data['prop_engagement_high'],
                                  alpha=0.6, label=condition.replace('_', ' ').title(),
                                  color=self.condition_colors.get(condition, 'gray'), s=80)

            axes[1, 1].set_xlabel('Chatbot Validation Proportion', fontsize=12)
            axes[1, 1].set_ylabel('Participant High Engagement Proportion', fontsize=12)
            axes[1, 1].set_title('Participant Engagement vs Chatbot Validation', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
            axes[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_folder / 'dyadic_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved: dyadic_patterns.png")

    def create_summary_report(self, stats: ConversationStatistics):
        """Create the text report"""
        report_path = self.output_folder / 'analysis_summary_report.txt'

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CONVERSATION ANALYSIS SUMMARY REPORT\n")
            f.write("(Including Chatbot and Interaction Analysis)\n")
            f.write("="*80 + "\n\n")

            # Overall summary
            f.write("OVERALL SUMMARY\n")
            f.write("-"*80 + "\n")
            f.write(stats.overall_summary().to_string() + "\n\n")

            # Participant strategies
            f.write("PARTICIPANT STRATEGY DISTRIBUTION\n")
            f.write("-"*80 + "\n")
            f.write("\nOverall:\n")
            f.write(stats.strategy_distribution(by_condition=False).to_string() + "\n\n")
            f.write("By Condition (%):\n")
            f.write(stats.strategy_distribution(by_condition=True).to_string() + "\n\n")

            # Chatbot tactics
            f.write("CHATBOT LEADERSHIP TACTICS\n")
            f.write("-"*80 + "\n")
            f.write("\nOverall:\n")
            f.write(stats.chatbot_tactic_distribution(by_condition=False).to_string() + "\n\n")
            f.write("By Condition:\n")
            f.write(stats.chatbot_tactic_distribution(by_condition=True).to_string() + "\n\n")

            # Chatbot tones
            f.write("CHATBOT TONE DISTRIBUTION\n")
            f.write("-"*80 + "\n")
            f.write("\nOverall (%):\n")
            f.write(stats.chatbot_tone_distribution(by_condition=False).to_string() + "\n\n")
            f.write("By Condition (%):\n")
            f.write(stats.chatbot_tone_distribution(by_condition=True).to_string() + "\n\n")

            # Chatbot responses
            f.write("CHATBOT RESPONSE TYPES\n")
            f.write("-"*80 + "\n")
            f.write("\nOverall (%):\n")
            f.write(stats.chatbot_response_distribution(by_condition=False).to_string() + "\n\n")
            f.write("By Condition (%):\n")
            f.write(stats.chatbot_response_distribution(by_condition=True).to_string() + "\n\n")

            # Participant emotions
            f.write("PARTICIPANT EMOTIONAL PATTERNS\n")
            f.write("-"*80 + "\n")
            f.write("\nOverall:\n")
            f.write(stats.emotion_distribution(by_condition=False).to_string() + "\n\n")
            f.write("By Condition:\n")
            f.write(stats.emotion_distribution(by_condition=True).to_string() + "\n\n")

            # Engagement
            f.write("PARTICIPANT TASK ENGAGEMENT\n")
            f.write("-"*80 + "\n")
            f.write("\nOverall (%):\n")
            f.write(stats.engagement_distribution(by_condition=False).to_string() + "\n\n")
            f.write("By Condition (%):\n")
            f.write(stats.engagement_distribution(by_condition=True).to_string() + "\n\n")

            # Condition comparison
            f.write("CONDITION COMPARISON\n")
            f.write("-"*80 + "\n")
            comparisons = stats.condition_comparison()

            for name, df in comparisons.items():
                f.write(f"\n{name.replace('_', ' ').title()}:\n")
                df_display = df.copy()
                df_display.columns = [col.replace('prop_', '').replace('_', ' ').upper() for col in df_display.columns]
                df_display.index = [idx.replace('_', ' ').title() for idx in df_display.index]
                f.write(df_display.to_string() + "\n\n")

            # Interaction patterns
            f.write("INTERACTION PATTERNS\n")
            f.write("-"*80 + "\n")
            patterns = stats.interaction_patterns()
            for name, df in patterns.items():
                f.write(f"\n{name.replace('_', ' ').title()}:\n")
                f.write(df.to_string() + "\n\n")

            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")

        print("Saved: analysis_summary_report.txt")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("\n" + "="*80)
    print("CONVERSATION ANALYSIS RESULTS PROCESSOR")
    print("(Extended: Participant + Chatbot + Interaction Analysis + Duration)")
    print("="*80 + "\n")

    # Step 1: Load data
    print("Step 1: Loading analyzed conversations...")
    loader = AnalysisResultsLoader(ANALYSES_FOLDER, CONVERSATION_LOGS_FOLDER)
    conversations = loader.load_all_analyses()

    if len(conversations) == 0:
        print("ERROR: No analyzed conversations found")
        return

    # Step 1b: Load timestamp data from original conversation logs
    print("\nStep 1b: Loading timestamp data from conversation logs...")
    loader.load_conversation_timestamps()

    # Step 2: Create dataframes
    print("\nStep 2: Creating analysis dataframes...")
    messages_df = loader.create_message_dataframe()
    chatbot_messages_df = loader.create_chatbot_message_dataframe()
    interactions_df = loader.create_interactions_dataframe()
    conversations_df = loader.create_conversation_dataframe()

    print(f"  - Participant messages: {len(messages_df)} rows")
    print(f"  - Chatbot messages: {len(chatbot_messages_df)} rows")
    print(f"  - Interaction turns: {len(interactions_df)} rows")
    print(f"  - Conversations: {len(conversations_df)} rows")

    # Check duration data
    if 'duration_minutes' in conversations_df.columns:
        valid_durations = conversations_df['duration_minutes'].notna().sum()
        print(f"  - Conversations with duration data: {valid_durations}/{len(conversations_df)}")

    # Step 3: Generate statistics
    print("\nStep 3: Computing statistics...")
    stats = ConversationStatistics(messages_df, chatbot_messages_df, conversations_df, interactions_df)

    print("\n--- Overall Summary ---")
    # Set pandas display options to show all rows
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None):
        # Print overall summary with all rows and columns visible
        print(stats.overall_summary())

        # Generate all paper-related statistical tests
        print("\n--- Paper Section Tests ---")
        paper_report = stats.paper_section_tests()
        print(paper_report)

    # Save section statistics to text file
    output_folder = Path(ANALYSES_FOLDER)
    with open(output_folder / 'paper_section_statistics.txt', 'w') as f:
        f.write(paper_report)
    print("Saved: paper_section_statistics.txt")

    # Step 4: Create visualizations
    print("\nStep 4: Creating visualizations...")
    visualizer = ConversationVisualizer(messages_df, chatbot_messages_df, conversations_df,
                                        interactions_df, ANALYSES_FOLDER)

    visualizer.plot_strategy_distribution()
    visualizer.plot_chatbot_tactics()
    visualizer.plot_chatbot_tones()
    visualizer.plot_interaction_dynamics()
    visualizer.plot_temporal_chatbot_patterns()
    visualizer.plot_emotion_patterns()
    visualizer.plot_engagement_analysis()
    visualizer.plot_temporal_patterns()
    visualizer.plot_conversation_characteristics()
    visualizer.plot_jailbreaking_analysis()
    visualizer.plot_correlation_matrix()
    visualizer.plot_dyadic_patterns()

    # Step 5: Create summary report
    print("\nStep 5: Generating summary report...")
    visualizer.create_summary_report(stats)

    # Step 6: Save processed dataframes
    print("\nStep 6: Saving processed dataframes...")
    output_folder = Path(ANALYSES_FOLDER)
    messages_df.to_csv(output_folder / 'participant_messages_dataframe.csv', index=False)
    chatbot_messages_df.to_csv(output_folder / 'chatbot_messages_dataframe.csv', index=False)
    interactions_df.to_csv(output_folder / 'interactions_dataframe.csv', index=False)
    conversations_df.to_csv(output_folder / 'conversations_dataframe.csv', index=False)
    print("  Saved: participant_messages_dataframe.csv")
    print("  Saved: chatbot_messages_dataframe.csv")
    print("  Saved: interactions_dataframe.csv")
    print("  Saved: conversations_dataframe.csv")

    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {output_folder / 'visualizations'}")
    print("\nGenerated visualizations:")
    print("  - strategy_distribution.png (participant strategies)")
    print("  - chatbot_tactics.png (chatbot leadership tactics)")
    print("  - chatbot_tones.png (chatbot tones & response types)")
    print("  - interaction_dynamics.png (participant-chatbot interactions)")
    print("  - temporal_chatbot_patterns.png (chatbot behavior over time)")
    print("  - emotion_patterns.png (participant emotions)")
    print("  - engagement_analysis.png (participant engagement)")
    print("  - temporal_patterns.png (participant behavior over time)")
    print("  - conversation_characteristics.png")
    print("  - jailbreaking_analysis.png")
    print("  - correlation_matrix.png (extended with chatbot features)")
    print("  - dyadic_patterns.png (participant-chatbot relationships)")
    print("  - analysis_summary_report.txt")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

# %%