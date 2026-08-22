# %% identify_missing_conversations.py
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
ANALYSES_FOLDER = os.path.join(DATA_ROOT, "CONVERSATION_ANALYSES")
conversations_df = pd.read_csv(Path(ANALYSES_FOLDER) / 'conversations_dataframe.csv')

print("="*80)
print("CONVERSATIONS WITHOUT AN INFERRED CONDITION")
print("="*80)
print(f"\nTotal conversations: {len(conversations_df)}")
print(f"Dark Triad: {(conversations_df['condition'] == 'dark_triad').sum()}")
print(f"Servant Leader: {(conversations_df['condition'] == 'servant_leader').sum()}")

# Find the "other" conversations
other_conditions = conversations_df[~conversations_df['condition'].isin(['dark_triad', 'servant_leader'])]

print(f"\n{'='*80}")
print(f"UNCLASSIFIED CONVERSATIONS: {len(other_conditions)}")
print(f"{'='*80}\n")

if len(other_conditions) > 0:
    print("Condition values for unclassified conversations:")
    print(other_conditions['condition'].value_counts())
    print("\nDetails of unclassified conversations:")
    print(other_conditions[['conversation_id', 'condition', 'total_participant_messages',
                            'total_chatbot_messages']].to_string())

    # Check what proportion of their messages contain certain patterns
    print("\n" + "="*80)
    print("STRATEGY PROFILES OF UNCLASSIFIED CONVERSATIONS")
    print("="*80)

    strategy_cols = [col for col in other_conditions.columns if col.startswith('prop_')
                     and not col.startswith('prop_emotion')
                     and not col.startswith('prop_engagement')
                     and not col.startswith('prop_tone')
                     and not col.startswith('prop_response')
                     and not col.startswith('prop_dark')
                     and not col.startswith('prop_servant')]

    if strategy_cols:
        print("\nParticipant strategy proportions:")
        display_df = other_conditions[['conversation_id'] + strategy_cols].copy()
        display_df.columns = ['ID'] + [col.replace('prop_', '').upper() for col in strategy_cols]
        print(display_df.to_string(index=False))

    # Check chatbot behavior
    chatbot_cols = ['prop_dark_triad_tactics', 'prop_servant_tactics']
    available_chatbot = [col for col in chatbot_cols if col in other_conditions.columns]

    if available_chatbot:
        print("\n" + "="*80)
        print("CHATBOT BEHAVIOR IN UNCLASSIFIED CONVERSATIONS")
        print("="*80)
        display_df = other_conditions[['conversation_id'] + available_chatbot].copy()
        display_df.columns = ['ID'] + [col.replace('prop_', '').replace('_', ' ').upper() for col in available_chatbot]
        print(display_df.to_string(index=False))

else:
    print("No unclassified conversations found")

print("\n" + "="*80)
print("CONDITION DISTRIBUTION SUMMARY")
print("="*80)
print("\nAll unique condition values:")
print(conversations_df['condition'].value_counts())

# Check if there are any 'unknown' or empty conditions
if (conversations_df['condition'] == 'unknown').any():
    print("\nWARNING: Some conversations labeled as 'unknown'")
    print("These likely failed condition inference during analysis.")

if conversations_df['condition'].isna().any():
    print("\nWARNING: Some conversations have NULL condition values")

# %%
