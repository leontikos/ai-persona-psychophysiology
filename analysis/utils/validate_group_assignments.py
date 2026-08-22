# %% validate_group_assignments

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")

def validate_group_assignments(csv_path, json_folder):
    """
    Validate that participant groups in CSV match chatbot versions in JSON files.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing participant data
    json_folder : str
        Path to folder containing conversation log JSON files
    """

    # Read CSV file
    print("Reading CSV file...")
    df = pd.read_csv(csv_path)

    # Create dictionary mapping participantID to Group
    participant_groups = dict(zip(df['participantID'], df['Group']))
    print(f"Found {len(participant_groups)} participants in CSV\n")

    # Read all JSON files
    print("Reading JSON files...")
    json_files = list(Path(json_folder).glob("*.json"))
    print(f"Found {len(json_files)} JSON files\n")

    # Store results
    mismatches = []
    matches = []
    csv_only = []
    json_only = []

    # Track which participants we've seen in JSON files
    seen_in_json = set()

    # Check each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            user_id = data['metadata']['user_id']
            chatbot_version = data['metadata']['chatbot_version']

            seen_in_json.add(user_id)

            # Check if user exists in CSV
            if user_id in participant_groups:
                expected_group = participant_groups[user_id]

                # Compare group with chatbot_version
                if str(expected_group) == str(chatbot_version):
                    matches.append({
                        'user_id': user_id,
                        'group': expected_group,
                        'chatbot_version': chatbot_version,
                        'json_file': json_file.name
                    })
                else:
                    mismatches.append({
                        'user_id': user_id,
                        'expected_group': expected_group,
                        'actual_chatbot_version': chatbot_version,
                        'json_file': json_file.name
                    })
            else:
                json_only.append({
                    'user_id': user_id,
                    'chatbot_version': chatbot_version,
                    'json_file': json_file.name
                })

        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")

    # Find participants in CSV but not in JSON
    csv_only = [{'user_id': uid, 'group': group}
                for uid, group in participant_groups.items()
                if uid not in seen_in_json]

    # Print results
    print("="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    print(f"\nMATCHES: {len(matches)}")
    if matches:
        for match in matches:
            print(f"  - {match['user_id']}: Group {match['group']} = Chatbot Version {match['chatbot_version']} ({match['json_file']})")

    print(f"\nERROR: MISMATCHES: {len(mismatches)}")
    if mismatches:
        for mismatch in mismatches:
            print(f"  - {mismatch['user_id']}: Expected Group {mismatch['expected_group']} but found Chatbot Version {mismatch['actual_chatbot_version']}")
            print(f"    JSON File: {mismatch['json_file']}")

    print(f"\nWARNING: PARTICIPANTS IN CSV ONLY (no JSON file found): {len(csv_only)}")
    if csv_only:
        for item in csv_only:
            print(f"  - {item['user_id']} (Group {item['group']})")

    print(f"\nWARNING: PARTICIPANTS IN JSON ONLY (not in CSV): {len(json_only)}")
    if json_only:
        for item in json_only:
            print(f"  - {item['user_id']} (Chatbot Version {item['chatbot_version']}, File: {item['json_file']})")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total participants in CSV: {len(participant_groups)}")
    print(f"Total JSON files: {len(json_files)}")
    print(f"Matching assignments: {len(matches)}")
    print(f"Mismatched assignments: {len(mismatches)}")
    print(f"CSV only: {len(csv_only)}")
    print(f"JSON only: {len(json_only)}")

    # Return results for further analysis if needed
    return {
        'matches': matches,
        'mismatches': mismatches,
        'csv_only': csv_only,
        'json_only': json_only
    }


# Run the validation
if __name__ == "__main__":
    csv_path = os.path.join(DATA_ROOT, "SURVEY_data", "AI_boss_qualtrics_quality.csv")
    json_folder = os.path.join(DATA_ROOT, "conversation_logs")

    results = validate_group_assignments(csv_path, json_folder)

# %%
