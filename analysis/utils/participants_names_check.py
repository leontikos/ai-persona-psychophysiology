# %% participants_names_check.py
# Extract participant names/IDs from conversation log files

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

import pandas as pd
from dotenv import load_dotenv

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
CONVERSATION_LOGS_FOLDER = os.path.join(DATA_ROOT, "conversation_logs")
OUTPUT_FOLDER = os.path.join(DATA_ROOT, "CONVERSATION_ANALYSES")

# ============================================================================
# PARTICIPANT EXTRACTOR CLASS
# ============================================================================

class ParticipantExtractor:
    """Extract participant information from conversation log files"""

    def __init__(self, logs_folder: str):
        self.logs_folder = Path(logs_folder)
        self.participants = []
        self.raw_data = []

    def find_log_files(self, extensions: List[str] = ['.json', '.jsonl', '.txt']) -> List[Path]:
        """Find all conversation log files in the folder"""
        files = []
        for ext in extensions:
            files.extend(self.logs_folder.glob(f"*{ext}"))
            files.extend(self.logs_folder.rglob(f"*{ext}"))  # Include subfolders

        # Remove duplicates and sort
        files = sorted(set(files))
        print(f"Found {len(files)} log files")
        return files

    def extract_from_filename(self, filepath: Path) -> Dict[str, Any]:
        """Extract participant info from filename patterns"""
        filename = filepath.stem
        info = {'filename': filename, 'filepath': str(filepath)}

        # Common patterns to try:
        # Pattern 1: participant_001, P001, subject_001, etc.
        participant_pattern = re.search(r'(participant|subject|user|p|s|u)[_-]?(\d+)', filename, re.IGNORECASE)
        if participant_pattern:
            info['participant_id'] = f"{participant_pattern.group(1)}_{participant_pattern.group(2)}"
            info['participant_number'] = int(participant_pattern.group(2))

        # Pattern 2: Just a number ID
        if 'participant_id' not in info:
            number_pattern = re.search(r'^(\d+)', filename)
            if number_pattern:
                info['participant_id'] = number_pattern.group(1)
                info['participant_number'] = int(number_pattern.group(1))

        # Pattern 3: UUID-like patterns
        uuid_pattern = re.search(r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})', filename, re.IGNORECASE)
        if uuid_pattern:
            info['uuid'] = uuid_pattern.group(1)
            if 'participant_id' not in info:
                info['participant_id'] = uuid_pattern.group(1)[:8]  # Short version

        # Pattern 4: Date patterns (YYYYMMDD, YYYY-MM-DD)
        date_pattern = re.search(r'(\d{4}[-_]?\d{2}[-_]?\d{2})', filename)
        if date_pattern:
            info['date_string'] = date_pattern.group(1)

        # Pattern 5: Condition indicators
        if 'dark' in filename.lower() or 'dt' in filename.lower():
            info['condition_hint'] = 'dark_triad'
        elif 'servant' in filename.lower() or 'sl' in filename.lower():
            info['condition_hint'] = 'servant_leader'

        return info

    def extract_from_json_content(self, filepath: Path) -> Dict[str, Any]:
        """Extract participant info from JSON file content"""
        info = {}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Store raw data for later use
            self.raw_data.append({'filepath': str(filepath), 'data': data})

            # Common JSON structures to check:

            # 1. Top-level metadata
            for key in ['participant_id', 'participant', 'user_id', 'user', 'subject_id',
                       'subject', 'id', 'conversation_id', 'session_id']:
                if key in data:
                    info['participant_id'] = str(data[key])
                    break

            # 2. Nested metadata object
            if 'metadata' in data:
                metadata = data['metadata']
                for key in ['participant_id', 'participant', 'user_id', 'user', 'subject_id',
                           'subject', 'id', 'name']:
                    if key in metadata:
                        info['participant_id'] = str(metadata[key])
                        break

                # Extract other useful metadata
                if 'condition' in metadata:
                    info['condition'] = metadata['condition']
                if 'timestamp' in metadata:
                    info['timestamp'] = metadata['timestamp']
                if 'date' in metadata:
                    info['date'] = metadata['date']

            # 3. Check for participant name in messages
            if 'messages' in data:
                info['num_messages'] = len(data['messages'])
                # Check first message for participant info
                if len(data['messages']) > 0:
                    first_msg = data['messages'][0]
                    if isinstance(first_msg, dict):
                        if 'user' in first_msg:
                            info['user_from_message'] = first_msg['user']
                        if 'sender' in first_msg:
                            info['sender_from_message'] = first_msg['sender']
                        if 'participant' in first_msg:
                            info['participant_from_message'] = first_msg['participant']

            # 4. Check for history/conversation array
            for history_key in ['history', 'conversation', 'chat', 'dialogue']:
                if history_key in data:
                    info['num_turns'] = len(data[history_key])
                    break

            # 5. Extract condition from content if present
            if 'condition' in data:
                info['condition'] = data['condition']
            elif 'bot_type' in data:
                info['condition'] = data['bot_type']
            elif 'personality' in data:
                info['condition'] = data['personality']

            # 6. Check for timestamps
            for ts_key in ['created_at', 'timestamp', 'start_time', 'date']:
                if ts_key in data:
                    info['timestamp'] = data[ts_key]
                    break

        except json.JSONDecodeError as e:
            info['error'] = f"JSON decode error: {e}"
        except Exception as e:
            info['error'] = f"Error reading file: {e}"

        return info

    def extract_from_jsonl_content(self, filepath: Path) -> Dict[str, Any]:
        """Extract participant info from JSONL file content"""
        info = {}
        messages = []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            msg = json.loads(line)
                            messages.append(msg)
                        except json.JSONDecodeError:
                            continue

            info['num_messages'] = len(messages)

            # Check first message for metadata
            if len(messages) > 0:
                first_msg = messages[0]
                if isinstance(first_msg, dict):
                    for key in ['participant_id', 'user_id', 'session_id', 'id']:
                        if key in first_msg:
                            info['participant_id'] = str(first_msg[key])
                            break

                    if 'metadata' in first_msg:
                        for key in ['participant_id', 'user_id', 'subject_id']:
                            if key in first_msg['metadata']:
                                info['participant_id'] = str(first_msg['metadata'][key])
                                break

        except Exception as e:
            info['error'] = f"Error reading JSONL: {e}"

        return info

    def extract_all_participants(self) -> pd.DataFrame:
        """Extract participant information from all log files"""
        files = self.find_log_files()

        for filepath in files:
            # Start with filename extraction
            participant_info = self.extract_from_filename(filepath)

            # Add content-based extraction
            if filepath.suffix == '.json':
                content_info = self.extract_from_json_content(filepath)
                participant_info.update(content_info)
            elif filepath.suffix == '.jsonl':
                content_info = self.extract_from_jsonl_content(filepath)
                participant_info.update(content_info)

            # Ensure we have some ID
            if 'participant_id' not in participant_info:
                participant_info['participant_id'] = filepath.stem

            self.participants.append(participant_info)

        # Create DataFrame
        df = pd.DataFrame(self.participants)

        # Reorder columns for clarity
        priority_cols = ['participant_id', 'filename', 'condition', 'num_messages',
                        'timestamp', 'filepath']
        other_cols = [c for c in df.columns if c not in priority_cols]
        ordered_cols = [c for c in priority_cols if c in df.columns] + other_cols
        df = df[ordered_cols]

        return df

    def find_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find participants that appear in multiple files"""
        # Count occurrences of each participant_id
        id_counts = df['participant_id'].value_counts()
        duplicate_ids = id_counts[id_counts > 1].index.tolist()

        if len(duplicate_ids) == 0:
            return pd.DataFrame()

        # Get all rows for duplicated participants
        duplicates_df = df[df['participant_id'].isin(duplicate_ids)].copy()
        duplicates_df = duplicates_df.sort_values(['participant_id', 'filename'])

        return duplicates_df

    def generate_duplicate_report(self, df: pd.DataFrame) -> str:
        """Generate a detailed report of duplicate participants"""
        duplicates_df = self.find_duplicates(df)

        if len(duplicates_df) == 0:
            return "\nNo duplicate participants found.\n"

        report = []
        report.append("\n" + "="*80)
        report.append("DUPLICATE PARTICIPANTS REPORT")
        report.append("="*80)

        # Count unique duplicated IDs
        duplicate_ids = duplicates_df['participant_id'].unique()
        report.append(f"\nFound {len(duplicate_ids)} participant(s) appearing in multiple files:")
        report.append(f"Total duplicate entries: {len(duplicates_df)}")

        report.append("\n" + "-"*80)
        report.append("DETAILED BREAKDOWN:")
        report.append("-"*80)

        for pid in sorted(duplicate_ids):
            pid_data = duplicates_df[duplicates_df['participant_id'] == pid]
            report.append(f"\n► {pid} (appears {len(pid_data)} times):")

            for idx, row in pid_data.iterrows():
                filename = row.get('filename', 'N/A')
                num_msgs = row.get('num_messages', 'N/A')
                timestamp = row.get('timestamp', row.get('date_string', 'N/A'))

                report.append(f"    • File: {filename}")
                report.append(f"      Messages: {num_msgs}")
                if timestamp != 'N/A':
                    report.append(f"      Timestamp: {timestamp}")
                report.append("")

        report.append("-"*80)

        return "\n".join(report)

    def generate_summary(self, df: pd.DataFrame) -> str:
        """Generate a summary of extracted participants"""
        summary = []
        summary.append("="*80)
        summary.append("PARTICIPANT EXTRACTION SUMMARY")
        summary.append("="*80)
        summary.append(f"\nTotal files processed: {len(df)}")
        summary.append(f"Unique participant IDs: {df['participant_id'].nunique()}")

        # Add duplicate count
        duplicates_df = self.find_duplicates(df)
        if len(duplicates_df) > 0:
            num_dup_ids = duplicates_df['participant_id'].nunique()
            summary.append(f"Duplicated participant IDs: {num_dup_ids}")

        if 'condition' in df.columns:
            summary.append(f"\nCondition distribution:")
            for cond, count in df['condition'].value_counts().items():
                summary.append(f"  - {cond}: {count}")

        if 'num_messages' in df.columns:
            summary.append(f"\nMessage statistics:")
            summary.append(f"  - Total messages: {df['num_messages'].sum()}")
            summary.append(f"  - Avg messages/conversation: {df['num_messages'].mean():.1f}")
            summary.append(f"  - Min messages: {df['num_messages'].min()}")
            summary.append(f"  - Max messages: {df['num_messages'].max()}")

        if 'error' in df.columns:
            errors = df[df['error'].notna()]
            if len(errors) > 0:
                summary.append(f"\nFiles with errors: {len(errors)}")
                for _, row in errors.iterrows():
                    summary.append(f"  - {row['filename']}: {row['error']}")

        summary.append("\n" + "="*80)

        return "\n".join(summary)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("\n" + "="*80)
    print("PARTICIPANT NAME EXTRACTOR")
    print("="*80 + "\n")

    # Initialize extractor
    extractor = ParticipantExtractor(CONVERSATION_LOGS_FOLDER)

    # Check if folder exists
    if not extractor.logs_folder.exists():
        print(f"ERROR: Folder not found: {CONVERSATION_LOGS_FOLDER}")
        print("\nPlease update CONVERSATION_LOGS_FOLDER variable with the correct path.")
        return

    # Extract all participants
    print("Extracting participant information...")
    participants_df = extractor.extract_all_participants()

    if len(participants_df) == 0:
        print("No conversation log files found")
        return

    # Generate and print summary
    summary = extractor.generate_summary(participants_df)
    print(summary)

    # Generate and print duplicate report
    duplicate_report = extractor.generate_duplicate_report(participants_df)
    print(duplicate_report)

    # Display participant list
    print("\nExtracted Participants:")
    print("-"*80)
    display_cols = ['participant_id', 'condition', 'num_messages', 'filename']
    display_cols = [c for c in display_cols if c in participants_df.columns]
    print(participants_df[display_cols].to_string(index=False))

    # Save results
    output_folder = Path(OUTPUT_FOLDER)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Save full dataframe
    output_path = output_folder / 'extracted_participants.csv'
    participants_df.to_csv(output_path, index=False)
    print(f"\nSaved full results to: {output_path}")

    # Save duplicates dataframe if any exist
    duplicates_df = extractor.find_duplicates(participants_df)
    if len(duplicates_df) > 0:
        duplicates_path = output_folder / 'duplicate_participants.csv'
        duplicates_df.to_csv(duplicates_path, index=False)
        print(f"Saved duplicates to: {duplicates_path}")

    # Save simple participant list (unique only)
    simple_list_path = output_folder / 'participant_list.txt'
    with open(simple_list_path, 'w') as f:
        f.write("Participant IDs (Unique)\n")
        f.write("="*40 + "\n")
        for pid in sorted(participants_df['participant_id'].unique()):
            # Mark duplicates
            count = (participants_df['participant_id'] == pid).sum()
            if count > 1:
                f.write(f"{pid}  [DUPLICATE - {count} files]\n")
            else:
                f.write(f"{pid}\n")
    print(f"Saved participant list to: {simple_list_path}")

    # Save summary (including duplicate report)
    summary_path = output_folder / 'participant_extraction_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
        f.write(duplicate_report)
    print(f"Saved summary to: {summary_path}")

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE")
    print("="*80 + "\n")

    return participants_df, duplicates_df


if __name__ == "__main__":
    df, duplicates = main()
# %%
