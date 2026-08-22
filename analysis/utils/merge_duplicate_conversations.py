# %% merge_duplicate_conversations.py
# Merge conversation logs from participants who had session interruptions
# Run this BEFORE the GPT analysis pipeline

import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from dotenv import load_dotenv

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
CONVERSATION_LOGS_FOLDER = os.path.join(DATA_ROOT, "conversation_logs")
BACKUP_FOLDER = os.path.join(DATA_ROOT, "conversation_logs_backup")

# Set to True to actually perform the merge, False for dry-run preview
PERFORM_MERGE = True

# Whether to remove redundant greeting from continuation files
REMOVE_REDUNDANT_GREETING = False

# ============================================================================
# CONVERSATION MERGER CLASS
# ============================================================================

class ConversationMerger:
    """Detect and merge duplicate participant conversations"""

    def __init__(self, logs_folder: str, backup_folder: str):
        self.logs_folder = Path(logs_folder)
        self.backup_folder = Path(backup_folder)
        self.conversations = {}  # filename -> data
        self.user_files = defaultdict(list)  # user_id -> [filenames]

    def load_all_conversations(self) -> int:
        """Load all conversation files and index by user_id"""
        json_files = sorted(self.logs_folder.glob("conversation_*.json"))
        print(f"Found {len(json_files)} conversation files")

        for filepath in json_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                filename = filepath.name
                self.conversations[filename] = {
                    'filepath': filepath,
                    'data': data,
                    'user_id': data.get('metadata', {}).get('user_id', 'unknown'),
                    'total_messages': data.get('metadata', {}).get('total_messages', 0),
                    'last_updated': data.get('metadata', {}).get('last_updated', ''),
                    'first_msg_time': self._get_first_message_time(data),
                    'last_msg_time': self._get_last_message_time(data),
                }

                user_id = self.conversations[filename]['user_id']
                self.user_files[user_id].append(filename)

            except Exception as e:
                print(f"ERROR loading {filepath.name}: {e}")

        return len(self.conversations)

    def _get_first_message_time(self, data: Dict) -> str:
        """Get timestamp of first message"""
        messages = data.get('messages', [])
        if messages:
            return messages[0].get('timestamp_start', '')
        return ''

    def _get_last_message_time(self, data: Dict) -> str:
        """Get timestamp of last message"""
        messages = data.get('messages', [])
        if messages:
            return messages[-1].get('timestamp_end', messages[-1].get('timestamp_start', ''))
        return ''

    def find_duplicates(self) -> Dict[str, List[str]]:
        """Find user_ids that appear in multiple files"""
        duplicates = {
            user_id: files
            for user_id, files in self.user_files.items()
            if len(files) > 1
        }
        return duplicates

    def generate_duplicate_report(self) -> str:
        """Generate detailed report of duplicates"""
        duplicates = self.find_duplicates()

        if not duplicates:
            return "\nNo duplicate participants found.\n"

        report = []
        report.append("\n" + "="*80)
        report.append("DUPLICATE PARTICIPANTS DETECTED")
        report.append("="*80)
        report.append(f"\nFound {len(duplicates)} participant(s) with multiple conversation files:\n")

        for user_id, files in sorted(duplicates.items()):
            report.append(f"\n{'─'*60}")
            report.append(f"► {user_id} ({len(files)} files)")
            report.append(f"{'─'*60}")

            # Sort files by first message timestamp
            sorted_files = sorted(files, key=lambda f: self.conversations[f]['first_msg_time'])

            for i, filename in enumerate(sorted_files, 1):
                conv = self.conversations[filename]
                report.append(f"\n  File {i}: {filename}")
                report.append(f"    Messages: {conv['total_messages']}")
                report.append(f"    First message: {conv['first_msg_time']}")
                report.append(f"    Last message: {conv['last_msg_time']}")

                # Show first user message for context
                messages = conv['data'].get('messages', [])
                first_user_msg = next((m for m in messages if m.get('role') == 'user'), None)
                if first_user_msg:
                    content = first_user_msg.get('content', '')[:80]
                    if len(first_user_msg.get('content', '')) > 80:
                        content += "..."
                    report.append(f"    First user msg: \"{content}\"")

        report.append("\n" + "="*80)
        return "\n".join(report)

    def _is_greeting_message(self, message: Dict) -> bool:
        """Check if message is a generic chatbot greeting"""
        if message.get('role') != 'assistant':
            return False
        content = message.get('content', '').strip().lower()
        greetings = [
            'how may i help you?',
            'how can i help you?',
            'hello! how can i help you?',
            'hi! how can i help you?',
        ]
        return content in greetings

    def merge_conversations(self, user_id: str, filenames: List[str]) -> Dict:
        """Merge multiple conversation files for the same user"""

        # Sort by first message timestamp (chronological order)
        sorted_files = sorted(filenames, key=lambda f: self.conversations[f]['first_msg_time'])

        print(f"\nMerging {len(sorted_files)} files for {user_id}:")
        for f in sorted_files:
            print(f"  - {f}")

        # Start with first conversation as base
        base_data = self.conversations[sorted_files[0]]['data']
        merged_messages = list(base_data.get('messages', []))
        total_tokens = base_data.get('metadata', {}).get('total_tokens', 0)

        # Track original files for metadata
        source_files = [sorted_files[0]]

        # Append messages from subsequent files
        for filename in sorted_files[1:]:
            conv_data = self.conversations[filename]['data']
            messages = conv_data.get('messages', [])
            tokens = conv_data.get('metadata', {}).get('total_tokens', 0)

            # Optionally skip redundant greeting
            start_idx = 0
            if REMOVE_REDUNDANT_GREETING and messages:
                if self._is_greeting_message(messages[0]):
                    start_idx = 1
                    print(f"    Skipping redundant greeting in {filename}")
                    # Adjust token count (approximate)
                    tokens -= messages[0].get('tokens', 0)

            # Add continuation marker as a system note (optional)
            continuation_marker = {
                "role": "system",
                "content": f"[SESSION CONTINUATION - Original file: {filename}]",
                "timestamp_start": messages[start_idx]['timestamp_start'] if start_idx < len(messages) else "",
                "timestamp_end": messages[start_idx]['timestamp_start'] if start_idx < len(messages) else "",
                "tokens": 0,
                "_merge_marker": True
            }
            merged_messages.append(continuation_marker)

            # Add remaining messages
            merged_messages.extend(messages[start_idx:])
            total_tokens += tokens
            source_files.append(filename)

        # Update metadata
        merged_metadata = {
            "user_id": user_id,
            "chatbot_version": base_data.get('metadata', {}).get('chatbot_version', '1'),
            "total_messages": len([m for m in merged_messages if not m.get('_merge_marker')]),
            "total_tokens": total_tokens,
            "model": base_data.get('metadata', {}).get('model', 'gpt-4o'),
            "last_updated": datetime.now().isoformat(),
            "_merged": True,
            "_source_files": source_files,
            "_merge_timestamp": datetime.now().isoformat(),
            "_original_message_counts": {
                f: self.conversations[f]['total_messages'] for f in sorted_files
            }
        }

        # Clean up merge markers from messages (keep metadata clean)
        for msg in merged_messages:
            if '_merge_marker' in msg:
                del msg['_merge_marker']

        merged_conversation = {
            "metadata": merged_metadata,
            "messages": merged_messages
        }

        print(f"    Merged: {merged_metadata['total_messages']} messages, {total_tokens} tokens")

        return merged_conversation

    def backup_files(self, filenames: List[str]):
        """Backup original files before modification"""
        self.backup_folder.mkdir(parents=True, exist_ok=True)

        for filename in filenames:
            src = self.logs_folder / filename
            dst = self.backup_folder / filename
            shutil.copy2(src, dst)
            print(f"    Backed up: {filename}")

    def save_merged_and_cleanup(self, user_id: str, merged_data: Dict,
                                 original_files: List[str]) -> str:
        """Save merged file and remove originals"""

        # Sort to get the first file (chronologically)
        sorted_files = sorted(original_files,
                             key=lambda f: self.conversations[f]['first_msg_time'])

        # Use the first filename as the merged filename
        merged_filename = sorted_files[0]
        merged_filepath = self.logs_folder / merged_filename

        # Save merged conversation
        with open(merged_filepath, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"    Saved merged file: {merged_filename}")

        # Remove other original files
        for filename in sorted_files[1:]:
            filepath = self.logs_folder / filename
            filepath.unlink()
            print(f"    Removed: {filename}")

        return merged_filename

    def process_all_duplicates(self, dry_run: bool = True) -> Dict:
        """Process all duplicate participants"""
        duplicates = self.find_duplicates()

        if not duplicates:
            print("\nNo duplicates to process.")
            return {}

        results = {
            'processed': [],
            'skipped': [],
            'errors': []
        }

        print("\n" + "="*80)
        print("PROCESSING DUPLICATES" + (" (DRY RUN)" if dry_run else ""))
        print("="*80)

        for user_id, files in duplicates.items():
            print(f"\n{'─'*60}")
            print(f"Processing: {user_id}")
            print(f"{'─'*60}")

            try:
                # Generate merged conversation
                merged_data = self.merge_conversations(user_id, files)

                if not dry_run:
                    # Backup originals
                    print("  Backing up original files...")
                    self.backup_files(files)

                    # Save merged and cleanup
                    print("  Saving merged file and cleaning up...")
                    merged_filename = self.save_merged_and_cleanup(user_id, merged_data, files)

                    results['processed'].append({
                        'user_id': user_id,
                        'original_files': files,
                        'merged_filename': merged_filename,
                        'total_messages': merged_data['metadata']['total_messages']
                    })
                else:
                    print("  [DRY RUN] Would merge and save")
                    results['processed'].append({
                        'user_id': user_id,
                        'original_files': files,
                        'merged_filename': f"{files[0]} (would be kept)",
                        'total_messages': merged_data['metadata']['total_messages']
                    })

            except Exception as e:
                print(f"ERROR: {e}")
                results['errors'].append({
                    'user_id': user_id,
                    'files': files,
                    'error': str(e)
                })

        return results

    def generate_summary(self, results: Dict, dry_run: bool = True) -> str:
        """Generate processing summary"""
        summary = []
        summary.append("\n" + "="*80)
        summary.append("MERGE SUMMARY" + (" (DRY RUN)" if dry_run else ""))
        summary.append("="*80)

        if results.get('processed'):
            summary.append(f"\nSuccessfully {'would merge' if dry_run else 'merged'}: {len(results['processed'])} participant(s)")
            for item in results['processed']:
                summary.append(f"  - {item['user_id']}: {len(item['original_files'])} files → {item['total_messages']} messages")

        if results.get('errors'):
            summary.append(f"\nErrors: {len(results['errors'])}")
            for item in results['errors']:
                summary.append(f"  - {item['user_id']}: {item['error']}")

        if not dry_run and results.get('processed'):
            summary.append(f"\n Backups saved to: {self.backup_folder}")

        if dry_run:
            summary.append("\nWARNING: This was a DRY RUN. No files were modified.")
            summary.append("    Set PERFORM_MERGE = True to actually merge files.")

        summary.append("\n" + "="*80)
        return "\n".join(summary)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("CONVERSATION MERGER - DUPLICATE HANDLER")
    print("="*80 + "\n")

    # Initialize merger
    merger = ConversationMerger(CONVERSATION_LOGS_FOLDER, BACKUP_FOLDER)

    # Check if folder exists
    if not merger.logs_folder.exists():
        print(f"ERROR: Folder not found: {CONVERSATION_LOGS_FOLDER}")
        return

    # Load all conversations
    print("Loading conversations...")
    count = merger.load_all_conversations()
    print(f"Loaded {count} conversations")

    # Generate duplicate report
    duplicate_report = merger.generate_duplicate_report()
    print(duplicate_report)

    # Check for duplicates
    duplicates = merger.find_duplicates()
    if not duplicates:
        print("\nNo action needed - no duplicates found.")
        return

    # Process duplicates
    print(f"\nPERFORM_MERGE is set to: {PERFORM_MERGE}")

    if PERFORM_MERGE:
        confirm = input("\nWARNING: This will MODIFY files. Type 'yes' to confirm: ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            return

    results = merger.process_all_duplicates(dry_run=not PERFORM_MERGE)

    # Print summary
    summary = merger.generate_summary(results, dry_run=not PERFORM_MERGE)
    print(summary)

    # Save report
    report_path = merger.logs_folder.parent / "merge_report.txt"
    with open(report_path, 'w') as f:
        f.write(duplicate_report)
        f.write(summary)
    print(f"\n Report saved to: {report_path}")


if __name__ == "__main__":
    main()
# %%
