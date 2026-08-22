# %% delete_csv_files_from_cleaned_files.py

"""
CSV File Cleanup Script for BIOPAC Study

This script removes CSV files from participant directories in the cleaned signals folder.
It only removes CSV files that have corresponding NPY files, ensuring data safety.

The script includes:
1. Safety checks and dry-run mode
2. Logging of all operations
3. Size reporting to show space saved
4. Participant-by-participant processing with error handling
"""

import os
import glob
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
CLEANED_SIGNALS_FOLDER = os.path.join(DATA_ROOT, "cleaned_signals", "cleaned_signals")
LOG_FILE = "csv_cleanup_log.txt"

# Safety settings
DRY_RUN = False  # Set to False to actually delete files
REQUIRE_CONFIRMATION = False  # Require user confirmation before deletion

class CSVCleaner:
    """
    Safe CSV file cleanup for BIOPAC study data
    """

    def __init__(self, base_folder, dry_run=True):
        self.base_folder = base_folder
        self.dry_run = dry_run
        self.total_files_found = 0
        self.total_files_deleted = 0
        self.total_size_saved = 0
        self.errors = []

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def find_participants(self):
        """Find all participant directories"""
        if not os.path.exists(self.base_folder):
            self.logger.error(f"Base folder does not exist: {self.base_folder}")
            return []

        participants = []
        for item in os.listdir(self.base_folder):
            participant_dir = os.path.join(self.base_folder, item)
            if os.path.isdir(participant_dir):
                participants.append(item)

        return sorted(participants)

    def find_csv_files_with_npy_counterparts(self, participant_dir):
        """
        Find CSV files that have corresponding NPY files

        Returns:
        - List of CSV files that can be safely deleted
        - Statistics about files found
        """
        csv_files_to_delete = []

        # Find all CSV files in participant directory
        csv_pattern = os.path.join(participant_dir, "*.csv")
        csv_files = glob.glob(csv_pattern)

        for csv_file in csv_files:
            # Generate expected NPY filename
            csv_path = Path(csv_file)
            npy_file = csv_path.with_suffix('.npy')

            # Check if corresponding NPY file exists
            if npy_file.exists():
                csv_files_to_delete.append(csv_file)
                self.logger.debug(f"Found CSV with NPY counterpart: {os.path.basename(csv_file)}")
            else:
                self.logger.warning(f"CSV file without NPY counterpart: {os.path.basename(csv_file)}")

        return csv_files_to_delete

    def get_file_size(self, filepath):
        """Get file size in bytes"""
        try:
            return os.path.getsize(filepath)
        except OSError:
            return 0

    def format_size(self, size_bytes):
        """Format size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def process_participant(self, participant_id):
        """Process a single participant directory"""
        participant_dir = os.path.join(self.base_folder, participant_id)

        self.logger.info(f"Processing participant: {participant_id}")

        try:
            # Find CSV files that can be safely deleted
            csv_files_to_delete = self.find_csv_files_with_npy_counterparts(participant_dir)

            if not csv_files_to_delete:
                self.logger.info(f"  No CSV files with NPY counterparts found")
                return

            participant_size_saved = 0
            participant_files_deleted = 0

            # Process each CSV file
            for csv_file in csv_files_to_delete:
                file_size = self.get_file_size(csv_file)
                filename = os.path.basename(csv_file)

                if self.dry_run:
                    self.logger.info(f"  [DRY RUN] Would delete: {filename} ({self.format_size(file_size)})")
                else:
                    try:
                        os.remove(csv_file)
                        self.logger.info(f"  Deleted: {filename} ({self.format_size(file_size)})")
                        participant_files_deleted += 1
                    except OSError as e:
                        error_msg = f"Failed to delete {filename}: {e}"
                        self.logger.error(f"  {error_msg}")
                        self.errors.append(error_msg)
                        continue

                participant_size_saved += file_size
                self.total_files_found += 1

            self.total_files_deleted += participant_files_deleted
            self.total_size_saved += participant_size_saved

            self.logger.info(f"  Participant summary: {len(csv_files_to_delete)} files, {self.format_size(participant_size_saved)} saved")

        except Exception as e:
            error_msg = f"Error processing participant {participant_id}: {e}"
            self.logger.error(error_msg)
            self.errors.append(error_msg)

    def confirm_deletion(self, total_files, total_size):
        """Ask user to confirm deletion"""
        if not REQUIRE_CONFIRMATION:
            return True

        print(f"\nREADY TO DELETE:")
        print(f"  Files to delete: {total_files}")
        print(f"  Total size: {self.format_size(total_size)}")
        print(f"  Directory: {self.base_folder}")

        while True:
            response = input("\nProceed with deletion? (yes/no): ").lower().strip()
            if response in ['yes', 'y']:
                return True
            elif response in ['no', 'n']:
                return False
            else:
                print("Please enter 'yes' or 'no'")

    def run_cleanup(self):
        """Run the complete cleanup process"""
        self.logger.info("="*80)
        self.logger.info("CSV CLEANUP SCRIPT FOR BIOPAC STUDY")
        self.logger.info("="*80)
        self.logger.info(f"Base directory: {self.base_folder}")
        self.logger.info(f"Dry run mode: {self.dry_run}")
        self.logger.info(f"Log file: {LOG_FILE}")

        # Find participants
        participants = self.find_participants()
        if not participants:
            self.logger.error("No participant directories found!")
            return

        self.logger.info(f"Found {len(participants)} participant directories")

        # First pass: count files and sizes (always in dry run mode)
        self.logger.info("\nScanning for CSV files...")
        temp_dry_run = self.dry_run
        self.dry_run = True  # Force dry run for scanning

        for participant_id in participants:
            self.process_participant(participant_id)

        # Restore original dry run setting
        self.dry_run = temp_dry_run

        if self.total_files_found == 0:
            self.logger.info("No CSV files with NPY counterparts found!")
            return

        # Show summary and get confirmation if not in dry run mode
        self.logger.info("\n" + "="*60)
        self.logger.info("SCAN COMPLETE")
        self.logger.info("="*60)
        self.logger.info(f"Total CSV files found: {self.total_files_found}")
        self.logger.info(f"Total size to be freed: {self.format_size(self.total_size_saved)}")

        if not self.dry_run:
            if not self.confirm_deletion(self.total_files_found, self.total_size_saved):
                self.logger.info("Deletion cancelled by user")
                return

            # Reset counters for actual deletion
            self.total_files_deleted = 0
            self.total_size_saved = 0

            self.logger.info("\nStarting deletion...")
            for participant_id in participants:
                self.process_participant(participant_id)

        # Final summary
        self.logger.info("\n" + "="*60)
        self.logger.info("CLEANUP COMPLETE")
        self.logger.info("="*60)

        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would delete: {self.total_files_found} files")
            self.logger.info(f"[DRY RUN] Would free: {self.format_size(self.total_size_saved)}")
            self.logger.info("\nTo actually delete files, set DRY_RUN = False")
        else:
            self.logger.info(f"Files deleted: {self.total_files_deleted}/{self.total_files_found}")
            self.logger.info(f"Space freed: {self.format_size(self.total_size_saved)}")

            if self.errors:
                self.logger.warning(f"Errors encountered: {len(self.errors)}")
                for error in self.errors:
                    self.logger.warning(f"  {error}")

        self.logger.info(f"Log saved to: {LOG_FILE}")

def main():
    """Main execution function"""
    print("CSV Cleanup Script for BIOPAC Study")
    print("="*50)

    # Safety reminder
    if not DRY_RUN:
        print("WARNING: this script deletes files. Back up first.")
        print()
    else:
        print("ℹ  Running in DRY RUN mode - no files will be deleted")
        print()

    # Create cleaner and run
    cleaner = CSVCleaner(CLEANED_SIGNALS_FOLDER, dry_run=DRY_RUN)
    cleaner.run_cleanup()

    print("\nScript completed. Check the log file for details.")

if __name__ == "__main__":
    main()

# %%