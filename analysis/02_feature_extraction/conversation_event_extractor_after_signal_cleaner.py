# %% conversation_event_extractor_after_signal_cleaner.py

"""
Conversation Event Extractor for Psychophysiological Signal Analysis

This script extracts conversation timing events from conversation logs and aligns them
with the cleaned physiological signal data. It creates event metadata files that can
be used for downstream analysis without modifying the existing signal cleaning pipeline.

The script:
1. Finds conversation logs for each participant
2. Extracts conversation events (user messages, assistant responses)
3. Handles timezone adjustments (2-hour shift)
4. Aligns events with signal timing from cleaned signal metadata
5. Saves event files alongside cleaned signals for analysis

Output files per participant:
- {participant_id}_conversation_events.json: Detailed conversation timing
- {participant_id}_experimental_info.json: Group/condition metadata
- {participant_id}_event_summary.json: Summary statistics
"""

import os
import json
import glob
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import bioread
import re
from dotenv import load_dotenv

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
CLEANED_SIGNALS_FOLDER = os.path.join(DATA_ROOT, "cleaned_signals", "cleaned_signals")
CONVERSATION_LOGS_FOLDER = os.path.join(DATA_ROOT, "conversation_logs")
RAW_BIOPAC_FOLDER = os.path.join(DATA_ROOT, "psychophysiology_signal")
TIME_ADJUSTMENT_HOURS = 2  # Timezone adjustment ONLY for BIOPAC data (conversation logs have correct timing)

# ===== EVENT EXTRACTION CLASS =====

class ConversationEventExtractor:
    """
    Extract and align conversation events with physiological signal data
    """

    def __init__(self, cleaned_signals_folder, conversation_logs_folder, raw_biopac_folder, time_adjustment_hours=2):
        self.cleaned_signals_folder = cleaned_signals_folder
        self.conversation_logs_folder = conversation_logs_folder
        self.raw_biopac_folder = raw_biopac_folder
        self.time_adjustment_hours = time_adjustment_hours

        self.processing_log = {
            'start_time': datetime.now().isoformat(),
            'processed_participants': {},
            'errors': [],
            'summary': {}
        }

    def find_participants_with_cleaned_signals(self):
        """Find all participants who have cleaned signal data"""
        participants = []

        # Look for participant directories in cleaned signals folder
        if os.path.exists(self.cleaned_signals_folder):
            for item in os.listdir(self.cleaned_signals_folder):
                item_path = os.path.join(self.cleaned_signals_folder, item)
                if os.path.isdir(item_path):
                    # Check if this directory contains cleaned signal files
                    metadata_files = glob.glob(os.path.join(item_path, "*_metadata.json"))
                    if metadata_files:
                        participants.append(item)

        return sorted(participants)

    def find_conversation_log_for_participant(self, participant_id):
        """Find the conversation log file for a specific participant"""
        conversation_files = glob.glob(os.path.join(self.conversation_logs_folder, "*.json"))

        for log_file in conversation_files:
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)

                # Check if this log belongs to the participant
                if ('metadata' in data and
                    'user_id' in data['metadata'] and
                    data['metadata']['user_id'] == participant_id):
                    return log_file, data

            except Exception as e:
                print(f"Error reading {log_file}: {e}")
                continue

        return None, None

    def adjust_datetime_biopac(self, dt, hours=None):
        """
        Adjust a datetime object by a specified number of hours (for BIOPAC data only).

        Parameters:
        dt: datetime object to adjust
        hours: Number of hours to adjust by (if None, uses the global time_adjustment_hours)

        Returns:
        datetime: Adjusted datetime object
        """
        if dt is None:
            return None

        adjustment = hours if hours is not None else self.time_adjustment_hours
        return dt + timedelta(hours=adjustment)

    def parse_event_marker(self, marker_str):
        """
        Parse event marker string to extract meaningful information.

        Parameters:
        marker_str: String representation of event marker

        Returns:
        dict: Dictionary containing parsed marker information
        """
        marker_info = {}

        # Extract segment name/number
        segment_match = re.search(r'Segment (\d+)', marker_str)
        if segment_match:
            marker_info['segment'] = segment_match.group(1)

        # Extract date and time
        datetime_match = re.search(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)', marker_str)
        if datetime_match:
            dt_str = datetime_match.group(1)
            try:
                marker_info['timestamp'] = datetime.fromisoformat(dt_str)
            except ValueError:
                # Handle different datetime formats if needed
                marker_info['timestamp_str'] = dt_str

        # Extract device info
        device_match = re.search(r'MP\d+ ([0-9A-F]+)', marker_str)
        if device_match:
            marker_info['device_id'] = device_match.group(1)

        # Extract any additional info
        type_match = re.search(r'type_code: (\w+)', marker_str)
        if type_match:
            marker_info['type'] = type_match.group(1)

        return marker_info

    def extract_biopac_timestamps(self, data, adjust_hours=None):
        """
        Extract timestamp information from a bioread Datafile object and adjust for timezone differences.

        Parameters:
        data: bioread Datafile object
        adjust_hours: Optional number of hours to adjust timestamps by

        Returns:
        dict: Dictionary containing timestamp information (all as ISO strings)
        """
        timestamps = {}

        # Get the earliest marker timestamp
        if hasattr(data, 'earliest_marker_created_at'):
            adjusted_dt = self.adjust_datetime_biopac(data.earliest_marker_created_at, adjust_hours)
            timestamps['start_time'] = adjusted_dt.isoformat() if adjusted_dt else None

        # Extract event markers
        if hasattr(data, 'event_markers') and data.event_markers:
            timestamps['event_markers'] = []
            for i, marker in enumerate(data.event_markers):
                # Basic marker info
                marker_info = {
                    'index': i,
                    'raw_name': str(marker),
                    'sample_index': marker.sample_index if hasattr(marker, 'sample_index') else None
                }

                # Handle created_at timestamp
                if hasattr(marker, 'date_created_utc') and marker.date_created_utc:
                    adjusted_created_at = self.adjust_datetime_biopac(marker.date_created_utc, adjust_hours)
                    marker_info['created_at'] = adjusted_created_at.isoformat() if adjusted_created_at else None
                else:
                    marker_info['created_at'] = None

                # Parse the marker string for more structured information
                parsed_info = self.parse_event_marker(str(marker))

                # Adjust any timestamps in parsed info (BIOPAC data only)
                if 'timestamp' in parsed_info:
                    adjusted_timestamp = self.adjust_datetime_biopac(parsed_info['timestamp'], adjust_hours)
                    parsed_info['timestamp'] = adjusted_timestamp.isoformat() if adjusted_timestamp else None

                marker_info.update(parsed_info)
                timestamps['event_markers'].append(marker_info)

        # Calculate end time based on duration and start time
        if 'start_time' in timestamps and timestamps['start_time'] and data.channels and len(data.channels) > 0:
            channel = data.channels[0]
            duration_seconds = len(channel.data) / channel.samples_per_second

            try:
                # Parse the start time back to datetime to calculate end time
                start_dt = datetime.fromisoformat(timestamps['start_time'].replace('Z', '+00:00').split('+')[0])
                duration_delta = timedelta(seconds=duration_seconds)
                end_dt = start_dt + duration_delta
                timestamps['end_time'] = end_dt.isoformat()
            except Exception as e:
                print(f"Error calculating end time: {e}")
                timestamps['end_time'] = None

        # If channel has time_index attribute, get timing info
        for channel in data.channels:
            if hasattr(channel, 'time_index') and len(channel.time_index) > 0:
                timestamps['first_sample_time'] = float(channel.time_index[0])
                timestamps['last_sample_time'] = float(channel.time_index[-1])
                timestamps['sampling_interval'] = float(channel.time_index[1] - channel.time_index[0]) if len(channel.time_index) > 1 else None
                break

        return timestamps

    def find_raw_biopac_files(self, participant_id):
        """Find raw BIOPAC files for a participant"""
        biopac_files = {}

        # Look for main and baseline files
        for recording_type in ['main', 'baseline']:
            file_pattern = os.path.join(self.raw_biopac_folder, f"{participant_id}_{recording_type}.acq")
            files = glob.glob(file_pattern)

            if files:
                biopac_files[recording_type] = files[0]

        return biopac_files

    def get_biopac_recording_timestamps(self, participant_id):
        """Get BIOPAC recording start timestamps for both main and baseline recordings"""
        biopac_files = self.find_raw_biopac_files(participant_id)
        recording_timestamps = {}

        for recording_type, file_path in biopac_files.items():
            try:
                print(f"    Loading BIOPAC file: {os.path.basename(file_path)}")
                data = bioread.read(file_path)
                timestamps = self.extract_biopac_timestamps(data, self.time_adjustment_hours)

                # Calculate duration if we have both start and end times
                duration_seconds = None
                if timestamps.get('start_time') and timestamps.get('end_time'):
                    try:
                        start_dt = datetime.fromisoformat(timestamps['start_time'].replace('Z', '+00:00').split('+')[0])
                        end_dt = datetime.fromisoformat(timestamps['end_time'].replace('Z', '+00:00').split('+')[0])
                        duration_seconds = (end_dt - start_dt).total_seconds()
                    except:
                        duration_seconds = None

                recording_timestamps[recording_type] = {
                    'file_path': file_path,
                    'recording_start_time': timestamps.get('start_time'),
                    'recording_end_time': timestamps.get('end_time'),
                    'duration_seconds': duration_seconds,
                    'event_markers': timestamps.get('event_markers', []),
                    'sampling_info': {
                        'first_sample_time': timestamps.get('first_sample_time'),
                        'last_sample_time': timestamps.get('last_sample_time'),
                        'sampling_interval': timestamps.get('sampling_interval')
                    }
                }

                print(f"    Recording start time: {recording_timestamps[recording_type]['recording_start_time']}")

            except Exception as e:
                print(f"    Error loading {recording_type} BIOPAC file: {e}")
                recording_timestamps[recording_type] = {
                    'file_path': file_path,
                    'error': str(e),
                    'recording_start_time': None
                }

        return recording_timestamps

    def adjust_conversation_datetime(self, dt_str):
        """Parse conversation datetime string (NO timezone adjustment for conversation logs)"""
        try:
            # Remove timezone info if present and parse
            dt_str_clean = dt_str.split('+')[0].split('Z')[0]
            dt = datetime.fromisoformat(dt_str_clean)
            return dt

        except Exception as e:
            print(f"Error parsing conversation datetime {dt_str}: {e}")
            return None
        """Get timing information from cleaned signal metadata"""
        # Look for any metadata file for this participant and recording type
        metadata_pattern = os.path.join(
            self.cleaned_signals_folder,
            participant_id,
            f"{participant_id}_{recording_type}_*_metadata.json"
        )
        metadata_files = glob.glob(metadata_pattern)

        if not metadata_files:
            return None

        # Use the first metadata file to get timing info
        try:
            with open(metadata_files[0], 'r') as f:
                metadata = json.load(f)

            # Extract signal timing information
            signal_info = {
                'sampling_rate': metadata['original_channel_info']['samples_per_second'],
                'sample_count': metadata['original_channel_info']['sample_count'],
                'duration_seconds': metadata['original_channel_info']['sample_count'] / metadata['original_channel_info']['samples_per_second'],
                'export_timestamp': metadata.get('export_timestamp'),
                'available_channels': []
            }

            # Find all channels for this participant and recording type
            all_metadata_files = glob.glob(metadata_pattern)
            for meta_file in all_metadata_files:
                try:
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                    signal_info['available_channels'].append({
                        'channel_name': meta['channel_name'],
                        'channel_type': meta['cleaning_metrics']['channel_type'],
                        'quality_score': meta['cleaning_metrics']['quality_score'],
                        'file_paths': meta['file_paths']
                    })
                except:
                    continue

            return signal_info

        except Exception as e:
            print(f"Error reading metadata from {metadata_files[0]}: {e}")
            return None

    def extract_conversation_events(self, conversation_data):
        """Extract detailed conversation events from conversation log"""
        if not conversation_data or 'messages' not in conversation_data:
            return None

        events = {
            'metadata': conversation_data.get('metadata', {}),
            'conversation_timeline': [],
            'user_messages': [],
            'assistant_responses': [],
            'conversation_summary': {
                'total_messages': 0,
                'user_message_count': 0,
                'assistant_response_count': 0,
                'conversation_start': None,
                'conversation_end': None,
                'total_duration_seconds': 0,
                'avg_response_time_seconds': 0
            }
        }

        response_times = []

        for i, message in enumerate(conversation_data['messages']):
            try:
                role = message.get('role')
                content = message.get('content', '')
                tokens = message.get('tokens', 0)

                # Parse timestamps (NO adjustment for conversation logs - they have correct timing)
                timestamp_start = self.adjust_conversation_datetime(message['timestamp_start'])
                timestamp_end = self.adjust_conversation_datetime(message['timestamp_end'])

                if not timestamp_start or not timestamp_end:
                    continue

                # Calculate message duration
                duration_seconds = (timestamp_end - timestamp_start).total_seconds()

                event_info = {
                    'message_index': i,
                    'role': role,
                    'content_preview': content[:100] + '...' if len(content) > 100 else content,
                    'content_length': len(content),
                    'tokens': tokens,
                    'timestamp_start': timestamp_start.isoformat(),
                    'timestamp_end': timestamp_end.isoformat(),
                    'duration_seconds': duration_seconds,
                    'time_since_start': 0  # Will be calculated later
                }

                # Add to appropriate list
                if role == 'user':
                    events['user_messages'].append(event_info)
                    events['conversation_summary']['user_message_count'] += 1
                elif role == 'assistant':
                    events['assistant_responses'].append(event_info)
                    events['conversation_summary']['assistant_response_count'] += 1

                    # Calculate response time if there's a previous user message
                    if (i > 0 and
                        conversation_data['messages'][i-1].get('role') == 'user'):
                        prev_timestamp_end = self.adjust_conversation_datetime(
                            conversation_data['messages'][i-1]['timestamp_end']
                        )
                        if prev_timestamp_end:
                            response_time = (timestamp_start - prev_timestamp_end).total_seconds()
                            event_info['response_time_seconds'] = response_time
                            response_times.append(response_time)

                # Add to timeline
                events['conversation_timeline'].append(event_info)
                events['conversation_summary']['total_messages'] += 1

            except Exception as e:
                print(f"Error processing message {i}: {e}")
                continue

        # Calculate summary statistics
        if events['conversation_timeline']:
            # Find conversation start and end
            start_times = [datetime.fromisoformat(msg['timestamp_start'])
                          for msg in events['conversation_timeline']]
            end_times = [datetime.fromisoformat(msg['timestamp_end'])
                        for msg in events['conversation_timeline']]

            conversation_start = min(start_times)
            conversation_end = max(end_times)

            events['conversation_summary']['conversation_start'] = conversation_start.isoformat()
            events['conversation_summary']['conversation_end'] = conversation_end.isoformat()
            events['conversation_summary']['total_duration_seconds'] = (
                conversation_end - conversation_start
            ).total_seconds()

            # Calculate time since start for each message
            for event in events['conversation_timeline']:
                event_start = datetime.fromisoformat(event['timestamp_start'])
                event['time_since_start'] = (event_start - conversation_start).total_seconds()

            # Average response time
            if response_times:
                events['conversation_summary']['avg_response_time_seconds'] = np.mean(response_times)

        return events

    def get_signal_timing_info(self, participant_id, recording_type):
            """Get timing information from cleaned signal metadata"""
            # Look for any metadata file for this participant and recording type
            metadata_pattern = os.path.join(
                self.cleaned_signals_folder,
                participant_id,
                f"{participant_id}_{recording_type}_*_metadata.json"
            )
            metadata_files = glob.glob(metadata_pattern)

            if not metadata_files:
                return None

            # Use the first metadata file to get timing info
            try:
                with open(metadata_files[0], 'r') as f:
                    metadata = json.load(f)

                # Extract signal timing information
                signal_info = {
                    'sampling_rate': metadata['original_channel_info']['samples_per_second'],
                    'sample_count': metadata['original_channel_info']['sample_count'],
                    'duration_seconds': metadata['original_channel_info']['sample_count'] / metadata['original_channel_info']['samples_per_second'],
                    'export_timestamp': metadata.get('export_timestamp'),
                    'available_channels': []
                }

                # Find all channels for this participant and recording type
                all_metadata_files = glob.glob(metadata_pattern)
                for meta_file in all_metadata_files:
                    try:
                        with open(meta_file, 'r') as f:
                            meta = json.load(f)
                        signal_info['available_channels'].append({
                            'channel_name': meta['channel_name'],
                            'channel_type': meta['cleaning_metrics']['channel_type'],
                            'quality_score': meta['cleaning_metrics']['quality_score'],
                            'file_paths': meta['file_paths']
                        })
                    except:
                        continue

                return signal_info

            except Exception as e:
                print(f"Error reading metadata from {metadata_files[0]}: {e}")
                return None

    def create_experimental_info(self, participant_id, conversation_data, signal_info, biopac_timestamps):
        """Create experimental condition and group information"""
        experimental_info = {
            'participant_id': participant_id,
            'experimental_group': None,
            'chatbot_version': None,
            'study_date': None,
            'has_conversation_data': conversation_data is not None,
            'recording_files': {
                'baseline': {},
                'main': {}
            },
            'biopac_recording_timestamps': biopac_timestamps,
            'synchronization_info': {
                'timezone_adjustment_applied_to_biopac': f"{self.time_adjustment_hours} hours",
                'conversation_logs_timezone_adjustment': "None (correct timing)",
                'note': "BIOPAC timestamps adjusted by timezone_adjustment_hours, conversation timestamps are original"
            }
        }

        # Extract group information from conversation metadata
        if conversation_data and 'metadata' in conversation_data:
            metadata = conversation_data['metadata']
            experimental_info['chatbot_version'] = metadata.get('chatbot_version', 'unknown')
            experimental_info['experimental_group'] = f"group_{metadata.get('chatbot_version', 'unknown')}"

            # Extract study date from first message or last_updated (NO timezone adjustment)
            if 'messages' in conversation_data and conversation_data['messages']:
                first_message_time = conversation_data['messages'][0].get('timestamp_start')
                if first_message_time:
                    study_date = self.adjust_conversation_datetime(first_message_time)
                    if study_date:
                        experimental_info['study_date'] = study_date.date().isoformat()

            # If no date from messages, try metadata
            if not experimental_info['study_date'] and 'last_updated' in metadata:
                study_date = self.adjust_conversation_datetime(metadata['last_updated'])
                if study_date:
                    experimental_info['study_date'] = study_date.date().isoformat()

        # Add information about available recording files
        for recording_type in ['baseline', 'main']:
            signal_timing = self.get_signal_timing_info(participant_id, recording_type)
            biopac_timing = biopac_timestamps.get(recording_type, {})

            if signal_timing:
                experimental_info['recording_files'][recording_type] = {
                    'available': True,
                    'duration_seconds': signal_timing['duration_seconds'],
                    'sampling_rate': signal_timing['sampling_rate'],
                    'channels': [ch['channel_name'] for ch in signal_timing['available_channels']],
                    'channel_quality_scores': {
                        ch['channel_name']: ch['quality_score']
                        for ch in signal_timing['available_channels']
                    },
                    'biopac_recording_start': biopac_timing.get('recording_start_time'),
                    'biopac_recording_end': biopac_timing.get('recording_end_time'),
                    'biopac_duration_seconds': biopac_timing.get('duration_seconds'),
                    'has_event_markers': len(biopac_timing.get('event_markers', [])) > 0,
                    'event_markers_count': len(biopac_timing.get('event_markers', []))
                }
            else:
                experimental_info['recording_files'][recording_type] = {
                    'available': False,
                    'biopac_recording_start': biopac_timing.get('recording_start_time'),
                    'biopac_recording_end': biopac_timing.get('recording_end_time')
                }

        return experimental_info

    def create_event_summary(self, participant_id, conversation_events, experimental_info):
        """Create a summary of events for quick analysis"""
        summary = {
            'participant_id': participant_id,
            'processing_timestamp': datetime.now().isoformat(),
            'experimental_group': experimental_info.get('experimental_group'),
            'chatbot_version': experimental_info.get('chatbot_version'),
            'study_date': experimental_info.get('study_date'),
            'has_conversation_events': conversation_events is not None,
            'event_counts': {
                'total_messages': 0,
                'user_messages': 0,
                'assistant_responses': 0
            },
            'timing_info': {
                'conversation_duration_seconds': 0,
                'avg_response_time_seconds': 0,
                'conversation_start_time': None,
                'conversation_end_time': None
            },
            'available_recordings': [],
            'analysis_ready': False
        }

        # Fill in conversation event information
        if conversation_events:
            conv_summary = conversation_events['conversation_summary']
            summary['event_counts'] = {
                'total_messages': conv_summary['total_messages'],
                'user_messages': conv_summary['user_message_count'],
                'assistant_responses': conv_summary['assistant_response_count']
            }
            summary['timing_info'] = {
                'conversation_duration_seconds': conv_summary['total_duration_seconds'],
                'avg_response_time_seconds': conv_summary['avg_response_time_seconds'],
                'conversation_start_time': conv_summary['conversation_start'],
                'conversation_end_time': conv_summary['conversation_end']
            }

        # Fill in recording information
        for recording_type, info in experimental_info['recording_files'].items():
            if info.get('available', False):
                summary['available_recordings'].append({
                    'recording_type': recording_type,
                    'duration_seconds': info['duration_seconds'],
                    'channels': info['channels'],
                    'has_conversation_events': recording_type == 'main' and conversation_events is not None
                })

        # Determine if ready for analysis
        has_main_recording = any(rec['recording_type'] == 'main'
                               for rec in summary['available_recordings'])
        has_baseline_recording = any(rec['recording_type'] == 'baseline'
                                   for rec in summary['available_recordings'])

        summary['analysis_ready'] = (
            has_main_recording and
            has_baseline_recording and
            (conversation_events is not None or experimental_info.get('experimental_group'))
        )

        return summary

    def process_participant(self, participant_id):
        """Process a single participant to extract conversation events"""
        print(f"Processing participant: {participant_id}")

        participant_results = {
            'participant_id': participant_id,
            'success': False,
            'conversation_log_found': False,
            'events_extracted': False,
            'files_created': [],
            'errors': []
        }

        try:
            # Find conversation log
            log_file, conversation_data = self.find_conversation_log_for_participant(participant_id)

            if not log_file:
                print(f"  No conversation log found for {participant_id}")
                participant_results['errors'].append("No conversation log found")
            else:
                print(f"  Found conversation log: {os.path.basename(log_file)}")
                participant_results['conversation_log_found'] = True

            # Extract BIOPAC recording timestamps
            print(f"  Extracting BIOPAC recording timestamps...")
            biopac_timestamps = self.get_biopac_recording_timestamps(participant_id)

            # Get signal timing information (check if participant has cleaned signals)
            main_signal_info = self.get_signal_timing_info(participant_id, 'main')
            baseline_signal_info = self.get_signal_timing_info(participant_id, 'baseline')

            if not main_signal_info and not baseline_signal_info:
                error_msg = f"No cleaned signal data found for {participant_id}"
                print(f"  {error_msg}")
                participant_results['errors'].append(error_msg)
                return participant_results

            # Extract conversation events
            conversation_events = None
            if conversation_data:
                conversation_events = self.extract_conversation_events(conversation_data)
                if conversation_events:
                    participant_results['events_extracted'] = True
                    print(f"  Extracted {conversation_events['conversation_summary']['total_messages']} conversation events")

            # Create experimental information (now includes BIOPAC timestamps)
            experimental_info = self.create_experimental_info(
                participant_id, conversation_data, main_signal_info or baseline_signal_info, biopac_timestamps
            )

            # Create event summary
            event_summary = self.create_event_summary(
                participant_id, conversation_events, experimental_info
            )

            # Save files to participant directory
            participant_dir = os.path.join(self.cleaned_signals_folder, participant_id)

            # 1. Save conversation events (if available)
            if conversation_events:
                events_file = os.path.join(participant_dir, f"{participant_id}_conversation_events.json")
                with open(events_file, 'w') as f:
                    json.dump(conversation_events, f, indent=2)
                participant_results['files_created'].append(events_file)
                print(f"  Saved: {os.path.basename(events_file)}")

            # 2. Save experimental info
            exp_info_file = os.path.join(participant_dir, f"{participant_id}_experimental_info.json")
            with open(exp_info_file, 'w') as f:
                json.dump(experimental_info, f, indent=2)
            participant_results['files_created'].append(exp_info_file)
            print(f"  Saved: {os.path.basename(exp_info_file)}")

            # 3. Save event summary
            summary_file = os.path.join(participant_dir, f"{participant_id}_event_summary.json")
            with open(summary_file, 'w') as f:
                json.dump(event_summary, f, indent=2)
            participant_results['files_created'].append(summary_file)
            print(f"  Saved: {os.path.basename(summary_file)}")

            participant_results['success'] = True
            print(f"  Successfully processed {participant_id}")

        except Exception as e:
            error_msg = f"Error processing {participant_id}: {e}"
            print(f"ERROR: {error_msg}")
            participant_results['errors'].append(error_msg)

        return participant_results

    def process_all_participants(self):
        """Process all participants with cleaned signal data"""
        print("=" * 80)
        print("CONVERSATION EVENT EXTRACTION")
        print("=" * 80)

        # Find all participants
        participants = self.find_participants_with_cleaned_signals()

        if not participants:
            print("No participants with cleaned signal data found")
            return

        print(f"Found {len(participants)} participants with cleaned signals: {', '.join(participants)}")
        print(f"Looking for conversation logs in: {self.conversation_logs_folder}")
        print(f"Time adjustment: {self.time_adjustment_hours} hours")
        print()

        # Process each participant
        successful_count = 0
        for participant_id in participants:
            result = self.process_participant(participant_id)
            self.processing_log['processed_participants'][participant_id] = result

            if result['success']:
                successful_count += 1

        # Create summary
        print("\n" + "=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)

        participants_with_conversations = sum(
            1 for r in self.processing_log['processed_participants'].values()
            if r['conversation_log_found']
        )
        participants_with_events = sum(
            1 for r in self.processing_log['processed_participants'].values()
            if r['events_extracted']
        )

        print(f"Total participants processed: {len(participants)}")
        print(f"Participants with conversation logs: {participants_with_conversations}")
        print(f"Participants with extracted events: {participants_with_events}")
        print(f"Successfully processed: {successful_count}")

        # Show any errors
        all_errors = []
        for participant_id, result in self.processing_log['processed_participants'].items():
            if result['errors']:
                for error in result['errors']:
                    all_errors.append(f"{participant_id}: {error}")

        if all_errors:
            print(f"\nErrors encountered ({len(all_errors)}):")
            for error in all_errors:
                print(f"  - {error}")

        # Save processing log
        self.processing_log['end_time'] = datetime.now().isoformat()
        self.processing_log['summary'] = {
            'total_participants': len(participants),
            'participants_with_conversations': participants_with_conversations,
            'participants_with_events': participants_with_events,
            'successful_processing': successful_count,
            'errors_count': len(all_errors)
        }

        log_file = os.path.join(
            os.path.dirname(self.cleaned_signals_folder),
            f"event_extraction_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        try:
            with open(log_file, 'w') as f:
                json.dump(self.processing_log, f, indent=2)
            print(f"\nProcessing log saved to: {log_file}")
        except Exception as e:
            print(f"Could not save processing log: {e}")

        print("\nEvent extraction complete")

def main():
    """Main execution function"""
    extractor = ConversationEventExtractor(
        CLEANED_SIGNALS_FOLDER,
        CONVERSATION_LOGS_FOLDER,
        RAW_BIOPAC_FOLDER,
        TIME_ADJUSTMENT_HOURS
    )

    extractor.process_all_participants()

if __name__ == "__main__":
    main()

# %%