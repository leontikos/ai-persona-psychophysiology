# %% biopac_data_quality_check.py

"""
Biometric Data Analysis Script for Processing Multiple Participants
Memory-optimized version that preserves all original functionality
"""

import bioread
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to save memory
import matplotlib.pyplot as plt
from scipy import signal
import os
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import re
import json
from matplotlib.patches import Rectangle
import glob
from pathlib import Path
import gc  # For garbage collection
from dotenv import load_dotenv

# Global variables
save_figs = True  # Set to True to save figures, False to only display them
time_adjustment_hours = 2  # Default adjustment to handle timezone differences

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
biodata_folder = os.path.join(DATA_ROOT, "psychophysiology_signal")
conversations_folder = os.path.join(DATA_ROOT, "conversation_logs")
output_folder = os.path.join(DATA_ROOT, "analysis_results")

# Memory optimization settings
BATCH_SIZE = 10  # Process participants in batches
FORCE_GC_EVERY_N = 5  # Force garbage collection every N participants

def clear_memory():
    """Force garbage collection and clear matplotlib figures"""
    plt.close('all')
    gc.collect()

def adjust_datetime(dt, hours=None):
    """
    Adjust a datetime object by a specified number of hours.

    Parameters:
    dt: datetime object to adjust
    hours: Number of hours to adjust by (if None, uses the global time_adjustment_hours)

    Returns:
    datetime: Adjusted datetime object
    """
    if dt is None:
        return None

    adjustment = hours if hours is not None else time_adjustment_hours
    return dt + timedelta(hours=adjustment)

def parse_event_marker(marker_str):
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

def extract_timestamps(data, adjust_hours=None):
    """
    Extract timestamp information from a bioread Datafile object and adjust for timezone differences.

    Parameters:
    data: bioread Datafile object
    adjust_hours: Optional number of hours to adjust timestamps by

    Returns:
    dict: Dictionary containing timestamp information
    """
    timestamps = {}

    # Get the earliest marker timestamp
    if hasattr(data, 'earliest_marker_created_at'):
        timestamps['start_time'] = adjust_datetime(data.earliest_marker_created_at, adjust_hours)

    # Extract event markers
    if hasattr(data, 'event_markers') and data.event_markers:
        timestamps['event_markers'] = []
        for i, marker in enumerate(data.event_markers):
            # Basic marker info
            marker_info = {
                'index': i,
                'raw_name': str(marker),
                'created_at': adjust_datetime(marker.date_created_utc, adjust_hours) if hasattr(marker, 'date_created_utc') else None,
                'sample_index': marker.sample_index if hasattr(marker, 'sample_index') else None
            }

            # Parse the marker string for more structured information
            parsed_info = parse_event_marker(str(marker))

            # Adjust any timestamps in parsed info
            if 'timestamp' in parsed_info:
                parsed_info['timestamp'] = adjust_datetime(parsed_info['timestamp'], adjust_hours)

            marker_info.update(parsed_info)
            timestamps['event_markers'].append(marker_info)

    # Calculate end time based on duration and start time
    if 'start_time' in timestamps and data.channels and len(data.channels) > 0:
        channel = data.channels[0]
        duration_seconds = len(channel.data) / channel.samples_per_second

        if isinstance(timestamps['start_time'], datetime):
            # Calculate end time by adding duration to start time
            try:
                duration_delta = timedelta(seconds=duration_seconds)
                timestamps['end_time'] = timestamps['start_time'] + duration_delta
            except Exception as e:
                print(f"Error calculating end time: {e}")

    # If channel has time_index attribute, get timing info
    for channel in data.channels:
        if hasattr(channel, 'time_index') and len(channel.time_index) > 0:
            timestamps['first_sample_time'] = channel.time_index[0]
            timestamps['last_sample_time'] = channel.time_index[-1]
            timestamps['sampling_interval'] = channel.time_index[1] - channel.time_index[0] if len(channel.time_index) > 1 else None
            break

    return timestamps

def print_timestamps(timestamps):
    """
    Print timestamp information in a readable format.

    Parameters:
    timestamps: Dictionary containing timestamp information

    Returns:
    None: Results are printed
    """
    print("\n===== Timestamp Information =====")

    if 'start_time' in timestamps:
        print(f"Recording start time: {timestamps['start_time'].strftime('%Y-%m-%d %H:%M:%S')}")

    if 'end_time' in timestamps:
        print(f"Recording end time: {timestamps['end_time'].strftime('%Y-%m-%d %H:%M:%S')}")

    if 'first_sample_time' in timestamps:
        print(f"First sample time: {timestamps['first_sample_time']:.3f} seconds")

    if 'last_sample_time' in timestamps:
        print(f"Last sample time: {timestamps['last_sample_time']:.3f} seconds")
        print(f"Recording duration: {timestamps['last_sample_time']:.3f} seconds")

    if 'sampling_interval' in timestamps:
        print(f"Sampling interval: {timestamps['sampling_interval']:.6f} seconds")
        print(f"Sampling rate: {1/timestamps['sampling_interval']:.2f} Hz")

    if 'event_markers' in timestamps and timestamps['event_markers']:
        print("\nEvent Markers:")
        for i, marker in enumerate(timestamps['event_markers']):
            print(f"  {i+1}. {'Segment ' + marker['segment'] if 'segment' in marker else 'Marker'}")

            if 'created_at' in marker and marker['created_at']:
                print(f"     Time: {marker['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")

            if 'sample_index' in marker and marker['sample_index'] is not None:
                print(f"     Sample index: {marker['sample_index']}")

            if 'device_id' in marker:
                print(f"     Device ID: {marker['device_id']}")

            if 'type' in marker:
                print(f"     Type: {marker['type']}")

def load_conversation_data(json_file):
    """
    Load conversation data from a JSON file.

    Parameters:
    json_file: Path to the JSON file

    Returns:
    dict: Dictionary containing conversation data and metadata
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading conversation data from {json_file}: {e}")
        return None

def extract_conversation_timestamps(conversation_data):
    """
    Extract conversation timestamps from conversation data.

    Parameters:
    conversation_data: Dictionary containing conversation data

    Returns:
    dict: Dictionary containing user and assistant timestamps
    """
    conversation_timestamps = {
        'user_messages': [],
        'assistant_messages': []
    }

    if not conversation_data or 'messages' not in conversation_data:
        return conversation_timestamps

    for message in conversation_data['messages']:
        try:
            # Check if message has required fields
            if 'role' not in message or 'timestamp_start' not in message:
                continue

            role = message['role']

            # Parse timestamps - remove timezone info to make naive datetime objects
            # This is to avoid timezone comparison issues
            start_time_str = message['timestamp_start'].split('+')[0]
            start_time = datetime.fromisoformat(start_time_str)

            # For user messages, start and end are the same
            if role == 'user':
                conversation_timestamps['user_messages'].append({
                    'content': message.get('content', ''),
                    'timestamp': start_time,
                    'tokens': message.get('tokens', 0)
                })
            # For assistant messages, capture both start and end
            elif role == 'assistant':
                end_time_str = message['timestamp_end'].split('+')[0]
                end_time = datetime.fromisoformat(end_time_str)
                conversation_timestamps['assistant_messages'].append({
                    'content': message.get('content', ''),
                    'timestamp_start': start_time,
                    'timestamp_end': end_time,
                    'tokens': message.get('tokens', 0)
                })
        except Exception as e:
            print(f"Error parsing message: {e}")
            continue

    return conversation_timestamps

def plot_channel_with_conversations(data, timestamps, conversation_timestamps, channel_name,
                                   output_path, participant_id, chatbot_version, study_date):
    """
    Plot a specific channel with conversation markers.

    Parameters:
    data: bioread Datafile object
    timestamps: Dictionary containing timestamp information
    conversation_timestamps: Dictionary containing conversation timestamps
    channel_name: Channel name to plot
    output_path: Path to save the plot
    participant_id: ID of the participant
    chatbot_version: Version of the chatbot used
    study_date: Date of the study

    Returns:
    None: Results are plotted and saved
    """
    # Find the channel to plot
    channel = None

    if channel_name:
        # Find channel by name
        for ch in data.channels:
            if channel_name in ch.name:
                channel = ch
                break

    if not channel and data.channels:
        # Use the first channel
        channel = data.channels[0]

    if not channel:
        print(f"No valid channel found matching '{channel_name}'")
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 6))

    # Make start_time timezone-naive if it has timezone info
    if hasattr(timestamps['start_time'], 'tzinfo') and timestamps['start_time'].tzinfo is not None:
        # Convert to naive datetime by removing timezone
        start_time = timestamps['start_time'].replace(tzinfo=None)
    else:
        start_time = timestamps['start_time']

    # Create time vector in real datetime
    sample_count = len(channel.data)
    time_delta = np.arange(sample_count) / channel.samples_per_second
    real_times = [start_time + timedelta(seconds=t) for t in time_delta]

    # Format study date for display
    formatted_study_date = study_date.strftime('%Y-%m-%d')

    # Plot the channel data
    ax.plot(real_times, channel.data)
    ax.set_title(f"Participant: {participant_id} | Channel: {channel.name} | Chatbot v{chatbot_version} | Date: {formatted_study_date}")
    ax.set_ylabel("Amplitude")

    # Calculate y-position for conversation markers
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min

    # Get marker information if available
    if 'event_markers' in timestamps:
        for marker in timestamps['event_markers']:
            if 'created_at' in marker and marker['created_at']:
                # Make marker time timezone-naive
                if hasattr(marker['created_at'], 'tzinfo') and marker['created_at'].tzinfo is not None:
                    marker_time = marker['created_at'].replace(tzinfo=None)
                else:
                    marker_time = marker['created_at']

                marker_label = f"Segment {marker['segment']}" if 'segment' in marker else "Marker"

                ax.axvline(x=marker_time, color='r', linestyle='--', alpha=0.7)

                # Add the marker label with timestamp
                time_str = marker_time.strftime('%H:%M:%S')
                marker_text = f"{marker_label}\n{time_str}"

                ax.text(marker_time, y_max - 0.05 * y_range, marker_text,
                        rotation=90, verticalalignment='top', color='r')

    # Add user message markers (blue)
    for msg in conversation_timestamps['user_messages']:
        timestamp = msg['timestamp']

        # Check if timestamp is within the time range of the recording
        # To avoid timezone issues, compare by time of day
        msg_time = timestamp.time()
        recording_start_time = start_time.time()
        recording_end_time = real_times[-1].time()

        # Skip if outside recording time range
        if msg_time < recording_start_time or msg_time > recording_end_time:
            continue

        # Adjust date to match recording date if needed
        if timestamp.date() != start_time.date():
            timestamp = timestamp.replace(
                year=start_time.year,
                month=start_time.month,
                day=start_time.day
            )

        ax.axvline(x=timestamp, color='blue', linestyle='-', alpha=0.5, linewidth=1)

        # Add message label
        time_str = timestamp.strftime('%H:%M:%S')
        label = f"User ({time_str})"

        # Position text with offset for visibility - Above the line
        # User labels now above the line (visually to the left when rotated)
        ax.text(timestamp, y_max, label, rotation=90,
                verticalalignment='top', horizontalalignment='right',
                color='blue', fontsize=8)

    # Add assistant message markers (green for start, orange for end)
    for msg in conversation_timestamps['assistant_messages']:
        start_time_msg = msg['timestamp_start']
        end_time_msg = msg['timestamp_end']

        # Check if timestamps are within the time range of the recording
        msg_start_time = start_time_msg.time()
        msg_end_time = end_time_msg.time()
        recording_start_time = start_time.time()
        recording_end_time = real_times[-1].time()

        # Skip if outside recording time range
        if (msg_end_time < recording_start_time or
            msg_start_time > recording_end_time):
            continue

        # Adjust dates to match recording date if needed
        if start_time_msg.date() != start_time.date():
            start_time_msg = start_time_msg.replace(
                year=start_time.year,
                month=start_time.month,
                day=start_time.day
            )

        if end_time_msg.date() != start_time.date():
            end_time_msg = end_time_msg.replace(
                year=start_time.year,
                month=start_time.month,
                day=start_time.day
            )

        # Start time marker (green)
        ax.axvline(x=start_time_msg, color='green', linestyle='-', alpha=0.5, linewidth=1)

        # End time marker (orange)
        ax.axvline(x=end_time_msg, color='orange', linestyle='-', alpha=0.5, linewidth=1)

        # Add message duration highlight (light green background)
        # Convert datetime to matplotlib's numeric date format for rectangle
        start_num = mdates.date2num(start_time_msg)
        end_num = mdates.date2num(end_time_msg)
        width_num = end_num - start_num

        duration_rect = Rectangle(
            (start_num, y_min),
            width_num,  # Width in matplotlib's date units
            y_range,
            facecolor='lightgreen',
            alpha=0.1
        )
        ax.add_patch(duration_rect)

        # Add message labels with improved positioning
        # Start time label - Above the line, to the left
        start_time_str = start_time_msg.strftime('%H:%M:%S')
        start_label = f"Asst start ({start_time_str})"
        ax.text(start_time_msg, y_min, start_label, rotation=90,
                verticalalignment='bottom', horizontalalignment='right',
                color='green', fontsize=8)

        # End time label - Below the line, to the right
        end_time_str = end_time_msg.strftime('%H:%M:%S')
        end_label = f"Asst end ({end_time_str})"
        ax.text(end_time_msg, y_min, end_label, rotation=90,
                verticalalignment='bottom', horizontalalignment='left',
                color='orange', fontsize=8)

    # Format the x-axis for time
    plt.gcf().autofmt_xdate()

    # Add major ticks every 5 minutes and minor ticks every minute
    major_locator = mdates.MinuteLocator(byminute=range(0, 60, 5))
    minor_locator = mdates.MinuteLocator(byminute=range(0, 60, 1))

    formatter = mdates.DateFormatter('%H:%M:%S')

    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel("Time (HH:MM:SS)")

    # Add legend for conversation markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='r', linestyle='--', label='Segment Marker'),
        Line2D([0], [0], color='blue', label='User Message'),
        Line2D([0], [0], color='green', label='Assistant Start'),
        Line2D([0], [0], color='orange', label='Assistant End'),
        Rectangle((0, 0), 1, 1, fc='lightgreen', alpha=0.3, label='Assistant Response Duration')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.03), ncol=5)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)  # Make room for datetime labels

    # Save figure
    if save_figs:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {output_path}")

    plt.close(fig)  # Close to free memory

def plot_time_aligned_view(main_data, baseline_data, main_timestamps, baseline_timestamps,
                          channel_name, output_path, participant_id, chatbot_version, study_date):
    """
    Plot a time-aligned view of baseline and main recordings.

    Parameters:
    main_data: bioread Datafile object for main recording
    baseline_data: bioread Datafile object for baseline recording
    main_timestamps: Dictionary containing main file timestamp information
    baseline_timestamps: Dictionary containing baseline file timestamp information
    channel_name: Channel name to plot
    output_path: Path to save the plot
    participant_id: ID of the participant
    chatbot_version: Version of the chatbot used
    study_date: Date of the study

    Returns:
    None: Results are plotted and saved
    """
    # Find the channel to plot
    main_channel = None
    baseline_channel = None

    if channel_name:
        # Find channels by name
        for ch in main_data.channels:
            if channel_name in ch.name:
                main_channel = ch
                break

        for ch in baseline_data.channels:
            if channel_name in ch.name:
                baseline_channel = ch
                break
    else:
        # Use the first channel
        main_channel = main_data.channels[0]
        baseline_channel = baseline_data.channels[0]

    if not main_channel or not baseline_channel:
        print(f"Cannot find matching channels with name: {channel_name}")
        return

    # Format study date for display
    formatted_study_date = study_date.strftime('%Y-%m-%d')

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 6))

    # Make start_time timezone-naive if it has timezone info
    main_start = main_timestamps['start_time']
    baseline_start = baseline_timestamps['start_time']

    if hasattr(main_start, 'tzinfo') and main_start.tzinfo is not None:
        main_start = main_start.replace(tzinfo=None)

    if hasattr(baseline_start, 'tzinfo') and baseline_start.tzinfo is not None:
        baseline_start = baseline_start.replace(tzinfo=None)

    # Create time vector in real datetime
    main_sample_count = len(main_channel.data)
    baseline_sample_count = len(baseline_channel.data)

    main_time_delta = np.arange(main_sample_count) / main_channel.samples_per_second
    baseline_time_delta = np.arange(baseline_sample_count) / baseline_channel.samples_per_second

    main_times = [main_start + timedelta(seconds=t) for t in main_time_delta]
    baseline_times = [baseline_start + timedelta(seconds=t) for t in baseline_time_delta]

    # Plot both datasets
    ax.plot(main_times, main_channel.data, label=f"Main - {main_channel.name}")
    ax.plot(baseline_times, baseline_channel.data, label=f"Baseline - {baseline_channel.name}")

    ax.set_title(f"Participant: {participant_id} | Channel: {channel_name} | Chatbot v{chatbot_version} | Date: {formatted_study_date}")

    # Add markers with timestamps for main file
    if 'event_markers' in main_timestamps:
        for marker in main_timestamps['event_markers']:
            if 'created_at' in marker and marker['created_at']:
                marker_time = marker['created_at']
                if hasattr(marker_time, 'tzinfo') and marker_time.tzinfo is not None:
                    marker_time = marker_time.replace(tzinfo=None)

                ax.axvline(x=marker_time, color='r', linestyle='--', alpha=0.7)

                # Add label with timestamp
                time_str = marker_time.strftime('%H:%M:%S')
                label = f"Main: Segment {marker['segment']}\n{time_str}" if 'segment' in marker else f"Main Marker\n{time_str}"

                ax.text(marker_time, ax.get_ylim()[1] * 0.9, label,
                        rotation=90, verticalalignment='top', color='r', fontsize=8)

    # Add markers with timestamps for baseline file
    if 'event_markers' in baseline_timestamps:
        for marker in baseline_timestamps['event_markers']:
            if 'created_at' in marker and marker['created_at']:
                marker_time = marker['created_at']
                if hasattr(marker_time, 'tzinfo') and marker_time.tzinfo is not None:
                    marker_time = marker_time.replace(tzinfo=None)

                ax.axvline(x=marker_time, color='g', linestyle='--', alpha=0.7)

                # Add label with timestamp
                time_str = marker_time.strftime('%H:%M:%S')
                label = f"Baseline: Segment {marker['segment']}\n{time_str}" if 'segment' in marker else f"Baseline Marker\n{time_str}"

                ax.text(marker_time, ax.get_ylim()[1] * 0.85, label,
                        rotation=90, verticalalignment='top', color='g', fontsize=8)

    # Format the x-axis for time
    plt.gcf().autofmt_xdate()

    # Add major ticks every 5 minutes and minor ticks every minute
    major_locator = mdates.MinuteLocator(byminute=range(0, 60, 5))
    minor_locator = mdates.MinuteLocator(byminute=range(0, 60, 1))

    formatter = mdates.DateFormatter('%H:%M:%S')

    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel("Time (HH:MM:SS)")
    ax.set_ylabel("Amplitude")
    ax.legend()

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)  # Make room for datetime labels

    # Save figure
    if save_figs:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved time-aligned view to {output_path}")

    plt.close(fig)  # Close to free memory

def analyze_acq_file(filepath):
    """
    Analyze a Biopac .acq file, checking for data integrity.

    Parameters:
    filepath (str): Path to the .acq file

    Returns:
    data: bioread Datafile object
    """
    print(f"\nAnalyzing file: {os.path.basename(filepath)}")

    try:
        # Load the data
        data = bioread.read(filepath)
        print(f"Successfully loaded file with {len(data.channels)} channels")

        # Check basic file properties
        if data.channels and len(data.channels) > 0:
            sample_count = len(data.channels[0].data)
            sample_rate = data.channels[0].samples_per_second
            duration = sample_count / sample_rate if sample_rate > 0 else 0
            print(f"File duration: {duration:.2f} seconds")
            print(f"Sample rate: {sample_rate} Hz")

        # Print channel information
        print("Channel Information:")
        for i, channel in enumerate(data.channels):
            print(f"  {i+1}. {channel.name} - {channel.samples_per_second} Hz - {len(channel.data)} samples")

        return data

    except Exception as e:
        print(f"Error analyzing file: {e}")
        return None

def find_matching_log_for_participant(participant_id, logs_folder):
    """
    Find the most recent conversation log for a participant.

    Parameters:
    participant_id: ID of the participant
    logs_folder: Folder containing log files

    Returns:
    tuple: (log_file_path, log_data) or (None, None) if not found
    """
    # Get all JSON files in the logs folder
    json_files = glob.glob(os.path.join(logs_folder, "*.json"))

    matching_logs = []

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check if this log belongs to the participant
            if 'metadata' in data and 'user_id' in data['metadata'] and data['metadata']['user_id'] == participant_id:
                # Add to matching logs with timestamp
                timestamp_str = data.get('metadata', {}).get('last_updated', '')

                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.split('+')[0])
                        matching_logs.append((json_file, data, timestamp))
                    except (ValueError, TypeError):
                        # If timestamp parsing fails, still include the log but with no timestamp
                        matching_logs.append((json_file, data, None))
                else:
                    # If no timestamp in metadata, check first message timestamp
                    if 'messages' in data and len(data['messages']) > 0:
                        msg_time_str = data['messages'][0].get('timestamp_start', '')
                        if msg_time_str:
                            try:
                                timestamp = datetime.fromisoformat(msg_time_str.split('+')[0])
                                matching_logs.append((json_file, data, timestamp))
                                continue
                            except (ValueError, TypeError):
                                pass

                    # If all else fails, use file modification time
                    file_mtime = os.path.getmtime(json_file)
                    timestamp = datetime.fromtimestamp(file_mtime)
                    matching_logs.append((json_file, data, timestamp))

        except Exception as e:
            print(f"Error reading {json_file}: {e}")

    if not matching_logs:
        print(f"No matching logs found for participant {participant_id}")
        return None, None

    # Sort by timestamp (most recent first) and return the most recent
    matching_logs.sort(key=lambda x: x[2] if x[2] is not None else datetime.min, reverse=True)
    return matching_logs[0][0], matching_logs[0][1]

def process_participant_data(participant_id):
    """
    Process all data for a single participant.

    Parameters:
    participant_id: ID of the participant

    Returns:
    bool: True if successful, False otherwise
    """
    print(f"\n\n{'='*80}")
    print(f"PROCESSING PARTICIPANT: {participant_id}")
    print(f"{'='*80}")

    # Create output directory for this participant
    participant_output_dir = os.path.join(output_folder, participant_id)
    os.makedirs(participant_output_dir, exist_ok=True)

    # Find main and baseline files for this participant
    main_file_pattern = os.path.join(biodata_folder, f"{participant_id}_main.acq")
    baseline_file_pattern = os.path.join(biodata_folder, f"{participant_id}_baseline.acq")

    # Use glob to handle potential variations in filenames
    main_files = glob.glob(main_file_pattern)
    baseline_files = glob.glob(baseline_file_pattern)

    if not main_files:
        print(f"No main recording file found for participant {participant_id}")
        return False

    if not baseline_files:
        print(f"No baseline recording file found for participant {participant_id}")
        return False

    main_file_path = main_files[0]
    baseline_file_path = baseline_files[0]

    print(f"Found main file: {os.path.basename(main_file_path)}")
    print(f"Found baseline file: {os.path.basename(baseline_file_path)}")

    # Find matching conversation log
    log_file_path, conversation_data = find_matching_log_for_participant(participant_id, conversations_folder)

    if not log_file_path or not conversation_data:
        print(f"No conversation log found for participant {participant_id}")
        return False

    print(f"Found conversation log: {os.path.basename(log_file_path)}")

    # Extract metadata from conversation log
    try:
        metadata = conversation_data['metadata']
        chatbot_version = metadata.get('chatbot_version', 'unknown')

        # Extract study date from the log file (use timestamp of first message)
        study_date = None
        if 'messages' in conversation_data and len(conversation_data['messages']) > 0:
            timestamp_str = conversation_data['messages'][0].get('timestamp_start', '')
            if timestamp_str:
                study_date = datetime.fromisoformat(timestamp_str.split('+')[0])

        # If no date found in messages, use the last_updated field
        if not study_date and 'last_updated' in metadata:
            study_date = datetime.fromisoformat(metadata['last_updated'].split('+')[0])

        # If still no date, use current date
        if not study_date:
            study_date = datetime.now()

        print(f"Chatbot version: {chatbot_version}")
        print(f"Study date: {study_date.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"Error extracting metadata: {e}")
        # Use defaults if metadata extraction fails
        chatbot_version = 'unknown'
        study_date = datetime.now()

    try:
        # Analyze the main and baseline files
        main_data = analyze_acq_file(main_file_path)
        baseline_data = analyze_acq_file(baseline_file_path)

        if not main_data or not baseline_data:
            print("Failed to analyze one or both files")
            return False

        # Extract timestamps from both files
        main_timestamps = extract_timestamps(main_data)
        baseline_timestamps = extract_timestamps(baseline_data)

        # Print timestamp information
        print("\n===== Main File Timestamp Information =====")
        print_timestamps(main_timestamps)

        print("\n===== Baseline File Timestamp Information =====")
        print_timestamps(baseline_timestamps)

        # Extract conversation timestamps
        conversation_timestamps = extract_conversation_timestamps(conversation_data)
        print(f"\nExtracted {len(conversation_timestamps['user_messages'])} user messages and {len(conversation_timestamps['assistant_messages'])} assistant messages")

        # Plot each channel individually with conversation markers
        for channel in main_data.channels:
            # Skip non-data channels or marker channels
            if "marker" in channel.name.lower() or len(channel.data) < 10:
                continue

            channel_name = channel.name
            output_filename = f"{participant_id}_{channel_name}_with_conversations.png"
            output_path = os.path.join(participant_output_dir, output_filename)

            print(f"Plotting channel {channel_name} with conversation markers...")
            plot_channel_with_conversations(
                main_data,
                main_timestamps,
                conversation_timestamps,
                channel_name,
                output_path,
                participant_id,
                chatbot_version,
                study_date
            )

        # Create time-aligned views for important channels (EDA, HR, etc.)
        important_channels = ["EDA", "ECG", "HR", "RSP", "RESP"]

        for channel_name in important_channels:
            # Check if this channel exists in both recordings
            has_channel = False
            for ch in main_data.channels:
                if channel_name in ch.name:
                    has_channel = True
                    break

            if not has_channel:
                continue

            output_filename = f"{participant_id}_{channel_name}_time_aligned.png"
            output_path = os.path.join(participant_output_dir, output_filename)

            print(f"Creating time-aligned view of {channel_name} data...")
            plot_time_aligned_view(
                main_data,
                baseline_data,
                main_timestamps,
                baseline_timestamps,
                channel_name,
                output_path,
                participant_id,
                chatbot_version,
                study_date
            )

        print(f"\nCompleted processing for participant {participant_id}")

        # Clean up large objects
        del main_data
        del baseline_data
        del conversation_data
        gc.collect()

        return True

    except Exception as e:
        print(f"Error processing participant {participant_id}: {e}")
        return False
    finally:
        # Always clear memory after processing each participant
        clear_memory()

def find_all_participants():
    """
    Find all participants in the biodata folder.

    Returns:
    list: List of participant IDs
    """
    # Look for files matching the pattern *_main.acq or *_baseline.acq
    main_files = glob.glob(os.path.join(biodata_folder, "*_main.acq"))
    baseline_files = glob.glob(os.path.join(biodata_folder, "*_baseline.acq"))

    # Extract participant IDs from filenames
    participant_ids_from_main = [os.path.basename(f).replace("_main.acq", "") for f in main_files]
    participant_ids_from_baseline = [os.path.basename(f).replace("_baseline.acq", "") for f in baseline_files]

    # Find participants that have both files
    participant_ids = set(participant_ids_from_main).intersection(set(participant_ids_from_baseline))

    return sorted(list(participant_ids))

def collect_statistics(participant_ids):
    """
    Collect statistics across all participants.

    Parameters:
    participant_ids: List of participant IDs

    Returns:
    dict: Dictionary containing collected statistics
    """
    stats = {
        'total_participants': len(participant_ids),
        'successful_processing': 0,
        'chatbot_versions': {},
        'study_dates': {},
        'conversation_stats': {
            'avg_messages_per_participant': 0,
            'avg_user_messages': 0,
            'avg_assistant_messages': 0,
            'avg_tokens_per_conversation': 0,
            'max_messages': 0,
            'min_messages': float('inf') if participant_ids else 0
        },
        'recording_stats': {
            'avg_recording_duration': 0,
            'max_recording_duration': 0,
            'min_recording_duration': float('inf') if participant_ids else 0,
            'total_channels': set()
        }
    }

    total_messages = 0
    total_user_messages = 0
    total_assistant_messages = 0
    total_tokens = 0
    total_duration = 0
    successful_recordings = 0

    for participant_id in participant_ids:
        # Find log file for this participant
        log_file_path, conversation_data = find_matching_log_for_participant(participant_id, conversations_folder)

        if not log_file_path or not conversation_data:
            continue

        # Extract metadata
        metadata = conversation_data.get('metadata', {})
        chatbot_version = metadata.get('chatbot_version', 'unknown')

        # Count chatbot versions
        if chatbot_version in stats['chatbot_versions']:
            stats['chatbot_versions'][chatbot_version] += 1
        else:
            stats['chatbot_versions'][chatbot_version] = 1

        # Extract study date
        study_date = None
        if 'messages' in conversation_data and len(conversation_data['messages']) > 0:
            timestamp_str = conversation_data['messages'][0].get('timestamp_start', '')
            if timestamp_str:
                try:
                    study_date = datetime.fromisoformat(timestamp_str.split('+')[0])
                    date_str = study_date.strftime('%Y-%m-%d')

                    if date_str in stats['study_dates']:
                        stats['study_dates'][date_str] += 1
                    else:
                        stats['study_dates'][date_str] = 1
                except:
                    pass

        # Count messages
        if 'messages' in conversation_data:
            messages = conversation_data['messages']
            message_count = len(messages)
            total_messages += message_count

            # Update max/min messages
            stats['conversation_stats']['max_messages'] = max(stats['conversation_stats']['max_messages'], message_count)
            stats['conversation_stats']['min_messages'] = min(stats['conversation_stats']['min_messages'], message_count) if message_count > 0 else stats['conversation_stats']['min_messages']

            # Count user and assistant messages
            user_messages = [m for m in messages if m.get('role') == 'user']
            assistant_messages = [m for m in messages if m.get('role') == 'assistant']

            total_user_messages += len(user_messages)
            total_assistant_messages += len(assistant_messages)

            # Count tokens
            tokens = metadata.get('total_tokens', 0)
            total_tokens += tokens

        # Analyze recording file
        main_file_pattern = os.path.join(biodata_folder, f"{participant_id}_main.acq")
        main_files = glob.glob(main_file_pattern)

        if main_files:
            try:
                main_data = bioread.read(main_files[0])

                # Get recording duration
                if main_data.channels and len(main_data.channels) > 0:
                    channel = main_data.channels[0]
                    duration = len(channel.data) / channel.samples_per_second
                    total_duration += duration
                    successful_recordings += 1

                    # Update max/min duration
                    stats['recording_stats']['max_recording_duration'] = max(stats['recording_stats']['max_recording_duration'], duration)
                    stats['recording_stats']['min_recording_duration'] = min(stats['recording_stats']['min_recording_duration'], duration) if duration > 0 else stats['recording_stats']['min_recording_duration']

                # Collect unique channel names
                for channel in main_data.channels:
                    if not "marker" in channel.name.lower():
                        stats['recording_stats']['total_channels'].add(channel.name)

                # Clean up
                del main_data
                gc.collect()

            except:
                pass

        stats['successful_processing'] += 1

        # Clean up conversation data
        del conversation_data
        gc.collect()

    # Calculate averages
    if stats['successful_processing'] > 0:
        stats['conversation_stats']['avg_messages_per_participant'] = total_messages / stats['successful_processing']
        stats['conversation_stats']['avg_user_messages'] = total_user_messages / stats['successful_processing']
        stats['conversation_stats']['avg_assistant_messages'] = total_assistant_messages / stats['successful_processing']
        stats['conversation_stats']['avg_tokens_per_conversation'] = total_tokens / stats['successful_processing']

    if successful_recordings > 0:
        stats['recording_stats']['avg_recording_duration'] = total_duration / successful_recordings

    # Convert set to list for easier processing
    stats['recording_stats']['total_channels'] = list(stats['recording_stats']['total_channels'])

    return stats

def print_statistics(stats):
    """
    Print summary statistics.

    Parameters:
    stats: Dictionary containing collected statistics
    """
    print("\n\n" + "="*80)
    print("STUDY STATISTICS SUMMARY".center(80))
    print("="*80)

    print(f"\nPARTICIPANT STATISTICS:")
    print(f"  Total participants: {stats['total_participants']}")
    print(f"  Successfully processed: {stats['successful_processing']} ({stats['successful_processing']/stats['total_participants']*100:.1f}% of total)")

    print(f"\nCHATBOT VERSION DISTRIBUTION:")
    total_with_version = sum(stats['chatbot_versions'].values())
    for version, count in sorted(stats['chatbot_versions'].items()):
        print(f"  Version {version}: {count} participants ({count/total_with_version*100:.1f}%)")

    print(f"\nSTUDY DATE DISTRIBUTION:")
    for date, count in sorted(stats['study_dates'].items()):
        print(f"  {date}: {count} participants")

    print(f"\nCONVERSATION STATISTICS:")
    print(f"  Average messages per conversation: {stats['conversation_stats']['avg_messages_per_participant']:.1f}")
    print(f"  Average user messages: {stats['conversation_stats']['avg_user_messages']:.1f}")
    print(f"  Average assistant messages: {stats['conversation_stats']['avg_assistant_messages']:.1f}")
    print(f"  Average tokens per conversation: {stats['conversation_stats']['avg_tokens_per_conversation']:.1f}")
    print(f"  Message count range: {stats['conversation_stats']['min_messages']} - {stats['conversation_stats']['max_messages']}")

    print(f"\nRECORDING STATISTICS:")
    print(f"  Average recording duration: {stats['recording_stats']['avg_recording_duration']:.1f} seconds ({stats['recording_stats']['avg_recording_duration']/60:.1f} minutes)")
    print(f"  Recording duration range: {stats['recording_stats']['min_recording_duration']:.1f} - {stats['recording_stats']['max_recording_duration']:.1f} seconds")
    print(f"  Physiological channels recorded: {', '.join(sorted(stats['recording_stats']['total_channels']))}")

def generate_summary_report(stats, output_folder):
    """
    Generate a summary report file.

    Parameters:
    stats: Dictionary containing collected statistics
    output_folder: Folder to save the report
    """
    report_path = os.path.join(output_folder, "study_summary_report.txt")

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BIOPAC STUDY SUMMARY REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write(f"PARTICIPANT STATISTICS:\n")
        f.write(f"  Total participants: {stats['total_participants']}\n")
        f.write(f"  Successfully processed: {stats['successful_processing']} ({stats['successful_processing']/stats['total_participants']*100:.1f}% of total)\n\n")

        f.write(f"CHATBOT VERSION DISTRIBUTION:\n")
        total_with_version = sum(stats['chatbot_versions'].values())
        for version, count in sorted(stats['chatbot_versions'].items()):
            f.write(f"  Version {version}: {count} participants ({count/total_with_version*100:.1f}%)\n")
        f.write("\n")

        f.write(f"STUDY DATE DISTRIBUTION:\n")
        for date, count in sorted(stats['study_dates'].items()):
            f.write(f"  {date}: {count} participants\n")
        f.write("\n")

        f.write(f"CONVERSATION STATISTICS:\n")
        f.write(f"  Average messages per conversation: {stats['conversation_stats']['avg_messages_per_participant']:.1f}\n")
        f.write(f"  Average user messages: {stats['conversation_stats']['avg_user_messages']:.1f}\n")
        f.write(f"  Average assistant messages: {stats['conversation_stats']['avg_assistant_messages']:.1f}\n")
        f.write(f"  Average tokens per conversation: {stats['conversation_stats']['avg_tokens_per_conversation']:.1f}\n")
        f.write(f"  Message count range: {stats['conversation_stats']['min_messages']} - {stats['conversation_stats']['max_messages']}\n\n")

        f.write(f"RECORDING STATISTICS:\n")
        f.write(f"  Average recording duration: {stats['recording_stats']['avg_recording_duration']:.1f} seconds ({stats['recording_stats']['avg_recording_duration']/60:.1f} minutes)\n")
        f.write(f"  Recording duration range: {stats['recording_stats']['min_recording_duration']:.1f} - {stats['recording_stats']['max_recording_duration']:.1f} seconds\n")
        f.write(f"  Physiological channels recorded: {', '.join(sorted(stats['recording_stats']['total_channels']))}\n\n")

        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")

    print(f"\nSummary report saved to: {report_path}")

def create_comparative_visualizations(stats, output_folder):
    """
    Create comparative visualizations across chatbot versions.

    Parameters:
    stats: Dictionary containing collected statistics
    output_folder: Folder to save the visualizations
    """
    # Only create visualizations if we have data for multiple versions
    if len(stats['chatbot_versions']) <= 1:
        print("Not creating comparative visualizations (only one chatbot version found)")
        return

    try:
        # Create bar chart for chatbot versions
        plt.figure(figsize=(10, 6))
        versions = sorted(stats['chatbot_versions'].keys())
        counts = [stats['chatbot_versions'][v] for v in versions]

        bars = plt.bar(versions, counts, color='skyblue')

        # Add count labels above bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.0f}', ha='center', va='bottom')

        plt.title('Distribution of Participants by Chatbot Version')
        plt.xlabel('Chatbot Version')
        plt.ylabel('Number of Participants')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Save the figure
        version_chart_path = os.path.join(output_folder, "chatbot_version_distribution.png")
        plt.savefig(version_chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved chatbot version distribution chart to: {version_chart_path}")

        # Clear memory after creating visualization
        clear_memory()

    except Exception as e:
        print(f"Error creating comparative visualizations: {e}")

def main():
    """
    Main function to process all participant data.
    """
    print("Biometric Data Analysis Script (Memory Optimized)")
    print("=================================================")
    print(f"Biodata folder: {biodata_folder}")
    print(f"Conversations folder: {conversations_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Batch size: {BATCH_SIZE}")

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Find all participants
    participant_ids = find_all_participants()

    if not participant_ids:
        print("No participants found with both main and baseline recordings")
        return

    print(f"Found {len(participant_ids)} participants: {', '.join(participant_ids)}")

    # Process each participant in batches
    successful_count = 0
    total_participants = len(participant_ids)

    for i in range(0, total_participants, BATCH_SIZE):
        batch = participant_ids[i:i+BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (total_participants + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n{'='*80}")
        print(f"Processing batch {batch_num}/{total_batches} (participants {i+1}-{min(i+BATCH_SIZE, total_participants)})")
        print(f"{'='*80}")

        for j, participant_id in enumerate(batch):
            success = process_participant_data(participant_id)
            if success:
                successful_count += 1

            # Force garbage collection every few participants
            if (j + 1) % FORCE_GC_EVERY_N == 0:
                clear_memory()
                print(f"Memory cleared after processing {j+1} participants in this batch")

        # Clear memory after each batch
        clear_memory()
        print(f"Completed batch {batch_num}/{total_batches}. Memory cleared.")

    print("\n==============================")
    print(f"Processing complete: Successfully processed {successful_count}/{len(participant_ids)} participants")
    print(f"Results saved to {output_folder}")

    # Collect and print statistics
    print("\nGenerating study statistics...")
    stats = collect_statistics(participant_ids)
    print_statistics(stats)

    # Generate summary report
    generate_summary_report(stats, output_folder)

    # Create comparative visualizations for chatbot versions
    print("\nCreating comparative visualizations...")
    create_comparative_visualizations(stats, output_folder)

    # Final memory cleanup
    clear_memory()

if __name__ == "__main__":
    main()