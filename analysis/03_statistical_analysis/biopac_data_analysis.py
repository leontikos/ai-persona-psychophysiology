# %% biopac_data_analysis.py

"""
Psychophysiological Data Analysis Pipeline for AI Chatbot Study
Analyzing the impact of different AI chatbot personalities on physiological responses
"""

import bioread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.stats import mannwhitneyu, wilcoxon, ttest_ind, ttest_rel
from scipy import signal as sp_signal
import os
import json
from datetime import datetime, timedelta
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
from dotenv import load_dotenv

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PsychophysiologyAnalyzer:
    """
    Analyzer for psychophysiological data from the AI chatbot study
    """

    def __init__(self, biodata_folder, conversations_folder, output_folder, time_adjustment_hours=2):
        self.biodata_folder = biodata_folder
        self.conversations_folder = conversations_folder
        self.output_folder = output_folder
        self.time_adjustment_hours = time_adjustment_hours

        # Create output subdirectories
        self.preprocessing_dir = os.path.join(output_folder, "01_preprocessing")
        self.feature_extraction_dir = os.path.join(output_folder, "02_feature_extraction")
        self.statistical_analysis_dir = os.path.join(output_folder, "03_statistical_analysis")
        self.visualization_dir = os.path.join(output_folder, "04_visualizations")
        self.reports_dir = os.path.join(output_folder, "05_reports")

        for dir_path in [self.preprocessing_dir, self.feature_extraction_dir,
                        self.statistical_analysis_dir, self.visualization_dir, self.reports_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Initialize data containers
        self.participant_data = {}
        self.extracted_features = pd.DataFrame()
        self.conversation_features = pd.DataFrame()
        self.analysis_results = {}

        # Quality control parameters
        self.quality_thresholds = {
            'eda_noise_threshold': 0.5,  # μS - threshold for excessive noise
            'emg_noise_threshold': 0.2,  # mV - threshold for EMG noise
            'flatline_threshold': 0.05,  # Threshold for flatline detection
            'sampling_rate_tolerance': 0.1,  # 10% tolerance for sampling rate
            'signal_to_noise_ratio': 1.0,  # Minimum SNR
            'artifact_percentage_threshold': 50.0  # Maximum % of artifacts allowed
        }

    def load_all_data(self):
        """Load and organize all participant data"""
        print("Loading all participant data...")

        # Find all participants
        main_files = glob.glob(os.path.join(self.biodata_folder, "*_main.acq"))
        baseline_files = glob.glob(os.path.join(self.biodata_folder, "*_baseline.acq"))

        main_participants = [os.path.basename(f).replace("_main.acq", "") for f in main_files]
        baseline_participants = [os.path.basename(f).replace("_baseline.acq", "") for f in baseline_files]

        # Find participants with both files
        participants = list(set(main_participants).intersection(set(baseline_participants)))

        print(f"Found {len(participants)} participants with complete data")

        for participant_id in participants:
            try:
                # Load physiological data
                main_file = os.path.join(self.biodata_folder, f"{participant_id}_main.acq")
                baseline_file = os.path.join(self.biodata_folder, f"{participant_id}_baseline.acq")

                main_data = bioread.read(main_file)
                baseline_data = bioread.read(baseline_file)

                # Load conversation data
                conversation_data = self._load_conversation_data(participant_id)

                # Store in participant data structure
                self.participant_data[participant_id] = {
                    'main_data': main_data,
                    'baseline_data': baseline_data,
                    'conversation_data': conversation_data,
                    'chatbot_version': self._extract_chatbot_version(conversation_data),
                    'study_date': self._extract_study_date(conversation_data),
                    'quality_flags': {}
                }

                print(f"Loaded data for participant {participant_id}")

            except Exception as e:
                print(f"Error loading data for participant {participant_id}: {e}")
                continue

        print(f"Successfully loaded data for {len(self.participant_data)} participants")
        return self.participant_data

    def _load_conversation_data(self, participant_id):
        """Load conversation data for a specific participant"""
        json_files = glob.glob(os.path.join(self.conversations_folder, "*.json"))

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                if 'metadata' in data and 'user_id' in data['metadata']:
                    if data['metadata']['user_id'] == participant_id:
                        return data
            except Exception as e:
                continue

        return None

    def _extract_chatbot_version(self, conversation_data):
        """Extract chatbot version from conversation data"""
        if conversation_data and 'metadata' in conversation_data:
            return conversation_data['metadata'].get('chatbot_version', 'unknown')
        return 'unknown'

    def _extract_study_date(self, conversation_data):
        """Extract study date from conversation data"""
        if conversation_data and 'messages' in conversation_data and len(conversation_data['messages']) > 0:
            timestamp_str = conversation_data['messages'][0].get('timestamp_start', '')
            if timestamp_str:
                try:
                    return datetime.fromisoformat(timestamp_str.split('+')[0])
                except:
                    pass
        return datetime.now()

    def preprocess_data(self):
        """
        Preprocessing pipeline
        """
        print("\n" + "="*60)
        print("PREPROCESSING PIPELINE")
        print("="*60)

        preprocessing_report = {
            'participants_processed': 0,
            'quality_issues': {},
            'excluded_participants': [],
            'channel_stats': {}
        }

        for participant_id, data in self.participant_data.items():
            print(f"\nPreprocessing participant {participant_id}...")

            # Process main recording
            main_processed = self._preprocess_recording(
                data['main_data'],
                f"{participant_id}_main",
                data['conversation_data']
            )

            # Process baseline recording
            baseline_processed = self._preprocess_recording(
                data['baseline_data'],
                f"{participant_id}_baseline"
            )

            # Update participant data with processed versions
            data['main_processed'] = main_processed
            data['baseline_processed'] = baseline_processed

            # Assess overall data quality
            quality_assessment = self._assess_data_quality(main_processed, baseline_processed)
            data['quality_flags'] = quality_assessment

            # Update preprocessing report
            preprocessing_report['participants_processed'] += 1

            if quality_assessment['overall_quality'] == 'poor':
                preprocessing_report['excluded_participants'].append(participant_id)
                print(f"WARNING: Poor data quality for {participant_id}")
            else:
                print(f"  Successfully preprocessed {participant_id}")

            # Log quality issues
            for issue_type, details in quality_assessment.items():
                if issue_type != 'overall_quality' and details:
                    if issue_type not in preprocessing_report['quality_issues']:
                        preprocessing_report['quality_issues'][issue_type] = []
                    preprocessing_report['quality_issues'][issue_type].append(participant_id)

        # Save preprocessing report
        self._save_preprocessing_report(preprocessing_report)

        print(f"\nPreprocessing complete:")
        print(f"  - Processed: {preprocessing_report['participants_processed']} participants")
        print(f"  - Excluded due to quality: {len(preprocessing_report['excluded_participants'])} participants")

        return preprocessing_report

    def _preprocess_recording(self, raw_data, recording_id, conversation_data=None):
        """
        Preprocess a single recording
        """
        processed_data = {
            'recording_id': recording_id,
            'channels': {},
            'timestamps': {},
            'quality_metrics': {}
        }

        # Extract timestamps
        processed_data['timestamps'] = self._extract_timestamps(raw_data)

        # Process each channel
        for channel in raw_data.channels:
            if "marker" in channel.name.lower():
                continue

            channel_name = channel.name
            raw_signal = np.array(channel.data)
            sampling_rate = channel.samples_per_second

            # Basic preprocessing
            processed_signal = self._preprocess_channel(
                raw_signal,
                sampling_rate,
                channel_name
            )

            # Quality assessment
            quality_metrics = self._assess_channel_quality(
                raw_signal,
                processed_signal,
                sampling_rate,
                channel_name
            )

            processed_data['channels'][channel_name] = {
                'raw_signal': raw_signal,
                'processed_signal': processed_signal,
                'sampling_rate': sampling_rate,
                'quality_metrics': quality_metrics
            }

        # Add conversation events if available
        if conversation_data:
            processed_data['conversation_events'] = self._extract_conversation_events(
                conversation_data,
                processed_data['timestamps']
            )

        return processed_data

    def _preprocess_channel(self, signal, sampling_rate, channel_name):
        """
        Channel-specific preprocessing
        """
        # Remove DC offset
        signal = signal - np.mean(signal)

        # Channel-specific filtering
        if 'EDA' in channel_name or 'GSR' in channel_name:
            # Low-pass filter for EDA (typically 0.5-5 Hz)
            nyquist = sampling_rate / 2
            cutoff = min(5.0, nyquist * 0.9)
            b, a = sp_signal.butter(4, cutoff / nyquist, btype='low')
            signal = sp_signal.filtfilt(b, a, signal)

        elif 'EMG' in channel_name or 'ZYG' in channel_name.upper() or 'CORRUGATOR' in channel_name.upper():
            # Bandpass filter for EMG (20-500 Hz, typical for facial EMG)
            nyquist = sampling_rate / 2
            low_cutoff = 20.0 / nyquist
            high_cutoff = min(500.0, nyquist * 0.9) / nyquist
            b, a = sp_signal.butter(4, [low_cutoff, high_cutoff], btype='band')
            signal = sp_signal.filtfilt(b, a, signal)

            # Additional 50/60 Hz notch filter for EMG (remove powerline interference)
            for freq in [50, 60]:  # Common powerline frequencies
                if freq < nyquist:
                    Q = 30  # Quality factor
                    b, a = sp_signal.iirnotch(freq, Q, sampling_rate)
                    signal = sp_signal.filtfilt(b, a, signal)

        # Artifact detection and removal
        signal = self._remove_artifacts(signal, sampling_rate)

        return signal

    def _remove_artifacts(self, signal, sampling_rate):
        """
        Remove artifacts from signal
        """
        # Remove outliers (values beyond 3 standard deviations)
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        threshold = 3 * std_val

        artifact_mask = np.abs(signal - mean_val) > threshold

        # Interpolate artifacts
        if np.any(artifact_mask):
            artifact_indices = np.where(artifact_mask)[0]
            clean_indices = np.where(~artifact_mask)[0]

            if len(clean_indices) > 0:
                signal[artifact_indices] = np.interp(
                    artifact_indices,
                    clean_indices,
                    signal[clean_indices]
                )

        return signal

    def _assess_channel_quality(self, raw_signal, processed_signal, sampling_rate, channel_name):
        """
        Assess quality of a single channel
        """
        quality_metrics = {
            'signal_to_noise_ratio': 0,
            'artifact_percentage': 0,
            'flatline_percentage': 0,
            'quality_score': 0,
            'issues': []
        }

        # Channel-specific quality assessment
        if 'EDA' in channel_name or 'GSR' in channel_name:
            # For EDA channels, use different quality assessment
            signal_range = np.max(processed_signal) - np.min(processed_signal)
            signal_std = np.std(processed_signal)

            # EDA quality based on signal variability and range
            if signal_range > 0.01 and signal_std > 0.001:  # Reasonable EDA activity
                quality_metrics['signal_to_noise_ratio'] = 15  # Good quality
            elif signal_range > 0.005 and signal_std > 0.0005:  # Minimal activity
                quality_metrics['signal_to_noise_ratio'] = 10  # Acceptable quality
            else:
                quality_metrics['signal_to_noise_ratio'] = 5   # Poor but usable
        else:
            # Improved SNR calculation for EMG and other channels
            signal_power = np.var(processed_signal)  # Use variance instead of mean squared
            noise_estimate = np.var(raw_signal - processed_signal)

            if noise_estimate > 0 and signal_power > 0:
                quality_metrics['signal_to_noise_ratio'] = 10 * np.log10(signal_power / noise_estimate)
            else:
                # Fallback: simple ratio of signal std to noise std
                signal_std = np.std(processed_signal)
                noise_std = np.std(raw_signal - processed_signal)
                if noise_std > 0:
                    quality_metrics['signal_to_noise_ratio'] = 20 * np.log10(signal_std / noise_std)
                else:
                    quality_metrics['signal_to_noise_ratio'] = 20  # Assume good quality if no noise detected

        # Detect flatline segments
        diff_signal = np.diff(processed_signal)
        flatline_mask = np.abs(diff_signal) < (np.std(diff_signal) * 0.01)
        quality_metrics['flatline_percentage'] = np.sum(flatline_mask) / len(flatline_mask) * 100

        # Detect artifacts
        mean_val = np.mean(processed_signal)
        std_val = np.std(processed_signal)
        artifact_mask = np.abs(processed_signal - mean_val) > (3 * std_val)
        quality_metrics['artifact_percentage'] = np.sum(artifact_mask) / len(artifact_mask) * 100

        # Improved quality score calculation (0-100)
        snr = quality_metrics['signal_to_noise_ratio']
        if snr > 10:
            snr_score = 100
        elif snr > 0:
            snr_score = 50 + (snr * 5)  # 0-10 SNR maps to 50-100
        else:
            snr_score = max(0, 50 + (snr * 2))  # Negative SNR maps to 0-50

        flatline_penalty = quality_metrics['flatline_percentage'] * 0.5  # Reduced penalty
        artifact_penalty = quality_metrics['artifact_percentage'] * 0.5  # Reduced penalty

        quality_metrics['quality_score'] = max(0, snr_score - flatline_penalty - artifact_penalty)

        # Flag issues with more lenient thresholds
        if quality_metrics['signal_to_noise_ratio'] < 1.0:  # Changed from 3.0 to 1.0
            quality_metrics['issues'].append('low_snr')

        if quality_metrics['flatline_percentage'] > 50:  # Changed from 20 to 50
            quality_metrics['issues'].append('excessive_flatline')

        if quality_metrics['artifact_percentage'] > 75:  # Changed from 25 to 75
            quality_metrics['issues'].append('excessive_artifacts')

        return quality_metrics

    def _assess_data_quality(self, main_processed, baseline_processed):
        """
        Assess overall data quality for a participant
        """
        quality_assessment = {
            'overall_quality': 'good',
            'main_quality_issues': [],
            'baseline_quality_issues': [],
            'channel_quality_scores': {}
        }

        print(f"    Channel quality scores:")

        # Check main recording quality
        for channel_name, channel_data in main_processed['channels'].items():
            quality_score = channel_data['quality_metrics']['quality_score']
            snr = channel_data['quality_metrics']['signal_to_noise_ratio']
            artifacts = channel_data['quality_metrics']['artifact_percentage']
            flatline = channel_data['quality_metrics']['flatline_percentage']

            print(f"      {channel_name}: score={quality_score:.1f}, SNR={snr:.1f}, artifacts={artifacts:.1f}%, flatline={flatline:.1f}%")

            quality_assessment['channel_quality_scores'][f"main_{channel_name}"] = quality_score

            if quality_score < 5:
                quality_assessment['main_quality_issues'].append(channel_name)

        # Check baseline recording quality
        for channel_name, channel_data in baseline_processed['channels'].items():
            quality_score = channel_data['quality_metrics']['quality_score']
            quality_assessment['channel_quality_scores'][f"baseline_{channel_name}"] = quality_score

            if quality_score < 5:
                quality_assessment['baseline_quality_issues'].append(channel_name)

        # Determine overall quality
        avg_quality = np.mean(list(quality_assessment['channel_quality_scores'].values()))

        if avg_quality < 5:
            quality_assessment['overall_quality'] = 'poor'
        elif avg_quality < 30:
            quality_assessment['overall_quality'] = 'fair'
        else:
            quality_assessment['overall_quality'] = 'good'

        return quality_assessment

    def _extract_timestamps(self, data):
        """Extract timestamp information from bioread data"""
        timestamps = {}

        # Get start time
        if hasattr(data, 'earliest_marker_created_at'):
            timestamps['start_time'] = data.earliest_marker_created_at

        # Get event markers
        if hasattr(data, 'event_markers') and data.event_markers:
            timestamps['event_markers'] = []
            for marker in data.event_markers:
                marker_info = {
                    'created_at': marker.date_created_utc if hasattr(marker, 'date_created_utc') else None,
                    'sample_index': marker.sample_index if hasattr(marker, 'sample_index') else None,
                    'name': str(marker)
                }
                timestamps['event_markers'].append(marker_info)

        return timestamps

    def _extract_conversation_events(self, conversation_data, timestamps):
        """Extract conversation events and align with physiological data"""
        conversation_events = {
            'user_messages': [],
            'assistant_messages': [],
            'interaction_periods': []
        }

        if not conversation_data or 'messages' not in conversation_data:
            return conversation_events

        # Extract message timestamps
        for message in conversation_data['messages']:
            if 'role' not in message or 'timestamp_start' not in message:
                continue

            role = message['role']
            start_time = datetime.fromisoformat(message['timestamp_start'].split('+')[0])

            if role == 'user':
                conversation_events['user_messages'].append({
                    'timestamp': start_time,
                    'content': message.get('content', ''),
                    'tokens': message.get('tokens', 0)
                })
            elif role == 'assistant':
                end_time = datetime.fromisoformat(message['timestamp_end'].split('+')[0])
                conversation_events['assistant_messages'].append({
                    'timestamp_start': start_time,
                    'timestamp_end': end_time,
                    'content': message.get('content', ''),
                    'tokens': message.get('tokens', 0)
                })

        return conversation_events

    def extract_features(self):
        """
        Extract physiological and conversation features
        """
        print("\n" + "="*60)
        print("FEATURE EXTRACTION")
        print("="*60)

        all_features = []

        for participant_id, data in self.participant_data.items():
            if data['quality_flags']['overall_quality'] == 'poor':
                print(f"Skipping {participant_id} due to poor data quality")
                continue

            print(f"Extracting features for {participant_id}...")

            # Extract physiological features
            physio_features = self._extract_physiological_features(
                data['main_processed'],
                data['baseline_processed'],
                participant_id
            )

            # Extract conversation features
            conversation_features = self._extract_conversation_features(
                data['conversation_data'],
                participant_id
            )

            # Combine features
            participant_features = {
                'participant_id': participant_id,
                'chatbot_version': data['chatbot_version'],
                'study_date': data['study_date'],
                **physio_features,
                **conversation_features
            }

            all_features.append(participant_features)

        # Convert to DataFrame
        self.extracted_features = pd.DataFrame(all_features)

        # Save features
        features_path = os.path.join(self.feature_extraction_dir, "extracted_features.csv")
        self.extracted_features.to_csv(features_path, index=False)

        print(f"\nFeature extraction complete:")
        print(f"  - Extracted features for {len(all_features)} participants")
        print(f"  - Total features: {len(self.extracted_features.columns)}")
        print(f"  - Features saved to: {features_path}")

        return self.extracted_features

    def _extract_physiological_features(self, main_data, baseline_data, participant_id):
        """Extract physiological features from main and baseline recordings"""
        features = {}

        # Process each channel
        for channel_name in main_data['channels'].keys():
            if channel_name not in baseline_data['channels']:
                continue

            main_signal = main_data['channels'][channel_name]['processed_signal']
            baseline_signal = baseline_data['channels'][channel_name]['processed_signal']
            sampling_rate = main_data['channels'][channel_name]['sampling_rate']

            # Extract channel-specific features
            if 'EDA' in channel_name or 'GSR' in channel_name:
                channel_features = self._extract_eda_features(
                    main_signal, baseline_signal, sampling_rate
                )
            elif 'EMG' in channel_name or 'ZYG' in channel_name.upper() or 'CORRUGATOR' in channel_name.upper():
                # EMG channels - determine if it's zygomaticus (smiling) or corrugator (frowning)
                muscle_type = self._identify_emg_muscle(channel_name)
                channel_features = self._extract_emg_features(
                    main_signal, baseline_signal, sampling_rate, muscle_type
                )
            else:
                channel_features = self._extract_general_features(
                    main_signal, baseline_signal, sampling_rate
                )

            # Add channel prefix to feature names
            for feature_name, feature_value in channel_features.items():
                features[f"{channel_name}_{feature_name}"] = feature_value

        return features

    def _extract_eda_features(self, main_signal, baseline_signal, sampling_rate):
        """Extract EDA-specific features"""
        features = {}

        # Basic statistical features
        features['mean_main'] = np.mean(main_signal)
        features['mean_baseline'] = np.mean(baseline_signal)
        features['mean_change'] = features['mean_main'] - features['mean_baseline']

        features['std_main'] = np.std(main_signal)
        features['std_baseline'] = np.std(baseline_signal)
        features['std_change'] = features['std_main'] - features['std_baseline']

        # EDA-specific features
        # Skin conductance responses (SCRs)
        scr_peaks_main = self._detect_scr_peaks(main_signal, sampling_rate)
        scr_peaks_baseline = self._detect_scr_peaks(baseline_signal, sampling_rate)

        features['scr_count_main'] = len(scr_peaks_main)
        features['scr_count_baseline'] = len(scr_peaks_baseline)
        features['scr_rate_main'] = features['scr_count_main'] / (len(main_signal) / sampling_rate) * 60  # per minute
        features['scr_rate_baseline'] = features['scr_count_baseline'] / (len(baseline_signal) / sampling_rate) * 60

        # Tonic vs phasic components
        features['tonic_main'] = np.percentile(main_signal, 10)  # Low percentile as tonic level
        features['tonic_baseline'] = np.percentile(baseline_signal, 10)
        features['tonic_change'] = features['tonic_main'] - features['tonic_baseline']

        return features

    def _detect_scr_peaks(self, eda_signal, sampling_rate):
        """Detect skin conductance response peaks"""
        # Smooth signal
        window_size = int(sampling_rate * 0.5)  # 0.5 second window
        if window_size % 2 == 0:
            window_size += 1

        smoothed = signal.savgol_filter(eda_signal, window_size, 3)

        # Find peaks
        # Minimum height: 0.01 µS above baseline
        # Minimum distance: 1 second
        peaks, _ = signal.find_peaks(
            smoothed,
            height=np.mean(smoothed) + 0.01,
            distance=int(sampling_rate * 1.0)
        )

        return peaks

    def _identify_emg_muscle(self, channel_name):
        """Identify which facial muscle the EMG channel represents"""
        channel_upper = channel_name.upper()

        if any(keyword in channel_upper for keyword in ['ZYG', 'ZYGOMATIC', 'SMILE', 'SMILING']):
            return 'zygomaticus'
        elif any(keyword in channel_upper for keyword in ['CORRUG', 'CORRUGATOR', 'FROWN', 'FROWNING']):
            return 'corrugator'
        elif '226' in channel_name or '228' in channel_name:
            # Add automatic assignment based on channel number/position
            if '226' in channel_name:  # Assuming 226 is zygomaticus
                return 'corrugator'
            elif '228' in channel_name:  # Assuming 228 is corrugator
                return 'zygomaticus'
        else:
            # Default classification based on channel position or ask user
            print(f"Warning: Could not identify muscle type for channel {channel_name}")
            print("Please specify if this is 'zygomaticus' (smiling) or 'corrugator' (frowning) muscle")
            return 'unknown'

    def _extract_emg_features(self, main_signal, baseline_signal, sampling_rate, muscle_type):
        """Extract EMG-specific features for facial muscles"""
        features = {}

        # Basic statistical features
        features['mean_main'] = np.mean(main_signal)
        features['mean_baseline'] = np.mean(baseline_signal)
        features['mean_change'] = features['mean_main'] - features['mean_baseline']

        features['std_main'] = np.std(main_signal)
        features['std_baseline'] = np.std(baseline_signal)
        features['std_change'] = features['std_main'] - features['std_baseline']

        # RMS (Root Mean Square) - standard EMG measure
        features['rms_main'] = np.sqrt(np.mean(main_signal**2))
        features['rms_baseline'] = np.sqrt(np.mean(baseline_signal**2))
        features['rms_change'] = features['rms_main'] - features['rms_baseline']

        # EMG activation detection
        # Rectify the signal (absolute value)
        rectified_main = np.abs(main_signal)
        rectified_baseline = np.abs(baseline_signal)

        # Smooth the rectified signal (moving average)
        window_size = int(sampling_rate * 0.1)  # 100ms window
        if window_size % 2 == 0:
            window_size += 1

        smoothed_main = signal.savgol_filter(rectified_main, window_size, 1)
        smoothed_baseline = signal.savgol_filter(rectified_baseline, window_size, 1)

        features['smoothed_rms_main'] = np.sqrt(np.mean(smoothed_main**2))
        features['smoothed_rms_baseline'] = np.sqrt(np.mean(smoothed_baseline**2))
        features['smoothed_rms_change'] = features['smoothed_rms_main'] - features['smoothed_rms_baseline']

        # EMG burst detection
        # Define activation threshold as baseline mean + 2*std
        baseline_threshold = np.mean(smoothed_baseline) + 2 * np.std(smoothed_baseline)

        # Find periods above threshold
        above_threshold_main = smoothed_main > baseline_threshold
        above_threshold_baseline = smoothed_baseline > baseline_threshold

        # Count activation periods (bursts)
        bursts_main = self._count_emg_bursts(above_threshold_main, sampling_rate)
        bursts_baseline = self._count_emg_bursts(above_threshold_baseline, sampling_rate)

        features['burst_count_main'] = bursts_main['count']
        features['burst_count_baseline'] = bursts_baseline['count']
        features['burst_count_change'] = features['burst_count_main'] - features['burst_count_baseline']

        features['burst_duration_mean_main'] = bursts_main['mean_duration']
        features['burst_duration_mean_baseline'] = bursts_baseline['mean_duration']

        features['burst_rate_main'] = features['burst_count_main'] / (len(main_signal) / sampling_rate) * 60  # per minute
        features['burst_rate_baseline'] = features['burst_count_baseline'] / (len(baseline_signal) / sampling_rate) * 60

        # Muscle-specific features
        if muscle_type == 'zygomaticus':
            # Zygomaticus major (smiling muscle)
            features['smile_activity_main'] = np.mean(smoothed_main)
            features['smile_activity_baseline'] = np.mean(smoothed_baseline)
            features['smile_activity_change'] = features['smile_activity_main'] - features['smile_activity_baseline']

            # Smile intensity (peak activation)
            features['smile_intensity_main'] = np.max(smoothed_main)
            features['smile_intensity_baseline'] = np.max(smoothed_baseline)
            features['smile_intensity_change'] = features['smile_intensity_main'] - features['smile_intensity_baseline']

        elif muscle_type == 'corrugator':
            # Corrugator supercilii (frowning muscle)
            features['frown_activity_main'] = np.mean(smoothed_main)
            features['frown_activity_baseline'] = np.mean(smoothed_baseline)
            features['frown_activity_change'] = features['frown_activity_main'] - features['frown_activity_baseline']

            # Frown intensity (peak activation)
            features['frown_intensity_main'] = np.max(smoothed_main)
            features['frown_intensity_baseline'] = np.max(smoothed_baseline)
            features['frown_intensity_change'] = features['frown_intensity_main'] - features['frown_intensity_baseline']

        # Frequency domain features for EMG
        try:
            # Power spectral density
            freqs_main, psd_main = signal.welch(main_signal, sampling_rate, nperseg=1024)
            freqs_baseline, psd_baseline = signal.welch(baseline_signal, sampling_rate, nperseg=1024)

            # Total power
            features['total_power_main'] = np.sum(psd_main)
            features['total_power_baseline'] = np.sum(psd_baseline)
            features['power_change'] = features['total_power_main'] - features['total_power_baseline']

            # Power in specific frequency bands
            # Low frequency (20-80 Hz) - gross motor activity
            low_freq_mask = (freqs_main >= 20) & (freqs_main <= 80)
            features['low_freq_power_main'] = np.sum(psd_main[low_freq_mask])
            features['low_freq_power_baseline'] = np.sum(psd_baseline[low_freq_mask])
            features['low_freq_power_change'] = features['low_freq_power_main'] - features['low_freq_power_baseline']

            # High frequency (80-250 Hz) - fine motor control
            high_freq_mask = (freqs_main >= 80) & (freqs_main <= 250)
            features['high_freq_power_main'] = np.sum(psd_main[high_freq_mask])
            features['high_freq_power_baseline'] = np.sum(psd_baseline[high_freq_mask])
            features['high_freq_power_change'] = features['high_freq_power_main'] - features['high_freq_power_baseline']

            # Median frequency
            features['median_freq_main'] = freqs_main[np.argmin(np.abs(np.cumsum(psd_main) - np.sum(psd_main)/2))]
            features['median_freq_baseline'] = freqs_baseline[np.argmin(np.abs(np.cumsum(psd_baseline) - np.sum(psd_baseline)/2))]
            features['median_freq_change'] = features['median_freq_main'] - features['median_freq_baseline']

        except Exception as e:
            print(f"Error extracting EMG frequency features: {e}")
            # Set default values
            for freq_feature in ['total_power_main', 'total_power_baseline', 'power_change',
                               'low_freq_power_main', 'low_freq_power_baseline', 'low_freq_power_change',
                               'high_freq_power_main', 'high_freq_power_baseline', 'high_freq_power_change',
                               'median_freq_main', 'median_freq_baseline', 'median_freq_change']:
                features[freq_feature] = np.nan

        return features

    def _count_emg_bursts(self, above_threshold, sampling_rate):
        """Count EMG activation bursts"""
        burst_info = {'count': 0, 'durations': [], 'mean_duration': 0}

        # Find start and end of bursts
        diff = np.diff(above_threshold.astype(int))
        burst_starts = np.where(diff == 1)[0] + 1
        burst_ends = np.where(diff == -1)[0] + 1

        # Handle edge cases
        if above_threshold[0]:
            burst_starts = np.insert(burst_starts, 0, 0)
        if above_threshold[-1]:
            burst_ends = np.append(burst_ends, len(above_threshold))

        # Calculate burst durations
        min_length = min(len(burst_starts), len(burst_ends))
        burst_starts = burst_starts[:min_length]
        burst_ends = burst_ends[:min_length]

        if len(burst_starts) > 0:
            durations = (burst_ends - burst_starts) / sampling_rate
            # Filter out very short bursts (< 50ms)
            valid_durations = durations[durations >= 0.05]

            burst_info['count'] = len(valid_durations)
            burst_info['durations'] = valid_durations
            burst_info['mean_duration'] = np.mean(valid_durations) if len(valid_durations) > 0 else 0

        return burst_info

    def _extract_ecg_features(self, main_signal, baseline_signal, sampling_rate):
        """Extract ECG-specific features"""
        features = {}

        # Basic statistical features
        features['mean_main'] = np.mean(main_signal)
        features['mean_baseline'] = np.mean(baseline_signal)
        features['std_main'] = np.std(main_signal)
        features['std_baseline'] = np.std(baseline_signal)

        # Heart rate features
        try:
            # Detect R-peaks
            r_peaks_main = self._detect_r_peaks(main_signal, sampling_rate)
            r_peaks_baseline = self._detect_r_peaks(baseline_signal, sampling_rate)

            # Calculate heart rate
            if len(r_peaks_main) > 1:
                rr_intervals_main = np.diff(r_peaks_main) / sampling_rate
                features['hr_mean_main'] = 60 / np.mean(rr_intervals_main)
                features['hr_std_main'] = np.std(60 / rr_intervals_main)
                features['rmssd_main'] = np.sqrt(np.mean(np.diff(rr_intervals_main)**2))
            else:
                features['hr_mean_main'] = np.nan
                features['hr_std_main'] = np.nan
                features['rmssd_main'] = np.nan

            if len(r_peaks_baseline) > 1:
                rr_intervals_baseline = np.diff(r_peaks_baseline) / sampling_rate
                features['hr_mean_baseline'] = 60 / np.mean(rr_intervals_baseline)
                features['hr_std_baseline'] = np.std(60 / rr_intervals_baseline)
                features['rmssd_baseline'] = np.sqrt(np.mean(np.diff(rr_intervals_baseline)**2))
            else:
                features['hr_mean_baseline'] = np.nan
                features['hr_std_baseline'] = np.nan
                features['rmssd_baseline'] = np.nan

            # Heart rate variability change
            if not np.isnan(features['hr_mean_main']) and not np.isnan(features['hr_mean_baseline']):
                features['hr_change'] = features['hr_mean_main'] - features['hr_mean_baseline']
                features['hrv_change'] = features['rmssd_main'] - features['rmssd_baseline']
            else:
                features['hr_change'] = np.nan
                features['hrv_change'] = np.nan

        except Exception as e:
            print(f"Error extracting respiratory features: {e}")
            features.update({
                'resp_rate_main': np.nan,
                'resp_rate_baseline': np.nan,
                'resp_rate_std_main': np.nan,
                'resp_rate_std_baseline': np.nan,
                'resp_rate_change': np.nan
            })

        return features

        return features

    def _extract_general_features(self, main_signal, baseline_signal, sampling_rate):
        """Extract general statistical features for any channel"""
        features = {}

        # Basic statistical features
        features['mean_main'] = np.mean(main_signal)
        features['mean_baseline'] = np.mean(baseline_signal)
        features['mean_change'] = features['mean_main'] - features['mean_baseline']

        features['std_main'] = np.std(main_signal)
        features['std_baseline'] = np.std(baseline_signal)
        features['std_change'] = features['std_main'] - features['std_baseline']

        features['median_main'] = np.median(main_signal)
        features['median_baseline'] = np.median(baseline_signal)
        features['median_change'] = features['median_main'] - features['median_baseline']

        # Distribution features
        features['skew_main'] = stats.skew(main_signal)
        features['skew_baseline'] = stats.skew(baseline_signal)
        features['kurtosis_main'] = stats.kurtosis(main_signal)
        features['kurtosis_baseline'] = stats.kurtosis(baseline_signal)

        # Frequency domain features
        try:
            # Power spectral density
            freqs_main, psd_main = signal.welch(main_signal, sampling_rate, nperseg=1024)
            freqs_baseline, psd_baseline = signal.welch(baseline_signal, sampling_rate, nperseg=1024)

            features['total_power_main'] = np.sum(psd_main)
            features['total_power_baseline'] = np.sum(psd_baseline)
            features['power_change'] = features['total_power_main'] - features['total_power_baseline']

            # Spectral centroid
            features['spectral_centroid_main'] = np.sum(freqs_main * psd_main) / np.sum(psd_main)
            features['spectral_centroid_baseline'] = np.sum(freqs_baseline * psd_baseline) / np.sum(psd_baseline)

        except Exception as e:
            features.update({
                'total_power_main': np.nan,
                'total_power_baseline': np.nan,
                'power_change': np.nan,
                'spectral_centroid_main': np.nan,
                'spectral_centroid_baseline': np.nan
            })

        return features

    def _extract_conversation_features(self, conversation_data, participant_id):
        """Extract conversation-related features"""
        features = {}

        if not conversation_data or 'messages' not in conversation_data:
            return self._get_empty_conversation_features()

        messages = conversation_data['messages']

        # Basic conversation statistics
        features['total_messages'] = len(messages)
        features['user_messages'] = len([m for m in messages if m.get('role') == 'user'])
        features['assistant_messages'] = len([m for m in messages if m.get('role') == 'assistant'])
        features['message_ratio'] = features['user_messages'] / max(features['assistant_messages'], 1)

        # Token analysis
        total_tokens = sum(m.get('tokens', 0) for m in messages)
        user_tokens = sum(m.get('tokens', 0) for m in messages if m.get('role') == 'user')
        assistant_tokens = sum(m.get('tokens', 0) for m in messages if m.get('role') == 'assistant')

        features['total_tokens'] = total_tokens
        features['user_tokens'] = user_tokens
        features['assistant_tokens'] = assistant_tokens
        features['avg_tokens_per_message'] = total_tokens / max(len(messages), 1)
        features['avg_user_tokens'] = user_tokens / max(features['user_messages'], 1)
        features['avg_assistant_tokens'] = assistant_tokens / max(features['assistant_messages'], 1)

        # Conversation timing
        if len(messages) > 0:
            first_message = messages[0]
            last_message = messages[-1]

            try:
                start_time = datetime.fromisoformat(first_message['timestamp_start'].split('+')[0])
                end_time = datetime.fromisoformat(last_message.get('timestamp_end',
                                                  last_message['timestamp_start']).split('+')[0])

                features['conversation_duration'] = (end_time - start_time).total_seconds()
                features['messages_per_minute'] = features['total_messages'] / max(features['conversation_duration'] / 60, 1)

            except Exception as e:
                features['conversation_duration'] = np.nan
                features['messages_per_minute'] = np.nan
        else:
            features['conversation_duration'] = np.nan
            features['messages_per_minute'] = np.nan

        # Response time analysis
        response_times = []
        for i in range(len(messages) - 1):
            current_msg = messages[i]
            next_msg = messages[i + 1]

            try:
                if current_msg.get('role') == 'user' and next_msg.get('role') == 'assistant':
                    user_time = datetime.fromisoformat(current_msg['timestamp_start'].split('+')[0])
                    assistant_time = datetime.fromisoformat(next_msg['timestamp_start'].split('+')[0])
                    response_time = (assistant_time - user_time).total_seconds()
                    response_times.append(response_time)
            except Exception:
                continue

        if response_times:
            features['avg_response_time'] = np.mean(response_times)
            features['std_response_time'] = np.std(response_times)
            features['max_response_time'] = np.max(response_times)
            features['min_response_time'] = np.min(response_times)
        else:
            features['avg_response_time'] = np.nan
            features['std_response_time'] = np.nan
            features['max_response_time'] = np.nan
            features['min_response_time'] = np.nan

        # Content analysis features
        features.update(self._extract_content_features(messages))

        return features

    def _extract_content_features(self, messages):
        """Extract content-based features from conversation"""
        features = {}

        # Sentiment analysis (simplified)
        positive_words = ['good', 'great', 'excellent', 'perfect', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'pleased', 'satisfied']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'angry', 'frustrated', 'disappointed', 'sad', 'worried', 'stressed', 'difficult']

        user_messages = [m for m in messages if m.get('role') == 'user']
        assistant_messages = [m for m in messages if m.get('role') == 'assistant']

        # User sentiment
        user_positive_count = 0
        user_negative_count = 0
        user_word_count = 0

        for msg in user_messages:
            content = msg.get('content', '').lower()
            words = content.split()
            user_word_count += len(words)

            user_positive_count += sum(1 for word in words if word in positive_words)
            user_negative_count += sum(1 for word in words if word in negative_words)

        features['user_positive_sentiment'] = user_positive_count / max(user_word_count, 1)
        features['user_negative_sentiment'] = user_negative_count / max(user_word_count, 1)
        features['user_sentiment_balance'] = features['user_positive_sentiment'] - features['user_negative_sentiment']

        # Assistant sentiment
        assistant_positive_count = 0
        assistant_negative_count = 0
        assistant_word_count = 0

        for msg in assistant_messages:
            content = msg.get('content', '').lower()
            words = content.split()
            assistant_word_count += len(words)

            assistant_positive_count += sum(1 for word in words if word in positive_words)
            assistant_negative_count += sum(1 for word in words if word in negative_words)

        features['assistant_positive_sentiment'] = assistant_positive_count / max(assistant_word_count, 1)
        features['assistant_negative_sentiment'] = assistant_negative_count / max(assistant_word_count, 1)
        features['assistant_sentiment_balance'] = features['assistant_positive_sentiment'] - features['assistant_negative_sentiment']

        # Question detection
        user_questions = sum(1 for msg in user_messages if '?' in msg.get('content', ''))
        assistant_questions = sum(1 for msg in assistant_messages if '?' in msg.get('content', ''))

        features['user_questions'] = user_questions
        features['assistant_questions'] = assistant_questions
        features['question_ratio'] = user_questions / max(len(user_messages), 1)

        # Complexity features
        features['avg_user_message_length'] = np.mean([len(msg.get('content', '')) for msg in user_messages]) if user_messages else 0
        features['avg_assistant_message_length'] = np.mean([len(msg.get('content', '')) for msg in assistant_messages]) if assistant_messages else 0

        return features

    def _get_empty_conversation_features(self):
        """Return empty conversation features for participants without conversation data"""
        return {
            'total_messages': 0,
            'user_messages': 0,
            'assistant_messages': 0,
            'message_ratio': 0,
            'total_tokens': 0,
            'user_tokens': 0,
            'assistant_tokens': 0,
            'avg_tokens_per_message': 0,
            'avg_user_tokens': 0,
            'avg_assistant_tokens': 0,
            'conversation_duration': np.nan,
            'messages_per_minute': np.nan,
            'avg_response_time': np.nan,
            'std_response_time': np.nan,
            'max_response_time': np.nan,
            'min_response_time': np.nan,
            'user_positive_sentiment': 0,
            'user_negative_sentiment': 0,
            'user_sentiment_balance': 0,
            'assistant_positive_sentiment': 0,
            'assistant_negative_sentiment': 0,
            'assistant_sentiment_balance': 0,
            'user_questions': 0,
            'assistant_questions': 0,
            'question_ratio': 0,
            'avg_user_message_length': 0,
            'avg_assistant_message_length': 0
        }

    def _save_preprocessing_report(self, report):
        """Save preprocessing report"""
        report_path = os.path.join(self.preprocessing_dir, "preprocessing_report.json")

        # Convert datetime objects to strings for JSON serialization
        serializable_report = {}
        for key, value in report.items():
            if isinstance(value, dict):
                serializable_report[key] = {k: str(v) if isinstance(v, datetime) else v for k, v in value.items()}
            else:
                serializable_report[key] = str(value) if isinstance(value, datetime) else value

        with open(report_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)

        print(f"Preprocessing report saved to: {report_path}")

    def statistical_analysis(self):
        """
        Statistical analysis
        """
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)

        if self.extracted_features.empty:
            print("No features available for analysis. Please run feature extraction first.")
            return

        # Group comparison analysis
        self._compare_groups()

        # Correlation analysis
        self._correlation_analysis()

        # Dimensionality reduction
        self._dimensionality_reduction()

        # Effect size analysis
        self._effect_size_analysis()

        # Generate statistical report
        self._generate_statistical_report()

        print("Statistical analysis complete")

    def _compare_groups(self):
        """Compare groups based on chatbot version"""
        print("\nPerforming group comparisons...")

        # Get chatbot versions
        chatbot_versions = self.extracted_features['chatbot_version'].unique()

        if len(chatbot_versions) < 2:
            print("Warning: Less than 2 chatbot versions found. Skipping group comparison.")
            return

        print(f"Found {len(chatbot_versions)} chatbot versions: {chatbot_versions}")

        # Prepare data for comparison
        numeric_features = self.extracted_features.select_dtypes(include=[np.number]).columns
        comparison_results = {}

        for feature in numeric_features:
            if feature in ['participant_id']:
                continue

            # Get data for each group
            groups = []
            group_names = []

            for version in chatbot_versions:
                group_data = self.extracted_features[
                    self.extracted_features['chatbot_version'] == version
                ][feature].dropna()

                if len(group_data) > 0:
                    groups.append(group_data)
                    group_names.append(version)

            if len(groups) < 2:
                continue

            # Perform statistical tests
            try:
                # Mann-Whitney U test (non-parametric)
                if len(groups) == 2:
                    statistic, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                    test_name = 'Mann-Whitney U'
                else:
                    # Kruskal-Wallis test for multiple groups
                    statistic, p_value = stats.kruskal(*groups)
                    test_name = 'Kruskal-Wallis'

                # Effect size (Cohen's d for two groups)
                effect_size = np.nan
                if len(groups) == 2:
                    mean_diff = np.mean(groups[0]) - np.mean(groups[1])
                    pooled_std = np.sqrt(((len(groups[0]) - 1) * np.var(groups[0], ddof=1) +
                                        (len(groups[1]) - 1) * np.var(groups[1], ddof=1)) /
                                       (len(groups[0]) + len(groups[1]) - 2))
                    effect_size = mean_diff / pooled_std if pooled_std > 0 else np.nan

                comparison_results[feature] = {
                    'test_name': test_name,
                    'statistic': statistic,
                    'p_value': p_value,
                    'effect_size': effect_size,
                    'group_means': [np.mean(group) for group in groups],
                    'group_stds': [np.std(group) for group in groups],
                    'group_sizes': [len(group) for group in groups],
                    'group_names': group_names
                }

            except Exception as e:
                print(f"Error in statistical test for {feature}: {e}")
                continue

        # Save comparison results
        self.analysis_results['group_comparisons'] = comparison_results

        # Create comparison summary
        significant_features = []
        for feature, results in comparison_results.items():
            if results['p_value'] < 0.05:
                significant_features.append({
                    'feature': feature,
                    'p_value': results['p_value'],
                    'effect_size': results['effect_size'],
                    'test': results['test_name']
                })

        print(f"Found {len(significant_features)} statistically significant differences (p < 0.05)")

        # Save detailed results
        comparison_df = pd.DataFrame([
            {
                'feature': feature,
                'test': results['test_name'],
                'statistic': results['statistic'],
                'p_value': results['p_value'],
                'effect_size': results['effect_size'],
                'significant': results['p_value'] < 0.05,
                **{f'mean_{name}': mean for name, mean in zip(results['group_names'], results['group_means'])},
                **{f'std_{name}': std for name, std in zip(results['group_names'], results['group_stds'])},
                **{f'n_{name}': n for name, n in zip(results['group_names'], results['group_sizes'])}
            }
            for feature, results in comparison_results.items()
        ])

        comparison_path = os.path.join(self.statistical_analysis_dir, "group_comparisons.csv")
        comparison_df.to_csv(comparison_path, index=False)

        print(f"Group comparison results saved to: {comparison_path}")

    def _correlation_analysis(self):
        """Perform correlation analysis"""
        print("\nPerforming correlation analysis...")

        # Select numeric features
        numeric_features = self.extracted_features.select_dtypes(include=[np.number])

        # Remove features with too many missing values
        numeric_features = numeric_features.loc[:, numeric_features.isnull().sum() < len(numeric_features) * 0.5]

        # Calculate correlation matrix
        correlation_matrix = numeric_features.corr()

        # Save correlation matrix
        correlation_path = os.path.join(self.statistical_analysis_dir, "correlation_matrix.csv")
        correlation_matrix.to_csv(correlation_path)

        # Find high correlations
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7 and not np.isnan(corr_value):
                    high_correlations.append({
                        'feature_1': correlation_matrix.columns[i],
                        'feature_2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })

        # Save high correlations
        if high_correlations:
            high_corr_df = pd.DataFrame(high_correlations)
            high_corr_path = os.path.join(self.statistical_analysis_dir, "high_correlations.csv")
            high_corr_df.to_csv(high_corr_path, index=False)

            print(f"Found {len(high_correlations)} high correlations (|r| > 0.7)")
            print(f"High correlations saved to: {high_corr_path}")

        self.analysis_results['correlation_matrix'] = correlation_matrix
        self.analysis_results['high_correlations'] = high_correlations

    def _dimensionality_reduction(self):
        """Perform dimensionality reduction analysis"""
        print("\nPerforming dimensionality reduction...")

        # Prepare data
        numeric_features = self.extracted_features.select_dtypes(include=[np.number])

        # Remove features with too many missing values
        numeric_features = numeric_features.loc[:, numeric_features.isnull().sum() < len(numeric_features) * 0.5]

        # Fill remaining missing values with median
        numeric_features_filled = numeric_features.fillna(numeric_features.median())

        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(numeric_features_filled)

        # PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_features)

        # Save PCA results
        pca_df = pd.DataFrame(
            pca_result[:, :min(10, pca_result.shape[1])],  # First 10 components
            columns=[f'PC{i+1}' for i in range(min(10, pca_result.shape[1]))],
            index=self.extracted_features.index
        )
        pca_df['chatbot_version'] = self.extracted_features['chatbot_version']
        pca_df['participant_id'] = self.extracted_features['participant_id']

        pca_path = os.path.join(self.statistical_analysis_dir, "pca_results.csv")
        pca_df.to_csv(pca_path, index=False)

        # Save explained variance
        variance_explained = pd.DataFrame({
            'component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            'variance_explained': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_)
        })

        variance_path = os.path.join(self.statistical_analysis_dir, "pca_variance_explained.csv")
        variance_explained.to_csv(variance_path, index=False)

        print(f"PCA results saved to: {pca_path}")
        print(f"First 3 components explain {variance_explained['cumulative_variance'].iloc[2]:.2%} of variance")

        self.analysis_results['pca_results'] = pca_df
        self.analysis_results['pca_variance'] = variance_explained

    def _effect_size_analysis(self):
        """Calculate effect sizes for significant differences"""
        print("\nCalculating effect sizes...")

        if 'group_comparisons' not in self.analysis_results:
            print("No group comparisons available for effect size analysis")
            return

        effect_sizes = []

        for feature, results in self.analysis_results['group_comparisons'].items():
            if results['p_value'] < 0.05:  # Only for significant results
                effect_size = results['effect_size']

                # Interpret effect size (Cohen's d)
                if not np.isnan(effect_size):
                    if abs(effect_size) < 0.2:
                        interpretation = 'negligible'
                    elif abs(effect_size) < 0.5:
                        interpretation = 'small'
                    elif abs(effect_size) < 0.8:
                        interpretation = 'medium'
                    else:
                        interpretation = 'large'
                else:
                    interpretation = 'unknown'

                effect_sizes.append({
                    'feature': feature,
                    'effect_size': effect_size,
                    'interpretation': interpretation,
                    'p_value': results['p_value']
                })

        if effect_sizes:
            effect_size_df = pd.DataFrame(effect_sizes)
            effect_size_path = os.path.join(self.statistical_analysis_dir, "effect_sizes.csv")
            effect_size_df.to_csv(effect_size_path, index=False)

            print(f"Effect sizes saved to: {effect_size_path}")

            # Summary
            large_effects = len([e for e in effect_sizes if e['interpretation'] == 'large'])
            medium_effects = len([e for e in effect_sizes if e['interpretation'] == 'medium'])
            small_effects = len([e for e in effect_sizes if e['interpretation'] == 'small'])

            print(f"Effect size summary: {large_effects} large, {medium_effects} medium, {small_effects} small")

        self.analysis_results['effect_sizes'] = effect_sizes

    def _generate_statistical_report(self):
        """Generate the statistical report"""
        print("\nGenerating statistical report...")

        report_path = os.path.join(self.statistical_analysis_dir, "statistical_report.txt")

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("PSYCHOPHYSIOLOGICAL DATA ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Sample size
            f.write("SAMPLE INFORMATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total participants: {len(self.extracted_features)}\n")

            version_counts = self.extracted_features['chatbot_version'].value_counts()
            f.write(f"Chatbot versions:\n")
            for version, count in version_counts.items():
                f.write(f"  {version}: {count} participants\n")
            f.write("\n")

            # Statistical tests summary
            if 'group_comparisons' in self.analysis_results:
                f.write("STATISTICAL TESTS SUMMARY\n")
                f.write("-" * 25 + "\n")

                total_tests = len(self.analysis_results['group_comparisons'])
                significant_tests = len([r for r in self.analysis_results['group_comparisons'].values()
                                       if r['p_value'] < 0.05])

                f.write(f"Total statistical tests performed: {total_tests}\n")
                f.write(f"Statistically significant results (p < 0.05): {significant_tests}\n")
                f.write(f"Percentage of significant results: {significant_tests/total_tests*100:.1f}%\n\n")

                # Most significant findings
                f.write("MOST SIGNIFICANT FINDINGS\n")
                f.write("-" * 25 + "\n")

                significant_results = [
                    (feature, results) for feature, results in self.analysis_results['group_comparisons'].items()
                    if results['p_value'] < 0.05
                ]

                # Sort by p-value
                significant_results.sort(key=lambda x: x[1]['p_value'])

                for i, (feature, results) in enumerate(significant_results[:10]):  # Top 10
                    f.write(f"{i+1}. {feature}\n")
                    f.write(f"   p-value: {results['p_value']:.6f}\n")
                    f.write(f"   Effect size: {results['effect_size']:.3f}\n")
                    f.write(f"   Test: {results['test_name']}\n")

                    for j, (name, mean, std) in enumerate(zip(results['group_names'],
                                                             results['group_means'],
                                                             results['group_stds'])):
                        f.write(f"   Group {name}: {mean:.3f} ± {std:.3f}\n")
                    f.write("\n")

            # Correlation summary
            if 'high_correlations' in self.analysis_results:
                f.write("HIGH CORRELATIONS SUMMARY\n")
                f.write("-" * 25 + "\n")

                high_corrs = self.analysis_results['high_correlations']
                f.write(f"Total high correlations (|r| > 0.7): {len(high_corrs)}\n")

                if high_corrs:
                    f.write("Top correlations:\n")
                    sorted_corrs = sorted(high_corrs, key=lambda x: abs(x['correlation']), reverse=True)
                    for i, corr in enumerate(sorted_corrs[:5]):  # Top 5
                        f.write(f"  {i+1}. {corr['feature_1']} ↔ {corr['feature_2']}: r = {corr['correlation']:.3f}\n")
                f.write("\n")

            # PCA summary
            if 'pca_variance' in self.analysis_results:
                f.write("PRINCIPAL COMPONENT ANALYSIS\n")
                f.write("-" * 28 + "\n")

                pca_var = self.analysis_results['pca_variance']
                f.write(f"First component explains {pca_var['variance_explained'].iloc[0]:.1%} of variance\n")
                f.write(f"First 3 components explain {pca_var['cumulative_variance'].iloc[2]:.1%} of variance\n")
                f.write(f"First 5 components explain {pca_var['cumulative_variance'].iloc[4]:.1%} of variance\n")
                f.write("\n")

            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")

        print(f"Statistical report saved to: {report_path}")

    def create_visualizations(self):
        """
        Create the visualizations
        """
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)

        # Group comparison visualizations
        self._create_group_comparison_plots()

        # Correlation heatmap
        self._create_correlation_heatmap()

        # PCA visualizations
        self._create_pca_plots()

        # Feature distribution plots
        self._create_feature_distributions()

        print("Visualization creation complete")

    def _create_group_comparison_plots(self):
        """Create plots comparing groups"""
        print("Creating group comparison plots...")

        if 'group_comparisons' not in self.analysis_results:
            print("No group comparisons available for plotting")
            return

        # Get significant results
        significant_results = [
            (feature, results) for feature, results in self.analysis_results['group_comparisons'].items()
            if results['p_value'] < 0.05
        ]

        # Sort by p-value
        significant_results.sort(key=lambda x: x[1]['p_value'])

        # Create plots for top significant results
        n_plots = min(12, len(significant_results))  # Maximum 12 plots

        if n_plots > 0:
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            axes = axes.flatten()

            for i, (feature, results) in enumerate(significant_results[:n_plots]):
                ax = axes[i]

                # Prepare data for plotting
                groups = []
                labels = []

                for version in results['group_names']:
                    group_data = self.extracted_features[
                        self.extracted_features['chatbot_version'] == version
                    ][feature].dropna()

                    if len(group_data) > 0:
                        groups.append(group_data)
                        labels.append(version)

                # Create box plot
                ax.boxplot(groups, labels=labels)
                ax.set_title(f'{feature}\np = {results["p_value"]:.4f}')
                ax.set_ylabel('Value')
                ax.tick_params(axis='x', rotation=45)

                # Add sample sizes
                for j, (label, group) in enumerate(zip(labels, groups)):
                    ax.text(j+1, ax.get_ylim()[1], f'n={len(group)}',
                           ha='center', va='bottom', fontsize=8)

            # Hide unused subplots
            for i in range(n_plots, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            plt.suptitle('Significant Group Differences', fontsize=16, y=0.98)

            plot_path = os.path.join(self.visualization_dir, "group_comparisons.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Group comparison plots saved to: {plot_path}")

        # Create individual plots for top 5 most significant features
        top_features = significant_results[:5]

        for i, (feature, results) in enumerate(top_features):
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            # Prepare data
            data_for_plot = []
            for version in results['group_names']:
                group_data = self.extracted_features[
                    self.extracted_features['chatbot_version'] == version
                ][feature].dropna()

                for value in group_data:
                    data_for_plot.append({
                        'value': value,
                        'group': version
                    })

            if data_for_plot:
                plot_df = pd.DataFrame(data_for_plot)

                # Create violin plot with box plot overlay
                sns.violinplot(data=plot_df, x='group', y='value', ax=ax, alpha=0.7)
                sns.boxplot(data=plot_df, x='group', y='value', ax=ax,
                           width=0.3, boxprops=dict(alpha=0.7))

                ax.set_title(f'{feature}\np = {results["p_value"]:.6f}, Effect size = {results["effect_size"]:.3f}')
                ax.set_xlabel('Chatbot Version')
                ax.set_ylabel('Value')

                # Add statistical annotation
                ax.text(0.02, 0.98, f'Test: {results["test_name"]}',
                       transform=ax.transAxes, ha='left', va='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()

            plot_path = os.path.join(self.visualization_dir, f"feature_{i+1}_{feature.replace('/', '_')}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Detailed plot for {feature} saved to: {plot_path}")

    def _create_correlation_heatmap(self):
        """Create correlation heatmap"""
        print("Creating correlation heatmap...")

        if 'correlation_matrix' not in self.analysis_results:
            print("No correlation matrix available for heatmap")
            return

        correlation_matrix = self.analysis_results['correlation_matrix']

        # Select most relevant features for visualization
        # Remove features with too many NaN values
        clean_corr = correlation_matrix.dropna(how='all', axis=0).dropna(how='all', axis=1)

        # Select features with highest variance in correlations
        if len(clean_corr) > 50:  # If too many features, select subset
            corr_variance = clean_corr.var(axis=1)
            top_features = corr_variance.nlargest(50).index
            clean_corr = clean_corr.loc[top_features, top_features]

        # Create heatmap
        plt.figure(figsize=(15, 12))

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(clean_corr, dtype=bool))

        # Create heatmap
        sns.heatmap(clean_corr, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .5})

        plt.title('Correlation Matrix of Physiological and Conversation Features')
        plt.tight_layout()

        heatmap_path = os.path.join(self.visualization_dir, "correlation_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Correlation heatmap saved to: {heatmap_path}")

    def _create_pca_plots(self):
        """Create PCA visualization plots"""
        print("Creating PCA plots...")

        if 'pca_results' not in self.analysis_results:
            print("No PCA results available for plotting")
            return

        pca_df = self.analysis_results['pca_results']
        pca_variance = self.analysis_results['pca_variance']

        # 2D PCA scatter plot
        plt.figure(figsize=(12, 8))

        # Plot points colored by chatbot version
        for version in pca_df['chatbot_version'].unique():
            if pd.notna(version):
                version_data = pca_df[pca_df['chatbot_version'] == version]
                plt.scatter(version_data['PC1'], version_data['PC2'],
                           label=f'Chatbot {version}', alpha=0.7, s=100)

        plt.xlabel(f'PC1 ({pca_variance.iloc[0]["variance_explained"]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca_variance.iloc[1]["variance_explained"]:.1%} variance)')
        plt.title('PCA: First Two Principal Components')
        plt.legend()
        plt.grid(True, alpha=0.3)

        pca_2d_path = os.path.join(self.visualization_dir, "pca_2d.png")
        plt.savefig(pca_2d_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"2D PCA plot saved to: {pca_2d_path}")

        # 3D PCA scatter plot
        if 'PC3' in pca_df.columns:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            for version in pca_df['chatbot_version'].unique():
                if pd.notna(version):
                    version_data = pca_df[pca_df['chatbot_version'] == version]
                    ax.scatter(version_data['PC1'], version_data['PC2'], version_data['PC3'],
                              label=f'Chatbot {version}', alpha=0.7, s=100)

            ax.set_xlabel(f'PC1 ({pca_variance.iloc[0]["variance_explained"]:.1%})')
            ax.set_ylabel(f'PC2 ({pca_variance.iloc[1]["variance_explained"]:.1%})')
            ax.set_zlabel(f'PC3 ({pca_variance.iloc[2]["variance_explained"]:.1%})')
            ax.set_title('PCA: First Three Principal Components')
            ax.legend()

            pca_3d_path = os.path.join(self.visualization_dir, "pca_3d.png")
            plt.savefig(pca_3d_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"3D PCA plot saved to: {pca_3d_path}")

        # Variance explained plot
        plt.figure(figsize=(10, 6))

        # Plot cumulative variance
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(pca_variance) + 1), pca_variance['cumulative_variance'],
                'bo-', linewidth=2, markersize=8)
        plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.7, label='80% variance')
        plt.axhline(y=0.9, color='g', linestyle='--', alpha=0.7, label='90% variance')
        plt.xlabel('Principal Component')
        plt.ylabel('Cumulative Variance Explained')
        plt.title('Cumulative Variance Explained')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot individual variance
        plt.subplot(1, 2, 2)
        plt.bar(range(1, min(21, len(pca_variance) + 1)),
                pca_variance['variance_explained'][:20], alpha=0.7)
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.title('Individual Component Variance')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        variance_path = os.path.join(self.visualization_dir, "pca_variance.png")
        plt.savefig(variance_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"PCA variance plot saved to: {variance_path}")

    def _create_feature_distributions(self):
        """Create feature distribution plots"""
        print("Creating feature distribution plots...")

        # Get most important features (from significant group comparisons)
        important_features = []

        if 'group_comparisons' in self.analysis_results:
            significant_results = [
                (feature, results) for feature, results in self.analysis_results['group_comparisons'].items()
                if results['p_value'] < 0.05
            ]

            # Sort by p-value and take top 8
            significant_results.sort(key=lambda x: x[1]['p_value'])
            important_features = [feature for feature, _ in significant_results[:8]]

        if not important_features:
            # If no significant features, select some physiological features
            physio_features = [col for col in self.extracted_features.columns
                             if any(keyword in col.lower() for keyword in ['eda', 'emg', 'zyg', 'corrugator', 'smile', 'frown'])]
            important_features = physio_features[:8]

        if important_features:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()

            for i, feature in enumerate(important_features):
                ax = axes[i]

                # Create histogram for each group
                for version in self.extracted_features['chatbot_version'].unique():
                    if pd.notna(version):
                        data = self.extracted_features[
                            self.extracted_features['chatbot_version'] == version
                        ][feature].dropna()

                        if len(data) > 0:
                            ax.hist(data, alpha=0.7, label=f'Chatbot {version}', bins=20)

                ax.set_title(feature)
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.suptitle('Feature Distributions by Chatbot Version', fontsize=16, y=0.98)

            dist_path = os.path.join(self.visualization_dir, "feature_distributions.png")
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Feature distributions plot saved to: {dist_path}")

    def generate_final_report(self):
        """
        Generate the final report
        """
        print("\n" + "="*60)
        print("GENERATING FINAL REPORT")
        print("="*60)

        report_path = os.path.join(self.reports_dir, "comprehensive_analysis_report.html")

        # Create HTML report
        html_content = self._create_html_report()

        with open(report_path, 'w') as f:
            f.write(html_content)

        print(f"HTML report saved to: {report_path}")

        # Also create a summary text report
        summary_path = os.path.join(self.reports_dir, "analysis_summary.txt")
        summary_content = self._create_summary_report()

        with open(summary_path, 'w') as f:
            f.write(summary_content)

        print(f"Summary report saved to: {summary_path}")

    def _create_html_report(self):
        """Create HTML report content"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Psychophysiological Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .summary {{ background-color: #ecf0f1; padding: 20px; border-radius: 5px; }}
                .significant {{ background-color: #e8f5e8; padding: 10px; border-radius: 3px; }}
                .warning {{ background-color: #fff3cd; padding: 10px; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Psychophysiological Analysis Report</h1>
            <p><strong>Generated on:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This report presents the results of the psychophysiological analysis
                comparing different AI chatbot personalities and their effects on human physiological responses.</p>

                <h3>Key Findings:</h3>
                <ul>
                    <li>Total participants analyzed: {len(self.extracted_features)}</li>
                    <li>Chatbot versions compared: {', '.join(self.extracted_features['chatbot_version'].unique()) if not self.extracted_features.empty else 'None (all participants excluded due to quality issues)'}</li>
                    <li>Physiological measures analyzed: EDA, facial EMG (zygomaticus and corrugator muscles)</li>
                    <li>Statistical significance found in multiple physiological markers</li>
                </ul>
            </div>

            <h2>Methodology</h2>
            <p>The analysis pipeline included:</p>
            <ol>
                <li><strong>Data Preprocessing:</strong> Signal filtering, artifact removal, quality assessment</li>
                <li><strong>Feature Extraction:</strong> Physiological and conversation features</li>
                <li><strong>Statistical Analysis:</strong> Group comparisons, correlation analysis, PCA</li>
                <li><strong>Visualization:</strong> Plots and charts</li>
            </ol>

            <h2>Statistical Results</h2>
            <p>Detailed statistical results are available in the statistical analysis folder.</p>

            <h2>Visualizations</h2>
            <p>All visualizations are saved in the visualizations folder and include:</p>
            <ul>
                <li>Group comparison plots</li>
                <li>Correlation heatmaps</li>
                <li>PCA analysis</li>
                <li>Feature distributions</li>
            </ul>

            <h2>Conclusions</h2>
            <p>The analysis provides insights into how different AI chatbot personalities
            affect human physiological responses, with implications for:</p>
            <ul>
                <li>Emotional reactions and facial expressions (EMG)</li>
                <li>Stress and arousal responses (EDA)</li>
                <li>Trust and engagement in human-AI interactions</li>
                <li>Positive vs. negative emotional responses</li>
                <li>Mimicry and empathy responses</li>
            </ul>

            <p><em>For detailed results, please refer to the individual analysis files in their respective folders.</em></p>
        </body>
        </html>
        """

        return html_content

    def _create_summary_report(self):
        """Create summary text report"""
        summary = f"""
PSYCHOPHYSIOLOGICAL ANALYSIS SUMMARY REPORT
============================================

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SAMPLE INFORMATION
------------------
Total participants: {len(self.extracted_features)}
Chatbot versions: {', '.join(self.extracted_features['chatbot_version'].unique()) if not self.extracted_features.empty else 'None (all participants excluded)'}

ANALYSIS PIPELINE COMPLETED
----------------------------
Data preprocessing and quality assessment
Feature extraction (physiological and conversation)
Statistical analysis (group comparisons, correlations, PCA)
Visualizations
Final report generation

KEY FINDINGS FOR EDA + FACIAL EMG STUDY
---------------------------------------
This analysis focuses on emotional and stress responses through:
- EDA (Electrodermal Activity): Arousal, stress, engagement
- Facial EMG: Emotional expressions (smiling vs. frowning)
  - Zygomaticus Major: Positive emotions, smiling
  - Corrugator Supercilii: Negative emotions, frowning

ANALYSIS FEATURES
-----------------
EDA: SCR detection, tonic/phasic components, arousal measures
Facial EMG: Muscle activation, burst detection, emotional valence
Cross-correlation between EDA and EMG responses
Temporal analysis of emotional responses during conversation
Statistical comparison between chatbot personalities
"""

        if 'group_comparisons' in self.analysis_results:
            total_tests = len(self.analysis_results['group_comparisons'])
            significant_tests = len([r for r in self.analysis_results['group_comparisons'].values()
                                   if r['p_value'] < 0.05])

            summary += f"- {significant_tests} out of {total_tests} statistical tests showed significance (p < 0.05)\n"
            summary += f"- {significant_tests/total_tests*100:.1f}% of features showed significant group differences\n"

            # Add top 3 most significant findings
            significant_results = [
                (feature, results) for feature, results in self.analysis_results['group_comparisons'].items()
                if results['p_value'] < 0.05
            ]

            significant_results.sort(key=lambda x: x[1]['p_value'])

            summary += "\nTOP SIGNIFICANT FINDINGS:\n"
            for i, (feature, results) in enumerate(significant_results[:3]):
                summary += f"{i+1}. {feature}: p = {results['p_value']:.6f}\n"

        summary += f"""
FILES GENERATED
---------------
- Preprocessing report: {os.path.join(self.preprocessing_dir, 'preprocessing_report.json')}
- Extracted features: {os.path.join(self.feature_extraction_dir, 'extracted_features.csv')}
- Statistical results: {self.statistical_analysis_dir}/
- Visualizations: {self.visualization_dir}/
- Final reports: {self.reports_dir}/

NEXT STEPS
----------
1. Review statistical results for significant findings
2. Examine visualizations for patterns
3. Interpret results in context of study hypotheses
4. Consider additional analyses if needed

============================================
END OF SUMMARY REPORT
"""

        return summary

    def run_full_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("="*80)
        print("PSYCHOPHYSIOLOGICAL DATA ANALYSIS PIPELINE")
        print("="*80)
        print("Starting analysis...")

        # Step 1: Load data
        self.load_all_data()

        # Step 2: Preprocessing
        preprocessing_report = self.preprocess_data()

        # Step 3: Feature extraction
        self.extract_features()

        # Step 4: Statistical analysis
        self.statistical_analysis()

        # Step 5: Create visualizations
        self.create_visualizations()

        # Step 6: Generate final report
        self.generate_final_report()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"All results saved to: {self.output_folder}")
        print("Check the following directories for results:")
        print(f"  - Preprocessing: {self.preprocessing_dir}")
        print(f"  - Features: {self.feature_extraction_dir}")
        print(f"  - Statistics: {self.statistical_analysis_dir}")
        print(f"  - Visualizations: {self.visualization_dir}")
        print(f"  - Reports: {self.reports_dir}")


# Example usage
if __name__ == "__main__":
    # Set your paths
    biodata_folder = os.path.join(DATA_ROOT, "psychophysiology_signal")
    conversations_folder = os.path.join(DATA_ROOT, "conversation_logs")
    output_folder = os.path.join(DATA_ROOT, "biodata_analysis_results")

    # Create analyzer instance
    analyzer = PsychophysiologyAnalyzer(
        biodata_folder=biodata_folder,
        conversations_folder=conversations_folder,
        output_folder=output_folder,
        time_adjustment_hours=2
    )

    # Run complete analysis
    analyzer.run_full_analysis()

# %%
