# %% event_related_analysis.py

"""
Event-Related Analysis of Psychophysiological Responses to Chatbot Messages

This script performs event-related analysis (ERA) on physiological signals:
1. Extracts epochs time-locked to assistant (chatbot) message onsets
2. Processes each channel separately (EDA, fEMG channels)
3. Groups participants by experimental condition (chatbot version)
4. Computes group averages and statistical comparisons
5. Creates visualizations and statistical reports

Time windows:
- EDA: 20 seconds (pre: 2s, post: 18s) - accounts for slow EDA dynamics
- fEMG: 15 seconds (pre: 2s, post: 13s) - captures facial muscle responses

The statistics written by this script pool epochs across participants (permutation of epoch
labels; point-wise t-tests), so they treat non-independent epochs as independent and are
descriptive only. The event-related results reported in the paper are computed with the
participant as the unit of analysis by participant_level_extract.py,
participant_level_stats.py and participant_level_figures.py in this folder.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats, signal as sp_signal
from sklearn.preprocessing import StandardScaler
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
CLEANED_SIGNALS_FOLDER = os.path.join(DATA_ROOT, "cleaned_signals", "cleaned_signals")
OUTPUT_FOLDER = os.path.join(DATA_ROOT, "event_related_analysis")

# Event-related analysis parameters
ERA_CONFIG = {
    'EDA': {
        'pre_event_seconds': 2.0,
        'post_event_seconds': 18.0,
        'baseline_window_seconds': 2.0,  # Use pre-event period for baseline
        'smoothing_window': 0.5,  # Smoothing window in seconds
        'min_events_per_participant': 3
    },
    'fEMG': {
        'pre_event_seconds': 2.0,
        'post_event_seconds': 13.0,
        'baseline_window_seconds': 2.0,
        'smoothing_window': 0.1,
        'min_events_per_participant': 3
    }
}

# Statistical analysis parameters
STATS_CONFIG = {
    'alpha_level': 0.05,
    'multiple_comparisons_correction': 'fdr_bh',  # False discovery rate
    'effect_size_threshold': 0.5,  # Cohen's d
    'permutation_tests': True,
    'n_permutations': 1000
}

# Visualization parameters
VIZ_CONFIG = {
    'figure_size': (15, 10),
    'dpi': 300,
    'color_palette': ['#E57373', '#64B5F6', '#81C784', '#FFB74D'],  # Softer red, blue, green, orange
    'confidence_alpha': 0.3,
    'line_width': 2.0
}

# ===== EVENT-RELATED ANALYSIS CLASS =====

class EventRelatedAnalyzer:
    """
    Event-related analysis for psychophysiological data
    """

    def __init__(self, cleaned_signals_folder, output_folder):
        self.cleaned_signals_folder = cleaned_signals_folder
        self.output_folder = output_folder

        # Create output directories
        self.results_dir = os.path.join(output_folder, "results")
        self.figures_dir = os.path.join(output_folder, "figures")
        self.data_dir = os.path.join(output_folder, "extracted_data")
        self.stats_dir = os.path.join(output_folder, "statistics")

        for dir_path in [self.results_dir, self.figures_dir, self.data_dir, self.stats_dir]:
            os.makedirs(dir_path, exist_ok=True)

        self.analysis_log = {
            'start_time': datetime.now().isoformat(),
            'participants_processed': {},
            'group_statistics': {},
            'errors': []
        }

        # Group name mapping for display purposes
        self.group_display_names = {
            'group_1': 'Servant Leader',
            'group_2': 'Dark Triad'
        }

    def get_display_name(self, group_name):
        """Get display-friendly group name"""
        return self.group_display_names.get(group_name, group_name)

    def find_participants_with_events(self):
        """Find participants who have both cleaned signals and event data"""
        participants = []

        if not os.path.exists(self.cleaned_signals_folder):
            print(f"Cleaned signals folder not found: {self.cleaned_signals_folder}")
            return participants

        for item in os.listdir(self.cleaned_signals_folder):
            participant_dir = os.path.join(self.cleaned_signals_folder, item)
            if not os.path.isdir(participant_dir):
                continue

            # Check for required files
            event_file = os.path.join(participant_dir, f"{item}_conversation_events.json")
            exp_info_file = os.path.join(participant_dir, f"{item}_experimental_info.json")

            # Check for signal files (at least main recording)
            signal_files = glob.glob(os.path.join(participant_dir, f"{item}_main_*.npy"))

            if (os.path.exists(event_file) and
                os.path.exists(exp_info_file) and
                signal_files):
                participants.append(item)

        return sorted(participants)

    def load_participant_data(self, participant_id):
        """Load all necessary data for a participant"""
        participant_dir = os.path.join(self.cleaned_signals_folder, participant_id)

        # Load conversation events
        events_file = os.path.join(participant_dir, f"{participant_id}_conversation_events.json")
        with open(events_file, 'r') as f:
            events_data = json.load(f)

        # Load experimental info
        exp_info_file = os.path.join(participant_dir, f"{participant_id}_experimental_info.json")
        with open(exp_info_file, 'r') as f:
            exp_info = json.load(f)

        # Load cleaned signals
        signal_data = {}

        # Find all signal files for main recording
        signal_files = glob.glob(os.path.join(participant_dir, f"{participant_id}_main_*.npy"))

        for signal_file in signal_files:
            # Extract channel name from filename
            filename = os.path.basename(signal_file)
            parts = filename.replace('.npy', '').split('_')
            if len(parts) >= 3:
                channel_name = '_'.join(parts[2:])  # Everything after participant_main_

                # Load signal
                signal = np.load(signal_file)

                # Load corresponding metadata
                metadata_file = signal_file.replace('.npy', '_metadata.json')
                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    signal_data[channel_name] = {
                        'signal': signal,
                        'sampling_rate': metadata['original_channel_info']['samples_per_second'],
                        'channel_type': metadata['cleaning_metrics']['channel_type'],
                        'quality_score': metadata['cleaning_metrics']['quality_score']
                    }

        return events_data, exp_info, signal_data

    def extract_event_epochs(self, signal, sampling_rate, event_times, biopac_start_time,
                           pre_seconds, post_seconds):
        """
        Extract signal epochs around events

        Parameters:
        signal: 1D numpy array of signal data
        sampling_rate: Sampling rate in Hz
        event_times: List of event timestamps (ISO strings)
        biopac_start_time: BIOPAC recording start time (ISO string)
        pre_seconds: Seconds before event
        post_seconds: Seconds after event

        Returns:
        epochs: 2D array (n_epochs, n_timepoints)
        time_axis: Time axis relative to event (in seconds)
        valid_events: List of valid event information
        """
        # Convert timestamps
        biopac_start = datetime.fromisoformat(biopac_start_time.replace('Z', '+00:00').split('+')[0])

        # Calculate epoch parameters
        pre_samples = int(pre_seconds * sampling_rate)
        post_samples = int(post_seconds * sampling_rate)
        epoch_length = pre_samples + post_samples

        # Time axis relative to event
        time_axis = np.arange(-pre_samples, post_samples) / sampling_rate

        epochs = []
        valid_events = []

        for i, event_time_str in enumerate(event_times):
            try:
                # Parse event time (no timezone adjustment - conversation logs have correct timing)
                event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00').split('+')[0])

                # Calculate time relative to BIOPAC start
                time_diff = (event_time - biopac_start).total_seconds()
                event_sample = int(time_diff * sampling_rate)

                # Check if we can extract a full epoch
                start_sample = event_sample - pre_samples
                end_sample = event_sample + post_samples

                if start_sample >= 0 and end_sample < len(signal):
                    epoch = signal[start_sample:end_sample]
                    if len(epoch) == epoch_length:
                        epochs.append(epoch)
                        valid_events.append({
                            'event_index': i,
                            'event_time': event_time_str,
                            'time_from_start': time_diff,
                            'sample_index': event_sample
                        })

            except Exception as e:
                print(f"    Error processing event {i}: {e}")
                continue

        if epochs:
            epochs = np.array(epochs)
        else:
            epochs = np.array([]).reshape(0, epoch_length)

        return epochs, time_axis, valid_events

    def baseline_correct_epochs(self, epochs, time_axis, baseline_window_seconds):
        """
        Apply baseline correction to epochs.

        Baseline correction subtracts the mean of the pre-stimulus period from each epoch,
        removing tonic/baseline differences between epochs and participants.

        Parameters:
        -----------
        epochs : np.ndarray
            2D array of shape (n_epochs, n_timepoints)
        time_axis : np.ndarray
            1D array of time values relative to stimulus onset (negative = pre-stimulus)
        baseline_window_seconds : float
            Duration of baseline window to use (seconds before stimulus)
            If this equals the pre-event period, uses entire pre-stimulus period

        Returns:
        --------
        baseline_corrected : np.ndarray
            Baseline-corrected epochs with same shape as input
        """
        if len(epochs) == 0:
            return epochs

        print(f"    DEBUG - Baseline correction:")
        print(f"      Input epochs shape: {epochs.shape}")
        print(f"      Input epochs range: [{np.min(epochs):.6f}, {np.max(epochs):.6f}]")
        print(f"      Time axis range: [{np.min(time_axis):.3f}, {np.max(time_axis):.3f}]")
        print(f"      Baseline window: {baseline_window_seconds} seconds")

        # Find baseline samples (pre-stimulus period, i.e., negative time values)
        # Use the specified baseline window duration before stimulus onset (time = 0)
        baseline_mask = (time_axis >= -baseline_window_seconds) & (time_axis < 0)
        baseline_indices = np.where(baseline_mask)[0]

        print(f"      Baseline mask sum: {np.sum(baseline_mask)} points")

        # If no samples in specified window, fall back to all pre-stimulus samples
        if len(baseline_indices) == 0:
            print(f"WARNING: No samples in specified baseline window, using all pre-stimulus samples")
            baseline_mask = time_axis < 0
            baseline_indices = np.where(baseline_mask)[0]

        if len(baseline_indices) == 0:
            print("ERROR: No baseline samples found")
            return epochs

        # Get start and end indices for slicing
        baseline_start_idx = baseline_indices[0]
        baseline_end_idx = baseline_indices[-1] + 1  # +1 because Python slicing is exclusive
        n_baseline_samples = baseline_end_idx - baseline_start_idx

        print(f"      Baseline indices: {baseline_start_idx} to {baseline_end_idx} ({n_baseline_samples} samples)")
        print(f"      Baseline time range: [{time_axis[baseline_start_idx]:.3f}, {time_axis[baseline_end_idx-1]:.3f}] seconds")

        # Apply baseline correction to each epoch
        baseline_corrected = epochs.copy()
        baseline_means = []

        for i in range(epochs.shape[0]):
            # Extract baseline segment
            baseline_segment = epochs[i, baseline_start_idx:baseline_end_idx]

            # Calculate and store baseline mean
            baseline_mean = np.mean(baseline_segment)
            baseline_means.append(baseline_mean)

            # Subtract baseline mean from entire epoch
            baseline_corrected[i, :] -= baseline_mean

            # Debug first few epochs
            if i < 3:
                print(f"      Epoch {i}: baseline_mean = {baseline_mean:.6f}, "
                      f"baseline_segment range = [{np.min(baseline_segment):.6f}, {np.max(baseline_segment):.6f}]")

        # Summary statistics
        baseline_means = np.array(baseline_means)
        print(f"      Baseline means: range=[{np.min(baseline_means):.6f}, {np.max(baseline_means):.6f}], "
              f"mean={np.mean(baseline_means):.6f}, std={np.std(baseline_means):.6f}")
        print(f"      Output epochs range: [{np.min(baseline_corrected):.6f}, {np.max(baseline_corrected):.6f}]")

        # Sanity check: verify baseline correction worked
        post_correction_baseline_mean = np.mean([
            np.mean(baseline_corrected[i, baseline_start_idx:baseline_end_idx])
            for i in range(len(baseline_corrected))
        ])
        print(f"      Verification - mean of baselines after correction: {post_correction_baseline_mean:.9f} (should be ~0)")

        return baseline_corrected

    def convert_numpy_types(self, obj):
        """
        Recursively convert numpy data types to native Python types for JSON serialization
        """
        import numpy as np

        if isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'dtype'):  # Catch any other numpy objects
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj.item()  # Convert scalar numpy objects
        else:
            return obj

    def smooth_signal(self, signal, sampling_rate, window_seconds):
        """Apply smoothing to signal"""
        if window_seconds <= 0:
            return signal

        window_samples = int(window_seconds * sampling_rate)
        if window_samples < 3:
            return signal

        # Use a simple moving average
        kernel = np.ones(window_samples) / window_samples
        if signal.ndim == 1:
            return np.convolve(signal, kernel, mode='same')
        else:
            # Apply to each epoch
            smoothed = np.zeros_like(signal)
            for i in range(signal.shape[0]):
                smoothed[i, :] = np.convolve(signal[i, :], kernel, mode='same')
            return smoothed

    def preprocess_femg_for_era(self, raw_signal, sampling_rate):
        """
        Preprocess fEMG signal specifically for event-related analysis.
        Returns the RMS ENVELOPE, not the raw filtered signal.

        This is the KEY change - we analyze the envelope, not the raw signal!
        """
        nyquist = sampling_rate / 2

        # Step 1: Bandpass filter (20-500 Hz)
        low = 20 / nyquist
        high = min(500, nyquist * 0.95) / nyquist
        b, a = sp_signal.butter(4, [low, high], btype='band')
        filtered = sp_signal.filtfilt(b, a, raw_signal)

        # Step 2: Notch filters
        for freq in [50, 60]:
            if freq < nyquist:
                b, a = sp_signal.iirnotch(freq, 30, sampling_rate)
                filtered = sp_signal.filtfilt(b, a, filtered)

        # Step 3: Full-wave rectification
        rectified = np.abs(filtered)

        # Step 4: RMS envelope extraction (50ms window is standard)
        window_ms = 50
        window_samples = int(window_ms * sampling_rate / 1000)
        if window_samples % 2 == 0:
            window_samples += 1

        # Efficient sliding RMS
        half_window = window_samples // 2
        padded = np.pad(rectified, (half_window, half_window), mode='reflect')
        squared = padded ** 2
        cumsum = np.cumsum(squared)
        envelope = np.sqrt((cumsum[window_samples:] - cumsum[:-window_samples]) / window_samples)
        envelope = envelope[:len(raw_signal)]

        return envelope


    def extract_femg_epochs(self, envelope, sampling_rate, event_times, biopac_start_time,
                            pre_seconds=2.0, post_seconds=13.0):
        """
        Extract epochs from the EMG ENVELOPE (not raw signal).

        Key difference from EDA: We're working with the amplitude envelope.
        """
        biopac_start = datetime.fromisoformat(biopac_start_time.replace('Z', '+00:00').split('+')[0])

        pre_samples = int(pre_seconds * sampling_rate)
        post_samples = int(post_seconds * sampling_rate)
        epoch_length = pre_samples + post_samples
        time_axis = np.arange(-pre_samples, post_samples) / sampling_rate

        epochs = []
        valid_events = []

        for i, event_time_str in enumerate(event_times):
            try:
                event_time = datetime.fromisoformat(event_time_str.replace('Z', '+00:00').split('+')[0])
                time_diff = (event_time - biopac_start).total_seconds()
                event_sample = int(time_diff * sampling_rate)

                start_sample = event_sample - pre_samples
                end_sample = event_sample + post_samples

                if start_sample >= 0 and end_sample < len(envelope):
                    epoch = envelope[start_sample:end_sample]
                    if len(epoch) == epoch_length:
                        epochs.append(epoch)
                        valid_events.append({
                            'event_index': i,
                            'event_time': event_time_str,
                            'time_from_start': time_diff,
                            'sample_index': event_sample
                        })
            except Exception as e:
                continue

        return np.array(epochs) if epochs else np.array([]).reshape(0, epoch_length), time_axis, valid_events


    def baseline_correct_femg_epochs(self, epochs, time_axis, baseline_seconds=2.0):
        """
        Baseline correction for fEMG epochs.

        For EMG envelopes, we typically:
        1. Calculate baseline MEAN (not subtract raw values)
        2. Can use subtraction OR division (percent change)
        """
        if len(epochs) == 0:
            return epochs, epochs

        # Find baseline samples
        baseline_mask = time_axis < 0
        baseline_samples = np.sum(baseline_mask)

        # Method 1: Subtraction (absolute change)
        epochs_subtracted = np.zeros_like(epochs)

        # Method 2: Percent change (often more meaningful for EMG)
        epochs_percent = np.zeros_like(epochs)

        for i in range(len(epochs)):
            baseline = epochs[i, :baseline_samples]
            baseline_mean = np.mean(baseline)

            # Subtraction
            epochs_subtracted[i] = epochs[i] - baseline_mean

            # Percent change
            if baseline_mean > 0:
                epochs_percent[i] = ((epochs[i] - baseline_mean) / baseline_mean) * 100
            else:
                epochs_percent[i] = 0

        return epochs_subtracted, epochs_percent


    def extract_femg_window_features(self, epochs, time_axis, sampling_rate):
        """
        Extract features from specific time windows.

        This is more meaningful than point-by-point comparison for EMG!
        """
        # Define analysis windows (in seconds relative to stimulus)
        windows = {
            'early': (0, 1.0),      # 0-1 second post-stimulus
            'mid': (1.0, 3.0),      # 1-3 seconds
            'late': (3.0, 6.0),     # 3-6 seconds
            'sustained': (6.0, 13.0), # 6-13 seconds
            'overall': (0, 13.0),   # Overall response window (full post-stimulus period)
        }

        features = []
        zero_idx = np.argmin(np.abs(time_axis))  # Index of stimulus onset

        for epoch_idx in range(len(epochs)):
            epoch_features = {'epoch_idx': epoch_idx}

            # Baseline features
            baseline_mask = time_axis < 0
            baseline = epochs[epoch_idx, baseline_mask]
            epoch_features['baseline_mean'] = np.mean(baseline)
            epoch_features['baseline_rms'] = np.sqrt(np.mean(baseline**2))

            # Window features
            for window_name, (start_sec, end_sec) in windows.items():
                start_idx = zero_idx + int(start_sec * sampling_rate)
                end_idx = zero_idx + int(end_sec * sampling_rate)

                if end_idx <= len(epochs[epoch_idx]):
                    window_data = epochs[epoch_idx, start_idx:end_idx]

                    # Mean amplitude
                    epoch_features[f'{window_name}_mean'] = np.mean(window_data)

                    # RMS (root mean square)
                    epoch_features[f'{window_name}_rms'] = np.sqrt(np.mean(window_data**2))

                    # Peak amplitude
                    epoch_features[f'{window_name}_peak'] = np.max(window_data)

                    # Area under curve (integrated EMG)
                    epoch_features[f'{window_name}_auc'] = np.trapz(window_data) / sampling_rate

                    # Change from baseline
                    epoch_features[f'{window_name}_change'] = epoch_features[f'{window_name}_mean'] - epoch_features['baseline_mean']

                    # Percent change from baseline
                    if epoch_features['baseline_mean'] > 0:
                        epoch_features[f'{window_name}_pct_change'] = (
                            (epoch_features[f'{window_name}_mean'] - epoch_features['baseline_mean'])
                            / epoch_features['baseline_mean'] * 100
                        )
                    else:
                        epoch_features[f'{window_name}_pct_change'] = 0

            features.append(epoch_features)

        return pd.DataFrame(features)


    def compare_femg_groups_windowed(self, group1_features, group2_features,
                                      group1_name='Servant Leader', group2_name='Dark Triad'):
        """
        Compare fEMG features between groups using window-based analysis.

        This is the RECOMMENDED approach for EMG group comparisons.
        """
        results = {
            'comparison_name': f'{group1_name}_vs_{group2_name}',
            'n_group1': len(group1_features),
            'n_group2': len(group2_features),
            'window_comparisons': {},
            'summary': {}
        }

        # Features to compare
        comparison_features = [col for col in group1_features.columns
                              if any(x in col for x in ['_mean', '_rms', '_peak', '_auc', '_change', '_pct_change'])]

        for feature in comparison_features:
            g1 = group1_features[feature].dropna().values
            g2 = group2_features[feature].dropna().values

            if len(g1) < 3 or len(g2) < 3:
                continue

            # Descriptive stats
            comp = {
                'g1_mean': np.mean(g1),
                'g1_std': np.std(g1),
                'g1_median': np.median(g1),
                'g2_mean': np.mean(g2),
                'g2_std': np.std(g2),
                'g2_median': np.median(g2),
            }

            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(g1)-1)*np.var(g1, ddof=1) + (len(g2)-1)*np.var(g2, ddof=1)) / (len(g1)+len(g2)-2))
            comp['cohens_d'] = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0

            # Mann-Whitney U (non-parametric, recommended for EMG)
            u_stat, p_mann = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            comp['mann_whitney_u'] = u_stat
            comp['mann_whitney_p'] = p_mann

            # T-test (for comparison)
            t_stat, p_ttest = stats.ttest_ind(g1, g2)
            comp['ttest_t'] = t_stat
            comp['ttest_p'] = p_ttest

            # Permutation test (most robust)
            comp['permutation_p'] = self._run_permutation_test(g1, g2, n_permutations=5000)

            # Effect interpretation
            d = abs(comp['cohens_d'])
            if d >= 0.8:
                comp['effect_interpretation'] = 'large'
            elif d >= 0.5:
                comp['effect_interpretation'] = 'medium'
            elif d >= 0.2:
                comp['effect_interpretation'] = 'small'
            else:
                comp['effect_interpretation'] = 'negligible'

            results['window_comparisons'][feature] = comp

        # Summary: count significant findings
        sig_count = sum(1 for f, c in results['window_comparisons'].items()
                        if c['mann_whitney_p'] < 0.05)
        large_effects = sum(1 for f, c in results['window_comparisons'].items()
                           if c['effect_interpretation'] == 'large')

        results['summary'] = {
            'total_comparisons': len(results['window_comparisons']),
            'significant_at_05': sig_count,
            'large_effects': large_effects,
            'most_significant': sorted(
                [(f, c['mann_whitney_p'], c['cohens_d'])
                 for f, c in results['window_comparisons'].items()],
                key=lambda x: x[1]
            )[:5]
        }

        return results


    def _run_permutation_test(self, group1, group2, n_permutations=5000):
        """Run permutation test for difference in means."""
        combined = np.concatenate([group1, group2])
        n1 = len(group1)
        observed_diff = abs(np.mean(group1) - np.mean(group2))

        count = 0
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_diff = abs(np.mean(combined[:n1]) - np.mean(combined[n1:]))
            if perm_diff >= observed_diff:
                count += 1

        return count / n_permutations


    def create_femg_era_visualization(self, group1_epochs, group2_epochs, time_axis,
                                       group1_features, group2_features,
                                       group1_name='Servant Leader', group2_name='Dark Triad',
                                       save_path=None):
        """
        Create the fEMG event-related visualization.

        Key elements of fEMG visualization:
        - Show RMS envelope, not raw signal
        - Include window-based bar plots
        - Focus on meaningful time windows
        """
        fig = plt.figure(figsize=(16, 12))

        # Calculate group statistics
        g1_mean = np.mean(group1_epochs, axis=0)
        g1_sem = np.std(group1_epochs, axis=0) / np.sqrt(len(group1_epochs))
        g2_mean = np.mean(group2_epochs, axis=0)
        g2_sem = np.std(group2_epochs, axis=0) / np.sqrt(len(group2_epochs))

        # Plot 1: Grand average envelope (TOP LEFT)
        ax1 = fig.add_subplot(2, 2, 1)
        # Use softer colors: red for group1, blue for group2 to match the desired color scheme
        color1 = '#E57373'  # Soft red
        color2 = '#64B5F6'  # Soft blue
        ax1.plot(time_axis * 1000, g1_mean, color=color1, linewidth=2.5,
                 label=f'{group1_name} (n={len(group1_epochs)})')
        ax1.fill_between(time_axis * 1000, g1_mean - g1_sem, g1_mean + g1_sem,
                        color=color1, alpha=0.25)
        ax1.plot(time_axis * 1000, g2_mean, color=color2, linewidth=2.5,
                 label=f'{group2_name} (n={len(group2_epochs)})')
        ax1.fill_between(time_axis * 1000, g2_mean - g2_sem, g2_mean + g2_sem,
                        color=color2, alpha=0.25)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.set_xlabel('Time from Message Onset (ms)', fontsize=11)
        ax1.set_ylabel('EMG Amplitude (µV RMS)', fontsize=11)
        ax1.set_title('fEMG Event-Related Response (RMS Envelope)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Shade analysis windows
        window_colors = {'early': 'yellow', 'mid': 'orange', 'late': 'green'}
        windows = {'early': (0, 1000), 'mid': (1000, 3000), 'late': (3000, 6000)}
        for name, (start, end) in windows.items():
            ax1.axvspan(start, end, alpha=0.1, color=window_colors[name], label=f'{name} window')

        # Plot 2: Difference wave (TOP RIGHT)
        ax2 = fig.add_subplot(2, 2, 2)
        diff = g2_mean - g1_mean
        diff_sem = np.sqrt(g1_sem**2 + g2_sem**2)
        ax2.plot(time_axis * 1000, diff, 'purple', linewidth=2.5)
        ax2.fill_between(time_axis * 1000, diff - 1.96*diff_sem, diff + 1.96*diff_sem,
                        color='purple', alpha=0.25, label='95% CI')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Time from Message Onset (ms)', fontsize=11)
        ax2.set_ylabel('Amplitude Difference (µV)', fontsize=11)
        ax2.set_title(f'Difference ({group2_name} - {group1_name})', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Window-based comparison (BOTTOM LEFT)
        ax3 = fig.add_subplot(2, 2, 3)

        window_names = ['early', 'mid', 'late', 'sustained', 'overall']
        x_pos = np.arange(len(window_names))
        width = 0.35

        g1_means = [group1_features[f'{w}_mean'].mean() for w in window_names]
        g1_sems = [group1_features[f'{w}_mean'].std() / np.sqrt(len(group1_features)) for w in window_names]
        g2_means = [group2_features[f'{w}_mean'].mean() for w in window_names]
        g2_sems = [group2_features[f'{w}_mean'].std() / np.sqrt(len(group2_features)) for w in window_names]

        bars1 = ax3.bar(x_pos - width/2, g1_means, width, yerr=g1_sems,
                        label=group1_name, color=color1, alpha=0.7, capsize=5)
        bars2 = ax3.bar(x_pos + width/2, g2_means, width, yerr=g2_sems,
                        label=group2_name, color=color2, alpha=0.7, capsize=5)

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(['Early\n(0-1s)', 'Mid\n(1-3s)', 'Late\n(3-6s)', 'Sustained\n(6-13s)', 'Overall\n(0-13s)'])
        ax3.set_ylabel('Mean EMG Amplitude (µV)', fontsize=11)
        ax3.set_title('Time Window Analysis', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')

        # Plot 4: Effect sizes (BOTTOM RIGHT)
        ax4 = fig.add_subplot(2, 2, 4)

        effect_features = [f'{w}_mean' for w in window_names]
        effect_sizes = []
        for feat in effect_features:
            g1 = group1_features[feat].values
            g2 = group2_features[feat].values
            pooled_std = np.sqrt(((len(g1)-1)*np.var(g1, ddof=1) + (len(g2)-1)*np.var(g2, ddof=1)) / (len(g1)+len(g2)-2))
            d = (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std > 0 else 0
            effect_sizes.append(d)

        colors = ['forestgreen' if abs(d) > 0.8 else 'orange' if abs(d) > 0.5 else 'gray' for d in effect_sizes]
        bars = ax4.barh(x_pos, effect_sizes, color=colors, alpha=0.7)

        ax4.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax4.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax4.axvline(x=-0.5, color='orange', linestyle='--', alpha=0.5, linewidth=1)
        ax4.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, linewidth=1)
        ax4.axvline(x=-0.8, color='green', linestyle='--', alpha=0.5, linewidth=1)

        ax4.set_yticks(x_pos)
        ax4.set_yticklabels(['Early\n(0-1s)', 'Mid\n(1-3s)', 'Late\n(3-6s)', 'Sustained\n(6-13s)', 'Overall\n(0-13s)'])
        ax4.set_xlabel("Cohen's d (Effect Size)", fontsize=11)
        ax4.set_title('Effect Sizes by Time Window', fontsize=12, fontweight='bold')

        # Add legend for effect sizes
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='forestgreen', alpha=0.7, label='Large (|d| > 0.8)'),
            Patch(facecolor='orange', alpha=0.7, label='Medium (|d| > 0.5)'),
            Patch(facecolor='gray', alpha=0.7, label='Small (|d| < 0.5)')
        ]
        ax4.legend(handles=legend_elements, loc='lower right', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            return fig

    def process_participant_femg(self, participant_id):
        """
        Modified process_participant specifically for fEMG channels.
        Uses envelope extraction and window-based features.
        """
        print(f"Processing fEMG for participant: {participant_id}")

        results = {
            'participant_id': participant_id,
            'success': False,
            'experimental_group': None,
            'channels': {},
            'errors': []
        }

        try:
            events_data, exp_info, signal_data = self.load_participant_data(participant_id)
            results['experimental_group'] = exp_info.get('experimental_group', 'unknown')

            biopac_start_time = exp_info.get('biopac_recording_timestamps', {}).get('main', {}).get('recording_start_time')
            if not biopac_start_time:
                results['errors'].append("No BIOPAC start time")
                return results

            assistant_responses = events_data.get('assistant_responses', [])
            event_times = [msg['timestamp_start'] for msg in assistant_responses]

            if len(event_times) < 3:
                results['errors'].append("Insufficient events")
                return results

            for channel_name, channel_data in signal_data.items():
                if channel_data['channel_type'] != 'fEMG':
                    continue

                print(f"  Processing fEMG channel: {channel_name}")

                raw_signal = channel_data['signal']
                sampling_rate = channel_data['sampling_rate']

                # KEY CHANGE: Extract RMS envelope first
                envelope = self.preprocess_femg_for_era(raw_signal, sampling_rate)

                # Extract epochs from envelope
                epochs, time_axis, valid_events = self.extract_femg_epochs(
                    envelope, sampling_rate, event_times, biopac_start_time
                )

                if len(epochs) < 3:
                    results['errors'].append(f"Insufficient epochs for {channel_name}")
                    continue

                # Baseline correction
                epochs_corrected, epochs_percent = self.baseline_correct_femg_epochs(epochs, time_axis)

                # Extract window features (THIS IS WHAT YOU SHOULD COMPARE!)
                window_features = self.extract_femg_window_features(epochs_corrected, time_axis, sampling_rate)

                results['channels'][channel_name] = {
                    'n_epochs': len(epochs),
                    'time_axis': time_axis.tolist(),
                    'epochs_envelope': epochs.tolist(),
                    'epochs_corrected': epochs_corrected.tolist(),
                    'epochs_percent_change': epochs_percent.tolist(),
                    'window_features': window_features.to_dict('records'),
                    'grand_average': np.mean(epochs_corrected, axis=0).tolist(),
                    'sem': (np.std(epochs_corrected, axis=0) / np.sqrt(len(epochs))).tolist()
                }

                print(f"    Extracted {len(epochs)} epochs with window features")

            if results['channels']:
                results['success'] = True

        except Exception as e:
            results['errors'].append(str(e))
            import traceback
            traceback.print_exc()

        return results

    def process_participant(self, participant_id):
        """Process a single participant for event-related analysis"""
        print(f"Processing participant: {participant_id}")

        participant_results = {
            'participant_id': participant_id,
            'success': False,
            'experimental_group': None,
            'channels_processed': {},
            'errors': []
        }

        try:
            # Load participant data
            events_data, exp_info, signal_data = self.load_participant_data(participant_id)

            # Get experimental group
            experimental_group = exp_info.get('experimental_group', 'unknown')
            participant_results['experimental_group'] = experimental_group
            print(f"  Group: {experimental_group}")

            # Get BIOPAC recording start time for main recording
            biopac_timestamps = exp_info.get('biopac_recording_timestamps', {})
            main_timestamps = biopac_timestamps.get('main', {})
            biopac_start_time = main_timestamps.get('recording_start_time')

            if not biopac_start_time:
                error_msg = "No BIOPAC start time found for main recording"
                print(f"  Error: {error_msg}")
                participant_results['errors'].append(error_msg)
                return participant_results

            # Extract assistant message onset times
            assistant_responses = events_data.get('assistant_responses', [])
            if len(assistant_responses) == 0:
                error_msg = "No assistant responses found"
                print(f"  Error: {error_msg}")
                participant_results['errors'].append(error_msg)
                return participant_results

            assistant_onset_times = [msg['timestamp_start'] for msg in assistant_responses]
            print(f"  Found {len(assistant_onset_times)} assistant message onsets")

            # Process each channel
            for channel_name, channel_data in signal_data.items():
                try:
                    channel_type = channel_data['channel_type']
                    if channel_type not in ERA_CONFIG:
                        print(f"  Skipping {channel_name} - unsupported channel type: {channel_type}")
                        continue

                    print(f"  Processing channel: {channel_name} ({channel_type})")

                    config = ERA_CONFIG[channel_type]
                    signal = channel_data['signal']
                    sampling_rate = channel_data['sampling_rate']

                    # Check minimum events requirement
                    if len(assistant_onset_times) < config['min_events_per_participant']:
                        error_msg = f"Insufficient events for {channel_name}"
                        print(f"    Warning: {error_msg}")
                        participant_results['errors'].append(error_msg)
                        continue

                    # ============================================
                    # DIFFERENT PROCESSING FOR fEMG vs EDA
                    # ============================================
                    if channel_type == 'fEMG':
                        # fEMG: Extract RMS envelope FIRST, then epochs
                        print(f"    Extracting RMS envelope for fEMG...")
                        envelope = self.preprocess_femg_for_era(signal, sampling_rate)

                        # Extract epochs from the envelope
                        epochs, time_axis, valid_events = self.extract_femg_epochs(
                            envelope, sampling_rate, assistant_onset_times, biopac_start_time,
                            config['pre_event_seconds'], config['post_event_seconds']
                        )

                        if len(epochs) == 0:
                            error_msg = f"No valid epochs extracted for {channel_name}"
                            print(f"    Error: {error_msg}")
                            participant_results['errors'].append(error_msg)
                            continue

                        print(f"    Extracted {len(epochs)} envelope epochs")

                        # Baseline correction for fEMG
                        epochs_corrected, epochs_percent = self.baseline_correct_femg_epochs(
                            epochs, time_axis, config['baseline_window_seconds']
                        )

                        # Extract window-based features (KEY FOR fEMG ANALYSIS!)
                        window_features = self.extract_femg_window_features(
                            epochs_corrected, time_axis, sampling_rate
                        )

                        # Store results with fEMG-specific data
                        participant_results['channels_processed'][channel_name] = {
                            'channel_type': channel_type,
                            'quality_score': channel_data['quality_score'],
                            'sampling_rate': sampling_rate,
                            'n_epochs': len(epochs),
                            'epoch_length': len(time_axis),
                            'time_axis': time_axis.tolist(),
                            'epochs_envelope': epochs.tolist(),  # Raw envelope epochs
                            'epochs_corrected': epochs_corrected.tolist(),  # Baseline corrected
                            'epochs_percent_change': epochs_percent.tolist(),  # Percent change
                            'epochs_final': epochs_corrected.tolist(),  # For compatibility
                            'window_features': window_features.to_dict('records'),  # KEY DATA!
                            'valid_events': valid_events,
                            'config_used': config,
                            'processing_method': 'rms_envelope'  # Flag for downstream
                        }

                    else:
                        # EDA and other channels: Use original processing
                        epochs, time_axis, valid_events = self.extract_event_epochs(
                            signal, sampling_rate, assistant_onset_times, biopac_start_time,
                            config['pre_event_seconds'], config['post_event_seconds']
                        )

                        if len(epochs) == 0:
                            error_msg = f"No valid epochs extracted for {channel_name}"
                            print(f"    Error: {error_msg}")
                            participant_results['errors'].append(error_msg)
                            continue

                        print(f"    Extracted {len(epochs)} valid epochs")

                        # Apply baseline correction (original method)
                        epochs_baseline = self.baseline_correct_epochs(
                            epochs, time_axis, config['baseline_window_seconds']
                        )

                        # Apply smoothing (original method)
                        epochs_smooth = self.smooth_signal(
                            epochs_baseline, sampling_rate, config['smoothing_window']
                        )

                        # Store results (original format)
                        participant_results['channels_processed'][channel_name] = {
                            'channel_type': channel_type,
                            'quality_score': channel_data['quality_score'],
                            'sampling_rate': sampling_rate,
                            'n_epochs': len(epochs_smooth),
                            'epoch_length': len(time_axis),
                            'time_axis': time_axis.tolist(),
                            'epochs_raw': epochs.tolist(),
                            'epochs_baseline_corrected': epochs_baseline.tolist(),
                            'epochs_final': epochs_smooth.tolist(),
                            'valid_events': valid_events,
                            'config_used': config,
                            'processing_method': 'standard'
                        }

                    print(f"    Successfully processed {channel_name}")

                except Exception as e:
                    error_msg = f"Error processing channel {channel_name}: {e}"
                    print(f"ERROR: {error_msg}")
                    participant_results['errors'].append(error_msg)
                    import traceback
                    traceback.print_exc()

            if participant_results['channels_processed']:
                participant_results['success'] = True
                print(f"  Successfully processed {len(participant_results['channels_processed'])} channels")

        except Exception as e:
            error_msg = f"Error processing participant {participant_id}: {e}"
            print(f"ERROR: {error_msg}")
            participant_results['errors'].append(error_msg)
            import traceback
            traceback.print_exc()

        return participant_results

    def aggregate_group_data(self, participant_results):
        """Aggregate data by experimental groups and channels - with fEMG window features"""
        print("\nAggregating group data...")

        group_data = {}

        for participant_id, results in participant_results.items():
            if not results['success']:
                continue

            group = results['experimental_group']
            if group not in group_data:
                group_data[group] = {}

            for channel_name, channel_results in results['channels_processed'].items():
                if channel_name not in group_data[group]:
                    group_data[group][channel_name] = {
                        'participants': [],
                        'epochs_all': [],
                        'time_axis': channel_results['time_axis'],
                        'channel_type': channel_results['channel_type'],
                        'sampling_rates': [],
                        'quality_scores': [],
                        'processing_method': channel_results.get('processing_method', 'standard'),
                        # For fEMG window features
                        'window_features_all': []
                    }

                group_data[group][channel_name]['participants'].append(participant_id)

                # Use epochs_corrected for fEMG, epochs_final for others
                if channel_results.get('processing_method') == 'rms_envelope':
                    group_data[group][channel_name]['epochs_all'].extend(channel_results['epochs_corrected'])
                    # Collect window features for fEMG
                    if 'window_features' in channel_results:
                        for feat in channel_results['window_features']:
                            feat['participant_id'] = participant_id
                            group_data[group][channel_name]['window_features_all'].append(feat)
                else:
                    group_data[group][channel_name]['epochs_all'].extend(channel_results['epochs_final'])

                group_data[group][channel_name]['sampling_rates'].append(channel_results['sampling_rate'])
                group_data[group][channel_name]['quality_scores'].append(channel_results['quality_score'])

        # Convert to numpy arrays and calculate statistics
        for group in group_data:
            for channel_name in group_data[group]:
                channel_data = group_data[group][channel_name]
                epochs_array = np.array(channel_data['epochs_all'])

                if len(epochs_array) > 0:
                    channel_data['epochs_array'] = epochs_array
                    channel_data['mean_response'] = np.mean(epochs_array, axis=0)
                    channel_data['std_response'] = np.std(epochs_array, axis=0)
                    channel_data['sem_response'] = channel_data['std_response'] / np.sqrt(len(epochs_array))
                    channel_data['n_epochs'] = len(epochs_array)
                    channel_data['n_participants'] = len(channel_data['participants'])

                    # Convert window features to DataFrame for fEMG
                    if channel_data['window_features_all']:
                        channel_data['window_features_df'] = pd.DataFrame(channel_data['window_features_all'])
                        print(f"  {group} - {channel_name}: {channel_data['n_participants']} participants, "
                                f"{channel_data['n_epochs']} epochs, "
                                f"{len(channel_data['window_features_df'])} window feature sets")
                    else:
                        print(f"  {group} - {channel_name}: {channel_data['n_participants']} participants, "
                                f"{channel_data['n_epochs']} epochs")

        return group_data

    def perform_statistical_analysis(self, group_data):
        """Perform statistical comparisons between groups - with fEMG window analysis"""
        print("\nPerforming statistical analysis...")

        stats_results = {}

        # Get all unique channels across groups
        all_channels = set()
        for group in group_data:
            all_channels.update(group_data[group].keys())

        for channel_name in all_channels:
            print(f"  Analyzing channel: {channel_name}")

            # Get groups that have this channel
            groups_with_channel = [g for g in group_data if channel_name in group_data[g]]

            if len(groups_with_channel) < 2:
                print(f"    Skipping - insufficient groups ({len(groups_with_channel)})")
                continue

            channel_type = group_data[groups_with_channel[0]][channel_name]['channel_type']
            processing_method = group_data[groups_with_channel[0]][channel_name].get('processing_method', 'standard')

            channel_stats = {
                'channel_name': channel_name,
                'channel_type': channel_type,
                'processing_method': processing_method,
                'groups_compared': groups_with_channel,
                'comparisons': {},
                'window_comparisons': {}  # NEW: For fEMG
            }

            # Pairwise comparisons between groups
            for i, group1 in enumerate(groups_with_channel):
                for j, group2 in enumerate(groups_with_channel[i+1:], i+1):
                    comparison_key = f"{group1}_vs_{group2}"
                    print(f"    Comparing {group1} vs {group2}")

                    # ============================================
                    # fEMG: Use window-based comparison
                    # ============================================
                    if channel_type == 'fEMG' and processing_method == 'rms_envelope':
                        # Get window features DataFrames
                        g1_features = group_data[group1][channel_name].get('window_features_df')
                        g2_features = group_data[group2][channel_name].get('window_features_df')

                        if g1_features is not None and g2_features is not None:
                            # Use window-based comparison (MORE MEANINGFUL FOR fEMG!)
                            window_results = self.compare_femg_groups_windowed(
                                g1_features, g2_features, group1, group2
                            )
                            channel_stats['window_comparisons'][comparison_key] = window_results

                            print(f"      Window analysis: {window_results['summary']['significant_at_05']} "
                                    f"significant features, {window_results['summary']['large_effects']} large effects")

                    # Standard point-by-point comparison (for EDA and as secondary for fEMG)
                    epochs1 = group_data[group1][channel_name]['epochs_array']
                    epochs2 = group_data[group2][channel_name]['epochs_array']

                    comparison_results = self.compare_groups(
                        epochs1, epochs2, group1, group2,
                        group_data[group1][channel_name]['time_axis']
                    )

                    channel_stats['comparisons'][comparison_key] = comparison_results

            stats_results[channel_name] = channel_stats

        return stats_results

    def compare_groups(self, epochs1, epochs2, group1_name, group2_name, time_axis):
        """Compare two groups of epochs statistically"""

        mean1 = np.mean(epochs1, axis=0)
        mean2 = np.mean(epochs2, axis=0)

        # Overall statistics
        comparison = {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_n_epochs': len(epochs1),
            'group2_n_epochs': len(epochs2),
            'mean_difference': mean2 - mean1,
            'tests': {}
        }

        # T-test at each time point
        t_stats = []
        p_values = []

        for i in range(min(epochs1.shape[1], epochs2.shape[1])):
            t_stat, p_val = stats.ttest_ind(epochs1[:, i], epochs2[:, i])
            t_stats.append(t_stat)
            p_values.append(p_val)

        t_stats = np.array(t_stats)
        p_values = np.array(p_values)

        # Multiple comparisons correction
        from statsmodels.stats.multitest import multipletests
        rejected, p_corrected, _, _ = multipletests(
            p_values, alpha=STATS_CONFIG['alpha_level'],
            method=STATS_CONFIG['multiple_comparisons_correction']
        )

        comparison['tests']['pointwise_ttest'] = {
            't_statistics': t_stats.tolist(),
            'p_values': p_values.tolist(),
            'p_corrected': p_corrected.tolist(),
            'significant_points': rejected.tolist(),
            'n_significant': int(np.sum(rejected))
        }

        # Effect size (Cohen's d)
        # Calculate pooled standard deviation with better numerical stability
        var1 = np.var(epochs1, axis=0, ddof=1)
        var2 = np.var(epochs2, axis=0, ddof=1)
        n1, n2 = len(epochs1), len(epochs2)

        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_var)

        # Use a reasonable epsilon based on the data scale
        # Calculate epsilon as a fraction of the overall signal variance
        overall_var = np.var(np.concatenate([epochs1.flatten(), epochs2.flatten()]))
        epsilon = np.sqrt(overall_var) * 1e-6  # Much smaller relative threshold

        # Only set to zero if pooled_std is truly negligible
        cohens_d = np.divide(mean2 - mean1, pooled_std,
                            out=np.zeros_like(mean2 - mean1),
                            where=pooled_std > epsilon)

        # Handle any remaining NaN or infinite values
        cohens_d = np.where(np.isnan(cohens_d) | np.isinf(cohens_d), 0.0, cohens_d)

        comparison['effect_size'] = {
            'cohens_d': cohens_d.tolist(),
            'large_effects': (np.abs(cohens_d) > 0.8).tolist(),
            'medium_effects': ((np.abs(cohens_d) > 0.5) & (np.abs(cohens_d) <= 0.8)).tolist(),
            'pooled_std_stats': {
                'min': float(np.min(pooled_std)),
                'max': float(np.max(pooled_std)),
                'mean': float(np.mean(pooled_std)),
                'epsilon_used': float(epsilon)
            }
        }

        # Permutation test (if enabled)
        if STATS_CONFIG['permutation_tests']:
            perm_p = self.permutation_test(epochs1, epochs2, STATS_CONFIG['n_permutations'])
            comparison['tests']['permutation_test'] = {
                'p_value': float(perm_p),
                'n_permutations': STATS_CONFIG['n_permutations'],
                'significant': bool(perm_p < STATS_CONFIG['alpha_level'])
            }

        return comparison

    def permutation_test(self, epochs1, epochs2, n_permutations):
        """Perform permutation test between two groups"""
        # Combine all epochs
        all_epochs = np.vstack([epochs1, epochs2])
        n1 = len(epochs1)
        n2 = len(epochs2)

        # Calculate observed difference in means
        observed_diff = np.mean(epochs2, axis=0) - np.mean(epochs1, axis=0)
        observed_stat = np.mean(np.abs(observed_diff))

        # Permutation test
        null_stats = []
        for _ in range(n_permutations):
            # Randomly shuffle group labels
            perm_indices = np.random.permutation(len(all_epochs))
            perm_group1 = all_epochs[perm_indices[:n1]]
            perm_group2 = all_epochs[perm_indices[n1:]]

            # Calculate null statistic
            null_diff = np.mean(perm_group2, axis=0) - np.mean(perm_group1, axis=0)
            null_stat = np.mean(np.abs(null_diff))
            null_stats.append(null_stat)

        # Calculate p-value
        null_stats = np.array(null_stats)
        p_value = np.mean(null_stats >= observed_stat)

        return p_value

    def smooth_for_visualization(self, signal, window_size=None):
        """Apply heavy smoothing for visualization purposes only"""
        from scipy import signal as sp_signal

        # Auto-determine window size based on signal length and type
        if window_size is None:
            window_size = max(len(signal) // 100, 51)  # At least 51, up to 1% of signal length

        if len(signal) < window_size:
            window_size = len(signal) // 10
            if window_size < 5:
                return signal  # Too short to smooth meaningfully

        # Ensure odd window size for Savitzky-Golay
        if window_size % 2 == 0:
            window_size += 1

        try:
            # Use Savitzky-Golay filter for smooth curves
            smoothed = sp_signal.savgol_filter(signal, window_size, 3)
        except:
            # Fallback to simple moving average if Savitzky-Golay fails
            kernel = np.ones(min(window_size, len(signal))) / min(window_size, len(signal))
            smoothed = np.convolve(signal, kernel, mode='same')

        return smoothed

    def create_visualization(self, group_data, stats_results):
        """Create the visualizations, including the fEMG-specific plots"""
        print("\nCreating visualizations...")

        # Get all unique channels
        all_channels = set()
        for group in group_data:
            all_channels.update(group_data[group].keys())

        for channel_name in all_channels:
            print(f"  Creating plots for: {channel_name}")

            # Get groups that have this channel
            groups_with_channel = [g for g in group_data if channel_name in group_data[g]]

            if len(groups_with_channel) < 2:
                continue

            channel_type = group_data[groups_with_channel[0]][channel_name]['channel_type']
            processing_method = group_data[groups_with_channel[0]][channel_name].get('processing_method', 'standard')
            time_axis = np.array(group_data[groups_with_channel[0]][channel_name]['time_axis'])

            # ============================================
            # fEMG: Use specialized visualization
            # ============================================
            if channel_type == 'fEMG' and processing_method == 'rms_envelope' and len(groups_with_channel) == 2:
                group1, group2 = groups_with_channel

                # Get data
                g1_epochs = group_data[group1][channel_name]['epochs_array']
                g2_epochs = group_data[group2][channel_name]['epochs_array']
                g1_features = group_data[group1][channel_name].get('window_features_df')
                g2_features = group_data[group2][channel_name].get('window_features_df')

                if g1_features is not None and g2_features is not None:
                    # Use the specialized fEMG visualization
                    save_path = os.path.join(self.figures_dir, f"{channel_name}_femg_analysis.png")
                    self.create_femg_era_visualization(
                        g1_epochs, g2_epochs, time_axis,
                        g1_features, g2_features,
                        self.get_display_name(group1), self.get_display_name(group2),
                        save_path=save_path
                    )
                    print(f"    Saved fEMG visualization: {os.path.basename(save_path)}")
                    continue  # Skip the standard visualization for fEMG

            # ============================================
            # Standard visualization (for EDA and fallback)
            # ============================================
            fig, axes = plt.subplots(2, 2, figsize=VIZ_CONFIG['figure_size'])
            fig.suptitle(f'Event-Related Analysis: {channel_name}', fontsize=16, fontweight='bold')

            colors = VIZ_CONFIG['color_palette'][:len(groups_with_channel)]

            # PLOT 1: Group averages
            ax1 = axes[0, 0]
            for i, group in enumerate(groups_with_channel):
                channel_data = group_data[group][channel_name]
                mean_response = np.array(channel_data['mean_response'])
                sem_response = np.array(channel_data['sem_response'])

                if len(mean_response) != len(time_axis):
                    continue

                ax1.plot(time_axis, mean_response,
                        color=colors[i], linewidth=VIZ_CONFIG['line_width'],
                        label=f'{self.get_display_name(group)} (n={channel_data["n_participants"]})')
                ax1.fill_between(time_axis, mean_response - 1.96 * sem_response,
                                mean_response + 1.96 * sem_response,
                                color=colors[i], alpha=VIZ_CONFIG['confidence_alpha'])

            ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='Message Onset')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel(f'{channel_type} Response')
            ax1.set_title(f'Group Averages ± 95% CI')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # PLOT 2: Difference wave
            ax2 = axes[0, 1]
            if len(groups_with_channel) == 2:
                group1, group2 = groups_with_channel
                mean1 = np.array(group_data[group1][channel_name]['mean_response'])
                mean2 = np.array(group_data[group2][channel_name]['mean_response'])

                if len(mean1) == len(mean2) == len(time_axis):
                    diff_wave = mean2 - mean1
                    ax2.plot(time_axis, diff_wave, color='purple', linewidth=2)
                    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                    ax2.set_xlabel('Time (seconds)')
                    ax2.set_ylabel(f'{channel_type} Difference')
                    ax2.set_title(f'Difference Wave ({self.get_display_name(group2)} - {self.get_display_name(group1)})')
                    ax2.grid(True, alpha=0.3)

            # PLOT 3: Sample epochs
            ax3 = axes[1, 0]
            for i, group in enumerate(groups_with_channel):
                epochs = group_data[group][channel_name]['epochs_array']
                if len(epochs) > 0:
                    sample_size = min(10, len(epochs))
                    sample_indices = np.random.choice(len(epochs), sample_size, replace=False)
                    for j in sample_indices:
                        if len(epochs[j]) == len(time_axis):
                            ax3.plot(time_axis, epochs[j], color=colors[i], alpha=0.3, linewidth=0.5)

            ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel(f'{channel_type} Response')
            ax3.set_title('Sample Individual Epochs')
            ax3.grid(True, alpha=0.3)

            # PLOT 4: Effect size
            ax4 = axes[1, 1]
            if channel_name in stats_results and len(groups_with_channel) == 2:
                comparison_key = f"{groups_with_channel[0]}_vs_{groups_with_channel[1]}"
                if comparison_key in stats_results[channel_name]['comparisons']:
                    comp_data = stats_results[channel_name]['comparisons'][comparison_key]
                    cohens_d = np.array(comp_data['effect_size']['cohens_d'])

                    if len(cohens_d) == len(time_axis):
                        ax4.plot(time_axis, cohens_d, color='purple', linewidth=VIZ_CONFIG['line_width'])
                        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                        ax4.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5)
                        ax4.axhline(y=0.8, color='red', linestyle=':', alpha=0.5)
                        ax4.axhline(y=-0.5, color='orange', linestyle=':', alpha=0.5)
                        ax4.axhline(y=-0.8, color='red', linestyle=':', alpha=0.5)
                        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
                        ax4.set_xlabel('Time (seconds)')
                        ax4.set_ylabel("Cohen's d")
                        ax4.set_title('Effect Size')
                        ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            fig_path = os.path.join(self.figures_dir, f"{channel_name}_event_related_analysis.png")
            plt.savefig(fig_path, dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
            plt.close()
            print(f"    Saved: {os.path.basename(fig_path)}")

        print("  Visualization complete")

    def create_summary_report(self, participant_results, group_data, stats_results):
        """Create the summary report"""
        print("\nCreating summary report...")

        report = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'total_participants': len(participant_results),
                'successful_participants': sum(1 for r in participant_results.values() if r['success']),
                'era_config': ERA_CONFIG,
                'stats_config': STATS_CONFIG
            },
            'participant_summary': {},
            'group_summary': {},
            'statistical_results': {},
            'key_findings': []
        }

        # Participant summary
        groups = {}
        for participant_id, results in participant_results.items():
            if results['success']:
                group = results['experimental_group']
                if group not in groups:
                    groups[group] = []
                groups[group].append(participant_id)

                report['participant_summary'][participant_id] = {
                    'group': group,
                    'channels_processed': list(results['channels_processed'].keys()),
                    'total_epochs_extracted': sum(
                        ch['n_epochs'] for ch in results['channels_processed'].values()
                    ),
                    'errors': results['errors']
                }

        # Group summary
        for group, channel_data in group_data.items():
            report['group_summary'][group] = {
                'n_participants': len(groups.get(group, [])),
                'participants': groups.get(group, []),
                'channels': {}
            }

            for channel_name, data in channel_data.items():
                report['group_summary'][group]['channels'][channel_name] = {
                    'channel_type': data['channel_type'],
                    'n_epochs': data['n_epochs'],
                    'n_participants': data['n_participants'],
                    'mean_quality_score': np.mean(data['quality_scores'])
                }

        # Statistical results summary
        for channel_name, channel_stats in stats_results.items():
            report['statistical_results'][channel_name] = {
                'channel_type': channel_stats['channel_type'],
                'groups_compared': channel_stats['groups_compared'],
                'significant_comparisons': []
            }

            for comp_name, comp_data in channel_stats['comparisons'].items():
                # Check for significance
                if 'pointwise_ttest' in comp_data['tests']:
                    n_sig_points = int(comp_data['tests']['pointwise_ttest']['n_significant'])  # Convert to int
                    perm_significant = (comp_data['tests'].get('permutation_test', {}).get('significant', False)
                                      if 'permutation_test' in comp_data['tests'] else False)

                    max_effect_size = float(np.max(np.abs(comp_data['effect_size']['cohens_d'])))  # Convert to float

                    if n_sig_points > 0 or perm_significant or max_effect_size > STATS_CONFIG['effect_size_threshold']:
                        report['statistical_results'][channel_name]['significant_comparisons'].append({
                            'comparison': comp_name,
                            'n_significant_timepoints': n_sig_points,
                            'permutation_significant': perm_significant,
                            'max_effect_size': max_effect_size,
                            'groups': [comp_data['group1_name'], comp_data['group2_name']]
                        })

        # Generate key findings
        report['key_findings'] = self.generate_key_findings(report, stats_results)

        # Save report
        report_path = os.path.join(self.results_dir, "event_related_analysis_report.json")
        report_clean = self.convert_numpy_types(report)
        with open(report_path, 'w') as f:
            json.dump(report_clean, f, indent=2)

        # Create human-readable summary
        self.create_readable_summary(report)

        return report

    def create_femg_summary_report(self, stats_results, output_path=None):
        """Create a detailed summary of fEMG window-based analysis results."""

        if output_path is None:
            output_path = os.path.join(self.results_dir, "femg_window_analysis_summary.txt")

        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("fEMG WINDOW-BASED ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            for channel_name, channel_stats in stats_results.items():
                if channel_stats['channel_type'] != 'fEMG':
                    continue

                f.write(f"\n{'='*60}\n")
                f.write(f"CHANNEL: {channel_name}\n")
                f.write(f"{'='*60}\n")

                for comp_name, window_results in channel_stats.get('window_comparisons', {}).items():
                    f.write(f"\nComparison: {comp_name}\n")
                    f.write(f"  Group 1 epochs: {window_results['n_group1']}\n")
                    f.write(f"  Group 2 epochs: {window_results['n_group2']}\n")
                    f.write(f"\n  SUMMARY:\n")
                    f.write(f"    Total comparisons: {window_results['summary']['total_comparisons']}\n")
                    f.write(f"    Significant (p<0.05): {window_results['summary']['significant_at_05']}\n")
                    f.write(f"    Large effects: {window_results['summary']['large_effects']}\n")

                    f.write(f"\n  TOP 5 MOST SIGNIFICANT FEATURES:\n")
                    for feat, p_val, d in window_results['summary']['most_significant']:
                        sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                        f.write(f"    {feat}: p={p_val:.4f} {sig_marker}, Cohen's d={d:.3f}\n")

                    f.write(f"\n  DETAILED WINDOW RESULTS:\n")
                    f.write(f"  {'-'*50}\n")

                    # Group by window
                    windows = ['early', 'mid', 'late', 'overall']
                    for window in windows:
                        f.write(f"\n  {window.upper()} WINDOW:\n")
                        window_feats = {k: v for k, v in window_results['window_comparisons'].items()
                                        if k.startswith(window)}

                        for feat_name, feat_stats in window_feats.items():
                            p_val = feat_stats['mann_whitney_p']
                            d = feat_stats['cohens_d']
                            g1_mean = feat_stats['g1_mean']
                            g2_mean = feat_stats['g2_mean']

                            sig = "SIGNIFICANT" if p_val < 0.05 else ""
                            f.write(f"    {feat_name}:\n")
                            f.write(f"      Group1: {g1_mean:.4f} ± {feat_stats['g1_std']:.4f}\n")
                            f.write(f"      Group2: {g2_mean:.4f} ± {feat_stats['g2_std']:.4f}\n")
                            f.write(f"      Mann-Whitney p: {p_val:.4f} {sig}\n")
                            f.write(f"      Cohen's d: {d:.3f} ({feat_stats['effect_interpretation']})\n")

            f.write("\n" + "="*80 + "\n")
            f.write("INTERPRETATION GUIDE:\n")
            f.write("="*80 + "\n")
            f.write("Window Definitions:\n")
            f.write("  - EARLY: 0-1 second post-stimulus (immediate response)\n")
            f.write("  - MID: 1-3 seconds (sustained response)\n")
            f.write("  - LATE: 3-6 seconds (late response)\n")
            f.write("  - OVERALL: 0-6 seconds (full response window)\n")
            f.write("\nEffect Size Interpretation (Cohen's d):\n")
            f.write("  - |d| < 0.2: Negligible\n")
            f.write("  - |d| 0.2-0.5: Small\n")
            f.write("  - |d| 0.5-0.8: Medium\n")
            f.write("  - |d| > 0.8: Large\n")
            f.write("\nKey Features:\n")
            f.write("  - _mean: Average amplitude in window\n")
            f.write("  - _rms: Root mean square (gold standard for EMG)\n")
            f.write("  - _peak: Maximum amplitude\n")
            f.write("  - _auc: Area under curve (integrated activity)\n")
            f.write("  - _change: Absolute change from baseline\n")
            f.write("  - _pct_change: Percent change from baseline\n")

        print(f"fEMG summary report saved to: {output_path}")
        return output_path

    def create_eda_summary_report(self, group_data, stats_results, output_path=None):
        """Create a detailed summary of EDA event-related analysis results."""

        if output_path is None:
            output_path = os.path.join(self.results_dir, "eda_event_related_summary.txt")

        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EDA EVENT-RELATED ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")

            # Find EDA channels in the data
            eda_channels = []
            for group in group_data:
                for channel_name, channel_info in group_data[group].items():
                    if channel_info['channel_type'] == 'EDA' and channel_name not in eda_channels:
                        eda_channels.append(channel_name)

            if not eda_channels:
                f.write("No EDA channels found in the analysis.\n")
                print("No EDA channels to summarize.")
                return output_path

            for channel_name in eda_channels:
                f.write("\n" + "="*80 + "\n")
                f.write(f"CHANNEL: {channel_name}\n")
                f.write("="*80 + "\n\n")

                # Extract group data for this channel
                groups_with_channel = [g for g in group_data if channel_name in group_data[g]]

                if len(groups_with_channel) < 2:
                    f.write(f"Insufficient groups for comparison (only {len(groups_with_channel)} group(s))\n")
                    continue

                # GROUP DESCRIPTIVE STATISTICS
                f.write("-"*80 + "\n")
                f.write("GROUP DESCRIPTIVE STATISTICS\n")
                f.write("-"*80 + "\n\n")

                for group in groups_with_channel:
                    channel_data = group_data[group][channel_name]
                    epochs_array = channel_data['epochs_array']
                    time_axis = np.array(channel_data['time_axis'])

                    f.write(f"Group: {self.get_display_name(group)}\n")
                    f.write(f"  Participants: {channel_data['n_participants']}\n")
                    f.write(f"  Total epochs: {len(epochs_array)}\n")
                    f.write(f"  Epochs per participant: {len(epochs_array) / channel_data['n_participants']:.1f}\n")

                    # Calculate key EDA metrics
                    mean_response = np.array(channel_data['mean_response'])

                    # Find baseline (pre-stimulus) and post-stimulus periods
                    baseline_mask = time_axis < 0
                    post_mask = time_axis >= 0

                    baseline_mean = np.mean(mean_response[baseline_mask])
                    post_mean = np.mean(mean_response[post_mask])

                    # Find peak response (in post-stimulus period)
                    post_response = mean_response[post_mask]
                    post_time = time_axis[post_mask]

                    if len(post_response) > 0:
                        peak_idx = np.argmax(post_response)
                        peak_amplitude = post_response[peak_idx]
                        peak_latency = post_time[peak_idx]

                        # Calculate response amplitude (peak - baseline)
                        response_amplitude = peak_amplitude - baseline_mean

                        # Area under curve (post-stimulus)
                        auc = np.trapz(post_response - baseline_mean, post_time)

                        f.write(f"\n  Baseline mean: {baseline_mean:.6f}\n")
                        f.write(f"  Post-stimulus mean: {post_mean:.6f}\n")
                        f.write(f"  Peak amplitude: {peak_amplitude:.6f}\n")
                        f.write(f"  Peak latency: {peak_latency:.3f} seconds\n")
                        f.write(f"  Response amplitude (peak - baseline): {response_amplitude:.6f}\n")
                        f.write(f"  Area under curve (0-{post_time[-1]:.1f}s): {auc:.6f}\n")

                    f.write("\n")

                # STATISTICAL COMPARISONS
                if channel_name in stats_results and len(groups_with_channel) == 2:
                    f.write("-"*80 + "\n")
                    f.write("STATISTICAL COMPARISONS\n")
                    f.write("-"*80 + "\n\n")

                    group1, group2 = groups_with_channel
                    comparison_key = f"{group1}_vs_{group2}"

                    if comparison_key in stats_results[channel_name]['comparisons']:
                        comp_data = stats_results[channel_name]['comparisons'][comparison_key]

                        f.write(f"Comparison: {self.get_display_name(group1)} vs {self.get_display_name(group2)}\n\n")

                        # Overall comparison metrics
                        g1_epochs = group_data[group1][channel_name]['epochs_array']
                        g2_epochs = group_data[group2][channel_name]['epochs_array']
                        g1_mean = np.array(group_data[group1][channel_name]['mean_response'])
                        g2_mean = np.array(group_data[group2][channel_name]['mean_response'])

                        # Calculate overall effect size
                        pooled_std = np.sqrt(((len(g1_epochs)-1)*np.var(g1_epochs) + (len(g2_epochs)-1)*np.var(g2_epochs)) / (len(g1_epochs)+len(g2_epochs)-2))
                        overall_cohens_d = (np.mean(g2_epochs) - np.mean(g1_epochs)) / pooled_std if pooled_std > 0 else 0

                        f.write(f"Overall Effect Size (Cohen's d): {overall_cohens_d:.3f}\n")

                        if abs(overall_cohens_d) >= 0.8:
                            interpretation = "LARGE"
                        elif abs(overall_cohens_d) >= 0.5:
                            interpretation = "MEDIUM"
                        elif abs(overall_cohens_d) >= 0.2:
                            interpretation = "SMALL"
                        else:
                            interpretation = "NEGLIGIBLE"
                        f.write(f"Effect Interpretation: {interpretation}\n\n")

                        # Permutation test result
                        if 'permutation_test' in comp_data.get('tests', {}):
                            perm_p = comp_data['tests']['permutation_test']['p_value']
                            f.write(f"Permutation Test p-value: {perm_p:.4f}")
                            if perm_p < 0.001:
                                f.write(" ***\n")
                            elif perm_p < 0.01:
                                f.write(" **\n")
                            elif perm_p < 0.05:
                                f.write(" *\n")
                            else:
                                f.write(" (not significant)\n")

                        # Point-wise t-test results
                        if 'pointwise_ttest' in comp_data.get('tests', {}):
                            n_sig = comp_data['tests']['pointwise_ttest']['n_significant']
                            total_points = len(comp_data['tests']['pointwise_ttest']['p_values'])
                            pct_sig = (n_sig / total_points * 100) if total_points > 0 else 0

                            f.write(f"\nPoint-wise t-tests (FDR corrected):\n")
                            f.write(f"  Significant time points: {n_sig}/{total_points} ({pct_sig:.1f}%)\n")

                            # Find significant time windows
                            time_axis = np.array(group_data[group1][channel_name]['time_axis'])
                            sig_points = np.array(comp_data['tests']['pointwise_ttest']['significant_points'])

                            if np.any(sig_points):
                                sig_times = time_axis[sig_points]
                                f.write(f"  Significant time range: {sig_times[0]:.3f}s to {sig_times[-1]:.3f}s\n")

                                # Find peak difference in significant region
                                cohens_d = np.array(comp_data['effect_size']['cohens_d'])
                                sig_d = cohens_d[sig_points]
                                peak_d_idx = np.argmax(np.abs(sig_d))
                                peak_d_value = sig_d[peak_d_idx]
                                peak_d_time = sig_times[peak_d_idx]

                                f.write(f"  Peak effect at {peak_d_time:.3f}s: Cohen's d = {peak_d_value:.3f}\n")

                        f.write("\n")

                        # TIME-COURSE ANALYSIS
                        f.write("-"*80 + "\n")
                        f.write("TIME-COURSE ANALYSIS\n")
                        f.write("-"*80 + "\n\n")

                        time_axis = np.array(group_data[group1][channel_name]['time_axis'])

                        # Define EDA-specific time windows
                        windows = {
                            'immediate': (0, 3),      # 0-3s: immediate response (SCR onset)
                            'peak': (3, 6),           # 3-6s: peak response window
                            'sustained': (6, 12),     # 6-12s: sustained response
                            'late': (12, 18)          # 12-18s: late/recovery phase
                        }

                        for window_name, (start_sec, end_sec) in windows.items():
                            window_mask = (time_axis >= start_sec) & (time_axis < end_sec)

                            if np.any(window_mask):
                                g1_window = g1_mean[window_mask]
                                g2_window = g2_mean[window_mask]

                                g1_window_mean = np.mean(g1_window)
                                g2_window_mean = np.mean(g2_window)

                                diff = g2_window_mean - g1_window_mean

                                # Effect size for this window
                                g1_window_epochs = g1_epochs[:, window_mask]
                                g2_window_epochs = g2_epochs[:, window_mask]

                                g1_window_overall = np.mean(g1_window_epochs)
                                g2_window_overall = np.mean(g2_window_epochs)

                                pooled_std_window = np.sqrt(
                                    ((len(g1_epochs)-1)*np.var(g1_window_epochs) +
                                     (len(g2_epochs)-1)*np.var(g2_window_epochs)) /
                                    (len(g1_epochs)+len(g2_epochs)-2)
                                )

                                window_d = (g2_window_overall - g1_window_overall) / pooled_std_window if pooled_std_window > 0 else 0

                                f.write(f"{window_name.upper()} window ({start_sec}-{end_sec}s):\n")
                                f.write(f"  {self.get_display_name(group1)}: {g1_window_mean:.6f}\n")
                                f.write(f"  {self.get_display_name(group2)}: {g2_window_mean:.6f}\n")
                                f.write(f"  Difference: {diff:.6f}\n")
                                f.write(f"  Cohen's d: {window_d:.3f}\n\n")

            # INTERPRETATION GUIDE
            f.write("\n" + "="*80 + "\n")
            f.write("INTERPRETATION GUIDE\n")
            f.write("="*80 + "\n")
            f.write("\nEDA Time-Course Characteristics:\n")
            f.write("-"*40 + "\n")
            f.write("EDA responses to stimuli typically show:\n")
            f.write("  - LATENCY: 1-3 seconds (time to onset)\n")
            f.write("  - PEAK: 3-6 seconds (maximum response)\n")
            f.write("  - RECOVERY: Gradual return to baseline over 10-20 seconds\n")
            f.write("\nTime Window Definitions:\n")
            f.write("-"*40 + "\n")
            f.write("  - IMMEDIATE (0-3s): Early response onset, anticipatory arousal\n")
            f.write("  - PEAK (3-6s): Maximum sympathetic activation\n")
            f.write("  - SUSTAINED (6-12s): Maintained arousal, cognitive processing\n")
            f.write("  - LATE (12-18s): Recovery phase, residual arousal\n")
            f.write("\nEffect Size Interpretation (Cohen's d):\n")
            f.write("-"*40 + "\n")
            f.write("  - |d| < 0.2: Negligible difference\n")
            f.write("  - |d| 0.2-0.5: Small effect\n")
            f.write("  - |d| 0.5-0.8: Medium effect\n")
            f.write("  - |d| > 0.8: Large effect\n")
            f.write("\nKey Metrics:\n")
            f.write("-"*40 + "\n")
            f.write("  - Response amplitude: Peak value minus baseline (captures response magnitude)\n")
            f.write("  - Peak latency: Time to maximum response (speed of arousal)\n")
            f.write("  - Area under curve (AUC): Total arousal integrated over time\n")
            f.write("  - Baseline mean: Tonic level before stimulus\n")
            f.write("\nStatistical Significance:\n")
            f.write("-"*40 + "\n")
            f.write("  * p < 0.05\n")
            f.write("  ** p < 0.01\n")
            f.write("  *** p < 0.001\n")
            f.write("\nPhysiological Interpretation:\n")
            f.write("-"*40 + "\n")
            f.write("EDA (Electrodermal Activity) reflects sympathetic nervous system activation.\n")
            f.write("Higher EDA indicates greater emotional arousal, attention, or cognitive effort.\n")
            f.write("In the context of chatbot interactions:\n")
            f.write("  - Increased EDA may indicate engagement, stress, or emotional response\n")
            f.write("  - Sustained elevation suggests maintained cognitive/emotional processing\n")
            f.write("  - Group differences reveal distinct emotional/cognitive responses to\n")
            f.write("    different chatbot communication styles\n")

            f.write("\n" + "="*80 + "\n")

        print(f"EDA summary report saved to: {output_path}")
        return output_path

    def generate_key_findings(self, report, stats_results):
        """Generate key findings from the analysis"""
        findings = []

        # Overall participation
        total_participants = report['analysis_info']['total_participants']
        successful_participants = report['analysis_info']['successful_participants']

        findings.append(f"Analyzed {successful_participants}/{total_participants} participants successfully")

        # Group sizes
        group_sizes = {}
        for group, data in report['group_summary'].items():
            group_sizes[group] = data['n_participants']

        findings.append(f"Group sizes: {', '.join([f'{g}: {n}' for g, n in group_sizes.items()])}")

        # Significant effects
        significant_channels = []
        for channel_name, channel_results in report['statistical_results'].items():
            if channel_results['significant_comparisons']:
                significant_channels.append(channel_name)
                for comp in channel_results['significant_comparisons']:
                    # Handle NaN values in effect size
                    max_effect = comp['max_effect_size']
                    if np.isnan(max_effect) or np.isinf(max_effect):
                        # Only report permutation test results
                        if comp['permutation_significant']:
                            findings.append(f"Significant difference in {channel_name}: {comp['comparison']} (permutation test)")
                    else:
                        # Report effect size if valid
                        if max_effect > 0.8:
                            findings.append(f"Large effect found in {channel_name}: {comp['comparison']} (d={max_effect:.2f})")
                        elif comp['permutation_significant']:
                            findings.append(f"Significant difference in {channel_name}: {comp['comparison']} (permutation test, d={max_effect:.2f})")

        if not significant_channels:
            findings.append("No significant differences found between groups")
        else:
            findings.append(f"Significant effects found in channels: {', '.join(significant_channels)}")

        return findings

    def create_readable_summary(self, report):
        """Create a human-readable summary document"""
        summary_path = os.path.join(self.results_dir, "analysis_summary.txt")

        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EVENT-RELATED ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Analysis completed: {report['analysis_info']['timestamp']}\n\n")

            # Key findings
            f.write("KEY FINDINGS:\n")
            f.write("-" * 40 + "\n")
            for finding in report['key_findings']:
                f.write(f"• {finding}\n")
            f.write("\n")

            # Participant summary
            f.write("PARTICIPANT SUMMARY:\n")
            f.write("-" * 40 + "\n")
            for group, data in report['group_summary'].items():
                f.write(f"{group}: {data['n_participants']} participants\n")
                for channel, ch_data in data['channels'].items():
                    f.write(f"  {channel} ({ch_data['channel_type']}): {ch_data['n_epochs']} epochs, "
                           f"quality={ch_data['mean_quality_score']:.1f}\n")
            f.write("\n")

            # Statistical results
            f.write("STATISTICAL RESULTS:\n")
            f.write("-" * 40 + "\n")
            for channel, ch_stats in report['statistical_results'].items():
                f.write(f"{channel} ({ch_stats['channel_type']}):\n")
                if ch_stats['significant_comparisons']:
                    for comp in ch_stats['significant_comparisons']:
                        f.write(f"  {comp['comparison']}: max effect size = {comp['max_effect_size']:.3f}\n")
                        if comp['permutation_significant']:
                            f.write(f"    Permutation test: SIGNIFICANT\n")
                        f.write(f"    Significant timepoints: {comp['n_significant_timepoints']}\n")
                else:
                    f.write("  No significant differences found\n")
                f.write("\n")

            f.write("="*80 + "\n")

        print(f"  Readable summary saved: {os.path.basename(summary_path)}")

    def save_extracted_data(self, participant_results, group_data):
        """Save extracted epoch data for further analysis"""
        print("\nSaving extracted data...")

        # Save individual participant data
        participants_file = os.path.join(self.data_dir, "participant_epochs.json")

        # Convert numpy arrays to lists for JSON serialization
        participants_json = {}
        for participant_id, results in participant_results.items():
            if results['success']:
                participants_json[participant_id] = {
                    'experimental_group': results['experimental_group'],
                    'channels': {}
                }
                for channel_name, channel_data in results['channels_processed'].items():
                    participants_json[participant_id]['channels'][channel_name] = {
                        'channel_type': channel_data['channel_type'],
                        'time_axis': channel_data['time_axis'],
                        'epochs': channel_data['epochs_final'],
                        'n_epochs': channel_data['n_epochs'],
                        'quality_score': channel_data['quality_score']
                    }

        with open(participants_file, 'w') as f:
            json.dump(participants_json, f, indent=2)

        # Save group-averaged data
        group_file = os.path.join(self.data_dir, "group_averages.json")

        group_json = {}
        for group, channels in group_data.items():
            group_json[group] = {}
            for channel_name, channel_data in channels.items():
                group_json[group][channel_name] = {
                    'channel_type': channel_data['channel_type'],
                    'time_axis': channel_data['time_axis'],
                    'mean_response': channel_data['mean_response'].tolist(),
                    'std_response': channel_data['std_response'].tolist(),
                    'sem_response': channel_data['sem_response'].tolist(),
                    'n_epochs': channel_data['n_epochs'],
                    'n_participants': channel_data['n_participants'],
                    'participants': channel_data['participants']
                }

        with open(group_file, 'w') as f:
            json.dump(group_json, f, indent=2)

        print(f"  Participant data saved: {os.path.basename(participants_file)}")
        print(f"  Group averages saved: {os.path.basename(group_file)}")

    def save_statistical_results(self, stats_results):
        """Save statistical analysis results to the statistics folder"""
        print("\nSaving statistical results...")

        # Save complete statistical results as JSON
        stats_file = os.path.join(self.stats_dir, "statistical_comparisons.json")
        stats_clean = self.convert_numpy_types(stats_results)
        with open(stats_file, 'w') as f:
            json.dump(stats_clean, f, indent=2)
        print(f"  Statistical comparisons saved: {os.path.basename(stats_file)}")

        # Create a readable text summary of statistical results
        stats_summary_file = os.path.join(self.stats_dir, "statistical_summary.txt")
        with open(stats_summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("STATISTICAL ANALYSIS SUMMARY\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Correction method: {STATS_CONFIG['multiple_comparisons_correction']}\n")
            f.write(f"Alpha level: {STATS_CONFIG['alpha_level']}\n\n")

            for channel_name, channel_stats in stats_results.items():
                f.write("\n" + "="*80 + "\n")
                f.write(f"CHANNEL: {channel_name}\n")
                f.write("="*80 + "\n")
                f.write(f"Type: {channel_stats['channel_type']}\n")
                f.write(f"Processing: {channel_stats.get('processing_method', 'standard')}\n")
                f.write(f"Groups: {', '.join(channel_stats['groups_compared'])}\n\n")

                # Pairwise comparisons
                for comp_name, comp_data in channel_stats['comparisons'].items():
                    f.write("-"*80 + "\n")
                    f.write(f"COMPARISON: {comp_name}\n")
                    f.write("-"*80 + "\n")
                    f.write(f"Group 1 ({comp_data['group1_name']}): {comp_data['group1_n_epochs']} epochs\n")
                    f.write(f"Group 2 ({comp_data['group2_name']}): {comp_data['group2_n_epochs']} epochs\n\n")

                    # Permutation test
                    if 'permutation_test' in comp_data.get('tests', {}):
                        perm_test = comp_data['tests']['permutation_test']
                        f.write("Permutation Test:\n")
                        f.write(f"  p-value: {perm_test['p_value']:.6f}")
                        if perm_test['significant']:
                            f.write(" ***SIGNIFICANT***\n")
                        else:
                            f.write(" (not significant)\n")
                        f.write(f"  n_permutations: {perm_test['n_permutations']}\n\n")

                    # Point-wise t-tests
                    if 'pointwise_ttest' in comp_data.get('tests', {}):
                        pt_test = comp_data['tests']['pointwise_ttest']
                        f.write("Point-wise T-tests (FDR corrected):\n")
                        f.write(f"  Significant timepoints: {pt_test['n_significant']}\n")
                        f.write(f"  Total timepoints: {len(pt_test['p_values'])}\n")
                        if pt_test['n_significant'] > 0:
                            pct = (pt_test['n_significant'] / len(pt_test['p_values'])) * 100
                            f.write(f"  Percentage significant: {pct:.1f}%\n")
                        f.write("\n")

                    # Effect sizes
                    if 'effect_size' in comp_data:
                        effect_data = comp_data['effect_size']
                        cohens_d = np.array(effect_data['cohens_d'])

                        f.write("Effect Sizes (Cohen's d):\n")
                        f.write(f"  Mean: {np.mean(cohens_d):.3f}\n")
                        f.write(f"  Max (absolute): {np.max(np.abs(cohens_d)):.3f}\n")
                        f.write(f"  Min (absolute): {np.min(np.abs(cohens_d)):.3f}\n")

                        n_large = np.sum(np.abs(cohens_d) > 0.8)
                        n_medium = np.sum((np.abs(cohens_d) > 0.5) & (np.abs(cohens_d) <= 0.8))
                        n_small = np.sum((np.abs(cohens_d) > 0.2) & (np.abs(cohens_d) <= 0.5))

                        f.write(f"  Large effects (|d| > 0.8): {n_large} timepoints\n")
                        f.write(f"  Medium effects (0.5 < |d| ≤ 0.8): {n_medium} timepoints\n")
                        f.write(f"  Small effects (0.2 < |d| ≤ 0.5): {n_small} timepoints\n\n")

                # Window comparisons (for fEMG)
                if 'window_comparisons' in channel_stats and channel_stats['window_comparisons']:
                    f.write("\n" + "-"*80 + "\n")
                    f.write("WINDOW-BASED COMPARISONS (fEMG)\n")
                    f.write("-"*80 + "\n\n")

                    for comp_name, comp_data in channel_stats['window_comparisons'].items():
                        f.write(f"{comp_name}:\n")
                        f.write(f"  Groups: {comp_data['comparison_name']}\n")
                        f.write(f"  n_group1: {comp_data['n_group1']}, n_group2: {comp_data['n_group2']}\n\n")

                        # Summary of significant findings
                        if 'summary' in comp_data:
                            summary = comp_data['summary']
                            f.write(f"  Summary:\n")
                            f.write(f"    Total comparisons: {summary['total_comparisons']}\n")
                            f.write(f"    Significant at α=0.05: {summary['significant_at_05']}\n")
                            f.write(f"    Large effects: {summary['large_effects']}\n\n")

                            if summary['most_significant']:
                                f.write(f"  Top 5 most significant findings:\n")
                                for i, (feature, p_val, d) in enumerate(summary['most_significant'], 1):
                                    f.write(f"    {i}. {feature}: p={p_val:.4f}, d={d:.3f}\n")
                        f.write("\n")

        print(f"  Statistical summary saved: {os.path.basename(stats_summary_file)}")

    def run_complete_analysis(self):
        """Run the complete event-related analysis pipeline"""
        print("="*80)
        print("EVENT-RELATED ANALYSIS OF PSYCHOPHYSIOLOGICAL RESPONSES")
        print("="*80)
        print(f"Output directory: {self.output_folder}")
        print(f"Configuration:")
        for channel_type, config in ERA_CONFIG.items():
            print(f"  {channel_type}: {config['pre_event_seconds']}s pre + {config['post_event_seconds']}s post")
        print()

        # Step 1: Find participants
        participants = self.find_participants_with_events()
        if not participants:
            print("No participants with event data found")
            return

        print(f"Found {len(participants)} participants with event data")
        print()

        # Step 2: Process each participant
        participant_results = {}
        for participant_id in participants:
            result = self.process_participant(participant_id)
            participant_results[participant_id] = result
            self.analysis_log['participants_processed'][participant_id] = result

        # Step 3: Aggregate group data
        group_data = self.aggregate_group_data(participant_results)

        # Step 4: Statistical analysis
        stats_results = self.perform_statistical_analysis(group_data)

        # Step 5: Create visualizations
        self.create_visualization(group_data, stats_results)

        # Step 6a: Create fEMG-specific summary report
        self.create_femg_summary_report(stats_results)

        # Step 6b: Create EDA-specific summary report
        self.create_eda_summary_report(group_data, stats_results)

        # Step 7a: Save extracted data
        self.save_extracted_data(participant_results, group_data)

        # Step 7b: Save statistical results
        self.save_statistical_results(stats_results)

        # Step 8: Create summary report
        report = self.create_summary_report(participant_results, group_data, stats_results)

        # Final summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)

        successful_participants = sum(1 for r in participant_results.values() if r['success'])
        print(f"Successfully processed: {successful_participants}/{len(participants)} participants")
        print(f"Groups analyzed: {list(group_data.keys())}")
        print(f"Channels analyzed: {set().union(*[list(channels.keys()) for channels in group_data.values()])}")

        print(f"\nOutput files saved to: {self.output_folder}")
        print("  - Figures: event-related plots for each channel")
        print("  - Results: JSON report and readable summary")
        print("  - Data: extracted epochs and group averages")
        print("  - Statistics: statistical comparison results")

        print("\nKey findings:")
        for finding in report['key_findings']:
            print(f"  • {finding}")

def main():
    """Main execution function"""
    analyzer = EventRelatedAnalyzer(
        CLEANED_SIGNALS_FOLDER,
        OUTPUT_FOLDER
    )

    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()

# %%