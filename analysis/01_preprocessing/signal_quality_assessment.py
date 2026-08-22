# %% signal_quality_assessment.py

"""
Memory-Optimized Psychophysiological Signal Quality Analysis

This script analyzes the quality of psychophysiological signals with memory optimizations
to handle large datasets (50+ participants) without running out of memory.

Key optimizations:
1. Batch processing with garbage collection
2. Selective data retention
3. Optimized figure handling
4. Progressive result writing
5. Memory monitoring and cleanup

"""

# ===== ANALYSIS CONSTANTS =====
WINDOW_SIZE = 5.0  # Size of the sliding window for local analysis (in seconds)
MIN_SEGMENT_LENGTH = 0.5  # Minimum length of a segment to analyze (in seconds)

# Memory optimization settings
BATCH_SIZE = 5  # Process this many participants before garbage collection
MEMORY_LIMIT_GB = 40  # Maximum memory usage before forcing cleanup (in GB)
ENABLE_MEMORY_MONITORING = True  # Enable memory usage monitoring
# ===================================

import bioread
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
import os
import glob
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.dates as mdates
from scipy.fft import fft, fftfreq
import re
import time
import io
import base64
from matplotlib.patches import Rectangle
import gc
from dotenv import load_dotenv

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
DATA_FOLDER = os.path.join(DATA_ROOT, "psychophysiology_signal")
OUTPUT_FOLDER = os.path.join(DATA_ROOT, "signal_quality_analysis")
import psutil
import warnings

# Suppress matplotlib warnings to reduce memory overhead
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Define channel-specific parameters (same as original)
CHANNEL_PARAMETERS = {
    'EDA': {
        'normal_range': (0, 30),  # Normal range in microSiemens
        'min_variance': 0.001,     # Minimum variance for non-flatline
        'max_variance': 15,       # Maximum variance for non-noisy signal
        'max_jump': 5,            # Maximum allowed jump in microSiemens
        'expected_frequency_range': (0, 1),  # Expected frequency range in Hz
        'processing_filters': 'lowpass',     # Type of filter to apply
        'filter_cutoff': 1.0,               # Cutoff frequency for filter
        'min_peaks_per_minute': 0,          # Minimum peaks per minute
        'max_peaks_per_minute': 10,         # Maximum peaks per minute
        'scoring_weights': {
            'flatline': 0.30,
            'noise': 0.20,
            'range': 0.15,
            'jumps': 0.15,
            'peaks': 0.10,
            'frequency': 0.10
        }
    },
    'fEMG': {
        'normal_range': (-1000, 1000),      # Normal range in μV
        'min_variance': 10,                 # Minimum variance for non-flatline
        'max_variance': 100000,             # Maximum variance for non-noisy signal
        'max_jump': 500,                    # Maximum allowed jump
        'expected_frequency_range': (10, 500),  # Expected frequency range in Hz
        'processing_filters': 'bandpass',   # Type of filter to apply
        'filter_cutoff': [10, 500],         # Cutoff frequencies for filter
        'min_peaks_per_minute': 5,          # Minimum peaks per minute
        'max_peaks_per_minute': 300,        # Maximum peaks per minute
        'scoring_weights': {
            'flatline': 0.25,
            'noise': 0.30,
            'range': 0.15,
            'jumps': 0.15,
            'peaks': 0.05,
            'frequency': 0.10
        }
    }
}

# Custom colormap for quality scores
QUALITY_CMAP = LinearSegmentedColormap.from_list('quality_cmap',
                                                [(0.0, 'red'),
                                                 (0.5, 'yellow'),
                                                 (1.0, 'green')])

class MemoryMonitor:
    """Helper class to monitor and manage memory usage"""

    def __init__(self, limit_gb=8):
        self.limit_gb = limit_gb
        self.limit_bytes = limit_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()

    def get_memory_usage_gb(self):
        """Get current memory usage in GB"""
        return self.process.memory_info().rss / (1024 * 1024 * 1024)

    def check_memory_limit(self):
        """Check if memory usage exceeds the limit"""
        current_usage = self.get_memory_usage_gb()
        return current_usage >= self.limit_gb

    def force_cleanup(self):
        """Force garbage collection and memory cleanup"""
        plt.close('all')  # Close all matplotlib figures
        gc.collect()  # Force garbage collection

        # Additional cleanup
        if hasattr(gc, 'set_debug'):
            gc.set_debug(0)  # Disable debugging

    def print_memory_status(self, context=""):
        """Print current memory status"""
        usage = self.get_memory_usage_gb()
        print(f"Memory usage {context}: {usage:.2f} GB / {self.limit_gb} GB")

class OptimizedSignalQualityAnalyzer:
    """
    Memory-optimized version of SignalQualityAnalyzer for large datasets.
    """

    def __init__(self, data_folder, output_folder, min_segment_length=5, window_size=5,
                 batch_size=5, memory_limit_gb=8):
        """
        Initialize the OptimizedSignalQualityAnalyzer.

        Parameters:
        -----------
        data_folder : str
            Path to the folder containing BIOPAC .acq files
        output_folder : str
            Path to the folder where results will be saved
        min_segment_length : float
            Minimum length of a segment to analyze (in seconds)
        window_size : float
            Size of the sliding window for local analysis (in seconds)
        batch_size : int
            Number of participants to process before garbage collection
        memory_limit_gb : float
            Memory limit in GB before forcing cleanup
        """
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.min_segment_length = min_segment_length
        self.window_size = window_size
        self.batch_size = batch_size

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(memory_limit_gb)

        # Create output directories
        self.visualization_dir = os.path.join(output_folder, "visualizations")
        self.report_dir = os.path.join(output_folder, "reports")
        self.summary_dir = os.path.join(output_folder, "summaries")

        os.makedirs(self.visualization_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        os.makedirs(self.summary_dir, exist_ok=True)

        # Store minimal results for summary generation
        self.summary_data = {
            'participants': [],
            'channel_summary': [],  # Store only essential data
            'overall_stats': {}
        }

        # Create a file to progressively store results
        self.results_file = os.path.join(self.summary_dir, "progressive_results.json")
        self._initialize_results_file()

    def _initialize_results_file(self):
        """Initialize the progressive results file"""
        with open(self.results_file, 'w') as f:
            json.dump({"participants": {}, "metadata": {"created": datetime.now().isoformat()}}, f)

    def _save_participant_results(self, participant_id, results):
        """Save results for a single participant to the progressive file"""
        try:
            # Read existing data
            with open(self.results_file, 'r') as f:
                data = json.load(f)

            # Add new participant data (only essential info)
            data["participants"][participant_id] = self._extract_essential_data(results)

            # Write back to file
            with open(self.results_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save participant results: {e}")

    def _extract_essential_data(self, results):
        """Extract only essential data to reduce memory usage"""
        essential = {}
        for recording_type, channels in results.items():
            essential[recording_type] = {}
            for channel_name, analysis in channels.items():
                if 'quality' in analysis:
                    essential[recording_type][channel_name] = {
                        'channel_type': analysis['channel_type'],
                        'quality_score': analysis['quality']['quality_score'],
                        'quality_category': analysis['quality']['quality_category'],
                        'primary_issues': analysis['quality']['primary_issues'],
                        'subscores': analysis['quality']['subscores']
                    }
        return essential

    def find_all_recordings(self):
        """Find all BIOPAC .acq files in the data folder."""
        acq_files = glob.glob(os.path.join(self.data_folder, "*.acq"))
        recordings = []

        for file_path in acq_files:
            file_name = os.path.basename(file_path)

            # Extract participant ID and recording type from filename
            match = re.match(r'(.+?)_(main|baseline)\.acq', file_name)
            if match:
                participant_id = match.group(1)
                recording_type = match.group(2)

                recordings.append({
                    'file_path': file_path,
                    'file_name': file_name,
                    'participant_id': participant_id,
                    'recording_type': recording_type
                })

        # Group recordings by participant
        participants = {}
        for recording in recordings:
            participant_id = recording['participant_id']
            if participant_id not in participants:
                participants[participant_id] = []
            participants[participant_id].append(recording)

        self.summary_data['participants'] = list(participants.keys())

        return recordings

    def load_recording(self, file_path):
        """Load a BIOPAC .acq file with memory optimization."""
        try:
            data = bioread.read(file_path)
            return data
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return None

    def detect_channel_type(self, channel_name):
        """Detect the type of a channel based on its name."""
        if 'EDA' in channel_name:
            return 'EDA'
        elif 'EMG' in channel_name:
            return 'fEMG'
        else:
            return 'Unknown'

    def detect_flatline_segments(self, signal_data, fs, min_duration=0.5):
        """Detect segments where the signal is flat."""
        # Calculate the first derivative
        diff_signal = np.diff(signal_data)

        # Find points where the signal is not changing
        is_flat = np.abs(diff_signal) < 1e-10

        # Find the start and end indices of flatline segments
        flat_segments = []
        if any(is_flat):
            # Find the boundaries of flat segments
            flat_boundaries = np.diff(is_flat.astype(int))
            flat_starts = np.where(flat_boundaries == 1)[0] + 1
            flat_ends = np.where(flat_boundaries == -1)[0] + 1

            # Handle edge cases
            if is_flat[0]:
                flat_starts = np.insert(flat_starts, 0, 0)
            if is_flat[-1]:
                flat_ends = np.append(flat_ends, len(is_flat))

            # Filter segments by duration
            min_samples = int(fs * min_duration)
            for start, end in zip(flat_starts, flat_ends):
                if end - start >= min_samples:
                    flat_segments.append({
                        'start': start,
                        'end': end,
                        'duration': (end - start) / fs
                    })

        # Calculate metrics
        total_flat_duration = sum(seg['duration'] for seg in flat_segments)
        total_duration = len(signal_data) / fs
        percentage_flat = (total_flat_duration / total_duration) * 100 if total_duration > 0 else 0

        return {
            'segments': flat_segments,
            'total_flat_duration': total_flat_duration,
            'percentage_flat': percentage_flat,
            'has_significant_flatline': percentage_flat > 5
        }

    def detect_out_of_range(self, signal_data, channel_type):
        """Detect values that are outside the normal physiological range."""
        params = CHANNEL_PARAMETERS.get(channel_type, {})
        normal_range = params.get('normal_range', (-float('inf'), float('inf')))

        below_min = signal_data < normal_range[0]
        above_max = signal_data > normal_range[1]

        out_of_range = below_min | above_max
        percentage_out_of_range = (np.sum(out_of_range) / len(signal_data)) * 100

        return {
            'values_below_min': np.sum(below_min),
            'values_above_max': np.sum(above_max),
            'percentage_out_of_range': percentage_out_of_range,
            'has_significant_range_issues': percentage_out_of_range > 5
        }

    def detect_noise(self, signal_data, fs, channel_type):
        """Detect excessive noise in the signal."""
        params = CHANNEL_PARAMETERS.get(channel_type, {})
        expected_freq_range = params.get('expected_frequency_range', (0, fs/2))

        # Apply a high-pass filter to remove slow drift
        if channel_type == 'EDA':
            b, a = signal.butter(3, expected_freq_range[1]/(fs/2), 'highpass')
        else:
            b, a = signal.butter(3, [0.5/(fs/2), expected_freq_range[0]/(fs/2)], 'bandpass')

        try:
            noise_component = signal.filtfilt(b, a, signal_data)
        except Exception:
            noise_component = signal_data - np.convolve(signal_data,
                                                      np.ones(int(fs/2))/int(fs/2),
                                                      mode='same')

        # Calculate noise metrics
        signal_rms = np.sqrt(np.mean(signal_data**2))
        noise_rms = np.sqrt(np.mean(noise_component**2))

        if signal_rms > 0:
            snr_db = 20 * np.log10(signal_rms / (noise_rms if noise_rms > 0 else 1e-10))
        else:
            snr_db = 0

        # Power Spectral Density analysis (optimized)
        nperseg = min(len(signal_data), int(fs*2))
        freqs, psd = signal.welch(signal_data, fs, nperseg=nperseg)

        # Calculate power in expected and unexpected frequency bands
        mask_expected = (freqs >= expected_freq_range[0]) & (freqs <= expected_freq_range[1])
        mask_unexpected = ~mask_expected

        power_expected = np.sum(psd[mask_expected])
        power_unexpected = np.sum(psd[mask_unexpected])

        power_ratio = power_expected / (power_expected + power_unexpected) if (power_expected + power_unexpected) > 0 else 0

        return {
            'noise_rms': noise_rms,
            'snr_db': snr_db,
            'power_ratio': power_ratio,
            'has_significant_noise': snr_db < 10 or power_ratio < 0.7,
            'freq': freqs,
            'psd': psd,
            'expected_freq_range': expected_freq_range
        }

    def detect_abrupt_jumps(self, signal_data, fs, channel_type):
        """Detect abrupt jumps or motion artifacts in the signal."""
        params = CHANNEL_PARAMETERS.get(channel_type, {})
        max_jump = params.get('max_jump', float('inf'))

        # Calculate first derivative
        diff_signal = np.diff(signal_data)
        abs_diff = np.abs(diff_signal)

        # Identify jumps exceeding the threshold
        jump_locations = np.where(abs_diff > max_jump)[0]
        jump_magnitudes = abs_diff[jump_locations]

        # Group nearby jumps (within 100ms) as a single artifact
        artifacts = []
        if len(jump_locations) > 0:
            current_artifact = {
                'start': jump_locations[0],
                'end': jump_locations[0] + 1,
                'max_magnitude': jump_magnitudes[0]
            }

            for i in range(1, len(jump_locations)):
                if jump_locations[i] - current_artifact['end'] < int(fs * 0.1):
                    current_artifact['end'] = jump_locations[i] + 1
                    current_artifact['max_magnitude'] = max(current_artifact['max_magnitude'],
                                                           jump_magnitudes[i])
                else:
                    artifacts.append(current_artifact)
                    current_artifact = {
                        'start': jump_locations[i],
                        'end': jump_locations[i] + 1,
                        'max_magnitude': jump_magnitudes[i]
                    }

            artifacts.append(current_artifact)

        # Calculate metrics
        artifacts_per_minute = len(artifacts) / (len(signal_data) / fs / 60)
        max_jump_magnitude = max(jump_magnitudes) if len(jump_magnitudes) > 0 else 0

        return {
            'artifacts': artifacts,
            'artifacts_count': len(artifacts),
            'artifacts_per_minute': artifacts_per_minute,
            'max_jump_magnitude': max_jump_magnitude,
            'has_significant_jumps': artifacts_per_minute > 5
        }

    def analyze_peaks(self, signal_data, fs, channel_type):
        """Analyze peaks in the signal."""
        params = CHANNEL_PARAMETERS.get(channel_type, {})
        min_peaks_per_minute = params.get('min_peaks_per_minute', 0)
        max_peaks_per_minute = params.get('max_peaks_per_minute', float('inf'))

        # Preprocess signal based on channel type
        if channel_type == 'EDA':
            b, a = signal.butter(3, 1.0/(fs/2), 'lowpass')
            filtered_signal = signal.filtfilt(b, a, signal_data)
            peaks, properties = signal.find_peaks(filtered_signal, prominence=0.05, distance=int(fs*0.5))

        elif channel_type == 'fEMG':
            rectified_signal = np.abs(signal_data)
            b, a = signal.butter(3, 10.0/(fs/2), 'lowpass')
            envelope = signal.filtfilt(b, a, rectified_signal)
            peaks, properties = signal.find_peaks(envelope, height=np.std(envelope)*2, distance=int(fs*0.1))
            filtered_signal = envelope

        else:
            peaks, properties = signal.find_peaks(signal_data, distance=int(fs*0.5))
            filtered_signal = signal_data

        # Calculate metrics
        recording_duration_minutes = len(signal_data) / fs / 60
        peaks_per_minute = len(peaks) / recording_duration_minutes if recording_duration_minutes > 0 else 0

        too_few_peaks = peaks_per_minute < min_peaks_per_minute
        too_many_peaks = peaks_per_minute > max_peaks_per_minute

        return {
            'peaks': peaks,
            'filtered_signal': filtered_signal,
            'peak_properties': properties,
            'peaks_count': len(peaks),
            'peaks_per_minute': peaks_per_minute,
            'has_abnormal_peaks': too_few_peaks or too_many_peaks
        }

    def calculate_quality_score(self, analysis_results, channel_type):
        """Calculate an overall quality score for the signal."""
        params = CHANNEL_PARAMETERS.get(channel_type, {})
        weights = params.get('scoring_weights', {
            'flatline': 0.25,
            'noise': 0.25,
            'range': 0.20,
            'jumps': 0.20,
            'peaks': 0.10
        })

        # Initialize subscores
        subscores = {}

        # Flatline subscore
        flatline_pct = analysis_results.get('flatline', {}).get('percentage_flat', 0)
        subscores['flatline'] = max(0, 100 - flatline_pct)

        # Out-of-range subscore
        range_pct = analysis_results.get('range', {}).get('percentage_out_of_range', 0)
        subscores['range'] = max(0, 100 - range_pct)

        # Noise subscore
        snr_db = analysis_results.get('noise', {}).get('snr_db', 0)
        power_ratio = analysis_results.get('noise', {}).get('power_ratio', 0)

        if channel_type == 'EDA':
            subscores['noise'] = min(100, power_ratio * 100 * 1.3)
        else:
            subscores['noise'] = min(100, max(0, snr_db * 5))

        # Jumps/artifacts subscore
        artifacts_per_min = analysis_results.get('jumps', {}).get('artifacts_per_minute', 0)
        max_artifacts = 10
        subscores['jumps'] = max(0, 100 - (artifacts_per_min / max_artifacts) * 100)

        # Peaks subscore
        has_abnormal_peaks = analysis_results.get('peaks', {}).get('has_abnormal_peaks', False)
        peaks_per_min = analysis_results.get('peaks', {}).get('peaks_per_minute', 0)

        if channel_type == 'EDA':
            if peaks_per_min > 10:
                subscores['peaks'] = max(0, 100 - (peaks_per_min - 10) * 10)
            else:
                subscores['peaks'] = 100
        else:
            min_peaks = params.get('min_peaks_per_minute', 5)
            max_peaks = params.get('max_peaks_per_minute', 300)

            if peaks_per_min < min_peaks:
                subscores['peaks'] = (peaks_per_min / min_peaks) * 100
            elif peaks_per_min > max_peaks:
                subscores['peaks'] = max(0, 100 - ((peaks_per_min - max_peaks) / max_peaks) * 100)
            else:
                subscores['peaks'] = 100

        # Calculate weighted average
        total_weight = sum(weights.values())
        weighted_score = sum(subscores[metric] * weights.get(metric, 0) for metric in subscores) / total_weight

        quality_score = int(round(weighted_score))

        # Determine quality category
        if quality_score >= 80:
            quality_category = "Excellent"
        elif quality_score >= 60:
            quality_category = "Good"
        elif quality_score >= 40:
            quality_category = "Fair"
        elif quality_score >= 20:
            quality_category = "Poor"
        else:
            quality_category = "Very Poor"

        # Determine primary issues
        issues = []
        if subscores['flatline'] < 70:
            issues.append("Signal Flatline")
        if subscores['noise'] < 70:
            issues.append("Excessive Noise")
        if subscores['range'] < 70:
            issues.append("Out-of-Range Values")
        if subscores['jumps'] < 70:
            issues.append("Abrupt Jumps/Artifacts")
        if subscores['peaks'] < 70:
            issues.append("Abnormal Peak Pattern")

        return {
            'quality_score': quality_score,
            'quality_category': quality_category,
            'subscores': subscores,
            'primary_issues': issues
        }

    def analyze_channel(self, channel, recording_info):
        """Analyze the quality of a single channel with memory optimization."""
        channel_type = self.detect_channel_type(channel.name)
        if channel_type == 'Unknown':
            return None

        signal_data = channel.data
        fs = channel.samples_per_second

        # Skip non-data channels or marker channels
        if "marker" in channel.name.lower() or len(signal_data) < 10:
            return None

        print(f"Analyzing {channel_type} channel: {channel.name}")

        # Run various quality checks
        flatline_results = self.detect_flatline_segments(signal_data, fs)
        range_results = self.detect_out_of_range(signal_data, channel_type)
        noise_results = self.detect_noise(signal_data, fs, channel_type)
        jump_results = self.detect_abrupt_jumps(signal_data, fs, channel_type)
        peak_results = self.analyze_peaks(signal_data, fs, channel_type)

        # Combine results (without storing large arrays)
        analysis_results = {
            'channel_name': channel.name,
            'channel_type': channel_type,
            'sample_rate': fs,
            'signal_length': len(signal_data),
            'signal_duration': len(signal_data) / fs,
            'signal_data': signal_data,  # This will be removed after visualization
            'flatline': flatline_results,
            'range': range_results,
            'noise': noise_results,
            'jumps': jump_results,
            'peaks': peak_results
        }

        # Calculate overall quality
        quality_results = self.calculate_quality_score(analysis_results, channel_type)
        analysis_results['quality'] = quality_results

        return analysis_results

    def create_channel_visualization(self, analysis_results, recording_info):
        """Create the per-channel visualization, one figure at a time to limit memory."""
        channel_name = analysis_results['channel_name']
        channel_type = analysis_results['channel_type']
        signal_data = analysis_results['signal_data']
        fs = analysis_results['sample_rate']

        # Create a time axis
        time_axis = np.arange(len(signal_data)) / fs

        # Create figure with optimized settings
        plt.ioff()  # Turn off interactive mode to save memory
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig)

        # Extract quality information
        quality_score = analysis_results['quality']['quality_score']
        quality_category = analysis_results['quality']['quality_category']
        primary_issues = analysis_results['quality']['primary_issues']

        participant_id = recording_info['participant_id']
        recording_type = recording_info['recording_type']

        # Title with quality information
        title = (f"Channel Quality Analysis: {channel_name} ({channel_type})\n"
                f"Participant: {participant_id} | Recording: {recording_type}\n"
                f"Quality Score: {quality_score}/100 - {quality_category}")
        if primary_issues:
            title += f" | Issues: {', '.join(primary_issues)}"

        fig.suptitle(title, fontsize=16)

        # Panel 1: Raw signal with artifacts marked
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_axis, signal_data, 'b-', alpha=0.7)
        ax1.set_title("Raw Signal with Detected Issues")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")

        # Mark flatline segments
        for segment in analysis_results['flatline']['segments']:
            start_time = segment['start'] / fs
            end_time = segment['end'] / fs
            duration = segment['duration']

            rect = Rectangle((start_time, ax1.get_ylim()[0]),
                           duration,
                           ax1.get_ylim()[1] - ax1.get_ylim()[0],
                           facecolor='red', alpha=0.3)
            ax1.add_patch(rect)

            if duration > 1:
                ax1.text(start_time + duration/2, ax1.get_ylim()[1] * 0.9,
                       f"Flatline\n{duration:.1f}s",
                       ha='center', va='top',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Mark jump/motion artifacts
        for artifact in analysis_results['jumps']['artifacts']:
            start_time = artifact['start'] / fs
            magnitude = artifact['max_magnitude']

            ax1.axvline(x=start_time, color='orange', linestyle='--', alpha=0.7)
            ax1.text(start_time, ax1.get_ylim()[1] * 0.8,
                   f"Jump\n{magnitude:.1f}",
                   ha='right', va='top', rotation=90,
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # Panel 2: Signal distribution
        ax2 = fig.add_subplot(gs[0, 1])
        hist, bins, _ = ax2.hist(signal_data, bins=50, density=True, alpha=0.7, color='blue')

        # Fit a normal distribution
        mu, sigma = stats.norm.fit(signal_data)
        x = np.linspace(min(signal_data), max(signal_data), 100)
        pdf = stats.norm.pdf(x, mu, sigma)
        ax2.plot(x, pdf, 'r-', linewidth=2,
                label=f'Normal: μ={mu:.2f}, σ={sigma:.2f}')

        # Add channel-specific range guidelines
        params = CHANNEL_PARAMETERS.get(channel_type, {})
        normal_range = params.get('normal_range', (None, None))

        if normal_range[0] is not None:
            ax2.axvline(x=normal_range[0], color='green', linestyle='--', label=f'Min normal: {normal_range[0]}')
        if normal_range[1] is not None:
            ax2.axvline(x=normal_range[1], color='red', linestyle='--', label=f'Max normal: {normal_range[1]}')

        ax2.set_title("Signal Distribution")
        ax2.set_xlabel("Amplitude")
        ax2.set_ylabel("Density")
        ax2.legend()

        # Panel 3: Power spectral density analysis
        ax3 = fig.add_subplot(gs[1, 0])
        freqs = analysis_results['noise']['freq']
        psd = analysis_results['noise']['psd']
        expected_freq_range = analysis_results['noise']['expected_freq_range']

        ax3.semilogy(freqs, psd, 'b-', alpha=0.7)
        ax3.set_title("Power Spectral Density")
        ax3.set_xlabel("Frequency (Hz)")
        ax3.set_ylabel("Power Spectral Density (dB/Hz)")

        ax3.axvspan(expected_freq_range[0], expected_freq_range[1],
                   alpha=0.2, color='green', label='Expected frequency range')

        power_ratio = analysis_results['noise']['power_ratio']
        snr_db = analysis_results['noise']['snr_db']

        ax3.text(0.05, 0.95, f"Power in expected band: {power_ratio*100:.1f}%\nSNR: {snr_db:.1f} dB",
                transform=ax3.transAxes, va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax3.legend()

        # Panel 4: First derivative analysis
        ax4 = fig.add_subplot(gs[1, 1])
        diff_signal = np.diff(signal_data)
        diff_time = time_axis[:-1] + 1/(2*fs)

        ax4.plot(diff_time, diff_signal, 'g-', alpha=0.7)
        ax4.set_title("Rate of Change (First Derivative)")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Amplitude change")

        max_jump = CHANNEL_PARAMETERS.get(channel_type, {}).get('max_jump', None)
        if max_jump is not None:
            ax4.axhline(y=max_jump, color='r', linestyle='--', label=f'Jump threshold: {max_jump}')
            ax4.axhline(y=-max_jump, color='r', linestyle='--')

            artifacts_count = analysis_results['jumps']['artifacts_count']
            artifacts_per_min = analysis_results['jumps']['artifacts_per_minute']
            ax4.text(0.05, 0.95, f"Detected jumps: {artifacts_count}\n({artifacts_per_min:.1f} per minute)",
                    transform=ax4.transAxes, va='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax4.legend()

        # Panel 5: Signal variance over time
        ax5 = fig.add_subplot(gs[2, 0])

        window_samples = int(fs * self.window_size)
        step_samples = max(1, int(window_samples / 4))
        n_windows = (len(signal_data) - window_samples) // step_samples + 1

        variances = []
        window_times = []

        for i in range(n_windows):
            start_idx = i * step_samples
            end_idx = start_idx + window_samples

            window_data = signal_data[start_idx:end_idx]
            window_var = np.var(window_data)

            variances.append(window_var)
            window_times.append(time_axis[start_idx + window_samples//2])

        ax5.plot(window_times, variances, 'b-', alpha=0.7)
        ax5.set_title("Signal Variance Over Time")
        ax5.set_xlabel("Time (s)")
        ax5.set_ylabel("Variance")

        min_variance = CHANNEL_PARAMETERS.get(channel_type, {}).get('min_variance', None)
        max_variance = CHANNEL_PARAMETERS.get(channel_type, {}).get('max_variance', None)

        if min_variance is not None:
            ax5.axhline(y=min_variance, color='r', linestyle='--',
                       label=f'Min variance: {min_variance}')
        if max_variance is not None:
            ax5.axhline(y=max_variance, color='orange', linestyle='--',
                       label=f'Max variance: {max_variance}')

        ax5.legend()

        # Panel 6: Peak analysis
        ax6 = fig.add_subplot(gs[2, 1])

        filtered_signal = analysis_results['peaks']['filtered_signal']
        peaks = analysis_results['peaks']['peaks']

        ax6.plot(time_axis, filtered_signal, 'b-', alpha=0.7)

        if len(peaks) > 0:
            ax6.plot(time_axis[peaks], filtered_signal[peaks], 'ro', label='Detected peaks')

        ax6.set_title(f"Peak Analysis ({channel_type}-specific)")
        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Amplitude")

        peaks_count = analysis_results['peaks']['peaks_count']
        peaks_per_min = analysis_results['peaks']['peaks_per_minute']

        min_peaks = CHANNEL_PARAMETERS.get(channel_type, {}).get('min_peaks_per_minute', None)
        max_peaks = CHANNEL_PARAMETERS.get(channel_type, {}).get('max_peaks_per_minute', None)

        info_text = f"Detected peaks: {peaks_count}\n({peaks_per_min:.1f} per minute)"
        if min_peaks is not None and max_peaks is not None:
            info_text += f"\nExpected range: {min_peaks}-{max_peaks} per minute"

        ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax6.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    def create_html_report(self, analysis_results, recording_info):
        """Create an HTML report for a channel analysis with memory optimization."""
        channel_name = analysis_results['channel_name']
        channel_type = analysis_results['channel_type']

        # Extract quality information
        quality_score = analysis_results['quality']['quality_score']
        quality_category = analysis_results['quality']['quality_category']
        subscores = analysis_results['quality']['subscores']
        primary_issues = analysis_results['quality']['primary_issues']

        # Generate visualization and convert to base64 for embedding
        fig = self.create_channel_visualization(analysis_results, recording_info)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)  # Immediately close to free memory

        # Clear the buffer
        buf.close()
        del buf

        # Create quality badge color
        badge_color = "green"
        if quality_score < 60:
            badge_color = "orange"
        if quality_score < 40:
            badge_color = "red"

        # Create HTML report (same structure as original but optimized)
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Signal Quality Report: {channel_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ padding: 10px; margin-bottom: 20px; border-bottom: 1px solid #ddd; }}
                .quality-badge {{
                    display: inline-block;
                    padding: 8px 15px;
                    border-radius: 4px;
                    background-color: {badge_color};
                    color: white;
                    font-weight: bold;
                }}
                .quality-details {{
                    display: flex;
                    flex-wrap: wrap;
                    margin: 20px 0;
                }}
                .quality-details div {{
                    flex: 1 1 200px;
                    margin: 10px;
                    padding: 15px;
                    background-color: #f8f8f8;
                    border-radius: 4px;
                }}
                .issue-list {{
                    background-color: #fff3cd;
                    padding: 10px 20px;
                    border-radius: 4px;
                    margin-bottom: 20px;
                }}
                .progress-bar {{
                    height: 20px;
                    background-color: #e0e0e0;
                    border-radius: 10px;
                    margin-top: 5px;
                }}
                .progress-fill {{
                    height: 100%;
                    border-radius: 10px;
                    background-color: #4CAF50;
                }}
                .visualization {{
                    margin: 20px 0;
                    text-align: center;
                }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Signal Quality Analysis Report</h1>
                    <p>
                        <strong>Channel:</strong> {channel_name} ({channel_type}) |
                        <strong>Participant:</strong> {recording_info['participant_id']} |
                        <strong>Recording:</strong> {recording_info['recording_type']}
                    </p>
                </div>

                <h2>Quality Assessment: <span class="quality-badge">{quality_score}/100 - {quality_category}</span></h2>
        """

        # Add issue list if there are any
        if primary_issues:
            html += f"""
                <div class="issue-list">
                    <h3>Primary Issues Detected:</h3>
                    <ul>
            """

            for issue in primary_issues:
                html += f"<li>{issue}</li>"

            html += """
                    </ul>
                </div>
            """

        # Add quality subscores
        html += """
                <div class="quality-details">
        """

        for metric, score in subscores.items():
            html += f"""
                    <div>
                        <h3>{metric.capitalize()}</h3>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {score}%; background-color: {'#4CAF50' if score >= 70 else '#FFC107' if score >= 40 else '#F44336'}"></div>
                        </div>
                        <p>{score:.1f}/100</p>
                    </div>
            """

        html += """
                </div>
        """

        # Add signal statistics
        html += f"""
                <h2>Signal Statistics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Signal Duration</td>
                        <td>{analysis_results['signal_duration']:.2f} seconds</td>
                    </tr>
                    <tr>
                        <td>Sample Rate</td>
                        <td>{analysis_results['sample_rate']} Hz</td>
                    </tr>
                    <tr>
                        <td>Flatline Duration</td>
                        <td>{analysis_results['flatline']['total_flat_duration']:.2f} seconds ({analysis_results['flatline']['percentage_flat']:.1f}% of recording)</td>
                    </tr>
                    <tr>
                        <td>Out-of-Range Values</td>
                        <td>{analysis_results['range']['percentage_out_of_range']:.1f}% of samples</td>
                    </tr>
                    <tr>
                        <td>Signal-to-Noise Ratio</td>
                        <td>{analysis_results['noise']['snr_db']:.2f} dB</td>
                    </tr>
                    <tr>
                        <td>Detected Jumps/Artifacts</td>
                        <td>{analysis_results['jumps']['artifacts_count']} ({analysis_results['jumps']['artifacts_per_minute']:.1f} per minute)</td>
                    </tr>
                    <tr>
                        <td>Detected Peaks</td>
                        <td>{analysis_results['peaks']['peaks_count']} ({analysis_results['peaks']['peaks_per_minute']:.1f} per minute)</td>
                    </tr>
                </table>
        """

        # Add visualization
        html += f"""
                <div class="visualization">
                    <h2>Signal Visualization and Analysis</h2>
                    <img src="data:image/png;base64,{img_base64}" alt="Signal Analysis" style="max-width:100%">
                </div>

                <div class="header">
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Clear the base64 string from memory
        del img_base64

        return html

    def analyze_recording(self, recording_info):
        """Analyze all channels in a recording with memory optimization."""
        participant_id = recording_info['participant_id']
        recording_type = recording_info['recording_type']
        file_path = recording_info['file_path']

        print(f"\nAnalyzing recording: {os.path.basename(file_path)}")

        # Load recording
        data = self.load_recording(file_path)
        if not data:
            print(f"Failed to load {file_path}")
            return {}

        # Analyze each channel
        results = {}

        for channel in data.channels:
            # Skip non-data channels or marker channels
            if "marker" in channel.name.lower() or len(channel.data) < 10:
                continue

            channel_results = self.analyze_channel(channel, recording_info)
            if channel_results:
                results[channel.name] = channel_results

                # Create and save visualization
                fig = self.create_channel_visualization(channel_results, recording_info)

                viz_filename = f"{participant_id}_{recording_type}_{channel.name}.png"
                viz_path = os.path.join(self.visualization_dir, viz_filename)

                fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close(fig)  # Important: close immediately

                # Create and save HTML report
                html_report = self.create_html_report(channel_results, recording_info)

                report_filename = f"{participant_id}_{recording_type}_{channel.name}.html"
                report_path = os.path.join(self.report_dir, report_filename)

                with open(report_path, 'w') as f:
                    f.write(html_report)

                # Remove large data arrays from memory to reduce memory usage
                if 'signal_data' in channel_results:
                    del channel_results['signal_data']
                if 'filtered_signal' in channel_results.get('peaks', {}):
                    del channel_results['peaks']['filtered_signal']
                if 'freq' in channel_results.get('noise', {}):
                    del channel_results['noise']['freq']
                if 'psd' in channel_results.get('noise', {}):
                    del channel_results['noise']['psd']

        # Force cleanup of the loaded data
        del data
        gc.collect()

        return results

    def analyze_batch_of_participants(self, recordings_batch):
        """Analyze a batch of participants with memory management."""
        batch_results = {}

        # Group recordings by participant
        participants_in_batch = {}
        for recording in recordings_batch:
            participant_id = recording['participant_id']
            if participant_id not in participants_in_batch:
                participants_in_batch[participant_id] = []
            participants_in_batch[participant_id].append(recording)

        # Process each participant in the batch
        for participant_id, participant_recordings in participants_in_batch.items():
            batch_results[participant_id] = {}

            for recording in participant_recordings:
                recording_type = recording['recording_type']

                # Analyze recording
                recording_results = self.analyze_recording(recording)
                batch_results[participant_id][recording_type] = recording_results

            # Save participant results progressively
            self._save_participant_results(participant_id, batch_results[participant_id])

            # Memory check after each participant
            if self.memory_monitor.check_memory_limit():
                print(f"Memory limit reached after participant {participant_id}, forcing cleanup...")
                self.memory_monitor.force_cleanup()

        return batch_results

    def run_analysis(self):
        """Run the analysis on all recordings with memory optimization."""
        # Monitor initial memory
        if ENABLE_MEMORY_MONITORING:
            self.memory_monitor.print_memory_status("at start")

        # Find all recordings
        recordings = self.find_all_recordings()

        print(f"Found {len(recordings)} recordings from {len(self.summary_data['participants'])} participants")
        print(f"Processing in batches of {self.batch_size} participants")

        # Group recordings by participant
        participants = {}
        for recording in recordings:
            participant_id = recording['participant_id']
            if participant_id not in participants:
                participants[participant_id] = []
            participants[participant_id].append(recording)

        participant_list = list(participants.keys())
        total_participants = len(participant_list)

        # Process participants in batches
        all_results = {}

        for batch_start in range(0, total_participants, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_participants)
            batch_participants = participant_list[batch_start:batch_end]

            print(f"\nProcessing batch {batch_start//self.batch_size + 1}/{(total_participants + self.batch_size - 1)//self.batch_size}")
            print(f"Participants {batch_start + 1}-{batch_end} of {total_participants}")

            # Create batch of recordings
            batch_recordings = []
            for participant_id in batch_participants:
                batch_recordings.extend(participants[participant_id])

            # Process batch
            batch_results = self.analyze_batch_of_participants(batch_recordings)

            # Merge results (keeping only essential data)
            for participant_id, participant_results in batch_results.items():
                all_results[participant_id] = participant_results

            # Memory monitoring and cleanup after each batch
            if ENABLE_MEMORY_MONITORING:
                self.memory_monitor.print_memory_status(f"after batch {batch_start//self.batch_size + 1}")

            # Force cleanup after each batch
            self.memory_monitor.force_cleanup()

            print(f"Batch {batch_start//self.batch_size + 1} completed. Memory cleaned up.")

        print(f"\nAll individual analyses complete! Generating summaries...")

        # Generate summary using the optimized summary generator
        try:
            from pathlib import Path

            # Import and use the optimized summary generator
            summary_generator = OptimizedSummaryGenerator(self.output_folder)
            summary_generator.run()

        except Exception as e:
            print(f"Warning: Could not generate summary: {e}")
            print("You can run the summary generator separately later.")

        print("\nAnalysis complete")
        print(f"  - Individual reports saved to: {self.report_dir}")
        print(f"  - Visualizations saved to: {self.visualization_dir}")
        print(f"  - Progressive results saved to: {self.results_file}")

        if ENABLE_MEMORY_MONITORING:
            self.memory_monitor.print_memory_status("at completion")

        return all_results

class OptimizedSummaryGenerator:
    """
    Memory-optimized summary generator that reads from progressive results.
    """

    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.report_dir = os.path.join(output_folder, "reports")
        self.summary_dir = os.path.join(output_folder, "summaries")
        self.results_file = os.path.join(self.summary_dir, "progressive_results.json")

    def load_progressive_results(self):
        """Load results from the progressive results file."""
        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            return data.get("participants", {})
        except Exception as e:
            print(f"Could not load progressive results: {e}")
            return {}

    def create_summary_html_report(self, results):
        """Create a simplified summary HTML report."""
        # Calculate basic statistics
        total_channels = 0
        quality_scores = []
        channel_types = {}
        issues_count = {}

        for participant_id, rec_results in results.items():
            for recording_type, channels in rec_results.items():
                for channel_name, analysis in channels.items():
                    total_channels += 1
                    quality_scores.append(analysis['quality_score'])

                    ch_type = analysis['channel_type']
                    if ch_type not in channel_types:
                        channel_types[ch_type] = 0
                    channel_types[ch_type] += 1

                    for issue in analysis['primary_issues']:
                        if issue not in issues_count:
                            issues_count[issue] = 0
                        issues_count[issue] += 1

        avg_quality = np.mean(quality_scores) if quality_scores else 0

        # Create simplified HTML report
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Study-wide Signal Quality Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ padding: 10px; margin-bottom: 20px; border-bottom: 1px solid #ddd; }}
                .summary-stats {{
                    display: flex;
                    flex-wrap: wrap;
                    margin: 20px 0;
                }}
                .summary-stats div {{
                    flex: 1 1 200px;
                    margin: 10px;
                    padding: 15px;
                    background-color: #f8f8f8;
                    border-radius: 4px;
                    text-align: center;
                }}
                .quality-badge {{
                    display: inline-block;
                    padding: 8px 15px;
                    border-radius: 4px;
                    background-color: {'green' if avg_quality >= 60 else 'orange' if avg_quality >= 40 else 'red'};
                    color: white;
                    font-weight: bold;
                    font-size: 1.2em;
                }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Study-wide Signal Quality Summary</h1>
                    <p>Analysis of psychophysiological data across all participants and recordings</p>
                </div>

                <div class="summary-stats">
                    <div>
                        <h2>{len(results)}</h2>
                        <p>Participants</p>
                    </div>
                    <div>
                        <h2>{total_channels}</h2>
                        <p>Channels Analyzed</p>
                    </div>
                    <div>
                        <h2><span class="quality-badge">{avg_quality:.1f}/100</span></h2>
                        <p>Average Quality Score</p>
                    </div>
                </div>

                <h2>Channel Type Distribution</h2>
                <table>
                    <tr>
                        <th>Channel Type</th>
                        <th>Count</th>
                    </tr>
        """

        for ch_type, count in channel_types.items():
            html += f"""
                    <tr>
                        <td>{ch_type}</td>
                        <td>{count}</td>
                    </tr>
            """

        html += """
                </table>

                <h2>Common Issues Found</h2>
                <table>
                    <tr>
                        <th>Issue</th>
                        <th>Occurrences</th>
                        <th>Percentage of Channels</th>
                    </tr>
        """

        for issue, count in sorted(issues_count.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_channels) * 100 if total_channels > 0 else 0
            html += f"""
                    <tr>
                        <td>{issue}</td>
                        <td>{count}</td>
                        <td>{percentage:.1f}%</td>
                    </tr>
            """

        html += f"""
                </table>

                <div class="header">
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def run(self):
        """Run the optimized summary generation."""
        print("Generating optimized summary from progressive results...")

        # Load results
        results = self.load_progressive_results()

        if not results:
            print("No results found. Make sure the analysis completed successfully.")
            return

        # Create summary HTML report
        summary_html = self.create_summary_html_report(results)
        summary_report_path = os.path.join(self.report_dir, "study_summary.html")

        with open(summary_report_path, 'w', encoding='utf-8') as f:
            f.write(summary_html)

        print(f"Summary report created: {summary_report_path}")

def main():
    """Main function to run the optimized analysis."""
    print(f"Running memory-optimized signal quality analysis on data from: {DATA_FOLDER}")
    print(f"Results will be saved to: {OUTPUT_FOLDER}")
    print(f"Batch size: {BATCH_SIZE} participants")
    print(f"Memory limit: {MEMORY_LIMIT_GB} GB")

    # Create analyzer and run analysis
    analyzer = OptimizedSignalQualityAnalyzer(
        data_folder=DATA_FOLDER,
        output_folder=OUTPUT_FOLDER,
        window_size=WINDOW_SIZE,
        min_segment_length=MIN_SEGMENT_LENGTH,
        batch_size=BATCH_SIZE,
        memory_limit_gb=MEMORY_LIMIT_GB
    )

    analyzer.run_analysis()

if __name__ == "__main__":
    main()


# %%
