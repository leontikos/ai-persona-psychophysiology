# %% signal_cleaner.py

"""
Memory-Optimized Psychophysiological Signal Cleaning and Export Tool

This script takes the quality assessment results and applies signal cleaning
to all channels, then exports the cleaned signals to a separate folder
for downstream analysis.

MEMORY OPTIMIZATIONS:
- Process one channel at a time with immediate cleanup
- Use memory-mapped arrays for large datasets
- Minimize copies and intermediate arrays
- Explicit memory management and garbage collection
- Chunked processing for very large signals
"""

import bioread
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sp_signal
import os
import json
import glob
from pathlib import Path
from datetime import datetime
import warnings
import gc
import psutil
import tempfile
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
INPUT_DATA_FOLDER = os.path.join(DATA_ROOT, "psychophysiology_signal")
OUTPUT_FOLDER = os.path.join(DATA_ROOT, "cleaned_signals")
QUALITY_RESULTS_FILE = os.path.join(
    DATA_ROOT, "signal_quality_analysis", "summaries", "progressive_results.json"
)

# Memory management parameters
MEMORY_CONFIG = {
    'max_memory_usage_gb': 30,  # Maximum memory to use in GB
    'chunk_size_samples': 1000000,  # Process signals in chunks if larger than this
    'use_memmap': True,  # Use memory mapping for large arrays
    'temp_dir': '/tmp/signal_processing',  # Temporary directory for memory maps
    'force_gc_frequency': 5,  # Force garbage collection every N channels
}

# Cleaning parameters.
# Sample-wise artifact interpolation is disabled: replacing samples more than 3 SD from the
# mean by straight lines clips the peaks of genuine fEMG bursts and the tops of large skin
# conductance responses. Offset removal is applied to fEMG only, so EDA keeps its absolute
# level in microsiemens. The reported event-related results are computed from the raw
# recordings by analysis/03_statistical_analysis/participant_level_*.py.
CLEANING_CONFIG = {
    'remove_dc_offset': True,          # applied to fEMG only (see clean_channel)
    'apply_channel_filters': True,
    'remove_artifacts': False,
    'interpolate_artifacts': True,
    'quality_threshold': 10,
    'manual_review_threshold': 20,
    'artifact_threshold': 3,
    'max_interpolation_length': 1.0,
    'create_validation_plots': True,
}

# Channel-specific filter parameters.
# No software notch filter: the recordings already carry AcqKnowledge's real-time
# line-frequency comb filter (50 Hz and harmonics).
FILTER_PARAMS = {
    'EDA': {
        'type': 'lowpass',
        'cutoff': 5.0,
        'order': 4,
        'additional_filters': []
    },
    'fEMG': {
        'type': 'bandpass',
        'cutoff': [20, 500],
        'order': 4,
        'additional_filters': []
    }
}

TARGET_CHANNELS = ['EDA2111000305', 'fEMG2212000226', 'fEMG2212000228']

class MemoryEfficientSignalCleaner:
    """
    Memory-optimized signal cleaning utility for psychophysiological data
    """

    def __init__(self, input_folder, output_folder, quality_results_file=None):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.quality_results_file = quality_results_file

        # Create temp directory for memory maps
        os.makedirs(MEMORY_CONFIG['temp_dir'], exist_ok=True)

        # Create output directory structure
        self.cleaned_signals_dir = os.path.join(output_folder, "cleaned_signals")
        self.cleaning_reports_dir = os.path.join(output_folder, "cleaning_reports")
        self.export_logs_dir = os.path.join(output_folder, "export_logs")
        self.validation_plots_dir = os.path.join(output_folder, "validation_plots")
        self.manual_review_dir = os.path.join(output_folder, "manual_review_flagged")

        for dir_path in [self.cleaned_signals_dir, self.cleaning_reports_dir,
                        self.export_logs_dir, self.validation_plots_dir, self.manual_review_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # Load quality assessment results
        self.quality_data = self._load_quality_results()

        # Initialize cleaning log
        self.cleaning_log = {
            'start_time': datetime.now().isoformat(),
            'processed_files': {},
            'errors': [],
            'summary': {},
            'manual_review_flagged': [],
            'memory_stats': []
        }

        # Memory tracking
        self.process = psutil.Process()
        self.channel_counter = 0

    def get_memory_usage_gb(self):
        """Get current memory usage in GB"""
        return self.process.memory_info().rss / (1024**3)

    def log_memory_usage(self, context=""):
        """Log current memory usage"""
        memory_gb = self.get_memory_usage_gb()
        memory_info = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'memory_gb': memory_gb,
            'available_gb': psutil.virtual_memory().available / (1024**3)
        }
        self.cleaning_log['memory_stats'].append(memory_info)

        if memory_gb > MEMORY_CONFIG['max_memory_usage_gb']:
            print(f"WARNING: Memory usage high: {memory_gb:.1f}GB - {context}")

        return memory_gb

    def force_cleanup(self):
        """Aggressive memory cleanup"""
        # Close any open matplotlib figures
        plt.close('all')
        plt.clf()
        plt.cla()

        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()

        # Clear numpy cache if it exists
        if hasattr(np, 'seterrobj'):
            np.seterrobj(None)

    def _load_quality_results(self):
        """Load quality assessment results"""
        if not self.quality_results_file or not os.path.exists(self.quality_results_file):
            print("No quality results file found. Will clean all signals with default parameters.")
            return {}

        try:
            with open(self.quality_results_file, 'r') as f:
                data = json.load(f)
            print(f"Loaded quality results for {len(data.get('participants', {}))} participants")
            return data.get('participants', {})
        except Exception as e:
            print(f"Error loading quality results: {e}")
            return {}

    def detect_channel_type(self, channel_name):
        """Detect channel type from name"""
        channel_upper = channel_name.upper()

        if 'EDA' in channel_upper or 'GSR' in channel_upper:
            return 'EDA'
        elif ('EMG' in channel_upper or 'ZYGOMATIC' in channel_upper or
              'CORRUGATOR' in channel_upper or 'ZYG' in channel_upper or
              'CORRUG' in channel_upper):
            return 'fEMG'
        else:
            return 'unknown'

    def create_memmap_array(self, size, dtype=np.float64):
        """Create a memory-mapped array for large datasets"""
        if not MEMORY_CONFIG['use_memmap'] or size < MEMORY_CONFIG['chunk_size_samples']:
            return None

        temp_file = tempfile.NamedTemporaryFile(
            dir=MEMORY_CONFIG['temp_dir'],
            delete=False,
            suffix='.dat'
        )
        temp_file.close()

        return np.memmap(temp_file.name, dtype=dtype, mode='w+', shape=(size,))

    def process_signal_in_chunks(self, signal, processing_func, chunk_size=None, **kwargs):
        """Process large signals in chunks to manage memory"""
        if chunk_size is None:
            chunk_size = MEMORY_CONFIG['chunk_size_samples']

        if len(signal) <= chunk_size:
            return processing_func(signal, **kwargs)

        # Create output array (use memmap if large)
        output_array = self.create_memmap_array(len(signal), signal.dtype)
        if output_array is None:
            output_array = np.empty_like(signal)

        # Process in chunks
        for start_idx in range(0, len(signal), chunk_size):
            end_idx = min(start_idx + chunk_size, len(signal))

            # Add overlap for filtering continuity
            overlap = min(1000, chunk_size // 10)  # 10% overlap or 1000 samples
            chunk_start = max(0, start_idx - overlap)
            chunk_end = min(len(signal), end_idx + overlap)

            # Extract chunk
            chunk = signal[chunk_start:chunk_end]

            # Process chunk
            processed_chunk = processing_func(chunk, **kwargs)

            # Extract the relevant portion (removing overlap)
            if start_idx == 0:
                # First chunk
                relevant_part = processed_chunk[:end_idx-start_idx]
            elif end_idx == len(signal):
                # Last chunk
                relevant_part = processed_chunk[start_idx-chunk_start:]
            else:
                # Middle chunk
                relevant_part = processed_chunk[start_idx-chunk_start:end_idx-chunk_start]

            # Store in output
            output_array[start_idx:end_idx] = relevant_part

            # Clean up chunk data
            del chunk, processed_chunk, relevant_part
            gc.collect()

        return output_array

    def remove_dc_offset(self, signal):
        """Remove DC offset from signal (memory efficient)"""
        # Use online algorithm for mean calculation to handle large arrays
        if len(signal) > MEMORY_CONFIG['chunk_size_samples']:
            # Calculate mean in chunks
            chunk_size = MEMORY_CONFIG['chunk_size_samples']
            total_sum = 0.0
            for i in range(0, len(signal), chunk_size):
                chunk = signal[i:i+chunk_size]
                total_sum += np.sum(chunk)
                del chunk
            mean_value = total_sum / len(signal)
        else:
            mean_value = np.mean(signal)

        return signal - mean_value

    def apply_channel_specific_filter_chunked(self, signal, sampling_rate, channel_type):
        """Apply channel-specific filtering with chunked processing"""
        if channel_type not in FILTER_PARAMS:
            return signal

        def filter_func(chunk_signal, **kwargs):
            return self.apply_channel_specific_filter_single(chunk_signal, sampling_rate, channel_type)

        if len(signal) > MEMORY_CONFIG['chunk_size_samples']:
            return self.process_signal_in_chunks(signal, filter_func)
        else:
            return self.apply_channel_specific_filter_single(signal, sampling_rate, channel_type)

    def apply_channel_specific_filter_single(self, signal, sampling_rate, channel_type):
        """Apply filtering to a single chunk"""
        params = FILTER_PARAMS[channel_type]
        filtered_signal = signal.copy()

        try:
            nyquist = sampling_rate / 2

            if params['type'] == 'lowpass':
                cutoff = min(params['cutoff'], nyquist * 0.95)
                b, a = sp_signal.butter(params['order'], cutoff / nyquist, btype='low')
                filtered_signal = sp_signal.filtfilt(b, a, filtered_signal)

            elif params['type'] == 'bandpass':
                low_cutoff = max(params['cutoff'][0], 0.1) / nyquist
                high_cutoff = min(params['cutoff'][1], nyquist * 0.95) / nyquist

                if low_cutoff < high_cutoff:
                    b, a = sp_signal.butter(params['order'], [low_cutoff, high_cutoff], btype='band')
                    filtered_signal = sp_signal.filtfilt(b, a, filtered_signal)

            # Apply additional filters
            for additional_filter in params.get('additional_filters', []):
                if additional_filter['type'] == 'notch':
                    freq = additional_filter['freq']
                    if freq < nyquist:
                        Q = additional_filter.get('Q', 30)
                        b, a = sp_signal.iirnotch(freq, Q, sampling_rate)
                        filtered_signal = sp_signal.filtfilt(b, a, filtered_signal)

        except Exception as e:
            print(f"Error applying filter to {channel_type} channel: {e}")
            return signal

        return filtered_signal

    def detect_and_remove_artifacts(self, signal, sampling_rate, channel_name=None):
        """Memory-efficient artifact detection and removal"""
        # For very large signals, process in chunks
        if len(signal) > MEMORY_CONFIG['chunk_size_samples']:
            return self.detect_artifacts_chunked(signal, sampling_rate, channel_name)

        return self.detect_artifacts_single(signal, sampling_rate, channel_name)

    def detect_artifacts_single(self, signal, sampling_rate, channel_name=None):
        """Detect artifacts in a single array"""
        cleaned_signal = signal.copy()
        artifact_info = {
            'artifacts_detected': 0,
            'artifacts_interpolated': 0,
            'percentage_artifacts': 0,
            'artifact_locations': []
        }

        # Calculate statistics more efficiently
        signal_std = np.std(signal)
        signal_mean = np.mean(signal)
        threshold = CLEANING_CONFIG['artifact_threshold'] * signal_std

        # Detect artifacts
        artifact_mask = np.abs(signal - signal_mean) > threshold
        artifact_indices = np.where(artifact_mask)[0]

        if len(artifact_indices) > 0:
            artifact_segments = self._group_consecutive_indices(artifact_indices)
            artifact_info['artifacts_detected'] = len(artifact_segments)
            artifact_info['percentage_artifacts'] = np.sum(artifact_mask) / len(signal) * 100

            if CLEANING_CONFIG['interpolate_artifacts']:
                max_interp_samples = int(CLEANING_CONFIG['max_interpolation_length'] * sampling_rate)

                for start_idx, end_idx in artifact_segments:
                    segment_length = end_idx - start_idx + 1

                    if segment_length <= max_interp_samples:
                        clean_before = self._find_clean_point_before(signal, artifact_mask, start_idx)
                        clean_after = self._find_clean_point_after(signal, artifact_mask, end_idx)

                        if clean_before is not None and clean_after is not None:
                            x_interp = np.arange(start_idx, end_idx + 1)
                            x_points = np.array([clean_before, clean_after])
                            y_points = np.array([signal[clean_before], signal[clean_after]])

                            cleaned_signal[start_idx:end_idx + 1] = np.interp(x_interp, x_points, y_points)
                            artifact_info['artifacts_interpolated'] += 1

                        artifact_info['artifact_locations'].append({
                            'start_sample': int(start_idx),
                            'end_sample': int(end_idx),
                            'duration_samples': int(segment_length),
                            'duration_seconds': segment_length / sampling_rate,
                            'interpolated': clean_before is not None and clean_after is not None
                        })

        # Clean up intermediate arrays
        del artifact_mask, artifact_indices
        gc.collect()

        return cleaned_signal, artifact_info

    def detect_artifacts_chunked(self, signal, sampling_rate, channel_name=None):
        """Detect artifacts in large signals using chunked processing"""
        # This is a simplified version - for production, you'd want more sophisticated
        # cross-chunk artifact handling
        chunk_size = MEMORY_CONFIG['chunk_size_samples']
        cleaned_signal = np.empty_like(signal)

        total_artifact_info = {
            'artifacts_detected': 0,
            'artifacts_interpolated': 0,
            'percentage_artifacts': 0,
            'artifact_locations': []
        }

        for start_idx in range(0, len(signal), chunk_size):
            end_idx = min(start_idx + chunk_size, len(signal))
            chunk = signal[start_idx:end_idx].copy()

            cleaned_chunk, chunk_artifact_info = self.detect_artifacts_single(
                chunk, sampling_rate, channel_name
            )

            cleaned_signal[start_idx:end_idx] = cleaned_chunk

            # Aggregate artifact info
            total_artifact_info['artifacts_detected'] += chunk_artifact_info['artifacts_detected']
            total_artifact_info['artifacts_interpolated'] += chunk_artifact_info['artifacts_interpolated']

            # Adjust artifact locations to global coordinates
            for loc in chunk_artifact_info['artifact_locations']:
                loc['start_sample'] += start_idx
                loc['end_sample'] += start_idx
                total_artifact_info['artifact_locations'].append(loc)

            del chunk, cleaned_chunk
            gc.collect()

        # Calculate total percentage
        total_samples = sum(loc['duration_samples'] for loc in total_artifact_info['artifact_locations'])
        total_artifact_info['percentage_artifacts'] = total_samples / len(signal) * 100

        return cleaned_signal, total_artifact_info

    def _group_consecutive_indices(self, indices):
        """Group consecutive indices into segments"""
        if len(indices) == 0:
            return []

        segments = []
        start = indices[0]
        prev = indices[0]

        for current in indices[1:]:
            if current != prev + 1:
                segments.append((start, prev))
                start = current
            prev = current

        segments.append((start, prev))
        return segments

    def _find_clean_point_before(self, signal, artifact_mask, index, max_search=100):
        """Find clean point before the given index"""
        for i in range(index - 1, max(0, index - max_search), -1):
            if not artifact_mask[i]:
                return i
        return None

    def _find_clean_point_after(self, signal, artifact_mask, index, max_search=100):
        """Find clean point after the given index"""
        for i in range(index + 1, min(len(signal), index + max_search)):
            if not artifact_mask[i]:
                return i
        return None

    def get_channel_quality_score(self, participant_id, recording_type, channel_name):
        """Get quality score for a specific channel"""
        if not self.quality_data:
            return None

        participant_data = self.quality_data.get(participant_id, {})
        recording_data = participant_data.get(recording_type, {})
        channel_data = recording_data.get(channel_name, {})

        return channel_data.get('quality_score', None)

    def clean_channel(self, channel_data, sampling_rate, channel_name, participant_id=None, recording_type=None):
        """Clean a single channel with memory optimization"""
        self.log_memory_usage(f"Starting {channel_name}")

        # Work directly on the input array to minimize copies
        cleaned_signal = channel_data.copy()
        cleaning_steps = []

        channel_type = self.detect_channel_type(channel_name)
        cleaning_steps.append(f"Detected channel type: {channel_type}")

        quality_score = None
        if participant_id and recording_type:
            quality_score = self.get_channel_quality_score(participant_id, recording_type, channel_name)
            if quality_score:
                cleaning_steps.append(f"Quality score: {quality_score}")

        # Step 1: Remove DC offset
        if CLEANING_CONFIG['remove_dc_offset'] and channel_type == 'fEMG':
            original_mean = np.mean(cleaned_signal)
            cleaned_signal = self.remove_dc_offset(cleaned_signal)
            cleaning_steps.append(f"Removed DC offset: {original_mean:.6f}")
            self.force_cleanup()

        # Step 2: Apply channel-specific filtering
        if CLEANING_CONFIG['apply_channel_filters'] and channel_type != 'unknown':
            pre_filter_std = np.std(cleaned_signal)
            cleaned_signal = self.apply_channel_specific_filter_chunked(
                cleaned_signal, sampling_rate, channel_type
            )
            post_filter_std = np.std(cleaned_signal)
            cleaning_steps.append(f"Applied {channel_type} filter (std: {pre_filter_std:.6f} -> {post_filter_std:.6f})")
            self.force_cleanup()

        # Step 3: Artifact detection and removal
        artifact_info = {}
        if CLEANING_CONFIG['remove_artifacts']:
            cleaned_signal, artifact_info = self.detect_and_remove_artifacts(
                cleaned_signal, sampling_rate, channel_name
            )

            if artifact_info['artifacts_detected'] > 0:
                cleaning_steps.append(
                    f"Detected {artifact_info['artifacts_detected']} artifact segments "
                    f"({artifact_info['percentage_artifacts']:.1f}% of signal), "
                    f"interpolated {artifact_info['artifacts_interpolated']} segments"
                )

            self.force_cleanup()

        # Compute cleaning metrics (minimize intermediate arrays)
        cleaning_metrics = {
            'original_mean': float(np.mean(channel_data)),
            'original_std': float(np.std(channel_data)),
            'cleaned_mean': float(np.mean(cleaned_signal)),
            'cleaned_std': float(np.std(cleaned_signal)),
            'signal_change': float(np.std(cleaned_signal - channel_data)),
            'quality_score': quality_score,
            'channel_type': channel_type,
            'artifact_info': artifact_info
        }

        self.log_memory_usage(f"Finished {channel_name}")

        return cleaned_signal, cleaning_steps, cleaning_metrics

    def process_file(self, file_path):
        """Process a single BIOPAC file with memory management"""
        file_name = os.path.basename(file_path)
        participant_id = file_name.replace('_main.acq', '').replace('_baseline.acq', '')
        recording_type = 'main' if '_main.acq' in file_name else 'baseline'

        print(f"Processing {file_name}...")
        self.log_memory_usage(f"Starting file {file_name}")

        # Load BIOPAC file
        try:
            data = bioread.read(file_path)
        except Exception as e:
            error_msg = f"Error loading {file_name}: {e}"
            print(error_msg)
            self.cleaning_log['errors'].append(error_msg)
            return None

        file_results = {
            'file_name': file_name,
            'participant_id': participant_id,
            'recording_type': recording_type,
            'channels_processed': {},
            'export_paths': {},
            'processing_time': None,
            'errors': []
        }

        start_time = datetime.now()

        # Process each target channel individually
        for channel in data.channels:
            if (channel.name not in TARGET_CHANNELS or
                "marker" in channel.name.lower() or
                len(channel.data) < 10):
                continue

            channel_name = channel.name

            try:
                # Clean the channel
                cleaned_signal, cleaning_steps, cleaning_metrics = self.clean_channel(
                    channel.data,
                    channel.samples_per_second,
                    channel_name,
                    participant_id,
                    recording_type
                )

                # Store minimal results (don't keep large arrays)
                file_results['channels_processed'][channel_name] = {
                    'cleaning_steps': cleaning_steps,
                    'cleaning_metrics': cleaning_metrics,
                    'original_length': len(channel.data),
                    'cleaned_length': len(cleaned_signal),
                    'sampling_rate': channel.samples_per_second
                }

                # Check for manual review flag
                quality_score = cleaning_metrics.get('quality_score')
                if quality_score and quality_score < CLEANING_CONFIG['manual_review_threshold']:
                    self._flag_for_manual_review(
                        participant_id, recording_type, channel_name,
                        quality_score, cleaning_metrics
                    )
                    print(f"WARNING: {channel_name} flagged for manual review (quality: {quality_score})")

                # Create validation plot (only if requested and memory allows)
                memory_usage = self.get_memory_usage_gb()
                if (CLEANING_CONFIG['create_validation_plots'] and
                    memory_usage < MEMORY_CONFIG['max_memory_usage_gb'] * 0.8):
                    plot_path = self._create_validation_plot(
                        channel.data, cleaned_signal, channel,
                        participant_id, recording_type, cleaning_metrics
                    )
                    if plot_path:
                        file_results['export_paths'][f'{channel_name}_validation_plot'] = plot_path
                else:
                    print(f"  Skipping validation plot for {channel_name} (memory: {memory_usage:.1f}GB)")

                # Export cleaned signal
                export_path = self._export_cleaned_signal(
                    cleaned_signal,
                    channel,
                    participant_id,
                    recording_type,
                    channel_name,
                    cleaning_metrics
                )

                if export_path:
                    file_results['export_paths'][channel_name] = export_path
                    print(f"  Cleaned and exported {channel_name}")

                # IMMEDIATE CLEANUP after each channel
                del cleaned_signal

                self.channel_counter += 1
                if self.channel_counter % MEMORY_CONFIG['force_gc_frequency'] == 0:
                    print(f"   Forcing cleanup after {self.channel_counter} channels...")
                    self.force_cleanup()

            except Exception as e:
                error_msg = f"Error processing channel {channel_name} in {file_name}: {e}"
                print(f"ERROR: {error_msg}")
                file_results['errors'].append(error_msg)
                self.cleaning_log['errors'].append(error_msg)
                self.force_cleanup()  # Clean up even on error

        # Calculate processing time
        file_results['processing_time'] = (datetime.now() - start_time).total_seconds()

        # Save individual file report
        self._save_file_report(file_results)

        # Update main log
        self.cleaning_log['processed_files'][file_name] = file_results

        # FINAL CLEANUP for this file
        del data
        self.force_cleanup()
        self.log_memory_usage(f"Finished file {file_name}")

        return file_results

    # Export methods remain largely the same but with added cleanup
    def _export_cleaned_signal(self, cleaned_signal, original_channel, participant_id,
                              recording_type, channel_name, cleaning_metrics):
        """Export cleaned signal with memory management"""
        participant_dir = os.path.join(self.cleaned_signals_dir, participant_id)
        os.makedirs(participant_dir, exist_ok=True)

        base_filename = f"{participant_id}_{recording_type}_{channel_name}"

        try:
            # Export as NumPy array
            numpy_path = os.path.join(participant_dir, f"{base_filename}.npy")
            np.save(numpy_path, cleaned_signal)

            # Export as CSV with metadata (process in chunks for very large signals)
            csv_path = os.path.join(participant_dir, f"{base_filename}.csv")
            sampling_rate = original_channel.samples_per_second

            # Write header first
            with open(csv_path, 'w') as f:
                f.write(f"# Cleaned signal: {channel_name}\n")
                f.write(f"# Participant: {participant_id}\n")
                f.write(f"# Recording: {recording_type}\n")
                f.write(f"# Sampling rate: {sampling_rate} Hz\n")
                f.write(f"# Original length: {len(cleaned_signal)} samples\n")
                f.write(f"# Duration: {len(cleaned_signal)/sampling_rate:.2f} seconds\n")
                f.write(f"# Channel type: {cleaning_metrics['channel_type']}\n")
                f.write(f"# Quality score: {cleaning_metrics['quality_score']}\n")
                f.write(f"# Processing date: {datetime.now().isoformat()}\n")
                f.write("#\n")
                f.write("time_seconds,cleaned_signal,sample_index\n")

            # Write data in chunks to manage memory
            chunk_size = min(100000, len(cleaned_signal))  # 100k samples at a time
            for start_idx in range(0, len(cleaned_signal), chunk_size):
                end_idx = min(start_idx + chunk_size, len(cleaned_signal))

                time_chunk = np.arange(start_idx, end_idx) / sampling_rate
                signal_chunk = cleaned_signal[start_idx:end_idx]
                index_chunk = np.arange(start_idx, end_idx)

                chunk_df = pd.DataFrame({
                    'time_seconds': time_chunk,
                    'cleaned_signal': signal_chunk,
                    'sample_index': index_chunk
                })

                chunk_df.to_csv(csv_path, mode='a', header=False, index=False)

                # Clean up chunk
                del time_chunk, signal_chunk, index_chunk, chunk_df
                gc.collect()

            # Export metadata
            metadata_path = os.path.join(participant_dir, f"{base_filename}_metadata.json")
            metadata = {
                'participant_id': participant_id,
                'recording_type': recording_type,
                'channel_name': channel_name,
                'original_channel_info': {
                    'name': original_channel.name,
                    'units': getattr(original_channel, 'units', 'unknown'),
                    'samples_per_second': original_channel.samples_per_second,
                    'sample_count': len(cleaned_signal)
                },
                'cleaning_metrics': cleaning_metrics,
                'file_paths': {
                    'numpy': numpy_path,
                    'csv': csv_path,
                    'metadata': metadata_path
                },
                'export_timestamp': datetime.now().isoformat()
            }

            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            return {
                'numpy': numpy_path,
                'csv': csv_path,
                'metadata': metadata_path
            }

        except Exception as e:
            print(f"Error exporting {channel_name}: {e}")
            return None

def main():
    """Main execution function"""
    print("=" * 80)
    print("MEMORY-OPTIMIZED PSYCHOPHYSIOLOGICAL SIGNAL CLEANING")
    print("=" * 80)

    # Initialize the cleaner
    print(f"Initializing signal cleaner...")
    print(f"Input folder: {INPUT_DATA_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Quality results: {QUALITY_RESULTS_FILE}")

    cleaner = MemoryEfficientSignalCleaner(
        INPUT_DATA_FOLDER,
        OUTPUT_FOLDER,
        QUALITY_RESULTS_FILE
    )

    # Find all BIOPAC files
    acq_pattern = os.path.join(INPUT_DATA_FOLDER, "*.acq")
    acq_files = glob.glob(acq_pattern)

    if not acq_files:
        print(f"ERROR: No .acq files found in {INPUT_DATA_FOLDER}")
        return

    print(f"Found {len(acq_files)} .acq files to process")
    print(f"Target channels: {TARGET_CHANNELS}")
    print(f"Memory limit: {MEMORY_CONFIG['max_memory_usage_gb']}GB")
    print()

    # Process each file
    successful_files = 0
    failed_files = 0
    total_channels_processed = 0

    start_time = datetime.now()

    for i, file_path in enumerate(acq_files, 1):
        print(f"[{i}/{len(acq_files)}] Processing: {os.path.basename(file_path)}")

        try:
            result = cleaner.process_file(file_path)

            if result:
                successful_files += 1
                channels_in_file = len(result['channels_processed'])
                total_channels_processed += channels_in_file

                print(f"  Successfully processed {channels_in_file} channels")

                # Print any errors for this file
                if result['errors']:
                    print(f"WARNING: {len(result['errors'])} errors occurred:")
                    for error in result['errors']:
                        print(f"    - {error}")
            else:
                failed_files += 1
                print(f"ERROR: Failed to process file")

        except Exception as e:
            failed_files += 1
            print(f"ERROR: Unexpected error: {e}")
            cleaner.cleaning_log['errors'].append(f"Unexpected error processing {file_path}: {e}")

        # Memory status
        memory_usage = cleaner.get_memory_usage_gb()
        print(f"   Memory usage: {memory_usage:.1f}GB")

        # Force cleanup between files
        cleaner.force_cleanup()
        print()

    # Final summary
    total_time = datetime.now() - start_time
    cleaner.cleaning_log['end_time'] = datetime.now().isoformat()
    cleaner.cleaning_log['total_processing_time_seconds'] = total_time.total_seconds()

    # Create summary
    summary = {
        'total_files_found': len(acq_files),
        'successful_files': successful_files,
        'failed_files': failed_files,
        'total_channels_processed': total_channels_processed,
        'total_processing_time': str(total_time),
        'average_time_per_file': total_time.total_seconds() / len(acq_files) if acq_files else 0,
        'total_errors': len(cleaner.cleaning_log['errors']),
        'manual_review_flagged': len(cleaner.cleaning_log['manual_review_flagged']),
        'peak_memory_usage_gb': max([stat['memory_gb'] for stat in cleaner.cleaning_log['memory_stats']], default=0)
    }

    cleaner.cleaning_log['summary'] = summary

    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f" Files processed: {successful_files}/{len(acq_files)}")
    print(f" Channels cleaned: {total_channels_processed}")
    print(f"⏱  Total time: {total_time}")
    print(f" Peak memory: {summary['peak_memory_usage_gb']:.1f}GB")
    print(f"ERROR: Total errors: {summary['total_errors']}")
    print(f"WARNING: Manual review flagged: {summary['manual_review_flagged']}")

    if cleaner.cleaning_log['errors']:
        print("\nWARNING: ERRORS ENCOUNTERED:")
        for error in cleaner.cleaning_log['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(cleaner.cleaning_log['errors']) > 5:
            print(f"  ... and {len(cleaner.cleaning_log['errors']) - 5} more errors")

    # Save final log
    log_path = os.path.join(cleaner.export_logs_dir, f"cleaning_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    try:
        with open(log_path, 'w') as f:
            json.dump(cleaner.cleaning_log, f, indent=2)
        print(f" Full log saved to: {log_path}")
    except Exception as e:
        print(f"WARNING: Could not save log: {e}")

    print(f" Output directory: {OUTPUT_FOLDER}")
    print("=" * 80)

# Add the missing methods that are referenced but not defined
def _flag_for_manual_review(self, participant_id, recording_type, channel_name, quality_score, cleaning_metrics):
    """Flag a channel for manual review"""
    flag_info = {
        'participant_id': participant_id,
        'recording_type': recording_type,
        'channel_name': channel_name,
        'quality_score': quality_score,
        'cleaning_metrics': cleaning_metrics,
        'flagged_timestamp': datetime.now().isoformat()
    }

    self.cleaning_log['manual_review_flagged'].append(flag_info)

    # Save individual flag file
    flag_filename = f"{participant_id}_{recording_type}_{channel_name}_manual_review.json"
    flag_path = os.path.join(self.manual_review_dir, flag_filename)

    try:
        with open(flag_path, 'w') as f:
            json.dump(flag_info, f, indent=2)
    except Exception as e:
        print(f"Error saving manual review flag: {e}")

def _create_validation_plot(self, original_signal, cleaned_signal, channel, participant_id, recording_type, cleaning_metrics):
    """Create validation plot comparing original and cleaned signals"""
    try:
        plt.figure(figsize=(15, 8))

        # Calculate time axis
        sampling_rate = channel.samples_per_second
        time_axis = np.arange(len(original_signal)) / sampling_rate

        # Plot both signals (subsample if too large for plotting)
        max_plot_samples = 50000  # Limit for plotting performance
        if len(original_signal) > max_plot_samples:
            step = len(original_signal) // max_plot_samples
            plot_indices = slice(None, None, step)
            time_plot = time_axis[plot_indices]
            orig_plot = original_signal[plot_indices]
            clean_plot = cleaned_signal[plot_indices]
        else:
            time_plot = time_axis
            orig_plot = original_signal
            clean_plot = cleaned_signal

        plt.subplot(2, 1, 1)
        plt.plot(time_plot, orig_plot, 'b-', alpha=0.7, label='Original', linewidth=0.5)
        plt.plot(time_plot, clean_plot, 'r-', alpha=0.8, label='Cleaned', linewidth=0.5)
        plt.title(f'{channel.name} - Signal Comparison\n{participant_id} - {recording_type}')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot difference
        plt.subplot(2, 1, 2)
        diff_plot = orig_plot - clean_plot if len(orig_plot) == len(clean_plot) else np.zeros_like(orig_plot)
        plt.plot(time_plot, diff_plot, 'g-', alpha=0.7, linewidth=0.5)
        plt.title('Difference (Original - Cleaned)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude Difference')
        plt.grid(True, alpha=0.3)

        # Add metrics text
        metrics_text = f"Quality Score: {cleaning_metrics.get('quality_score', 'N/A')}\n"
        metrics_text += f"Channel Type: {cleaning_metrics['channel_type']}\n"
        metrics_text += f"Original STD: {cleaning_metrics['original_std']:.6f}\n"
        metrics_text += f"Cleaned STD: {cleaning_metrics['cleaned_std']:.6f}\n"
        if 'artifact_info' in cleaning_metrics:
            artifact_info = cleaning_metrics['artifact_info']
            metrics_text += f"Artifacts: {artifact_info['artifacts_detected']} detected, "
            metrics_text += f"{artifact_info['artifacts_interpolated']} interpolated"

        plt.figtext(0.02, 0.02, metrics_text, fontsize=8, verticalalignment='bottom')

        plt.tight_layout()

        # Save plot
        plot_filename = f"{participant_id}_{recording_type}_{channel.name}_validation.png"
        plot_path = os.path.join(self.validation_plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path

    except Exception as e:
        print(f"Error creating validation plot: {e}")
        plt.close('all')  # Make sure to close any open figures
        return None

def _save_file_report(self, file_results):
    """Save individual file processing report"""
    report_filename = f"{file_results['participant_id']}_{file_results['recording_type']}_cleaning_report.json"
    report_path = os.path.join(self.cleaning_reports_dir, report_filename)

    try:
        with open(report_path, 'w') as f:
            json.dump(file_results, f, indent=2)
    except Exception as e:
        print(f"Error saving file report: {e}")

# Add these methods to the MemoryEfficientSignalCleaner class
MemoryEfficientSignalCleaner._flag_for_manual_review = _flag_for_manual_review
MemoryEfficientSignalCleaner._create_validation_plot = _create_validation_plot
MemoryEfficientSignalCleaner._save_file_report = _save_file_report

# This is the key part - actually run the main function!
if __name__ == "__main__":
    main()

# %%
