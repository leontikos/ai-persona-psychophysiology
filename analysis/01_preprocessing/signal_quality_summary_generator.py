# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive HTML Summary Generator for Signal Quality Analysis

Creates interactive HTML summaries that allow exploration
of overall study statistics and individual participant data. It works with the
output from the memory-optimized signal quality analyzer.

Features:
1. Interactive tabs for different views
2. Filterable tables
3. Collapsible sections
4. Quality score visualizations
5. Participant-specific reports
6. Channel type analysis
7. Issue tracking across the study

Usage:
    python interactive_summary_generator.py

Make sure to set the OUTPUT_FOLDER path to match your analysis output.
"""

import os
import glob
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import io
import base64
from datetime import datetime
import time
from bs4 import BeautifulSoup
import warnings
from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
OUTPUT_FOLDER = os.path.join(DATA_ROOT, "signal_quality_analysis")

# Custom colormap for quality scores
QUALITY_CMAP = LinearSegmentedColormap.from_list('quality_cmap',
                                                [(0.0, 'red'),
                                                 (0.5, 'yellow'),
                                                 (1.0, 'green')])

class InteractiveSummaryGenerator:
    """
    Creates interactive HTML summaries for the signal quality analysis.
    """

    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.report_dir = os.path.join(output_folder, "reports")
        self.summary_dir = os.path.join(output_folder, "summaries")
        self.visualization_dir = os.path.join(output_folder, "visualizations")

        # Ensure summary directory exists
        os.makedirs(self.summary_dir, exist_ok=True)

        # Data storage
        self.results = {}
        self.study_stats = {}

    def extract_data_from_reports(self):
        """
        Extract data from individual HTML reports and progressive results.
        """
        print("Extracting data from reports...")

        # First, try to load from progressive results if available
        progressive_file = os.path.join(self.summary_dir, "progressive_results.json")
        if os.path.exists(progressive_file):
            print("Loading from progressive results file...")
            try:
                with open(progressive_file, 'r') as f:
                    data = json.load(f)
                self.results = data.get("participants", {})
                print(f"Loaded data for {len(self.results)} participants from progressive results")
            except Exception as e:
                print(f"Could not load progressive results: {e}")
                self.results = {}

        # If progressive results not available or incomplete, extract from HTML reports
        if not self.results:
            print("Extracting from HTML reports...")
            self.results = self._extract_from_html_reports()

        # Calculate study statistics
        self._calculate_study_stats()

    def _extract_from_html_reports(self):
        """Extract data from individual HTML reports."""
        html_files = glob.glob(os.path.join(self.report_dir, "*.html"))
        html_files = [f for f in html_files if os.path.basename(f) != "study_summary.html"]

        results = {}

        for html_file in html_files:
            try:
                # Parse filename to get participant info
                filename = os.path.basename(html_file)
                match = re.match(r'(.+?)_(main|baseline)_(.+?)\.html', filename)

                if not match:
                    continue

                participant_id = match.group(1)
                recording_type = match.group(2)
                channel_name = match.group(3)

                # Parse HTML content
                with open(html_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                soup = BeautifulSoup(html_content, 'html.parser')

                # Extract quality data
                quality_data = self._parse_quality_data(soup)

                # Store in results structure
                if participant_id not in results:
                    results[participant_id] = {}
                if recording_type not in results[participant_id]:
                    results[participant_id][recording_type] = {}

                results[participant_id][recording_type][channel_name] = quality_data

            except Exception as e:
                print(f"Error processing {html_file}: {e}")
                continue

        return results

    def _parse_quality_data(self, soup):
        """Parse quality data from HTML soup."""
        # Extract quality score
        quality_badge = soup.select_one('.quality-badge')
        if quality_badge:
            quality_text = quality_badge.text
            quality_score = int(quality_text.split('/')[0])
            quality_category = quality_text.split('-')[1].strip()
        else:
            quality_score = 0
            quality_category = "Unknown"

        # Extract primary issues
        issues = []
        issue_list = soup.select('.issue-list ul li')
        if issue_list:
            issues = [li.text.strip() for li in issue_list]

        # Extract subscores
        subscores = {}
        quality_details = soup.select('.quality-details div')
        for div in quality_details:
            h3 = div.find('h3')
            if h3:
                metric = h3.text.lower()
                p = div.find('p')
                if p:
                    score_text = p.text
                    try:
                        score = float(score_text.split('/')[0])
                        subscores[metric] = score
                    except:
                        pass

        # Determine channel type from header
        channel_type = "Unknown"
        header_text = soup.select_one('.header p')
        if header_text:
            channel_type_match = re.search(r'\((\w+)\)', header_text.text)
            if channel_type_match:
                channel_type = channel_type_match.group(1)

        return {
            'channel_type': channel_type,
            'quality_score': quality_score,
            'quality_category': quality_category,
            'subscores': subscores,
            'primary_issues': issues
        }

    def _calculate_study_stats(self):
        """Calculate overall study statistics."""
        total_channels = 0
        quality_scores = []
        channel_types = {}
        issues_count = {}
        recording_types = set()
        participants = list(self.results.keys())

        for participant_id, rec_results in self.results.items():
            for recording_type, channels in rec_results.items():
                recording_types.add(recording_type)
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

        self.study_stats = {
            'total_participants': len(participants),
            'total_channels': total_channels,
            'average_quality': np.mean(quality_scores) if quality_scores else 0,
            'quality_scores': quality_scores,
            'channel_types': channel_types,
            'issues_count': issues_count,
            'recording_types': list(recording_types),
            'participants': participants
        }

    def create_overview_visualizations(self):
        """Create overview visualizations and return as base64."""
        visualizations = {}

        # Set style
        plt.style.use('default')

        # 1. Quality Score Distribution
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create histogram
        ax.hist(self.study_stats['quality_scores'], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        ax.axvline(self.study_stats['average_quality'], color='red', linestyle='--',
                  label=f'Average: {self.study_stats["average_quality"]:.1f}')

        # Add quality thresholds
        ax.axvline(80, color='green', linestyle=':', alpha=0.7, label='Excellent (80+)')
        ax.axvline(60, color='orange', linestyle=':', alpha=0.7, label='Good (60+)')
        ax.axvline(40, color='red', linestyle=':', alpha=0.7, label='Fair (40+)')

        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Number of Channels')
        ax.set_title('Distribution of Signal Quality Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)

        visualizations['quality_distribution'] = self._fig_to_base64(fig)
        plt.close(fig)

        # 2. Channel Types Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Pie chart
        channel_types = self.study_stats['channel_types']
        if channel_types:
            ax1.pie(channel_types.values(), labels=channel_types.keys(), autopct='%1.1f%%', startangle=90)
            ax1.set_title('Channel Types Distribution')

        # Bar chart
        ax2.bar(channel_types.keys(), channel_types.values(), color=['lightblue', 'lightgreen', 'lightcoral'])
        ax2.set_title('Channel Types Count')
        ax2.set_ylabel('Number of Channels')

        plt.tight_layout()
        visualizations['channel_types'] = self._fig_to_base64(fig)
        plt.close(fig)

        # 3. Common Issues
        fig, ax = plt.subplots(figsize=(12, 6))

        issues = self.study_stats['issues_count']
        if issues:
            issues_sorted = dict(sorted(issues.items(), key=lambda x: x[1], reverse=True))
            top_issues = dict(list(issues_sorted.items())[:10])  # Top 10 issues

            bars = ax.bar(range(len(top_issues)), list(top_issues.values()),
                         color='salmon', alpha=0.7)
            ax.set_xticks(range(len(top_issues)))
            ax.set_xticklabels(list(top_issues.keys()), rotation=45, ha='right')
            ax.set_ylabel('Number of Occurrences')
            ax.set_title('Most Common Signal Quality Issues')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')

        plt.tight_layout()
        visualizations['common_issues'] = self._fig_to_base64(fig)
        plt.close(fig)

        # 4. Quality by Channel Type
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create data for boxplot
        channel_quality_data = []
        for participant_id, rec_results in self.results.items():
            for recording_type, channels in rec_results.items():
                for channel_name, analysis in channels.items():
                    channel_quality_data.append({
                        'channel_type': analysis['channel_type'],
                        'quality_score': analysis['quality_score']
                    })

        if channel_quality_data:
            df = pd.DataFrame(channel_quality_data)

            # Create boxplot
            try:
                sns.boxplot(data=df, x='channel_type', y='quality_score', ax=ax)
            except:
                # Fallback if seaborn not available
                grouped = df.groupby('channel_type')['quality_score']
                for i, (name, group) in enumerate(grouped):
                    ax.boxplot(group.values, positions=[i], labels=[name])

            ax.set_title('Quality Score Distribution by Channel Type')
            ax.set_xlabel('Channel Type')
            ax.set_ylabel('Quality Score')
            ax.set_ylim(0, 100)

        plt.tight_layout()
        visualizations['quality_by_channel'] = self._fig_to_base64(fig)
        plt.close(fig)

        return visualizations

    def create_participant_heatmap(self):
        """Create a heatmap showing quality scores for all participants."""
        # Prepare data for heatmap
        heatmap_data = []

        for participant_id in sorted(self.results.keys()):
            participant_row = {'Participant': participant_id}

            for recording_type in self.study_stats['recording_types']:
                if recording_type in self.results[participant_id]:
                    channels = self.results[participant_id][recording_type]
                    scores = [ch['quality_score'] for ch in channels.values()]
                    avg_score = np.mean(scores) if scores else 0
                    participant_row[recording_type] = avg_score
                else:
                    participant_row[recording_type] = np.nan

        heatmap_data.append(participant_row)

        # Create DataFrame
        df = pd.DataFrame(heatmap_data)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, max(6, len(self.results) * 0.3)))

        # Prepare data matrix
        data_cols = [col for col in df.columns if col != 'Participant']
        data_matrix = df[data_cols].values

        # Create heatmap
        im = ax.imshow(data_matrix, cmap=QUALITY_CMAP, aspect='auto', vmin=0, vmax=100)

        # Set labels
        ax.set_xticks(range(len(data_cols)))
        ax.set_xticklabels(data_cols)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['Participant'])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Quality Score')

        # Add text annotations
        for i in range(len(df)):
            for j in range(len(data_cols)):
                if not np.isnan(data_matrix[i, j]):
                    text = ax.text(j, i, f'{data_matrix[i, j]:.0f}',
                                 ha="center", va="center", color="black", fontsize=8)

        ax.set_title('Participant Quality Score Heatmap')
        plt.tight_layout()

        heatmap_b64 = self._fig_to_base64(fig)
        plt.close(fig)

        return heatmap_b64

    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_b64

    def generate_participant_summary_table(self):
        """Generate HTML for participant summary table."""
        html = """
        <table id="participantTable" class="sortable-table">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">Participant ID ↕</th>
                    <th onclick="sortTable(1)">Overall Quality ↕</th>
        """

        # Add recording type columns
        for rec_type in sorted(self.study_stats['recording_types']):
            html += f'<th onclick="sortTable({len(self.study_stats["recording_types"]) + 1})">{rec_type} ↕</th>'

        html += """
                    <th>Channel Count</th>
                    <th>Main Issues</th>
                </tr>
            </thead>
            <tbody>
        """

        # Generate participant rows
        for participant_id in sorted(self.results.keys()):
            # Calculate overall quality for participant
            all_scores = []
            channel_count = 0
            all_issues = []

            participant_data = {}

            for recording_type in self.study_stats['recording_types']:
                if recording_type in self.results[participant_id]:
                    channels = self.results[participant_id][recording_type]
                    scores = [ch['quality_score'] for ch in channels.values()]
                    channel_count += len(scores)
                    all_scores.extend(scores)

                    # Collect issues
                    for ch in channels.values():
                        all_issues.extend(ch['primary_issues'])

                    avg_score = np.mean(scores) if scores else 0
                    participant_data[recording_type] = avg_score
                else:
                    participant_data[recording_type] = None

            overall_quality = np.mean(all_scores) if all_scores else 0

            # Get most common issues for this participant
            issue_counts = {}
            for issue in all_issues:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

            top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            issues_text = ', '.join([f"{issue} ({count})" for issue, count in top_issues])

            # Determine quality class
            if overall_quality >= 80:
                quality_class = "excellent"
            elif overall_quality >= 60:
                quality_class = "good"
            elif overall_quality >= 40:
                quality_class = "fair"
            elif overall_quality >= 20:
                quality_class = "poor"
            else:
                quality_class = "very-poor"

            html += f"""
                <tr data-participant="{participant_id}">
                    <td>{participant_id}</td>
                    <td>
                        <div class="quality-cell {quality_class}">
                            {overall_quality:.1f}
                        </div>
                    </td>
            """

            # Add recording type scores
            for rec_type in sorted(self.study_stats['recording_types']):
                if participant_data[rec_type] is not None:
                    score = participant_data[rec_type]
                    if score >= 60:
                        score_class = "good"
                    elif score >= 40:
                        score_class = "fair"
                    else:
                        score_class = "poor"

                    html += f'<td><span class="score-{score_class}">{score:.1f}</span></td>'
                else:
                    html += '<td>N/A</td>'

            html += f"""
                    <td>{channel_count}</td>
                    <td class="issues-cell">{issues_text if issues_text else "None"}</td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """

        return html

    def generate_channel_details_table(self):
        """Generate detailed channel information table."""
        html = """
        <table id="channelTable" class="sortable-table">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">Participant ↕</th>
                    <th onclick="sortTable(1)">Recording ↕</th>
                    <th onclick="sortTable(2)">Channel ↕</th>
                    <th onclick="sortTable(3)">Type ↕</th>
                    <th onclick="sortTable(4)">Quality Score ↕</th>
                    <th onclick="sortTable(5)">Category ↕</th>
                    <th>Issues</th>
                    <th>Report</th>
                </tr>
            </thead>
            <tbody>
        """

        # Generate channel rows
        for participant_id in sorted(self.results.keys()):
            for recording_type in sorted(self.results[participant_id].keys()):
                for channel_name in sorted(self.results[participant_id][recording_type].keys()):
                    analysis = self.results[participant_id][recording_type][channel_name]

                    quality_score = analysis['quality_score']
                    quality_category = analysis['quality_category']

                    # Determine quality class
                    if quality_score >= 80:
                        quality_class = "excellent"
                    elif quality_score >= 60:
                        quality_class = "good"
                    elif quality_score >= 40:
                        quality_class = "fair"
                    elif quality_score >= 20:
                        quality_class = "poor"
                    else:
                        quality_class = "very-poor"

                    issues_text = ', '.join(analysis['primary_issues']) if analysis['primary_issues'] else "None"

                    html += f"""
                        <tr data-participant="{participant_id}" data-channel="{analysis['channel_type']}" data-recording="{recording_type}">
                            <td>{participant_id}</td>
                            <td>{recording_type}</td>
                            <td>{channel_name}</td>
                            <td>{analysis['channel_type']}</td>
                            <td>
                                <div class="quality-cell {quality_class}">
                                    {quality_score}
                                </div>
                            </td>
                            <td>{quality_category}</td>
                            <td class="issues-cell">{issues_text}</td>
                            <td>
                                <a href="{participant_id}_{recording_type}_{channel_name}.html"
                                   class="report-link" target="_blank">View Report</a>
                            </td>
                        </tr>
                    """

        html += """
            </tbody>
        </table>
        """

        return html

    def create_interactive_html_summary(self):
        """Create the main interactive HTML summary."""
        # Generate visualizations
        print("Creating overview visualizations...")
        overview_viz = self.create_overview_visualizations()

        print("Creating participant heatmap...")
        heatmap_viz = self.create_participant_heatmap()

        print("Generating data tables...")
        participant_table = self.generate_participant_summary_table()
        channel_table = self.generate_channel_details_table()

        # Get current timestamp
        generation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Create the HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interactive Signal Quality Analysis Summary</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}

        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}

        .header p {{
            margin: 10px 0 0 0;
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .stats-overview {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 20px;
            background-color: #f8f9fa;
        }}

        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
        }}

        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 0;
        }}

        .stat-label {{
            margin: 5px 0 0 0;
            color: #666;
            font-size: 1.1em;
        }}

        .tabs {{
            background-color: #e9ecef;
            border-bottom: 1px solid #dee2e6;
        }}

        .tab-nav {{
            display: flex;
            list-style: none;
            margin: 0;
            padding: 0;
            overflow-x: auto;
        }}

        .tab-nav li {{
            flex: none;
        }}

        .tab-nav button {{
            background: none;
            border: none;
            padding: 15px 25px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            color: #495057;
            transition: all 0.3s;
            white-space: nowrap;
        }}

        .tab-nav button:hover {{
            background-color: #dee2e6;
        }}

        .tab-nav button.active {{
            background-color: white;
            color: #667eea;
            border-bottom: 3px solid #667eea;
        }}

        .tab-content {{
            padding: 30px;
            min-height: 600px;
        }}

        .tab-pane {{
            display: none;
        }}

        .tab-pane.active {{
            display: block;
            animation: fadeIn 0.3s;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; }}
            to {{ opacity: 1; }}
        }}

        .visualization {{
            text-align: center;
            margin: 20px 0;
        }}

        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}

        .filter-section {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}

        .filter-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}

        .filter-group {{
            display: flex;
            flex-direction: column;
        }}

        .filter-group label {{
            font-weight: 500;
            margin-bottom: 5px;
            color: #495057;
        }}

        .filter-input {{
            padding: 8px 12px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            font-size: 14px;
        }}

        .filter-input:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.25);
        }}

        .sortable-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .sortable-table th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 10px;
            text-align: left;
            font-weight: 600;
            cursor: pointer;
            user-select: none;
            transition: background-color 0.2s;
        }}

        .sortable-table th:hover {{
            background-color: rgba(255,255,255,0.1);
        }}

        .sortable-table td {{
            padding: 12px 10px;
            border-bottom: 1px solid #dee2e6;
        }}

        .sortable-table tr:hover {{
            background-color: #f8f9fa;
        }}

        .quality-cell {{
            padding: 6px 12px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            text-align: center;
            min-width: 60px;
            display: inline-block;
        }}

        .excellent {{ background-color: #28a745; }}
        .good {{ background-color: #17a2b8; }}
        .fair {{ background-color: #ffc107; color: #212529; }}
        .poor {{ background-color: #fd7e14; }}
        .very-poor {{ background-color: #dc3545; }}

        .score-good {{ color: #28a745; font-weight: bold; }}
        .score-fair {{ color: #ffc107; font-weight: bold; }}
        .score-poor {{ color: #dc3545; font-weight: bold; }}

        .issues-cell {{
            max-width: 250px;
            word-wrap: break-word;
            font-size: 0.9em;
        }}

        .report-link {{
            background-color: #667eea;
            color: white;
            padding: 6px 12px;
            text-decoration: none;
            border-radius: 4px;
            font-size: 0.9em;
            transition: background-color 0.2s;
        }}

        .report-link:hover {{
            background-color: #5a6fd8;
        }}

        .collapsible {{
            background-color: #e9ecef;
            color: #495057;
            cursor: pointer;
            padding: 15px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 16px;
            font-weight: 500;
            border-radius: 4px;
            margin: 5px 0;
            transition: background-color 0.2s;
        }}

        .collapsible:hover {{
            background-color: #dee2e6;
        }}

        .collapsible:after {{
            content: '\\002B';
            color: #667eea;
            font-weight: bold;
            float: right;
            margin-left: 5px;
            font-size: 18px;
        }}

        .collapsible.active:after {{
            content: "\\2212";
        }}

        .collapsible-content {{
            padding: 0 15px;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease-out;
            background-color: #f8f9fa;
            border-radius: 0 0 4px 4px;
        }}

        .alert {{
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid transparent;
            border-radius: 4px;
        }}

        .alert-info {{
            color: #0c5460;
            background-color: #d1ecf1;
            border-color: #bee5eb;
        }}

        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        .grid-3 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}

        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .card h3 {{
            margin-top: 0;
            color: #495057;
        }}

        .footer {{
            background-color: #f8f9fa;
            padding: 20px;
            text-align: center;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
        }}

        @media (max-width: 768px) {{
            .container {{
                margin: 0;
            }}

            .tab-content {{
                padding: 15px;
            }}

            .stats-overview {{
                grid-template-columns: 1fr;
            }}

            .grid-2, .grid-3 {{
                grid-template-columns: 1fr;
            }}

            .sortable-table {{
                font-size: 14px;
            }}

            .sortable-table th,
            .sortable-table td {{
                padding: 8px 5px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1> Signal Quality Analysis Dashboard</h1>
            <p>Interactive exploration of psychophysiological data quality across all participants</p>
        </div>

        <!-- Stats Overview -->
        <div class="stats-overview">
            <div class="stat-card">
                <div class="stat-number">{self.study_stats['total_participants']}</div>
                <div class="stat-label">Participants</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{self.study_stats['total_channels']}</div>
                <div class="stat-label">Channels Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{self.study_stats['average_quality']:.1f}</div>
                <div class="stat-label">Average Quality Score</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len([s for s in self.study_stats['quality_scores'] if s >= 60])}</div>
                <div class="stat-label">Good Quality Channels</div>
            </div>
        </div>

        <!-- Navigation Tabs -->
        <div class="tabs">
            <ul class="tab-nav">
                <li><button class="tab-button active" onclick="openTab(event, 'overview')"> Overview</button></li>
                <li><button class="tab-button" onclick="openTab(event, 'participants')"> Participants</button></li>
                <li><button class="tab-button" onclick="openTab(event, 'channels')"> Channel Details</button></li>
                <li><button class="tab-button" onclick="openTab(event, 'analysis')"> Analysis</button></li>
                <li><button class="tab-button" onclick="openTab(event, 'issues')">Issues</button></li>
            </ul>
        </div>

        <!-- Tab Content -->
        <div class="tab-content">
            <!-- Overview Tab -->
            <div id="overview" class="tab-pane active">
                <h2> Study Overview</h2>

                <div class="alert alert-info">
                    <strong>Study Summary:</strong> This dashboard provides an interactive view of signal quality
                    across {self.study_stats['total_participants']} participants and {self.study_stats['total_channels']}
                    channels. The average quality score is {self.study_stats['average_quality']:.1f}/100.
                </div>

                <div class="grid-2">
                    <div class="visualization">
                        <h3>Quality Score Distribution</h3>
                        <img src="data:image/png;base64,{overview_viz['quality_distribution']}" alt="Quality Distribution">
                    </div>
                    <div class="visualization">
                        <h3>Channel Types</h3>
                        <img src="data:image/png;base64,{overview_viz['channel_types']}" alt="Channel Types">
                    </div>
                </div>

                <div class="grid-2">
                    <div class="visualization">
                        <h3>Quality by Channel Type</h3>
                        <img src="data:image/png;base64,{overview_viz['quality_by_channel']}" alt="Quality by Channel">
                    </div>
                    <div class="visualization">
                        <h3>Most Common Issues</h3>
                        <img src="data:image/png;base64,{overview_viz['common_issues']}" alt="Common Issues">
                    </div>
                </div>

                <div class="visualization">
                    <h3>Participant Quality Heatmap</h3>
                    <img src="data:image/png;base64,{heatmap_viz}" alt="Participant Heatmap">
                </div>
            </div>

            <!-- Participants Tab -->
            <div id="participants" class="tab-pane">
                <h2> Participant Summary</h2>

                <div class="alert alert-info">
                    Click on column headers to sort. Use the filters below to find specific participants
                    or filter by quality thresholds.
                </div>

                <div class="filter-section">
                    <div class="filter-grid">
                        <div class="filter-group">
                            <label for="participantFilter">Filter by Participant ID:</label>
                            <input type="text" id="participantFilter" class="filter-input"
                                   placeholder="Enter participant ID..." onkeyup="filterParticipantTable()">
                        </div>
                        <div class="filter-group">
                            <label for="qualityFilter">Minimum Quality Score:</label>
                            <input type="number" id="qualityFilter" class="filter-input"
                                   placeholder="0-100" min="0" max="100" onchange="filterParticipantTable()">
                        </div>
                        <div class="filter-group">
                            <label for="issueFilter">Filter by Issue:</label>
                            <select id="issueFilter" class="filter-input" onchange="filterParticipantTable()">
                                <option value="">All participants</option>
                                <option value="Signal Flatline">Signal Flatline</option>
                                <option value="Excessive Noise">Excessive Noise</option>
                                <option value="Out-of-Range Values">Out-of-Range Values</option>
                                <option value="Abrupt Jumps/Artifacts">Abrupt Jumps/Artifacts</option>
                                <option value="Abnormal Peak Pattern">Abnormal Peak Pattern</option>
                            </select>
                        </div>
                    </div>
                </div>

                {participant_table}
            </div>

            <!-- Channel Details Tab -->
            <div id="channels" class="tab-pane">
                <h2> Channel Details</h2>

                <div class="alert alert-info">
                    Detailed view of all channels across all participants. Click "View Report" to see
                    individual channel analysis reports.
                </div>

                <div class="filter-section">
                    <div class="filter-grid">
                        <div class="filter-group">
                            <label for="channelParticipantFilter">Participant:</label>
                            <input type="text" id="channelParticipantFilter" class="filter-input"
                                   placeholder="Filter by participant..." onkeyup="filterChannelTable()">
                        </div>
                        <div class="filter-group">
                            <label for="channelTypeFilter">Channel Type:</label>
                            <select id="channelTypeFilter" class="filter-input" onchange="filterChannelTable()">
                                <option value="">All types</option>"""

        # Add channel type options
        for ch_type in sorted(self.study_stats['channel_types'].keys()):
            html_content += f'<option value="{ch_type}">{ch_type}</option>'

        html_content += f"""
                            </select>
                        </div>
                        <div class="filter-group">
                            <label for="channelQualityFilter">Min Quality:</label>
                            <input type="number" id="channelQualityFilter" class="filter-input"
                                   placeholder="0-100" min="0" max="100" onchange="filterChannelTable()">
                        </div>
                        <div class="filter-group">
                            <label for="recordingTypeFilter">Recording Type:</label>
                            <select id="recordingTypeFilter" class="filter-input" onchange="filterChannelTable()">
                                <option value="">All recordings</option>"""

        # Add recording type options
        for rec_type in sorted(self.study_stats['recording_types']):
            html_content += f'<option value="{rec_type}">{rec_type}</option>'

        html_content += f"""
                            </select>
                        </div>
                    </div>
                </div>

                {channel_table}
            </div>

            <!-- Analysis Tab -->
            <div id="analysis" class="tab-pane">
                <h2> Advanced Analysis</h2>

                <div class="grid-3">
                    <div class="card">
                        <h3>Quality Distribution</h3>
                        <div id="qualityBreakdown"></div>
                    </div>
                    <div class="card">
                        <h3>Channel Type Performance</h3>
                        <div id="channelPerformance"></div>
                    </div>
                    <div class="card">
                        <h3>Recording Type Comparison</h3>
                        <div id="recordingComparison"></div>
                    </div>
                </div>

                <button class="collapsible">Detailed Statistics</button>
                <div class="collapsible-content">
                    <div id="detailedStats"></div>
                </div>

                <button class="collapsible">Quality Score Percentiles</button>
                <div class="collapsible-content">
                    <div id="percentileStats"></div>
                </div>
            </div>

            <!-- Issues Tab -->
            <div id="issues" class="tab-pane">
                <h2>Issue Analysis</h2>

                <div class="alert alert-info">
                    This section provides detailed analysis of signal quality issues across the study.
                </div>

                <div class="grid-2">
                    <div class="card">
                        <h3>Issue Frequency</h3>
                        <div id="issueFrequency"></div>
                    </div>
                    <div class="card">
                        <h3>Issues by Channel Type</h3>
                        <div id="issuesByChannel"></div>
                    </div>
                </div>

                <button class="collapsible">Participants with Multiple Issues</button>
                <div class="collapsible-content">
                    <div id="multipleIssues"></div>
                </div>

                <button class="collapsible">Issue Correlation Analysis</button>
                <div class="collapsible-content">
                    <div id="issueCorrelation"></div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>Generated on {generation_time} | Signal Quality Analysis Dashboard</p>
            <p>Total processing time and memory usage optimized for large datasets</p>
        </div>
    </div>

    <script>
        // Tab functionality
        function openTab(evt, tabName) {{
            var i, tabpanes, tabbuttons;

            // Hide all tab panes
            tabpanes = document.getElementsByClassName("tab-pane");
            for (i = 0; i < tabpanes.length; i++) {{
                tabpanes[i].classList.remove("active");
            }}

            // Remove active class from all tab buttons
            tabbuttons = document.getElementsByClassName("tab-button");
            for (i = 0; i < tabbuttons.length; i++) {{
                tabbuttons[i].classList.remove("active");
            }}

            // Show the selected tab pane and mark button as active
            document.getElementById(tabName).classList.add("active");
            evt.currentTarget.classList.add("active");

            // Initialize tab-specific content
            if (tabName === 'analysis') {{
                initializeAnalysisTab();
            }} else if (tabName === 'issues') {{
                initializeIssuesTab();
            }}
        }}

        // Table sorting functionality
        function sortTable(columnIndex) {{
            var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
            table = document.getElementById("participantTable") || document.getElementById("channelTable");
            switching = true;
            dir = "asc";

            while (switching) {{
                switching = false;
                rows = table.rows;

                for (i = 1; i < (rows.length - 1); i++) {{
                    shouldSwitch = false;
                    x = rows[i].getElementsByTagName("TD")[columnIndex];
                    y = rows[i + 1].getElementsByTagName("TD")[columnIndex];

                    var xContent = x.textContent || x.innerText;
                    var yContent = y.textContent || y.innerText;

                    // Try to parse as numbers
                    var xNum = parseFloat(xContent);
                    var yNum = parseFloat(yContent);

                    if (!isNaN(xNum) && !isNaN(yNum)) {{
                        if (dir == "asc") {{
                            if (xNum > yNum) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }} else if (dir == "desc") {{
                            if (xNum < yNum) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }}
                    }} else {{
                        if (dir == "asc") {{
                            if (xContent.toLowerCase() > yContent.toLowerCase()) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }} else if (dir == "desc") {{
                            if (xContent.toLowerCase() < yContent.toLowerCase()) {{
                                shouldSwitch = true;
                                break;
                            }}
                        }}
                    }}
                }}

                if (shouldSwitch) {{
                    rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                    switching = true;
                    switchcount++;
                }} else {{
                    if (switchcount == 0 && dir == "asc") {{
                        dir = "desc";
                        switching = true;
                    }}
                }}
            }}
        }}

        // Filter participant table
        function filterParticipantTable() {{
            var participantFilter = document.getElementById("participantFilter").value.toUpperCase();
            var qualityFilter = parseFloat(document.getElementById("qualityFilter").value) || 0;
            var issueFilter = document.getElementById("issueFilter").value;

            var table = document.getElementById("participantTable");
            var tr = table.getElementsByTagName("tr");

            for (var i = 1; i < tr.length; i++) {{
                var td = tr[i].getElementsByTagName("td");
                var showRow = true;

                // Check participant ID filter
                if (participantFilter && td[0] &&
                    td[0].textContent.toUpperCase().indexOf(participantFilter) === -1) {{
                    showRow = false;
                }}

                // Check quality filter
                if (qualityFilter > 0 && td[1]) {{
                    var qualityText = td[1].textContent || td[1].innerText;
                    var quality = parseFloat(qualityText);
                    if (isNaN(quality) || quality < qualityFilter) {{
                        showRow = false;
                    }}
                }}

                // Check issue filter
                if (issueFilter && td[td.length - 1]) {{
                    var issuesText = td[td.length - 1].textContent;
                    if (issuesText.indexOf(issueFilter) === -1) {{
                        showRow = false;
                    }}
                }}

                tr[i].style.display = showRow ? "" : "none";
            }}
        }}

        // Filter channel table
        function filterChannelTable() {{
            var participantFilter = document.getElementById("channelParticipantFilter").value.toUpperCase();
            var typeFilter = document.getElementById("channelTypeFilter").value;
            var qualityFilter = parseFloat(document.getElementById("channelQualityFilter").value) || 0;
            var recordingFilter = document.getElementById("recordingTypeFilter").value;

            var table = document.getElementById("channelTable");
            var tr = table.getElementsByTagName("tr");

            for (var i = 1; i < tr.length; i++) {{
                var td = tr[i].getElementsByTagName("td");
                var showRow = true;

                // Check participant filter
                if (participantFilter && td[0] &&
                    td[0].textContent.toUpperCase().indexOf(participantFilter) === -1) {{
                    showRow = false;
                }}

                // Check recording type filter
                if (recordingFilter && td[1] && td[1].textContent !== recordingFilter) {{
                    showRow = false;
                }}

                // Check channel type filter
                if (typeFilter && td[3] && td[3].textContent !== typeFilter) {{
                    showRow = false;
                }}

                // Check quality filter
                if (qualityFilter > 0 && td[4]) {{
                    var qualityText = td[4].textContent || td[4].innerText;
                    var quality = parseFloat(qualityText);
                    if (isNaN(quality) || quality < qualityFilter) {{
                        showRow = false;
                    }}
                }}

                tr[i].style.display = showRow ? "" : "none";
            }}
        }}

        // Initialize analysis tab content
        function initializeAnalysisTab() {{
            // Quality breakdown
            var qualityBreakdown = `
                <table class="sortable-table">
                    <tr><th>Quality Range</th><th>Count</th><th>Percentage</th></tr>
                    <tr><td>Excellent (80-100)</td><td>{len([s for s in self.study_stats['quality_scores'] if s >= 80])}</td><td>{len([s for s in self.study_stats['quality_scores'] if s >= 80])/len(self.study_stats['quality_scores'])*100:.1f}%</td></tr>
                    <tr><td>Good (60-79)</td><td>{len([s for s in self.study_stats['quality_scores'] if 60 <= s < 80])}</td><td>{len([s for s in self.study_stats['quality_scores'] if 60 <= s < 80])/len(self.study_stats['quality_scores'])*100:.1f}%</td></tr>
                    <tr><td>Fair (40-59)</td><td>{len([s for s in self.study_stats['quality_scores'] if 40 <= s < 60])}</td><td>{len([s for s in self.study_stats['quality_scores'] if 40 <= s < 60])/len(self.study_stats['quality_scores'])*100:.1f}%</td></tr>
                    <tr><td>Poor (20-39)</td><td>{len([s for s in self.study_stats['quality_scores'] if 20 <= s < 40])}</td><td>{len([s for s in self.study_stats['quality_scores'] if 20 <= s < 40])/len(self.study_stats['quality_scores'])*100:.1f}%</td></tr>
                    <tr><td>Very Poor (0-19)</td><td>{len([s for s in self.study_stats['quality_scores'] if s < 20])}</td><td>{len([s for s in self.study_stats['quality_scores'] if s < 20])/len(self.study_stats['quality_scores'])*100:.1f}%</td></tr>
                </table>
            `;
            document.getElementById('qualityBreakdown').innerHTML = qualityBreakdown;

            // Channel performance
            var channelPerformance = '<table class="sortable-table"><tr><th>Channel Type</th><th>Count</th><th>Avg Quality</th></tr>';"""

        # Calculate channel type averages
        channel_avg_quality = {}
        for participant_id, rec_results in self.results.items():
            for recording_type, channels in rec_results.items():
                for channel_name, analysis in channels.items():
                    ch_type = analysis['channel_type']
                    if ch_type not in channel_avg_quality:
                        channel_avg_quality[ch_type] = []
                    channel_avg_quality[ch_type].append(analysis['quality_score'])

        for ch_type, scores in channel_avg_quality.items():
            avg_score = np.mean(scores)
            html_content += f"""
            channelPerformance += '<tr><td>{ch_type}</td><td>{len(scores)}</td><td>{avg_score:.1f}</td></tr>';"""

        html_content += f"""
            channelPerformance += '</table>';
            document.getElementById('channelPerformance').innerHTML = channelPerformance;

            // Recording comparison
            var recordingComparison = '<table class="sortable-table"><tr><th>Recording Type</th><th>Count</th><th>Avg Quality</th></tr>';"""

        # Calculate recording type averages
        recording_avg_quality = {}
        for participant_id, rec_results in self.results.items():
            for recording_type, channels in rec_results.items():
                if recording_type not in recording_avg_quality:
                    recording_avg_quality[recording_type] = []
                for channel_name, analysis in channels.items():
                    recording_avg_quality[recording_type].append(analysis['quality_score'])

        for rec_type, scores in recording_avg_quality.items():
            avg_score = np.mean(scores)
            html_content += f"""
            recordingComparison += '<tr><td>{rec_type}</td><td>{len(scores)}</td><td>{avg_score:.1f}</td></tr>';"""

        # Calculate percentiles
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = [np.percentile(self.study_stats['quality_scores'], p) for p in percentiles]

        html_content += f"""
            recordingComparison += '</table>';
            document.getElementById('recordingComparison').innerHTML = recordingComparison;

            // Detailed statistics
            var detailedStats = `
                <table class="sortable-table">
                    <tr><th>Statistic</th><th>Value</th></tr>
                    <tr><td>Mean Quality Score</td><td>{np.mean(self.study_stats['quality_scores']):.2f}</td></tr>
                    <tr><td>Median Quality Score</td><td>{np.median(self.study_stats['quality_scores']):.2f}</td></tr>
                    <tr><td>Standard Deviation</td><td>{np.std(self.study_stats['quality_scores']):.2f}</td></tr>
                    <tr><td>Minimum Score</td><td>{np.min(self.study_stats['quality_scores']):.2f}</td></tr>
                    <tr><td>Maximum Score</td><td>{np.max(self.study_stats['quality_scores']):.2f}</td></tr>
                </table>
            `;
            document.getElementById('detailedStats').innerHTML = detailedStats;

            // Percentile statistics
            var percentileStats = `
                <table class="sortable-table">
                    <tr><th>Percentile</th><th>Quality Score</th></tr>"""

        for p, val in zip(percentiles, percentile_values):
            html_content += f"""
                    <tr><td>{p}th</td><td>{val:.1f}</td></tr>"""

        html_content += f"""
                </table>
            `;
            document.getElementById('percentileStats').innerHTML = percentileStats;
        }}

        // Initialize issues tab content
        function initializeIssuesTab() {{
            // Issue frequency
            var issueFrequency = '<table class="sortable-table"><tr><th>Issue</th><th>Count</th><th>% of Channels</th></tr>';"""

        for issue, count in sorted(self.study_stats['issues_count'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / self.study_stats['total_channels']) * 100
            html_content += f"""
            issueFrequency += '<tr><td>{issue}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>';"""

        html_content += f"""
            issueFrequency += '</table>';
            document.getElementById('issueFrequency').innerHTML = issueFrequency;

            // Issues by channel type analysis would go here
            document.getElementById('issuesByChannel').innerHTML = '<p>Issues breakdown by channel type analysis...</p>';
            document.getElementById('multipleIssues').innerHTML = '<p>Participants with multiple concurrent issues analysis...</p>';
            document.getElementById('issueCorrelation').innerHTML = '<p>Issue correlation and co-occurrence analysis...</p>';
        }}

        // Collapsible content
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {{
            coll[i].addEventListener("click", function() {{
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.maxHeight) {{
                    content.style.maxHeight = null;
                }} else {{
                    content.style.maxHeight = content.scrollHeight + "px";
                }}
            }});
        }}

        // Initialize analysis tab on load if it's the active tab
        document.addEventListener('DOMContentLoaded', function() {{
            // Any initialization code here
        }});
    </script>
</body>
</html>"""

        return html_content

    def run(self):
        """Run the interactive summary generation process."""
        print(" Starting Interactive Summary Generation...")
        print(f" Working with data from: {self.output_folder}")

        # Extract data from reports
        self.extract_data_from_reports()

        if not self.results:
            print("ERROR: No data found. Please ensure the signal quality analysis has been completed.")
            return

        print(f"Loaded data for {len(self.results)} participants")

        # Create interactive HTML summary
        print(" Creating interactive HTML summary...")
        html_content = self.create_interactive_html_summary()

        # Save the summary
        summary_path = os.path.join(self.summary_dir, "interactive_summary.html")

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print("Interactive summary created")
        print(f" Summary saved to: {summary_path}")
        print(f" Open the file in your web browser to explore the interactive dashboard")

        # Also create a simple index file
        index_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Signal Quality Analysis - Index</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; }}
        .link-card {{
            display: block;
            padding: 20px;
            margin: 15px 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            transition: transform 0.2s;
        }}
        .link-card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.2); }}
        .link-title {{ font-size: 1.4em; font-weight: bold; margin-bottom: 8px; }}
        .link-description {{ opacity: 0.9; }}
        .stats {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; text-align: center; }}
        .stat-item {{ background: white; padding: 15px; border-radius: 6px; }}
        .stat-number {{ font-size: 1.8em; font-weight: bold; color: #667eea; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1> Signal Quality Analysis Results</h1>

        <div class="stats">
            <h2>Study Overview</h2>
            <div class="stat-grid">
                <div class="stat-item">
                    <div class="stat-number">{self.study_stats['total_participants']}</div>
                    <div class="stat-label">Participants</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{self.study_stats['total_channels']}</div>
                    <div class="stat-label">Channels</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{self.study_stats['average_quality']:.1f}</div>
                    <div class="stat-label">Avg Quality</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{len([s for s in self.study_stats['quality_scores'] if s >= 60])}</div>
                    <div class="stat-label">Good Quality</div>
                </div>
            </div>
        </div>

        <a href="interactive_summary.html" class="link-card">
            <div class="link-title"> Interactive Dashboard</div>
            <div class="link-description">
                Interactive dashboard with filterable tables, visualizations,
                and detailed analysis of signal quality across all participants.
            </div>
        </a>

        <a href="../reports/study_summary.html" class="link-card">
            <div class="link-title"> Basic Summary Report</div>
            <div class="link-description">
                Simple summary report with overview statistics and links to individual channel reports.
            </div>
        </a>

        <a href="../reports/" class="link-card">
            <div class="link-title"> Individual Channel Reports</div>
            <div class="link-description">
                Browse individual HTML reports for each channel with detailed analysis and visualizations.
            </div>
        </a>

        <a href="../visualizations/" class="link-card">
            <div class="link-title"> Visualization Gallery</div>
            <div class="link-description">
                Collection of all generated visualization images for each channel analysis.
            </div>
        </a>

        <div style="margin-top: 40px; text-align: center; color: #666;">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Signal Quality Analysis Dashboard - Memory Optimized Version</p>
        </div>
    </div>
</body>
</html>"""

        index_path = os.path.join(self.summary_dir, "index.html")
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)

        print(f" Index page created: {index_path}")
        print(f"\n All done! You can start exploring your data at:")
        print(f"    Main Index: {index_path}")
        print(f"    Interactive Dashboard: {summary_path}")


def main():
    """
    Main function to run the interactive summary generator.
    """
    print("="*80)
    print("INTERACTIVE SIGNAL QUALITY SUMMARY GENERATOR")
    print("="*80)

    # Check if output folder exists
    if not os.path.exists(OUTPUT_FOLDER):
        print(f"ERROR: Output folder not found: {OUTPUT_FOLDER}")
        print("Please check the OUTPUT_FOLDER path and ensure the signal quality analysis has been completed.")
        return

    # Check if reports directory exists
    reports_dir = os.path.join(OUTPUT_FOLDER, "reports")
    if not os.path.exists(reports_dir):
        print(f"ERROR: Reports directory not found: {reports_dir}")
        print("Please ensure the signal quality analysis has been completed successfully.")
        return

    # Create generator and run
    start_time = time.time()

    try:
        generator = InteractiveSummaryGenerator(OUTPUT_FOLDER)
        generator.run()

        end_time = time.time()
        duration = end_time - start_time

        print(f"\n⏱  Total generation time: {duration:.2f} seconds")
        print(f" Results saved in: {os.path.join(OUTPUT_FOLDER, 'summaries')}")

        # Provide usage instructions
        print(f"\n" + "="*60)
        print(" USAGE INSTRUCTIONS:")
        print("="*60)
        print("1. Open the index.html file in your web browser")
        print("2. Click 'Interactive Dashboard' for the full experience")
        print("3. Use the tabs to navigate different views:")
        print("    Overview - Study-wide statistics and visualizations")
        print("    Participants - Participant-level summary with filtering")
        print("    Channel Details - Individual channel information")
        print("    Analysis - Advanced statistical analysis")
        print("  Issues - Common problems and issue analysis")
        print("4. Use filters and sorting to explore your data")
        print("5. Click 'View Report' links to see detailed channel analyses")

    except Exception as e:
        print(f"ERROR: An exception occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
# %%
