#!/usr/bin/env python3
"""
Simple metrics plotter for video stabilization CSV data
Usage: python plot_metrics.py [input.csv]
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
csv_file = sys.argv[1] if len(sys.argv) > 1 else 'output.csv'
print(f"Loading: {csv_file}")

# Define column names (in case CSV has no header)
column_names = ['frame', 'filter', 'tracked', 'total', 'trackRate', 
                'inliers', 'totalPts', 'inlierRate', 'rawShake', 
                'smoothShake', 'shakeReduct', 'rawVar', 'smoothVar', 
                'varReduct', 'cropPct']

# Try reading with header first, if that fails use manual column names
try:
    df = pd.read_csv(csv_file)
    # Check if first column looks like a header
    if 'frame' not in df.columns:
        df = pd.read_csv(csv_file, names=column_names)
except:
    df = pd.read_csv(csv_file, names=column_names)

print(f"Loaded {len(df)} frames")
print(f"Filters used: {df['filter'].unique()}")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Video Stabilization Metrics', fontsize=16, fontweight='bold')

# Plot 1: Feature Tracking Rate
ax = axes[0, 0]
ax.plot(df['frame'], df['trackRate'], color='#89b4fa', linewidth=2)
ax.set_xlabel('Frame')
ax.set_ylabel('Tracking Rate (%)')
ax.set_title('Feature Tracking Success')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

# Plot 2: RANSAC Inlier Rate
ax = axes[0, 1]
ax.plot(df['frame'], df['inlierRate'], color='#cba6f7', linewidth=2)
ax.set_xlabel('Frame')
ax.set_ylabel('Inlier Rate (%)')
ax.set_title('RANSAC Inlier Quality')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

# Plot 3: Shake Reduction
ax = axes[0, 2]
ax.plot(df['frame'], df['shakeReduct'], color='#a6e3a1', linewidth=2)
ax.axhline(y=df['shakeReduct'].mean(), color='red', linestyle='--', 
           label=f'Mean: {df["shakeReduct"].mean():.1f}%')
ax.set_xlabel('Frame')
ax.set_ylabel('Reduction (%)')
ax.set_title('Shake Reduction')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Motion Energy
ax = axes[1, 0]
ax.plot(df['frame'], df['rawShake'], color='#f38ba8', linewidth=2, label='Raw', alpha=0.7)
ax.plot(df['frame'], df['smoothShake'], color='#94e2d5', linewidth=2, label='Smoothed')
ax.set_xlabel('Frame')
ax.set_ylabel('Cumulative Motion')
ax.set_title('Motion Energy (Lower = Smoother)')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 5: Variance Reduction
ax = axes[1, 1]
ax.plot(df['frame'], df['varReduct'], color='#fab387', linewidth=2)
ax.axhline(y=df['varReduct'].mean(), color='red', linestyle='--',
           label=f'Mean: {df["varReduct"].mean():.1f}%')
ax.set_xlabel('Frame')
ax.set_ylabel('Reduction (%)')
ax.set_title('Variance Reduction (Jitter)')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 6: Crop Efficiency
ax = axes[1, 2]
ax.plot(df['frame'], df['cropPct'], color='#f9e2af', linewidth=2)
ax.axhline(y=df['cropPct'].mean(), color='red', linestyle='--',
           label=f'Mean: {df["cropPct"].mean():.2f}%')
ax.set_xlabel('Frame')
ax.set_ylabel('Crop Area (%)')
ax.set_title('Crop Efficiency')
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout()

# Save figure
output_png = csv_file.replace('.csv', '_plots.png')
plt.savefig(output_png, dpi=300, bbox_inches='tight')
print(f"\nSaved: {output_png}")

# Show summary statistics
print("\n=== Summary Statistics ===")
print(f"Feature Tracking Rate: {df['trackRate'].mean():.1f}% ± {df['trackRate'].std():.1f}%")
print(f"RANSAC Inlier Rate: {df['inlierRate'].mean():.1f}% ± {df['inlierRate'].std():.1f}%")
print(f"Shake Reduction: {df['shakeReduct'].mean():.1f}% ± {df['shakeReduct'].std():.1f}%")
print(f"Variance Reduction: {df['varReduct'].mean():.1f}% ± {df['varReduct'].std():.1f}%")
print(f"Crop Efficiency: {df['cropPct'].mean():.2f}% ± {df['cropPct'].std():.2f}%")

plt.show()
