"""
FracAtlas Dataset - Fracture Distribution Bar Chart
-----------------------------------------------------
Generates a bar chart showing the distribution of fractured images
by body region (Hand, Leg, Hip, Shoulder).

Usage:
    python fracture_distribution_chart.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration - Update these paths

CSV_PATH = "FracAtlas/dataset.csv"  # Update this path if needed
OUTPUT_PATH = "fracture_distribution.png"

# Main

def main():
    # Load the dataset
    print(f"Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    print(f"Total images in dataset: {len(df)}")
    
    # Filter to only fractured images
    fractured_df = df[df['fractured'] == 1]
    print(f"Total fractured images: {len(fractured_df)}")
    print(f"Total fracture instances: {fractured_df['fracture_count'].sum()}")
    
    # Count fractured images by body region
    # Note: An image can belong to multiple regions
    regions = ['hand', 'leg', 'hip', 'shoulder']
    region_labels = ['Hand', 'Leg', 'Hip', 'Shoulder']
    
    fractured_counts = []
    for region in regions:
        count = fractured_df[fractured_df[region] == 1].shape[0]
        fractured_counts.append(count)
        print(f"  {region.capitalize()}: {count} fractured images")
    
    # Also count total fracture instances by region
    instance_counts = []
    for region in regions:
        region_df = fractured_df[fractured_df[region] == 1]
        instances = region_df['fracture_count'].sum()
        instance_counts.append(instances)
        print(f"  {region.capitalize()}: {instances} fracture instances")
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(region_labels))
    width = 0.35
    
    #Just fractured images (matching the FracAtlas paper's figure style)
    bars = ax.bar(x, fractured_counts, width, color='#E74C3C', edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, count in zip(bars, fractured_counts):
        height = bar.get_height()
        ax.annotate(f'{count}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Body Region', fontsize=12)
    ax.set_ylabel('Number of X-rays Containing a Specific Region', fontsize=12)
    #ax.set_title('Distribution of Fractured X-rays by Body Region\n(FracAtlas Dataset)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, fontsize=11)
    ax.set_ylim(0, max(fractured_counts) * 1.15)  # Add some headroom for labels
    
    # Add grid for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nChart saved to: {OUTPUT_PATH}")
    
    # Also create a version with fracture instances
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(x, instance_counts, width, color='#3498DB', edgecolor='black', linewidth=1.2)
    
    for bar, count in zip(bars, instance_counts):
        height = bar.get_height()
        ax.annotate(f'{int(count)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Body Region', fontsize=12)
    ax.set_ylabel('Number of Fracture Instances', fontsize=12)
    ax.set_title('Distribution of Fracture Instances by Body Region\n(FracAtlas Dataset)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, fontsize=11)
    ax.set_ylim(0, max(instance_counts) * 1.15)
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    output_instances = OUTPUT_PATH.replace('.png', '_instances.png')
    plt.savefig(output_instances, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Instance chart saved to: {output_instances}")
    
    # Summary statistics
    print("\n" + "=" * 50)
    print("  SUMMARY")
    print("=" * 50)
    print(f"Total fractured images: {len(fractured_df)}")
    print(f"Total fracture instances: {int(fractured_df['fracture_count'].sum())}")
    print(f"\nNote: Some images contain multiple body regions,")
    print(f"so regional counts may sum to more than total images.")


if __name__ == "__main__":
    main()
