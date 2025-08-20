import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import io

# Set random seed for reproducible results
np.random.seed(42)

# Generate realistic synthetic data for marketing campaign effectiveness
n_campaigns = 120

# Generate data with clear patterns
np.random.seed(42)
campaign_data = {
    'marketing_spend': np.random.uniform(10, 100, n_campaigns),  # Marketing spend in thousands
    'conversion_rate': np.random.uniform(1, 20, n_campaigns),    # Conversion rate percentage
    'campaign_type': np.random.choice(['Social Media', 'Email', 'PPC', 'Display'], n_campaigns),
    'duration_days': np.random.randint(7, 60, n_campaigns)
}

# Create stronger correlation between spend and conversion
for i in range(n_campaigns):
    base_conversion = campaign_data['marketing_spend'][i] * 0.15 + np.random.normal(0, 2)
    campaign_data['conversion_rate'][i] = max(0.5, min(25, base_conversion))

# Create DataFrame
df = pd.DataFrame(campaign_data)

# Set Seaborn style and context
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)


# Create figure with exact dimensions for 512x512 output
plt.figure(figsize=(8, 8))

# Create a Seaborn barplot: average conversion rate by campaign type
bar_palette = 'Set2'
ax = sns.barplot(
    data=df,
    x='campaign_type',
    y='conversion_rate',
    estimator=np.mean,
    ci='sd',
    palette=bar_palette,
    edgecolor='black',
    errorbar='sd'
)

# Professional styling
plt.title('Average Conversion Rate by Campaign Type', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Campaign Type', fontsize=14, fontweight='semibold')
plt.ylabel('Average Conversion Rate (%)', fontsize=14, fontweight='semibold')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add grid and remove top/right spines
plt.grid(axis='y', alpha=0.3)
sns.despine()

# Tight layout for saving
plt.tight_layout()

# Save chart as PNG with exactly 512x512 pixels
plt.savefig('chart.png', dpi=64, bbox_inches='tight', facecolor='white', edgecolor='none')

# Display summary statistics
print("Marketing Campaign Effectiveness Analysis")
print("=" * 50)
print(f"Total Campaigns: {len(df)}")
print(f"Average Marketing Spend: ${df['marketing_spend'].mean():.2f}K")
print(f"Average Conversion Rate: {df['conversion_rate'].mean():.2f}%")
print(f"Correlation (Spend vs Conversion): {df['marketing_spend'].corr(df['conversion_rate']):.3f}")
print("\nChart generated successfully with Seaborn barplot!")

plt.show()
