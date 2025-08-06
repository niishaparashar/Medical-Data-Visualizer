import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Import the data
df = pd.read_csv('medical_examination.csv')

# 2: Add 'overweight' column
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)

# 3: Normalize data for cholesterol and glucose
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

def draw_cat_plot():
    # 4: Melt the DataFrame
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 5: Group, reformat, and rename count column
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    df_cat = df_cat.rename(columns={'size': 'total'})

    # 6: Create categorical plot
    fig = sns.catplot(
        x='variable', y='total', hue='value', col='cardio',
        data=df_cat, kind='bar'
    ).fig

    return fig

def draw_heat_map():
    # 7: Filter and clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 8: Calculate the correlation matrix
    corr = df_heat.corr()

    # 9: Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 10: Set up figure and draw heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.1f',
        center=0, vmax=.3, vmin=-.1, square=True, linewidths=.5, cbar_kws={"shrink": .5}
    )

    return fig
