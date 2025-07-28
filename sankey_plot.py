# %%

# Load necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict

# Load the dataset
df = pd.read_csv("path\\to\\your\\data.csv")

# Define color map
color_map = {
    1: "#0173b2",  # Mild
    2: "#de8f05",  # Moderate
    3: "#029e73",  # Severe
}

# Define function OUTSIDE of loops
def get_label_index_labeled(time, pain, label_index, labels, node_colors, counter):
    label = f"t{time}: pain {pain}"
    if label not in label_index:
        label_index[label] = counter
        label = f"t{time}: pain {pain}"
        labels.append(" ")  # empty text
        node_colors.append(color_map.get(int(pain)))
        counter += 1
    return label_index[label], counter

# OVERALL PLOT
pivot_df = df.pivot(index='id', columns='time', values='cat_pain')
pivot_df.columns = [f"time_{col}" for col in pivot_df.columns]
pivot_df.reset_index(inplace=True)

# Initialize structures
source, target, value = [], [], []
labels, node_colors, link_colors = [], [], []
label_index = {}
counter = 0
transition_dict = defaultdict(lambda: ['0 (0.0%)'] * 4)
totals = [0] * 4

# Build transitions and Sankey
for t in range(4):
    from_col, to_col = f"time_{t}", f"time_{t+1}"
    if from_col in pivot_df.columns and to_col in pivot_df.columns:
        sub_df = pivot_df[[from_col, to_col]].dropna()
        trans_counts = sub_df.groupby([from_col, to_col]).size().reset_index(name='count')
        total_count = trans_counts['count'].sum()
        totals[t] = total_count

        for _, row in trans_counts.iterrows():
            src, counter = get_label_index_labeled(t, row[from_col], label_index, labels, node_colors, counter)
            tgt, counter = get_label_index_labeled(t + 1, row[to_col], label_index, labels, node_colors, counter)
            source.append(src)
            target.append(tgt)
            value.append(row['count'])
            link_colors.append(color_map.get(int(row[from_col])))
            key = f"{int(row[from_col])} to {int(row[to_col])}"
            percent = (row['count'] / total_count) * 100 if total_count > 0 else 0
            transition_dict[key][t] = f"{int(row['count'])} ({percent:.1f}%)"

# Totals
for from_cat in [1, 2, 3]:
    total_row = []
    for t in range(4):
        total = sum(
            int(transition_dict[f"{from_cat} to {to_cat}"][t].split()[0])
            for to_cat in [1, 2, 3]
            if f"{from_cat} to {to_cat}" in transition_dict
        )
        percent = (total / totals[t]) * 100 if totals[t] > 0 else 0
        total_row.append(f"{total} ({percent:.1f}%)")
    transition_dict[f"Total ({from_cat})"] = total_row

# Save CSV
transition_df = pd.DataFrame.from_dict(transition_dict, orient='index', columns=[
    "time_0_to_1", "time_1_to_2", "time_2_to_3", "time_3_to_4"
])
transition_df.to_csv("path\\to\\your\\transition_data.csv")

# Plot
fig = go.Figure(data=[go.Sankey(
    node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=node_colors),
    link=dict(source=source, target=target, value=value, color=link_colors)
)])
fig.update_layout(title_text="Overall pain Transitions", height=600)
fig.show()

# CLUSTER-BY-CLUSTER PLOTS
clusters = sorted(df['clu#'].unique())
for cluster in clusters:
    df_cluster = df[df['clu#'] == cluster]
    pivot_cluster = df_cluster.pivot(index='id', columns='time', values='cat_pain')
    pivot_cluster.columns = [f"time_{col}" for col in pivot_cluster.columns]
    pivot_cluster.reset_index(inplace=True)

    # Reset structures per cluster
    source, target, value = [], [], []
    labels, node_colors, link_colors = [], [], []
    label_index = {}
    counter = 0

    for t in range(4):
        from_col = f"time_{t}"
        to_col = f"time_{t+1}"
        if from_col in pivot_cluster.columns and to_col in pivot_cluster.columns:
            transitions = pivot_cluster.groupby([from_col, to_col]).size().reset_index(name='count')
            for _, row in transitions.iterrows():
                src, counter = get_label_index_labeled(t, row[from_col], label_index, labels, node_colors, counter)
                tgt, counter = get_label_index_labeled(t + 1, row[to_col], label_index, labels, node_colors, counter)
                source.append(src)
                target.append(tgt)
                value.append(row['count'])
                link_colors.append(color_map.get(int(row[from_col])))

    fig = go.Figure(data=[go.Sankey(
        node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels, color=node_colors),
        link=dict(source=source, target=target, value=value, color=link_colors)
    )])
    fig.update_layout(title_text=f"Pain Transitions - Cluster {cluster}", height=600)
    fig.show()