# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# Load the survival data
survival_data = pd.read_csv('survival_data_v60.csv')

# Sidebar: Group selection
groups = survival_data['group'].unique()
selected_groups = st.sidebar.multiselect(
    'Select Groups to Plot', groups, default=groups.tolist()
)

# Filter data based on selected groups
filtered_data = survival_data[survival_data['group'].isin(selected_groups)]

# Create a placeholder for the figure
fig_placeholder = st.empty()

# Initial figure without vertical line
def create_figure(selected_time=None):
    fig = go.Figure()
    # Plot survival curves for each selected group
    for group in selected_groups:
        group_data = filtered_data[filtered_data['group'] == group]
        color = group_data['color'].iloc[0]
        linestyle = group_data['linestyle'].iloc[0]
        fig.add_trace(
            go.Scatter(
                x=group_data['timeline'],
                y=group_data['survival_probability'],
                mode='lines',
                name=group,
                line=dict(color=color, dash=linestyle),
                hovertemplate='%{y:.4f}<extra></extra>'
            )
        )
    # Add vertical line at selected time point
    if selected_time is not None:
        fig.add_vline(x=selected_time, line_dash="dash", line_color="gray")
    # Update layout to show vertical line on hover
    fig.update_layout(
        title='Kaplan-Meier Survival Curves for binarized V60cc(Gy): High Risk Groups > 59.2',
        xaxis_title='Time (Months)',
        yaxis_title='Survival Probability',
        hovermode='x',
        width=1000,
        height=600,
        xaxis=dict(
            showspikes=True,
            spikecolor="gray",
            spikethickness=1,
            spikedash='dot',
            spikemode='across',
        ),
        hoverdistance=100,  # Distance to show hover effect
        spikedistance=1000,  # Distance to show spike
    )
    return fig

# Display the figure initially without vertical line
fig = create_figure()
fig_display = fig_placeholder.plotly_chart(fig, use_container_width=True)

# Slider for selecting time point (now below the figure)
time_min = survival_data['timeline'].min()
time_max = survival_data['timeline'].max()
selected_time = st.slider(
    'Select Time Point',
    min_value=float(time_min),
    max_value=float(time_max),
    value=float(time_min),
    step=1.0,
)

# After slider is adjusted, update the figure with the vertical line
fig = create_figure(selected_time)
fig_placeholder.plotly_chart(fig, use_container_width=True)

# Compute survival probabilities at the selected time
probabilities = []
for group in selected_groups:
    group_data = filtered_data[filtered_data['group'] == group]
    # Interpolate survival probability at selected time
    survival_prob = np.interp(
        selected_time, group_data['timeline'], group_data['survival_probability']
    )
    probabilities.append({'Group': group, 'Survival Probability': survival_prob})

# Display the probabilities in a table
prob_df = pd.DataFrame(probabilities)
prob_df.set_index('Group', inplace=True)
st.write(f'#### Survival probabilities {selected_time:.0f} months after radiotherapy')
st.table(prob_df)
