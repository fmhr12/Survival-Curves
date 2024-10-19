# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

def create_figure(data, selected_groups, selected_time, title):
    fig = go.Figure()
    # Plot survival curves for each selected group
    for group in selected_groups:
        group_data = data[data['group'] == group]
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
        title=title,
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

def process_dataset(data, plot_title, dataset_id):
    st.header(plot_title)
    
    # Group selection with unique key
    groups = data['group'].unique()
    selected_groups = st.multiselect(
        f'Select Groups to Plot ({dataset_id})',
        options=groups,
        default=groups.tolist(),
        key=f'multiselect_{dataset_id}'
    )
    
    if not selected_groups:
        st.warning("Please select at least one group to display the survival curves.")
        st.markdown("---")
        return
    
    # Filter data based on selected groups
    filtered_data = data[data['group'].isin(selected_groups)]
    
    # Create a placeholder for the figure
    fig_placeholder = st.empty()
    
    # Plot the initial figure without vertical line
    initial_fig = create_figure(filtered_data, selected_groups, selected_time=None, title=plot_title)
    fig_placeholder.plotly_chart(initial_fig, use_container_width=True)
    
    # Slider for selecting time point with unique key
    time_min = data['timeline'].min()
    time_max = data['timeline'].max()
    selected_time = st.slider(
        f'Select Time Point ({dataset_id})',
        min_value=float(time_min),
        max_value=float(time_max),
        value=float(time_min),
        step=1.0,
        key=f'slider_{dataset_id}'
    )
    
    # Update the figure with the vertical line
    updated_fig = create_figure(filtered_data, selected_groups, selected_time, title=plot_title)
    fig_placeholder.plotly_chart(updated_fig, use_container_width=True)
    
    # Compute survival probabilities at the selected time
    probabilities = []
    for group in selected_groups:
        group_data = filtered_data[filtered_data['group'] == group]
        # Ensure the timeline is sorted for interpolation
        group_data = group_data.sort_values('timeline')
        # Interpolate survival probability at selected time
        survival_prob = np.interp(
            selected_time, group_data['timeline'], group_data['survival_probability']
        )
        probabilities.append({'Group': group, 'Survival Probability': survival_prob})
    
    # Display the probabilities in a table with unique key
    prob_df = pd.DataFrame(probabilities)
    prob_df.set_index('Group', inplace=True)
    st.write(f'#### Survival Probabilities {selected_time:.0f} Months After Radiotherapy')
    st.table(prob_df)
    st.markdown("---")  # Add a horizontal separator

# Load both datasets
survival_data_v60 = load_data('survival_data_v60.csv')
survival_data_D10 = load_data('survival_data_D10.csv')

# Process and display the first dataset
process_dataset(
    data=survival_data_v60,
    plot_title='Kaplan-Meier Survival Curves for Binarized V60Gy(cc): High Risk Group > 12.6',
    dataset_id='V60'
)

# Process and display the second dataset
process_dataset(
    data=survival_data_D10,
    plot_title='Kaplan-Meier Survival Curves for Binarized D10cc(Gy): High Risk Groups > 59.2',
    dataset_id='D10'
)
