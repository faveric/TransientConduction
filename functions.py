#import fmpy
from fmpy import simulate_fmu
from fmpy.model_description import  read_model_description
import streamlit as st
from fmpy.util import compile_platform_binary
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#import io
#import numpy as np
#from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly.express as px



# Function to simulate FMU
def read_model_variables(fmu_path):
    vars = read_model_description(fmu_path)
    vars = [var.name for var in vars.modelVariables]
    return vars
def simulate_transient_conduction(fmu_path, start_time, stop_time, tolerance, parameters,call_count):
    if call_count == 0:
        print('First simulation!')
        compile_platform_binary(fmu_path)
    result = simulate_fmu(
        fmu_path,
        start_time=start_time,
        stop_time=stop_time,
        relative_tolerance=tolerance,
        output= read_model_variables(fmu_path),
        start_values=parameters,
        fmi_type = 'ModelExchange'
    )
    return result


# Function to plot theta for both Lumped and 1D formulations
# Function to plot lumped and 1D formulation from the DataFrame
def plot_theta_lumped_and_1d(df):
    time = df['time']  # Time values from the DataFrame

    # Create the plot
    fig, ax = plt.subplots()

    # Plot Lumped parameter result (thick line)
    ax.plot(time, df['theta_lumped'], label='Lumped', linewidth=3, color='blue')

    # Plot 1D formulation (dashed lines for theta[1] to theta[N])
    for i in range(1, int(df['N'][0]) + 1):
        ax.plot(time, df[f'theta[{i}]'], '--', label=f'1D: Volume {i}')

    # Customize plot
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Dimensionless Temperature (theta)')
    ax.set_title('Theta vs Time for Lumped and 1D Formulation')
    ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.0))  # Adjust bbox_to_anchor for exact position
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)


def plot_heisler_theta_chart(df):
    time_values = df['time'].unique()  # Unique time values from the DataFrame

    # Create a slider for time selection
    selected_time = st.select_slider(
        "Select Time for Heisler Chart",
        options=sorted(df['time'].unique()),  # Populate slider with unique time values
        value=df['time'].iloc[0],  # Set default value to the first time point
        format_func=lambda x: f"{x:.2f} s" # Format the time display on the slider
    )
    # Filter the DataFrame for the selected time
    df_selected = df[df['time'] == selected_time]

    # Create radial positions (x-axis of the Heisler chart)
    N = int(df['N'][0])
    x_values = [df_selected[f'x[{i}]'].values[0] for i in range(1, N + 1)]

    # Extract corresponding dimensionless temperatures (theta) for the selected time
    theta_values = [df_selected[f'theta[{i}]'].values[0] for i in range(1, N + 1)]

    # Plot using Plotly for interactivity
    fig = px.bar(
        x=x_values,
        y=theta_values,
        labels={'x': 'Dimensionless Radial Position (x)', 'y': 'Dimensionless Temperature (theta)'},
        title=f"Heisler Chart at Time {selected_time:.1f}",
        range_y=[0, 1]  # Ensures that the y-axis is between 0 and 1
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)




def plot_polar_theta_map(df):
    # Create a slider for selecting time
    selected_time_sph = st.select_slider(
        "Select Time for Spherical Temperature Distribution",
        options=sorted(df['time'].unique()),  # Populate slider with unique time values
        value=df['time'].iloc[0],  # Set default value to the first time point
        format_func=lambda x: f"{x:.2f} s"  # Format the time display on the slider
    )

    # Filter DataFrame for the selected time
    df_selected = df[df['time'] == selected_time_sph]

    # Get the number of radial divisions (N)
    N = int(df['N'][0])

    # Radial positions (x[1], x[2], ..., x[N]) - add x[0] = 0 manually
    x_values = np.array([0] + [df_selected[f'x[{i}]'].values[0] for i in range(1, N + 1)])

    # Theta values (temperature values for each radial division)
    theta_values = np.array([df_selected[f'theta[{i}]'].values[0] for i in range(1, N + 1)])

    # Define angular range for a full circle (0 to 2*pi)
    phi_angle = np.linspace(0, 2 * np.pi, 100)

    # Create a polar plot with filled circular crowns
    fig = go.Figure()

    # Plot concentric circular regions with color corresponding to the temperature in that region
    for i in range(1, len(x_values)):
        r_inner = x_values[i-1]  # Inner radius of the crown
        r_outer = x_values[i]    # Outer radius of the crown
        theta_value = theta_values[i-1]  # Corresponding temperature value

        # Manually define the colorscale between blue (lower) and red (higher)
        color = 'BuRd'
        normalized_theta = theta_value / max(theta_values)  # Normalize theta to [0,1] for color mapping

        # Create filled area between r_inner and r_outer
        fig.add_trace(go.Scatterpolar(
            r=np.concatenate([np.full_like(phi_angle, r_inner), np.full_like(phi_angle, r_outer)]),
            theta=np.concatenate([phi_angle, phi_angle[::-1]]) * (180 / np.pi),  # Convert radians to degrees
            fill='toself',
            fillcolor=f'rgba({255 * (normalized_theta):.0f}, {255 * (1-normalized_theta):.0f}, 128, 0.7)',  # Adjust colors dynamically
            line=dict(color='black', width=1),
            name=f'Theta [{r_inner:.2f}, {r_outer:.2f}]'
        ))

    # Update layout for polar plot
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, 1]),  # Hide radial axis
            angularaxis=dict(visible=False)  # Hide angular axis
        ),
        showlegend=True,
        title=f"Spherical Temperature Distribution at Time {selected_time_sph:.2f} s",
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

def plot_T_lumped_and_1d_pyplot(df):
    time = df['time']  # Time values from the DataFrame

    # Create the plot
    fig, ax = plt.subplots()

    # Plot Lumped parameter result (thick line for temperature T)
    ax.plot(time, df['T_lumped'], label='Lumped', linewidth=3, color='blue')

    # Plot 1D formulation (dashed lines for T[1] to T[N])
    for i in range(1, int(df['N'][0]) + 1):
        ax.plot(time, df[f'T[{i}]'], '--', label=f'1D: Volume {i}')

    # Customize plot
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Temperature (T)')
    ax.set_title('Temperature vs Time for Lumped and 1D Formulation')
    ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.0))  # Adjust bbox_to_anchor for exact position
    ax.grid(True)

    # Display the plot in Streamlit
    st.pyplot(fig)

import matplotlib.cm as cm

def plot_T_lumped_and_1d(df):
    time = df['time']  # Time values from the DataFrame

    # Create Plotly figure
    fig = go.Figure()

    # Plot Lumped parameter result (black line for temperature T)
    fig.add_trace(go.Scatter(
        x=time,
        y=df['T_lumped'],
        mode='lines',
        line=dict(color='black', width=3),
        name='Lumped'
    ))

    # Use viridis colormap for the 1D formulation lines
    cmap = cm.get_cmap('viridis', int(df['N'][0]))  # Get 'viridis' colormap with N distinct colors

    # Plot 1D formulation (dashed lines for T[1] to T[N] using viridis colormap)
    for i in range(1, int(df['N'][0]) + 1):
        color = cmap((i - 1) / (int(df['N'][0]) - 1))  # Normalize i to get color from colormap
        rgb_color = f'rgb({color[0]*255:.0f}, {color[1]*255:.0f}, {color[2]*255:.0f})'  # Convert to RGB for Plotly
        fig.add_trace(go.Scatter(
            x=time,
            y=df[f'T[{i}]'],
            mode='lines',
            line=dict(dash='dash', color=rgb_color),
            name=f'1D: Volume {i}'
        ))

    # Customize layout
    fig.update_layout(
        title='Temperature vs Time for Lumped and 1D Formulation',
        xaxis_title='Time (s)',
        yaxis_title='Temperature (T)',
        legend=dict(
            x=1.05,
            y=0,
            traceorder='normal',
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=True
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

def plot_heisler_T_chart(df):
    time_values = df['time'].unique()  # Unique time values from the DataFrame

    T_max=max(df['T_inf'].iloc[0], df['T_init'].iloc[0]) # Maximum Temperature
    T_min=min(df['T_inf'].iloc[0], df['T_init'].iloc[0])  # Minimum Temperature

    # Create a slider for time selection
    selected_time = st.select_slider(
        "Select Time for Heisler Chart",
        options=sorted(df['time'].unique()),  # Populate slider with unique time values
        value=df['time'].iloc[0],  # Set default value to the first time point
        format_func=lambda x: f"{x:.2f} s"  # Format the time display on the slider
    )

    # Filter the DataFrame for the selected time
    df_selected = df[df['time'] == selected_time]

    # Create radial positions (x-axis of the Heisler chart)
    N = int(df['N'][0])
    r_values = [df_selected[f'r[{i}]'].values[0] for i in range(1, N + 1)]  # Use 'r' instead of 'x'

    # Extract corresponding temperature values (T) for the selected time
    T_values = [df_selected[f'T[{i}]'].values[0] for i in range(1, N + 1)]  # Use 'T' instead of 'theta'

    # Normalize temperature values for color scaling (0 = blue, 1 = red)
    normalized_T = [(T - T_min) / (T_max - T_min) for T in T_values]

    # Plot using Plotly for interactivity
    fig = px.bar(
        x=r_values,
        y=T_values,
        labels={'x': 'Radial Position (r)', 'y': 'Temperature (T)'},  # Update axis labels
        title=f"Heisler Chart - Temperature Distribution at Time {selected_time:.1f} s",
        range_y=[T_min-5, T_max],  # Ensures that the y-axis range fits the data
        color=normalized_T,  # Use normalized temperatures for color mapping
        color_continuous_scale='RdBu_r',  # Red to Blue color scale, reversed
        range_color = [0, 1]  # Fix the range from 0 to 1 for the color scale
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)

def plot_polar_temperature_map(df):

    T_max = max(df['T_inf'].iloc[0], df['T_init'].iloc[0])  # Maximum Temperature
    T_min = min(df['T_inf'].iloc[0], df['T_init'].iloc[0])  # Minimum Temperature

    # Create a slider for selecting time
    selected_time_sph = st.select_slider(
        "Select Time for Polar Temperature Distribution",
        options=sorted(df['time'].unique()),  # Populate slider with unique time values
        value=df['time'].iloc[0],  # Set default value to the first time point
        format_func=lambda x: f"{x:.2f} s"  # Format the time display on the slider
    )

    # Filter DataFrame for the selected time
    df_selected = df[df['time'] == selected_time_sph]

    # Get the number of radial divisions (N)
    N = int(df['N'][0])

    # Radial positions (r[1], r[2], ..., r[N]) - add r[0] = 0 manually
    r_values = np.array([0] + [df_selected[f'r[{i}]'].values[0] for i in range(1, N + 1)])

    # Temperature values (T values for each radial division)
    T_values = np.array([df_selected[f'T[{i}]'].values[0] for i in range(1, N + 1)])

    # Define angular range for a full circle (0 to 2*pi)
    phi_angle = np.linspace(0, 2 * np.pi, 100)

    # Create a polar plot with filled circular crowns
    fig = go.Figure()

    # Plot concentric circular regions with color corresponding to the temperature in that region
    for i in range(1, len(r_values)):
        r_inner = r_values[i - 1]  # Inner radius of the crown
        r_outer = r_values[i]      # Outer radius of the crown
        T_value = T_values[i - 1]  # Corresponding temperature value

        # Normalize temperature for the color scale (0 = blue, 1 = red)
        normalized_T = (T_value - T_min) / (T_max - T_min)

        # Create a gradient from blue (low) to red (high) using rgba values
        # Blue (0, 0, 255), Red (255, 0, 0)
        red = int(255 * normalized_T)  # More red as T increases
        blue = int(255 * (1 - normalized_T))  # Less blue as T increases
        fill_color = f'rgba({red}, 0, {blue}, 0.7)'  # Use some transparency for better visibility

        # Create filled area between r_inner and r_outer
        fig.add_trace(go.Scatterpolar(
            r=np.concatenate([np.full_like(phi_angle, r_inner), np.full_like(phi_angle, r_outer)]),
            theta=np.concatenate([phi_angle, phi_angle[::-1]]) * (180 / np.pi),  # Convert radians to degrees
            fill='toself',
            fillcolor=fill_color,  # Apply color based on the normalized temperature
            line=dict(color='black', width=1),
            name=f'T[{i}] = {T_value:.2f} °C'
        ))

    # Update layout for polar plot
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=False, range=[0, max(r_values)]),  # Adjust the range based on radial values
            angularaxis=dict(visible=False)  # Hide angular axis
        ),
        showlegend=True,
        title=f"Polar Chart - Temperature Distribution at Time {selected_time_sph:.2f} s",
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)



