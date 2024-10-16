import streamlit as st
import numpy as np
import pandas as pd
from problem_description import problem_description
from functions import *
import seaborn as sns


# Initialize session state variables
if 'call_count' not in st.session_state:
    st.session_state['call_count'] = 0  # compile FMU if call count is 0
if 'result' not in st.session_state:
    st.session_state['result'] = None  # initialize result


# Geometric Parameters
st.sidebar.subheader('Geometric Parameters')
r0 = st.sidebar.number_input("Sphere Radius (m)", value=0.01)

st.sidebar.subheader('Material Parameters')
# Material presets
materials = {
    "Custom": {},
    "Steel": {"rho": 7850, "k": 45.0, "c": 466.0},
    "Wood": {"rho": 700, "k": 0.15, "c": 1760.0}
}

# Material selection
material = st.sidebar.selectbox("Select Material", list(materials.keys()))

if material != "Custom":
    st.sidebar.write(f"Using preset values for {material}")
    rho = st.sidebar.number_input("Material Density (kg/m³)", value=materials[material]["rho"], disabled=True)
    k = st.sidebar.number_input("Thermal Conductivity (W/(m·K))", value=materials[material]["k"], disabled=True)
    c = st.sidebar.number_input("Specific Heat (J/kg)", value=materials[material]["c"], disabled=True)
else:
    rho = st.sidebar.number_input("Material Density (kg/m³)", min_value=50.00, value=7850.0)
    k = st.sidebar.number_input("Thermal Conductivity (W/(m·K))", min_value = 0.001, value=45.0)
    c = st.sidebar.number_input("Specific Heat (J/kg)", min_value = 1.0, value=466.0)

st.sidebar.header("Heat Transfer Parameters")
h = st.sidebar.number_input("Heat Transfer Coefficient (W/(m²·K))", value=300.0)
T_init = st.sidebar.number_input("Initial Temperature (K)", value=300.0)
T_inf = st.sidebar.number_input("Fluid Temperature (K)", value=273.0)

problem_description(T_init=T_init)

# Calculate Biot number
Lc = r0 / 3
Bi = h * Lc / k
st.write(f"##### **Biot number value= {Bi:.4f}**")
if Bi < 0.1:
    st.success("Bi < 0.1: Both 0D and 1D formulations are accurate.")
else:
    st.warning("Bi >= 0.1: 1D Finite volume formulation is recommended for accurate results.")

# Calculate tau and suggest simulation time
# Calculate Fo
alpha = k / (rho * c)
Fo = alpha/Lc**2
tau = 1/(Bi*Fo)

# Display the concept
st.subheader("Simulation Time")
with st.expander("How long should the simulation last?"):
    st.write("### Time Constant and Simulation Stopping Time")
    st.write(f"The time constant $(τ)$ for this heat transfer problem can be expressed as:")
    st.latex(r"\tau = \frac{1}{Bi \cdot Fo}")
    st.write(f"For the selected parameters, the time constant is:")
    st.latex(f"\\tau = {tau:.2f} \, s")
    st.write(f"For 0D formulation it is suggested to stop the simulation at approximately $5τ$, which is:")
    st.latex(f"5\\tau = {5 * tau:.2f} \, s")
    st.write(f"This ensures that the system reaches near steady-state conditions.")
    st.write(f"The option on the sidebar is automatically set at $10τ$.")

st.write(f"##### **Time Constant $\\tau$ value= {tau:.2f}**")


st.sidebar.subheader('Simulation Parameters')
#N = st.sidebar.number_input("Number of Volumes (1D)", min_value=3, value=10, step=1)
stop_time = st.sidebar.number_input("Stop Time (s)", min_value = 1.0, value=10*tau)
if st.sidebar.button("Start Simulation"):
    fmu_path = "./ModelicaResources/TransientConduction_CS.fmu"
    parameters = {
        'rho': rho, 'c': c, 'h': h, 'k': k, 'r0': r0, 'T_init': T_init, 'T_inf': T_inf
    }
    st.session_state['result'] = simulate_transient_conduction(fmu_path,
                                           0,
                                           stop_time,
                                           1e-6,
                                           parameters,
                                           st.session_state['call_count'])
    st.session_state['call_count'] = 1

if st.session_state['result'] is not None:
    st.header('Simulation Results')


    df = pd.DataFrame(st.session_state['result'])
    with st.expander('Show simulation results table'):
        st.dataframe(df)

    st.subheader(r'Time Plot non-dimensional temperatures ($\theta$) - 0D and 1D')
    plot_T_lumped_and_1d(df)
    st.subheader('Heisler Chart for 1D non-dimensional temperatures')
    plot_heisler_T_chart(df)
    st.subheader('Polar Chart for 1D non-dimensional temperatures')
    plot_polar_temperature_map(df)