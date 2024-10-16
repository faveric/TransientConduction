import streamlit as st


def problem_description(T_init):
    st.title("Transient Heat Conduction in a Sphere in 0D and 1D")

    st.write("This section provides a comprehensive description of the problem being simulated.")

    st.header("Problem Overview")

    st.write(r"""
        The model simulates the **transient heat conduction** in a solid sphere immersed at time $t_{init} = 0$ in a large body of a cooling or heating fluid. 
        This scenario has various practical applications, such as heat treatment of materials, cooling of hot objects, 
        or modeling of thermal processes in spherical geometries.

        0-Dimensional (0D) and 1-Dimensional (1D) formulations are analyzed.  
    """)

    with st.expander("Physical Setup"):
        st.write("### Sphere")
        st.write(r"""
        - A solid sphere of radius $r_0$ (m)
        - Initially at a uniform temperature $T_{init}$ (K)
        - Made of a homogeneous, continuous, and isotropic material
        - Material properties:
          - Density: $ρ$ (kg/m³)
          - Specific heat capacity: $c$ (J/(kg·K))
          - Thermal conductivity: $k$ (W/(m·K))
        """)

        st.write("### Surrounding Medium")
        st.write(r"""
        - A large body of a fluid
        - Maintained at a constant temperature $T_∞$ (K)
        - The water body is large enough that its temperature remains unaffected by the presence of the sphere
        """)

        st.write("### Heat Transfer")
        st.write(r"""
        - Convective heat transfer occurs at the surface of the sphere
        - Characterized by constant heat transfer coefficient $h$ (W/(m²·K))
        """)

    with st.expander("Details of Modeling Assumptions"):
        st.write("### Modeling Assumptions")
        st.write(r"""
        1. **Material Properties**: The thermal properties of the sphere $(ρ, c, k)$ are assumed to be constant and independent of temperature within the range of temperatures encountered in the problem.
        2. **Isotropic Material**: The material properties are the same in all directions within the sphere.
        3. **Homogeneous Material**: The material properties are uniform throughout the sphere and do not vary with radius.
        4. **No Internal Heat Generation**: There are no heat sources or sinks within the sphere.
        5. **Spherical Symmetry**: Due to the uniform initial condition and spherical geometry, the temperature within the sphere is a function of radius and time only, allowing for a zero/one-dimensional analysis.
        6. **Constant Surface Condition**: The convective heat transfer coefficient ($h$) and the surrounding water temperature ($T_∞$) remain constant throughout the process.
        7. **Negligible Radiation**: Radiative heat transfer is assumed to be negligible compared to convection and conduction.
        """)

    with st.expander("Analysis Methods"):
        st.write("### Analysis Methods")
        st.write(r"""
        Two methods are employed to analyze this problem:

        1. **Lumped Capacitance Method (0D)**: 
           - Assumes uniform temperature throughout the sphere at any given time
           - Valid for low Biot numbers ($Bi < 0.1$)
           - Biot number:
        """)
        st.latex(r""" Bi = \frac{h L_c}{k} """)
        st.write(r"where $L_c$ is the characteristic length ($r_s / 3$ for a sphere)")

        st.write("""
        2. **Finite Volume Method (1D)**:
           - Discretizes the sphere into concentric shells (Ribando, 1998)
           - Accounts for temperature gradients within the sphere
           - More accurate, especially for higher Biot numbers

        The simulation allows comparison between these two methods, demonstrating the importance of boundary conditions in choosing the correct model.
        Both methods are solved using dimensionless variables.
        """)

    with st.expander("Dimensionless parameters definition"):
        st.write("### Dimensionless Parameters Definition")
        st.write("The following dimensionless parameters are defined:")
        st.write(r"$\theta = \frac{T - T_{\infty}}{T_{\text{init}} - T_{\infty}} \quad$ : dimensionless temperature")
        st.write(r"$\alpha = \frac{k}{\rho c_p} \quad$ : thermal diffusivity")
        st.write(r"$\text{Bi} = \frac{hL}{k} \quad $: Biot number")
        st.write(r"$\text{Fo} \cdot t = \frac{\alpha t}{L^2} \quad $: Fourier number")
        st.write("$L$: characteristic length ($radius/3$ for a sphere)")

    with st.expander("1. Lumped Parameters Formulation (0D)"):
        st.write("### 1. Lumped Parameters Formulation (0D)")
        st.write("""
           The lumped parameter method assumes uniform temperature throughout the sphere. The energy balance equation is:
           """)
        st.latex(r"\rho V c \frac{dT}{dt} = hA(T_\infty - T)")
        st.write("Where:")
        st.write(r"$ρ$: density of the sphere material")
        st.write(r"$V$: volume of the sphere")
        st.write(r"$c$: specific heat capacity")
        st.write(r"$h$: heat transfer coefficient")
        st.write(r"$A$: surface area of the sphere")
        st.write(r"$T_∞$: ambient temperature")
        st.write(r"$T$: sphere temperature")

        st.write("This equation can be non-dimensionalized to:")
        st.latex(r"\frac{d\theta_{lumped}}{dt} = -Bi \cdot Fo \cdot \theta_{lumped}")

    with st.expander("2. Finite Volume Formulation (1D)"):
        st.write("### 2. Finite Volume Formulation (1D)")
        st.write("""
        The finite volume method divides the sphere into N concentric shells. For each of the N volumes formed by the shells, the energy balance is:
        """)
        st.latex(
            r"\rho V_j c_p \frac{dT_j}{dt} = -k A_{j-1} \frac{T_j - T_{j-1}}{r_j - r_{j-1}} + k A_{j} \frac{T_{j+1} - T_j}{r_{j+1} - r_j}")
        st.write("Where:")
        st.write(f"$r_j = j/N \cdot r_0$")
        st.write(f"$j = 1,...,N$")
        st.write(r"$V_j$: volume of the j-th shell")
        st.write(r"$A_{j-1}, A_{j}$: surface areas at the inner and outer boundaries of the j-th shell")
        st.write(r"$\Delta r_j = r_{j} - r_{j-1}$: thickness of each shell")

        st.write("Non-dimensionalizing this equation leads to:")
        st.latex(r"\frac{d\theta_j}{dt} = Fo \cdot (q_{l,j} + q_{r,j})")
        st.write("Where:")
        st.latex(r"q_{l,j} = -\frac{x_{j-1}^2}{x_{j}^3 - x_{j-1}^3} \frac{\theta_j - \theta_{j-1}}{x_j - x_{j-1}}")
        st.latex(r"q_{r,j} = \frac{x_{j}^2}{x_{j}^3 - x_{j-1}^3} \frac{\theta_{j+1} - \theta_j}{x_{j+1} - x_j}")
        st.write(r"$x_j =r_j/r_0$ : dimensionless radial coordinate of the j-th node ")

        st.write("#### Boundary Conditions")
        st.write("1. At the center (r = 0): Symmetry condition (zero heat flux)")
        st.latex(r"\left.\frac{\partial T}{\partial r}\right|_{r=0} = 0")
        st.write("2. At the surface (r = r₀): Convective heat transfer")
        st.latex(r"-k\left.\frac{\partial T}{\partial r}\right|_{r=r_0} = h(T_{r_0} - T_\infty)")
        st.write(r"Where $T_{r_0}$ is the surface temperature.")

        st.write("In dimensionless form for the finite volume method:")
        st.latex(r"q_{l,1} = 0")
        st.latex(r"q_{r,N} = - \frac{x_{N}^2}{x_{N}^3 - x_{N-1}^3} 3 \cdot Bi \cdot \theta_N")

        st.write("#### Initial Conditions")
        st.latex(r"T(r, t=0) = T_{init}")
        st.write("In dimensionless form for the finite volume method:")
        st.latex(r"\theta(x_1, ..., x_N, t=0) = 1")

    # Simulation Analysis
    st.header("Simulation Analysis")
    st.subheader("Biot Number")
    with st.expander("What is really the Biot Number?"):
        st.markdown("""
    ### What is really the Biot Number?

    The **Biot number** (Bi) is a dimensionless quantity. 
    It represents the ratio of the thermal resistance inside a body to the thermal resistance at the surface (due to convection). 
    Mathematically, it is defined as:

    """)

        st.latex(r"Bi = \frac{hL}{k}")

        st.markdown("""
    Where:
    - \( h \) is the heat transfer coefficient (W/m²·K),
    - \( L \) is the characteristic length (m), often (and as in this case) the ratio of the body's volume to surface area,
    - \( k \) is the thermal conductivity of the material (W/m·K).

    ### Physical Meaning of the Biot Number

    The Biot number tells us how heat flows within an object relative to how heat is exchanged with the environment. Here's the interpretation:
    - **Small Biot Number (Bi < 0.1)**: The heat conduction inside the object is very fast compared to the heat transfer at the surface. This means the temperature inside the object is nearly uniform, and a **0D model** (lumped-capacity model) is typically sufficient to describe the system. In this case, we can assume that the entire body has a single, uniform temperature at any point in time.

    - **Large Biot Number (Bi > 0.1)**: The heat transfer within the object is slower, causing a temperature gradient inside the object. In this case, a **1D model** is needed to capture the spatial variations in temperature. The temperature distribution needs to be calculated as it varies across the object's geometry.

    ### When to Use 0D or 1D Simulation?

    - **0D Simulation (Lumped Model)**: Use this when **Bi < 0.1**, as it assumes uniform temperature throughout the object.
    - **1D Simulation**: Use this when **Bi > 0.1**, because temperature gradients within the object become significant, requiring a model that accounts for spatial variation along one dimension.

    In summary, the Biot number provides critical guidance in choosing between a simplified **0D model** for uniform temperature systems or a more detailed **1D model** for systems where temperature changes spatially.

    Let's calculate Biot for the parameters in the sidebar
    """)
