"""
Constant-parameter dual-porosity model for variably saturated water flow
and conservative solute transport in fractured chalk.

Repository usage
----------------
1. Set ``selected_case`` to "NS", "60", or "40".
2. Run this script from any location.
3. Numerical outputs are written automatically to the repository ``outputs/``
   directory.

This publication version contains only the constant-parameter formulation.
Plotting and observed-data comparison are intentionally excluded because
they are post-processing steps and are not required to run the numerical model.

The governing equations, boundary conditions, numerical discretization,
case-specific parameters, and solver settings are retained from the working
constant-parameter implementation used in the study.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

# Select one experimental initial-saturation condition:
#   "NS" : near-saturated
#   "60" : 60% initial saturation condition
#   "40" : 40% initial saturation condition
selected_case = "60"

# The script is stored in <repository>/model/. Outputs are therefore written
# to <repository>/outputs/ without any computer-specific absolute paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. HYDRAULIC PARAMETERS
# =============================================================================

# Mobile domain (fracture)
WCR = 0.0       # Residual mobile water content [-]
WCS = 0.08      # Saturated mobile water content [-]
alpha = 0.05    # van Genuchten alpha [1/cm]
n = 1.25        # van Genuchten n [-]
Ks = 0.08       # Saturated hydraulic conductivity [cm/min]
l_mualem = 0.5  # Mualem pore-connectivity parameter [-]

# Immobile domain (chalk matrix)
WCIR = 0.03     # Residual immobile water content [-]
WCIM = 0.4000   # Saturated immobile water content [-]


# =============================================================================
# 3. CASE-SPECIFIC CONSTANT PARAMETERS
# =============================================================================

# Each case uses a constant mobile-immobile water-transfer coefficient (omega)
# and a constant mobile-domain dispersivity. Initial water contents and
# time-step controls are also case specific.
CASES = {
    "NS": {
        "label": "NS",
        "theta_total_upper": 0.468,
        "theta_total_bottom": 0.225,
        "omega": 9.0e-5,
        "dispersivity": 1.0,
        "dt_initial": 0.004,
        "dt_min": 0.0004,
        "dt_max": 0.5,
    },
    "60": {
        "label": "60%",
        "theta_total_upper": 0.265,
        "theta_total_bottom": 0.225,
        "omega": 1.8e-3,
        "dispersivity": 9.0,
        "dt_initial": 0.04,
        "dt_min": 0.004,
        "dt_max": 0.5,
    },
    "40": {
        "label": "40%",
        "theta_total_upper": 0.174,
        "theta_total_bottom": 0.225,
        "omega": 7.0e-4,
        "dispersivity": 2.0,
        "dt_initial": 0.4,
        "dt_min": 0.04,
        "dt_max": 0.5,
    },
}

if selected_case not in CASES:
    raise ValueError(
        f"selected_case must be one of {tuple(CASES)}, not {selected_case!r}."
    )

case = CASES[selected_case]
omega = case["omega"]
dispersivity = case["dispersivity"]


# =============================================================================
# 4. SPATIAL AND TEMPORAL DISCRETIZATION
# =============================================================================

L = 9.0                  # Column length [cm]
dx = 0.25                # Node spacing [cm]
nodes = int(round(L / dx)) + 1
depth = np.linspace(0.0, L, nodes)

t_end = 390.0            # Total simulation time [min]
dt_initial = case["dt_initial"]
dt_min = case["dt_min"]
dt_max = case["dt_max"]
dt_output = 1.0          # Regular output interval [min]

# Water-content diagnostics are saved at this depth.
diagnostic_depth = 4.0   # cm
diagnostic_node = int(round(diagnostic_depth / dx))

if diagnostic_node < 0 or diagnostic_node >= nodes:
    raise ValueError(
        f"Diagnostic depth {diagnostic_depth} cm is outside the model domain."
    )

actual_diagnostic_depth = depth[diagnostic_node]


# =============================================================================
# 5. INITIAL AND BOUNDARY CONDITIONS
# =============================================================================

# Initial total water contents used in the working model.
theta_total_upper = case["theta_total_upper"]
theta_total_bottom = case["theta_total_bottom"]

# Constant upper water flux. Positive q is downward.
q_top = 0.006  # [cm/min]


# =============================================================================
# 6. SOLUTE-TRANSPORT SETTINGS
# =============================================================================

C0 = 0.0005                 # Injected tracer concentration
tracer_duration = 90.0      # Pulse duration [min]
initial_concentration = 0.0

# Mechanical dispersion only; molecular diffusion is neglected.
SOLUTE_TIME_SCHEME = "crank_nicolson"
SOLUTE_SPACE_SCHEME = "galerkin_finite_elements"
GALERKIN_STORAGE_MATRIX = "lumped"

NEGATIVE_CONCENTRATION_ABSOLUTE_TOLERANCE = 1.0e-12
NEGATIVE_CONCENTRATION_RELATIVE_TOLERANCE = 1.0e-6

if SOLUTE_TIME_SCHEME == "crank_nicolson":
    SOLUTE_TIME_WEIGHT = 0.5
elif SOLUTE_TIME_SCHEME == "implicit":
    SOLUTE_TIME_WEIGHT = 1.0
else:
    raise ValueError(
        "SOLUTE_TIME_SCHEME must be 'crank_nicolson' or 'implicit'."
    )

if SOLUTE_SPACE_SCHEME != "galerkin_finite_elements":
    raise ValueError(
        "This script implements SOLUTE_SPACE_SCHEME='galerkin_finite_elements'."
    )

if GALERKIN_STORAGE_MATRIX not in ("lumped", "consistent"):
    raise ValueError(
        "GALERKIN_STORAGE_MATRIX must be 'lumped' or 'consistent'."
    )

# =============================================================================
# 7. VAN GENUCHTEN AND HYDRAULIC FUNCTIONS
# =============================================================================

def vg_m(n_value):
    """Calculate the van Genuchten m parameter."""
    return 1.0 - 1.0 / n_value


def effective_saturation_from_h(h, alpha_value, n_value):
    """
    Calculate effective saturation from pressure head.

    Saturated conditions are imposed for h >= 0.
    """
    h = np.asarray(h, dtype=float)

    m_value = vg_m(n_value)

    Se = np.ones_like(h)

    unsaturated = h < 0.0

    Se[unsaturated] = (
        1.0
        + (
            alpha_value
            * np.abs(h[unsaturated])
        ) ** n_value
    ) ** (-m_value)

    return np.clip(Se, 0.0, 1.0)


def theta_from_h(
    h,
    theta_r,
    theta_s,
    alpha_value,
    n_value
):
    """Calculate volumetric water content from pressure head."""
    Se = effective_saturation_from_h(
        h,
        alpha_value,
        n_value
    )

    return theta_r + (
        theta_s - theta_r
    ) * Se


def effective_saturation_from_theta(
    theta,
    theta_r,
    theta_s
):
    """Calculate effective saturation directly from water content."""
    theta = np.asarray(theta, dtype=float)

    Se = (
        theta - theta_r
    ) / (
        theta_s - theta_r
    )

    return np.clip(Se, 0.0, 1.0)



def mobile_h_from_theta(theta):
    """Invert the mobile van Genuchten retention relation."""
    theta = np.asarray(theta, dtype=float)
    m_value = vg_m(n)
    Se = effective_saturation_from_theta(theta, WCR, WCS)
    Se = np.clip(Se, 1.0e-10, 1.0 - 1.0e-10)
    h_absolute = ((Se ** (-1.0 / m_value) - 1.0) ** (1.0 / n)) / alpha
    return -h_absolute


def hydraulic_conductivity(h):
    """
    Calculate mobile-domain hydraulic conductivity using the
    van Genuchten-Mualem relationship.
    """
    m_value = vg_m(n)

    Se = effective_saturation_from_h(
        h,
        alpha,
        n
    )

    conductivity_term = (
        1.0
        - (
            1.0
            - Se ** (1.0 / m_value)
        ) ** m_value
    ) ** 2

    K = (
        Ks
        * Se ** l_mualem
        * conductivity_term
    )

    return np.maximum(K, 1.0e-20)


def K_face_value(K_up, K_down):
    """
    Arithmetic hydraulic-conductivity averaging at a node interface.
    """
    return 0.5 * (
        np.asarray(K_up, dtype=float)
        + np.asarray(K_down, dtype=float)
    )


# =============================================================================
# 8. MOBILE-IMMOBILE WATER EXCHANGE
# =============================================================================

def gamma_w_exchange(h_mo, theta_im):
    """Calculate signed water exchange from effective-saturation difference.

    Mobile water content is obtained from the mobile pressure head using the
    mobile van Genuchten relationship. Effective saturation in both domains is
    then calculated directly from water content:

        Se_mo = (theta_mo - WCR) / (WCS - WCR)
        Se_im = (theta_im - WCIR) / (WCIM - WCIR)

    Exchange is:

        Gamma_w = omega * (Se_mo - Se_im)

    Sign convention
    ---------------
    Gamma_w > 0 : mobile-to-immobile water transfer.
    Gamma_w < 0 : immobile-to-mobile water transfer.

    Since the driving difference is dimensionless, omega has units of 1/min.
    """
    theta_mo = theta_from_h(h_mo, WCR, WCS, alpha, n)
    Se_mo = effective_saturation_from_theta(theta_mo, WCR, WCS)
    Se_im = effective_saturation_from_theta(theta_im, WCIR, WCIM)
    Gamma_w = omega * (Se_mo - Se_im)
    return Gamma_w, Se_mo, Se_im


# =============================================================================
# 9. INITIAL MOBILE/IMMOBILE WATER-CONTENT SPLIT
# =============================================================================

def hydrus_split_total_theta(theta_total):
    """Split total water content using the HYDRUS capacity ratio.

    The split is

        theta_mo = theta_total * WCS / (WCS + WCIM)
        theta_im = theta_total * WCIM / (WCS + WCIM)

    A small numerical tolerance is used because values located exactly at a
    physical limit, such as theta_im = WCIR = 0.1, can be represented
    internally as 0.09999999999999999.
    """
    theta_mo = theta_total * WCS / (WCS + WCIM)
    theta_im = theta_total * WCIM / (WCS + WCIM)

    tolerance = 1.0e-12

    if theta_mo < WCR - tolerance or theta_mo > WCS + tolerance:
        raise ValueError(
            f"HYDRUS split gives invalid mobile theta={theta_mo:.12f}. "
            f"Allowed range: {WCR:.12f} to {WCS:.12f}."
        )

    if theta_im < WCIR - tolerance or theta_im > WCIM + tolerance:
        raise ValueError(
            f"HYDRUS split gives invalid immobile theta={theta_im:.12f}. "
            f"Allowed range: {WCIR:.12f} to {WCIM:.12f}."
        )

    # Correct only roundoff-level deviations from the physical limits.
    theta_mo = np.clip(theta_mo, WCR, WCS)
    theta_im = np.clip(theta_im, WCIR, WCIM)

    return float(theta_mo), float(theta_im)


# Upper nodes: exact total theta from PROFILE.DAT, split as in HYDRUS.
theta_mo_upper_initial, theta_im_upper_initial = hydrus_split_total_theta(theta_total_upper)
h_upper_initial = float(
    mobile_h_from_theta(theta_mo_upper_initial)
)

# Bottom node: exact total theta from PROFILE.DAT, split as in HYDRUS.
theta_mo_bottom, theta_im_bottom = hydrus_split_total_theta(theta_total_bottom)
h_bottom = float(
    mobile_h_from_theta(theta_mo_bottom)
)

h_mo = np.full(nodes, h_upper_initial, dtype=float)
h_mo[-1] = h_bottom

theta_im = np.full(nodes, theta_im_upper_initial, dtype=float)
theta_im[-1] = theta_im_bottom

theta_mo_initial = theta_from_h(h_mo, WCR, WCS, alpha, n)
theta_im_initial = theta_im.copy()
theta_total_initial = theta_mo_initial + theta_im_initial

# Initial solute concentrations in both domains.
c_mo = np.full(nodes, initial_concentration, dtype=float)
c_im = np.full(nodes, initial_concentration, dtype=float)

# A first-type upper boundary is active from t=0. The boundary node must
# therefore begin at the prescribed concentration, especially because the
# Crank-Nicolson scheme evaluates both the old and new transport operators.
c_mo[0] = C0

# =============================================================================
# 10. DARCY FLUX FUNCTION
# =============================================================================

def downward_flux(h_up, h_down, K_up, K_down):
    """
    Calculate Darcy flux across a node interface.

    The depth coordinate increases downward, and positive q represents
    downward flow.

        q = K_face * (1 - dh/dz)
    """

    K_face = K_face_value(
        K_up,
        K_down
    )

    dh_dz = (
        h_down - h_up
    ) / dx

    return K_face * (
        1.0 - dh_dz
    )


# =============================================================================
# 11. IMPLICIT NONLINEAR WATER SYSTEM
# =============================================================================

def residual_system(
    unknown_vector,
    h_mo_old,
    theta_im_old,
    current_dt
):
    """
    Residual equations for one fully implicit time step.

    Unknowns
    --------
    First 'nodes' values:
        mobile pressure head h_mo

    Last 'nodes' values:
        immobile water content theta_im
    """

    h_mo_new = unknown_vector[:nodes]
    theta_im_new = unknown_vector[nodes:]

    residual = np.zeros(
        2 * nodes,
        dtype=float
    )

    # -------------------------------------------------------------------------
    # Water contents
    # -------------------------------------------------------------------------

    theta_mo_old = theta_from_h(
        h_mo_old,
        WCR,
        WCS,
        alpha,
        n
    )

    theta_mo_new = theta_from_h(
        h_mo_new,
        WCR,
        WCS,
        alpha,
        n
    )

    # -------------------------------------------------------------------------
    # Mobile hydraulic conductivity
    # -------------------------------------------------------------------------

    K_new = hydraulic_conductivity(
        h_mo_new
    )

    # -------------------------------------------------------------------------
    # Mobile-immobile exchange
    # -------------------------------------------------------------------------

    Gamma_w, _, _ = gamma_w_exchange(
        h_mo_new,
        theta_im_new
    )

    # -------------------------------------------------------------------------
    # Fluxes between mobile nodes
    # -------------------------------------------------------------------------

    q_faces = np.zeros(
        nodes - 1,
        dtype=float
    )

    for face in range(nodes - 1):
        q_faces[face] = downward_flux(
            h_up=h_mo_new[face],
            h_down=h_mo_new[face + 1],
            K_up=K_new[face],
            K_down=K_new[face + 1]
        )

    # -------------------------------------------------------------------------
    # Mobile equation at the upper node
    #
    # d(theta_mo)/dt + (q_out - q_in)/dx + Gamma_w = 0
    #
    # At the upper boundary:
    # q_in = q_top
    # -------------------------------------------------------------------------

    residual[0] = (
        (
            theta_mo_new[0]
            - theta_mo_old[0]
        ) / current_dt
        + (
            q_faces[0]
            - q_top
        ) / dx
        + Gamma_w[0]
    )

    # -------------------------------------------------------------------------
    # Mobile equations at internal nodes
    # -------------------------------------------------------------------------

    for i in range(1, nodes - 1):

        q_in = q_faces[i - 1]
        q_out = q_faces[i]

        residual[i] = (
            (
                theta_mo_new[i]
                - theta_mo_old[i]
            ) / current_dt
            + (
                q_out - q_in
            ) / dx
            + Gamma_w[i]
        )

    # -------------------------------------------------------------------------
    # Fixed mobile pressure head at the bottom
    # -------------------------------------------------------------------------

    residual[nodes - 1] = (
        h_mo_new[-1]
        - h_bottom
    )

    # -------------------------------------------------------------------------
    # Immobile-domain storage equation at every node
    #
    # d(theta_im)/dt - Gamma_w = 0
    #
    # No vertical immobile flow and no independent immobile boundary head.
    # -------------------------------------------------------------------------

    residual[nodes:] = (
        (
            theta_im_new
            - theta_im_old
        ) / current_dt
        - Gamma_w
    )

    return residual


# =============================================================================
# 12. SOLUTE-TRANSPORT SYSTEM
# =============================================================================

def inlet_concentration(time_value):
    """Pulse input: C0 through 90 min, followed by solute-free water."""
    return C0 if time_value <= tracer_duration + 1.0e-12 else 0.0


def assemble_weighted_consistent_mass_matrix(coefficient_nodes):
    r"""Assemble \int N_i * coefficient * N_j dz for linear elements.

    The coefficient is represented by the same linear nodal basis as the
    concentration. For one element with endpoint values a and b, exact
    integration gives

        dx/12 * [[3a+b, a+b],
                 [a+b, a+3b]].

    For a constant coefficient this reduces to the standard Galerkin
    consistent mass matrix coefficient*dx/6*[[2,1],[1,2]].
    """
    coefficient_nodes = np.asarray(coefficient_nodes, dtype=float)
    if coefficient_nodes.shape != (nodes,):
        raise ValueError("coefficient_nodes must contain one value per node.")

    matrix = np.zeros((nodes, nodes), dtype=float)

    for element in range(nodes - 1):
        left = element
        right = element + 1
        a = coefficient_nodes[left]
        b_value = coefficient_nodes[right]

        local = (dx / 12.0) * np.array([
            [3.0 * a + b_value, a + b_value],
            [a + b_value, a + 3.0 * b_value],
        ])

        indices = (left, right)
        for local_i, global_i in enumerate(indices):
            for local_j, global_j in enumerate(indices):
                matrix[global_i, global_j] += local[local_i, local_j]

    return matrix


def assemble_galerkin_advection_dispersion_operator(
    theta_mo_nodes,
    q_faces,
):
    """Assemble the standard Galerkin advection-dispersion operator.

    The conservative mobile equation is

        d(theta*C)/dt + d(q*C)/dz - d(theta*D*dC/dz)/dz + Gamma_s = 0.

    Linear Galerkin elements give, for each element,

        K_adv = q/2 * [[ 1,  1],
                       [-1, -1]]

        K_disp = theta*D/dx * [[ 1, -1],
                               [-1,  1]].

    Mechanical dispersion only is used:

        D = dispersivity * abs(q/theta).

    Consequently theta*D = dispersivity*abs(q), apart from the numerical
    protection used when theta is extremely small.
    """
    theta_mo_nodes = np.asarray(theta_mo_nodes, dtype=float)
    q_faces = np.asarray(q_faces, dtype=float)

    if theta_mo_nodes.shape != (nodes,):
        raise ValueError("theta_mo_nodes must contain one value per node.")
    if q_faces.shape != (nodes - 1,):
        raise ValueError("q_faces must contain one value per element face.")

    operator = np.zeros((nodes, nodes), dtype=float)

    for element in range(nodes - 1):
        left = element
        right = element + 1
        q_element = q_faces[element]

        theta_element = max(
            0.5 * (theta_mo_nodes[left] + theta_mo_nodes[right]),
            1.0e-15,
        )
        pore_velocity = q_element / theta_element
        D_element = dispersivity * abs(pore_velocity)
        theta_D = theta_element * D_element

        local_advection = 0.5 * q_element * np.array([
            [1.0, 1.0],
            [-1.0, -1.0],
        ])

        local_dispersion = (theta_D / dx) * np.array([
            [1.0, -1.0],
            [-1.0, 1.0],
        ])

        local = local_advection + local_dispersion
        indices = (left, right)
        for local_i, global_i in enumerate(indices):
            for local_j, global_j in enumerate(indices):
                operator[global_i, global_j] += local[local_i, local_j]

    # Natural lower boundary: zero dispersive gradient and advective outflow.
    # In the weak form, the remaining lower-boundary total flux is q*C.
    operator[-1, -1] += q_faces[-1]

    return operator


def assemble_coupled_transport_matrices(
    theta_mo,
    theta_im,
    q_faces,
    Gamma_w,
):
    """Return conservative storage and transport matrices for one time level.

    Water-driven solute exchange uses the donor concentration:

      Gamma_w >= 0: Gamma_s = Gamma_w*C_mo
      Gamma_w <  0: Gamma_s = Gamma_w*C_im

    Positive and negative parts are assembled separately. The mobile and
    immobile exchange blocks are exact opposites, so exchange cannot create or
    destroy total solute mass internally.
    """
    theta_mo = np.asarray(theta_mo, dtype=float)
    theta_im = np.asarray(theta_im, dtype=float)
    Gamma_w = np.asarray(Gamma_w, dtype=float)

    number_unknowns = 2 * nodes

    mobile_mass_consistent = assemble_weighted_consistent_mass_matrix(theta_mo)
    immobile_mass_consistent = assemble_weighted_consistent_mass_matrix(theta_im)

    if GALERKIN_STORAGE_MATRIX == "lumped":
        mobile_mass = np.diag(np.sum(mobile_mass_consistent, axis=1))
        immobile_mass = np.diag(np.sum(immobile_mass_consistent, axis=1))
    else:
        mobile_mass = mobile_mass_consistent
        immobile_mass = immobile_mass_consistent

    storage = np.zeros((number_unknowns, number_unknowns), dtype=float)
    storage[:nodes, :nodes] = mobile_mass
    storage[nodes:, nodes:] = immobile_mass

    transport = np.zeros((number_unknowns, number_unknowns), dtype=float)
    transport[:nodes, :nodes] += (
        assemble_galerkin_advection_dispersion_operator(theta_mo, q_faces)
    )

    gamma_positive = np.maximum(Gamma_w, 0.0)
    gamma_negative = np.minimum(Gamma_w, 0.0)

    G_positive = assemble_weighted_consistent_mass_matrix(gamma_positive)
    G_negative = assemble_weighted_consistent_mass_matrix(gamma_negative)

    # Mobile equation: +Gamma_s.
    transport[:nodes, :nodes] += G_positive
    transport[:nodes, nodes:] += G_negative

    # Immobile equation: -Gamma_s.
    transport[nodes:, :nodes] -= G_positive
    transport[nodes:, nodes:] -= G_negative

    return storage, transport


def solve_solute_linear_system(
    c_mo_old,
    c_im_old,
    theta_mo_old,
    theta_im_old,
    theta_mo_new,
    theta_im_new,
    q_faces_old,
    q_faces_new,
    Gamma_w_old,
    Gamma_w_new,
    current_dt,
    time_old,
    time_new,
):
    """Solve solute transport with Crank-Nicolson and Galerkin FE.

    For temporal weight beta, the conservative update is

      [M_new + dt*beta*K_new] C_new
        = [M_old - dt*(1-beta)*K_old] C_old.

    beta=0.5 gives Crank-Nicolson and beta=1 gives fully implicit time
    weighting. M contains the selected Galerkin storage matrices (lumped by
    default for stability); K contains Galerkin advection-dispersion and
    conservative interdomain solute exchange.
    """
    old_concentrations = np.concatenate([
        np.asarray(c_mo_old, dtype=float),
        np.asarray(c_im_old, dtype=float),
    ])

    # Ensure the old boundary value is mathematically consistent with the
    # prescribed first-type boundary before evaluating the old operator.
    old_concentrations[0] = inlet_concentration(time_old)

    storage_old, transport_old = assemble_coupled_transport_matrices(
        theta_mo=theta_mo_old,
        theta_im=theta_im_old,
        q_faces=q_faces_old,
        Gamma_w=Gamma_w_old,
    )

    storage_new, transport_new = assemble_coupled_transport_matrices(
        theta_mo=theta_mo_new,
        theta_im=theta_im_new,
        q_faces=q_faces_new,
        Gamma_w=Gamma_w_new,
    )

    beta = SOLUTE_TIME_WEIGHT

    A = storage_new + current_dt * beta * transport_new
    b = (
        storage_old - current_dt * (1.0 - beta) * transport_old
    ) @ old_concentrations

    # First-type upper concentration boundary. Row replacement is performed
    # after complete assembly. Interior rows retain their coupling to node 0,
    # so both advective and dispersive influx respond to the prescribed value.
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = inlet_concentration(time_new)

    concentrations = np.linalg.solve(A, b)

    if not np.all(np.isfinite(concentrations)):
        raise RuntimeError("The Galerkin solute solution contains non-finite values.")

    minimum_concentration = float(np.min(concentrations))

    # Do not reject a valid solution because of machine-scale Galerkin/CN
    # undershoots. For C0=5e-4, the default relative tolerance is 5e-10,
    # whereas the rejected NS values were only about -5e-12.
    negative_tolerance = max(
        NEGATIVE_CONCENTRATION_ABSOLUTE_TOLERANCE,
        NEGATIVE_CONCENTRATION_RELATIVE_TOLERANCE * abs(C0),
    )

    if minimum_concentration < -negative_tolerance:
        raise RuntimeError(
            "The Crank-Nicolson/Galerkin solution produced a materially "
            f"negative concentration: {minimum_concentration:.6e}; "
            f"allowed numerical tolerance: {-negative_tolerance:.6e}."
        )

    # Clip only accepted numerical undershoots to exactly zero.
    concentrations = np.maximum(concentrations, 0.0)
    return concentrations[:nodes], concentrations[nodes:]


# =============================================================================
# 13. NONLINEAR SOLVER BOUNDS
# =============================================================================

# Mobile pressure-head bounds
h_lower = np.full(
    nodes,
    -1.0e7,
    dtype=float
)

h_upper = np.full(
    nodes,
    100.0,
    dtype=float
)

# Immobile water-content bounds.
# The tiny extension permits initial values located exactly at WCIR or WCIM
# while effective saturation remains clipped to the physical interval [0, 1].
numerical_theta_tolerance = 1.0e-12

theta_im_lower = np.full(
    nodes,
    WCIR - numerical_theta_tolerance,
    dtype=float
)

theta_im_upper = np.full(
    nodes,
    WCIM + numerical_theta_tolerance,
    dtype=float
)

lower_bounds = np.concatenate([
    h_lower,
    theta_im_lower
])

upper_bounds = np.concatenate([
    h_upper,
    theta_im_upper
])


# =============================================================================
# 14. INITIAL BOTTOM FLUX
# =============================================================================

K_initial = hydraulic_conductivity(
    h_mo
)

q_bottom_initial = downward_flux(
    h_up=h_mo[-2],
    h_down=h_mo[-1],
    K_up=K_initial[-2],
    K_down=K_initial[-1]
)


# =============================================================================
# 15. SIMULTANEOUS WATER-SOLUTE TIME INTEGRATION
# =============================================================================

time_current = 0.0
dt = dt_initial
cumulative_bottom_flux = 0.0

# Initial state at the diagnostic node.
tracked_time = [0.0]
tracked_bottom_flux = [q_bottom_initial]
tracked_cumulative_flux = [0.0]
tracked_bottom_btc = [0.0]
tracked_bottom_concentration = [0.0]
tracked_average_Gamma = []
tracked_theta_mo_4cm = [float(theta_mo_initial[diagnostic_node])]
tracked_theta_im_4cm = [float(theta_im_initial[diagnostic_node])]
tracked_theta_total_4cm = [
    float(theta_mo_initial[diagnostic_node] + theta_im_initial[diagnostic_node])
]

# Bottom-node water-content diagnostics.
# The mobile bottom water content should remain constant because the mobile
# pressure head is fixed at h_bottom. The immobile bottom water content remains
# free to evolve through Gamma_w.
tracked_theta_mo_bottom = [float(theta_mo_initial[-1])]
tracked_theta_im_bottom = [float(theta_im_initial[-1])]
tracked_theta_total_bottom = [
    float(theta_mo_initial[-1] + theta_im_initial[-1])
]

# Speed and reporting controls.
PRINT_EVERY_ACCEPTED_STEPS = 50
DT_GROWTH_FACTOR = 1.5

accepted_steps = 0
rejected_water_steps = 0
rejected_solute_steps = 0
wall_start = time.perf_counter()

print(f"Starting coupled water-solute simulation for {case['label']}...", flush=True)
print(
    f"Grid nodes={nodes}, t_end={t_end} min, "
    f"dt_initial={dt_initial:g} min, dt_max={dt_max:g} min",
    flush=True,
)
print(
    f"Initial upper state: theta_total={theta_total_upper:.6f}, "
    f"theta_mo_upper={theta_mo_upper_initial:.6f}, "
    f"theta_im_upper={theta_im_upper_initial:.6f}, "
    f"h_upper={h_upper_initial:.3f} cm",
    flush=True,
)
print(
    f"Initial bottom state: theta_total={theta_total_bottom:.6f}, "
    f"theta_mo_bottom={theta_mo_bottom:.6f}, "
    f"theta_im_bottom={theta_im_bottom:.6f}, "
    f"h_bottom={h_bottom:.3f} cm",
    flush=True,
)
print(
    f"Constant-omega exchange: Gamma_w=omega*(Se_mo-Se_im), "
    f"omega={omega:.3e} 1/min",
    flush=True,
)
print("Solute transport is solved by a direct linear solve.", flush=True)
print(
    f"Solute time scheme: {SOLUTE_TIME_SCHEME}; "
    f"space scheme: {SOLUTE_SPACE_SCHEME}; "
    f"storage matrix: {GALERKIN_STORAGE_MATRIX}.",
    flush=True,
)
print(
    "Accepted negative-concentration numerical tolerance: "
    f"{max(NEGATIVE_CONCENTRATION_ABSOLUTE_TOLERANCE, NEGATIVE_CONCENTRATION_RELATIVE_TOLERANCE * abs(C0)):.3e}.",
    flush=True,
)
print(
    "Upper solute boundary: prescribed concentration "
    "C(0,t)=C_in(t) (first type).",
    flush=True,
)

while time_current < t_end - 1.0e-12:
    current_dt = min(dt, t_end - time_current)

    # A Crank-Nicolson step must not cross the abrupt end of the concentration
    # pulse. Force an accepted time level exactly at tracer_duration.
    if (
        time_current < tracer_duration - 1.0e-12
        and time_current + current_dt > tracer_duration
    ):
        current_dt = tracer_duration - time_current

    # Save the complete old state. The time step is accepted only after both
    # the water and solute solutions are available for this same time step.
    h_mo_old_step = h_mo.copy()
    theta_im_old_step = theta_im.copy()
    theta_mo_old_step = theta_from_h(
        h_mo_old_step,
        WCR,
        WCS,
        alpha,
        n,
    )
    c_mo_old_step = c_mo.copy()
    c_im_old_step = c_im.copy()

    # Old-time hydraulic quantities are needed by Crank-Nicolson.
    K_old_step = hydraulic_conductivity(h_mo_old_step)
    q_faces_old_step = np.array([
        downward_flux(
            h_mo_old_step[i],
            h_mo_old_step[i + 1],
            K_old_step[i],
            K_old_step[i + 1],
        )
        for i in range(nodes - 1)
    ])
    Gamma_old_step, _, _ = gamma_w_exchange(
        h_mo_old_step,
        theta_im_old_step,
    )

    water_guess = np.concatenate([h_mo_old_step, theta_im_old_step])

    water_solution = least_squares(
        residual_system,
        water_guess,
        bounds=(lower_bounds, upper_bounds),
        args=(h_mo_old_step, theta_im_old_step, current_dt),
        xtol=1.0e-8,
        ftol=1.0e-8,
        gtol=1.0e-8,
        max_nfev=200,
    )

    water_maximum_residual = np.max(np.abs(water_solution.fun))
    water_successful = (
        water_solution.success
        and np.isfinite(water_maximum_residual)
        and water_maximum_residual < 1.0e-7
    )

    if not water_successful:
        rejected_water_steps += 1
        dt = current_dt / 2.0
        print(
            f"Water step rejected at t={time_current:.6f} min; "
            f"residual={water_maximum_residual:.3e}; new dt={dt:.3e} min.",
            flush=True,
        )
        if dt < dt_min:
            raise RuntimeError(
                f"Water solver failed at t={time_current:.8f} min; "
                f"maximum residual={water_maximum_residual:.4e}."
            )
        continue

    # Candidate water state for the new time level.
    h_mo_candidate = water_solution.x[:nodes]
    theta_im_candidate = water_solution.x[nodes:]
    theta_mo_candidate = theta_from_h(
        h_mo_candidate, WCR, WCS, alpha, n
    )
    K_candidate = hydraulic_conductivity(h_mo_candidate)
    q_faces_candidate = np.array([
        downward_flux(
            h_mo_candidate[i],
            h_mo_candidate[i + 1],
            K_candidate[i],
            K_candidate[i + 1],
        )
        for i in range(nodes - 1)
    ])
    Gamma_candidate, Se_mo_candidate, Se_im_candidate = gamma_w_exchange(
        h_mo_candidate,
        theta_im_candidate
    )

    # Solve transport using the water state from this exact time step.
    try:
        c_mo_candidate, c_im_candidate = solve_solute_linear_system(
            c_mo_old=c_mo_old_step,
            c_im_old=c_im_old_step,
            theta_mo_old=theta_mo_old_step,
            theta_im_old=theta_im_old_step,
            theta_mo_new=theta_mo_candidate,
            theta_im_new=theta_im_candidate,
            q_faces_old=q_faces_old_step,
            q_faces_new=q_faces_candidate,
            Gamma_w_old=Gamma_old_step,
            Gamma_w_new=Gamma_candidate,
            current_dt=current_dt,
            time_old=time_current,
            time_new=time_current + current_dt,
        )
    except (np.linalg.LinAlgError, RuntimeError) as error:
        rejected_solute_steps += 1
        dt = current_dt / 2.0
        print(
            f"Solute step rejected at t={time_current:.6f} min; "
            f"reason={error}; new dt={dt:.3e} min.",
            flush=True,
        )
        if dt < dt_min:
            raise RuntimeError(
                f"Solute solution failed at t={time_current:.8f} min."
            ) from error
        continue

    # Both systems succeeded: accept the complete coupled state.
    q_bottom_old = q_faces_old_step[-1]

    h_mo = h_mo_candidate
    theta_im = theta_im_candidate
    c_mo = c_mo_candidate
    c_im = c_im_candidate
    time_current += current_dt
    accepted_steps += 1

    q_bottom_new = q_faces_candidate[-1]
    cumulative_bottom_flux += (
        0.5 * (q_bottom_old + q_bottom_new) * current_dt
    )

    bottom_concentration = c_mo[-1]
    bottom_relative_concentration = bottom_concentration / C0

    tracked_time.append(time_current)
    tracked_bottom_flux.append(q_bottom_new)
    tracked_cumulative_flux.append(cumulative_bottom_flux)
    tracked_bottom_concentration.append(bottom_concentration)
    tracked_bottom_btc.append(bottom_relative_concentration)
    tracked_average_Gamma.append(np.mean(Gamma_candidate))

    tracked_theta_mo_4cm.append(
        float(theta_mo_candidate[diagnostic_node])
    )
    tracked_theta_im_4cm.append(
        float(theta_im_candidate[diagnostic_node])
    )
    tracked_theta_total_4cm.append(
        float(theta_mo_candidate[diagnostic_node] + theta_im_candidate[diagnostic_node])
    )

    tracked_theta_mo_bottom.append(
        float(theta_mo_candidate[-1])
    )
    tracked_theta_im_bottom.append(
        float(theta_im_candidate[-1])
    )
    tracked_theta_total_bottom.append(
        float(theta_mo_candidate[-1] + theta_im_candidate[-1])
    )

    dt = min(current_dt * DT_GROWTH_FACTOR, dt_max)

    if accepted_steps == 1 or accepted_steps % PRINT_EVERY_ACCEPTED_STEPS == 0:
        elapsed = time.perf_counter() - wall_start
        estimated_total = elapsed * t_end / max(time_current, 1.0e-15)
        estimated_remaining = max(estimated_total - elapsed, 0.0)
        print(
            f"Step {accepted_steps}: t={time_current:.2f}/{t_end:.0f} min, "
            f"dt={current_dt:.3f} min, water_nfev={water_solution.nfev}, "
            f"BTC={bottom_relative_concentration:.4e}, "
            f"remaining~{estimated_remaining/60.0:.1f} min",
            flush=True,
        )

print("\nCoupled simulation finished successfully.", flush=True)
print(
    f"Accepted steps={accepted_steps}, "
    f"water rejections={rejected_water_steps}, "
    f"solute rejections={rejected_solute_steps}, "
    f"wall time={(time.perf_counter()-wall_start)/60.0:.2f} min",
    flush=True,
)


# =============================================================================
# 16. INTERPOLATE TO REGULAR OUTPUT TIMES
# =============================================================================

output_time = np.arange(
    0.0,
    t_end + dt_output,
    dt_output
)

output_time = output_time[
    output_time <= t_end + 1.0e-10
]

bottom_flux_output = np.interp(
    output_time,
    tracked_time,
    tracked_bottom_flux
)

cumulative_flux_output = np.interp(
    output_time,
    tracked_time,
    tracked_cumulative_flux
)

bottom_concentration_output = np.interp(
    output_time,
    tracked_time,
    tracked_bottom_concentration
)

BTC_output = np.interp(
    output_time,
    tracked_time,
    tracked_bottom_btc
)



theta_mo_4cm_output = np.interp(
    output_time,
    tracked_time,
    tracked_theta_mo_4cm
)

theta_im_4cm_output = np.interp(
    output_time,
    tracked_time,
    tracked_theta_im_4cm
)

theta_total_4cm_output = np.interp(
    output_time,
    tracked_time,
    tracked_theta_total_4cm
)

theta_mo_bottom_output = np.interp(
    output_time,
    tracked_time,
    tracked_theta_mo_bottom
)

theta_im_bottom_output = np.interp(
    output_time,
    tracked_time,
    tracked_theta_im_bottom
)

theta_total_bottom_output = np.interp(
    output_time,
    tracked_time,
    tracked_theta_total_bottom
)

# =============================================================================
# 17. FINAL WATER-CONTENT PROFILES
# =============================================================================

theta_mo_final = theta_from_h(
    h_mo,
    WCR,
    WCS,
    alpha,
    n
)

theta_im_final = theta_im.copy()

theta_total_final = (
    theta_mo_final
    + theta_im_final
)

Gamma_final, Se_mo_final, Se_im_final = gamma_w_exchange(
    h_mo,
    theta_im_final
)




# =============================================================================
# 18. WATER MASS BALANCE
# =============================================================================

initial_storage = np.trapezoid(
    theta_total_initial,
    depth,
)

final_storage = np.trapezoid(
    theta_total_final,
    depth,
)

cumulative_top_inflow = q_top * t_end
cumulative_bottom_outflow = cumulative_flux_output[-1]

mass_balance_error = (
    initial_storage
    + cumulative_top_inflow
    - cumulative_bottom_outflow
    - final_storage
)

mass_balance_reference = max(
    abs(initial_storage) + abs(cumulative_top_inflow),
    1.0e-15,
)

relative_mass_balance_error = (
    100.0 * mass_balance_error / mass_balance_reference
)


# =============================================================================
# 19. SAVE NUMERICAL OUTPUTS
# =============================================================================

# Complete regularly sampled bottom-flow and BTC output.
time_series_results = pd.DataFrame({
    "time_min": output_time,
    "bottom_flux_cm_per_min": bottom_flux_output,
    "cumulative_bottom_outflow_cm": cumulative_flux_output,
    "bottom_concentration": bottom_concentration_output,
    "bottom_C_over_C0": BTC_output,
    "theta_mobile_at_diagnostic_depth": theta_mo_4cm_output,
    "theta_immobile_at_diagnostic_depth": theta_im_4cm_output,
    "theta_total_at_diagnostic_depth": theta_total_4cm_output,
    "theta_mobile_bottom": theta_mo_bottom_output,
    "theta_immobile_bottom": theta_im_bottom_output,
    "theta_total_bottom": theta_total_bottom_output,
})

time_series_output_file = (
    OUTPUT_DIRECTORY / f"constant_{selected_case}_time_series.csv"
)
time_series_results.to_csv(time_series_output_file, index=False)


# BTC saved separately for convenient figure reproduction/post-processing.
btc_results = pd.DataFrame({
    "time_min": output_time,
    "bottom_concentration": bottom_concentration_output,
    "C_over_C0": BTC_output,
})

btc_output_file = (
    OUTPUT_DIRECTORY / f"constant_{selected_case}_BTC.csv"
)
btc_results.to_csv(btc_output_file, index=False)


# Bottom flux and cumulative outflow saved separately.
flux_results = pd.DataFrame({
    "time_min": output_time,
    "bottom_flux_cm_per_min": bottom_flux_output,
    "cumulative_bottom_outflow_cm": cumulative_flux_output,
})

flux_output_file = (
    OUTPUT_DIRECTORY / f"constant_{selected_case}_cumulative_flux.csv"
)
flux_results.to_csv(flux_output_file, index=False)


# Water-content evolution at the diagnostic depth.
diagnostic_water_results = pd.DataFrame({
    "time_min": output_time,
    "depth_cm": actual_diagnostic_depth,
    "theta_mobile": theta_mo_4cm_output,
    "theta_immobile": theta_im_4cm_output,
    "theta_total": theta_total_4cm_output,
})

diagnostic_water_output_file = (
    OUTPUT_DIRECTORY
    / f"constant_{selected_case}_water_content_{actual_diagnostic_depth:g}cm.csv"
)
diagnostic_water_results.to_csv(
    diagnostic_water_output_file,
    index=False,
)


# Water-content evolution at the bottom node.
bottom_water_results = pd.DataFrame({
    "time_min": output_time,
    "depth_cm": depth[-1],
    "theta_mobile": theta_mo_bottom_output,
    "theta_immobile": theta_im_bottom_output,
    "theta_total": theta_total_bottom_output,
})

bottom_water_output_file = (
    OUTPUT_DIRECTORY / f"constant_{selected_case}_water_content_bottom.csv"
)
bottom_water_results.to_csv(
    bottom_water_output_file,
    index=False,
)


# Initial and final depth profiles.
water_profile_results = pd.DataFrame({
    "depth_cm": depth,
    "theta_mobile_initial": theta_mo_initial,
    "theta_mobile_final": theta_mo_final,
    "theta_immobile_initial": theta_im_initial,
    "theta_immobile_final": theta_im_final,
    "theta_total_initial": theta_total_initial,
    "theta_total_final": theta_total_final,
    "Se_mobile_final": Se_mo_final,
    "Se_immobile_final": Se_im_final,
    "Gamma_w_final_per_min": Gamma_final,
})

water_profile_output_file = (
    OUTPUT_DIRECTORY / f"constant_{selected_case}_final_water_profile.csv"
)
water_profile_results.to_csv(
    water_profile_output_file,
    index=False,
)


# Compact run summary with parameters, numerical counters, and water balance.
run_summary = pd.DataFrame([{
    "case": selected_case,
    "case_label": case["label"],
    "omega_per_min": omega,
    "dispersivity_cm": dispersivity,
    "theta_total_upper_initial": theta_total_upper,
    "theta_total_bottom_initial": theta_total_bottom,
    "q_top_cm_per_min": q_top,
    "column_length_cm": L,
    "dx_cm": dx,
    "simulation_time_min": t_end,
    "dt_initial_min": dt_initial,
    "dt_min_min": dt_min,
    "dt_max_min": dt_max,
    "accepted_steps": accepted_steps,
    "rejected_water_steps": rejected_water_steps,
    "rejected_solute_steps": rejected_solute_steps,
    "initial_storage_cm": initial_storage,
    "cumulative_top_inflow_cm": cumulative_top_inflow,
    "cumulative_bottom_outflow_cm": cumulative_bottom_outflow,
    "final_storage_cm": final_storage,
    "mass_balance_error_cm": mass_balance_error,
    "relative_mass_balance_error_percent": relative_mass_balance_error,
}])

summary_output_file = (
    OUTPUT_DIRECTORY / f"constant_{selected_case}_run_summary.csv"
)
run_summary.to_csv(summary_output_file, index=False)


# =============================================================================
# 20. RUN SUMMARY
# =============================================================================

print("\n" + "=" * 72)
print("CONSTANT-PARAMETER DUAL-POROSITY MODEL")
print("=" * 72)
print(f"Case:                         {case['label']}")
print(f"Omega:                        {omega:.6e} 1/min")
print(f"Dispersivity:                 {dispersivity:.6f} cm")
print(f"Maximum simulated C/C0:       {np.max(BTC_output):.6f}")
print(f"Initial stored water:         {initial_storage:.8f} cm")
print(f"Cumulative upper inflow:      {cumulative_top_inflow:.8f} cm")
print(f"Cumulative bottom outflow:    {cumulative_bottom_outflow:.8f} cm")
print(f"Final stored water:           {final_storage:.8f} cm")
print(f"Relative mass-balance error:  {relative_mass_balance_error:.6f} %")

print("\nSaved outputs:")
for path in (
    time_series_output_file,
    btc_output_file,
    flux_output_file,
    diagnostic_water_output_file,
    bottom_water_output_file,
    water_profile_output_file,
    summary_output_file,
):
    print(f"  {path}")
