"""
Dual-porosity model with saturation-dependent transfer and dispersivity.

This repository version is derived from the working research script used in
the study. The internal constant-parameter simulation is intentionally kept:
its modeled initial and final states define the state-variable anchors used by
the saturation-dependent parameter relationships.

The script performs:
    1. one constant reference simulation;
    2. automatic extraction of state anchors from that simulation;
    3. one selected saturation-dependent simulation;
    4. saving of the numerical results for both simulations.

Plotting, observed-data comparison, RRMSE calculations, experimental
mass-recovery calculations, and computer-specific file paths are excluded
because they are not required to run the numerical model.

Usage
-----
Set ``selected_case`` and ``comparison_mode`` below, then run:

    python dual_porosity_saturation_dependent.py

Outputs are written to the repository-level ``outputs`` directory.
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


# Repository-cleaned version marker.
CODE_VERSION = "github-ready-2026-09-08"


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

# Initial-water-content case: "NS", "60", or "40".
selected_case = "60"

# Saturation-dependent formulation:
#   "omega_variable"        -> omega varies; dispersivity remains case-constant
#   "dispersivity_variable" -> dispersivity varies; omega remains case-constant
#   "both_variable"         -> both omega and dispersivity vary
comparison_mode = "both_variable"

COMPARISON_MODES = (
    "omega_variable",
    "dispersivity_variable",
    "both_variable",
)

if comparison_mode not in COMPARISON_MODES:
    raise ValueError(
        f"comparison_mode must be one of {COMPARISON_MODES}, "
        f"not {comparison_mode!r}."
    )

# Repository-relative output location.
#
# Intended GitHub layout:
#   <repository>/model/dual_porosity_saturation_dependent.py
#   <repository>/outputs/
#
# If this file is tested standalone (for example from Downloads), outputs are
# written to an "outputs" folder beside the script instead.
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

if SCRIPT_DIRECTORY.name.lower() == "model":
    PROJECT_ROOT = SCRIPT_DIRECTORY.parent
else:
    PROJECT_ROOT = SCRIPT_DIRECTORY

OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. HYDRAULIC PARAMETERS
# =============================================================================

# Mobile domain: fracture
WCR = 0.0             # Residual mobile water content [-]
WCS = 0.08            # Saturated mobile water content [-]
alpha = 0.05          # van Genuchten alpha [1/cm]
n = 1.25              # van Genuchten n [-]
Ks = 0.08             # Saturated hydraulic conductivity [cm/min]
l_mualem = 0.5        # Mualem pore-connectivity parameter [-]

# Immobile domain: matrix
WCIR = 0.03           # Residual immobile water content [-]
WCIM = 0.4000         # Saturated immobile water content [-]


# =============================================================================
# 3. CONSTANT REFERENCE PARAMETERS
# =============================================================================

# These calibrated values are used by the internal constant reference run.
OMEGA_REFERENCE = {
    "40": 7.0e-4,
    "60": 1.8e-4,
    "NS": 9.0e-5,
}

DISPERSIVITY_REFERENCE = {
    "40": 2.0,
    "60": 0.2,
    "NS": 1.0,
}


# =============================================================================
# 4. SATURATION-DEPENDENT PARAMETER SETTINGS
# =============================================================================

# Omega is a normalized late sigmoid of local immobile effective saturation.
OMEGA_VALUE_START = 2.5e-4
OMEGA_VALUE_END = 1.8e-3
OMEGA_TRANSITION_FRACTION = 0.60
OMEGA_STEEPNESS = 20.0

# Assigned automatically from the internal constant reference simulation.
OMEGA_SATURATION_START = None
OMEGA_SATURATION_END = None

# Dispersivity is a smooth high -> low -> high function of local total water
# content, implemented with two logistic transitions.
DISPERSIVITY_VALUE_EARLY = 1.0
DISPERSIVITY_VALUE_MIDDLE = 0.10
DISPERSIVITY_VALUE_LATE = 2.0

DISPERSIVITY_TRANSITION_1 = 0.08
DISPERSIVITY_TRANSITION_2 = 0.78
DISPERSIVITY_STEEPNESS_1 = 50.0
DISPERSIVITY_STEEPNESS_2 = 35.0

# Compatibility aliases used in the numerical implementation.
DISPERSIVITY_VALUE_START = DISPERSIVITY_VALUE_EARLY
DISPERSIVITY_VALUE_END = DISPERSIVITY_VALUE_LATE

# Assigned automatically from the internal constant reference simulation.
DISPERSIVITY_THETA_TOTAL_START = None
DISPERSIVITY_THETA_TOTAL_END = None


ALL_PARAMETER_MODES = (
    "constant",
    "omega_variable",
    "dispersivity_variable",
    "both_variable",
)

MODE_LABELS = {
    "constant": "Constant reference",
    "omega_variable": "Variable omega",
    "dispersivity_variable": "Variable dispersivity",
    "both_variable": "Variable omega and dispersivity",
}


# =============================================================================
# 5. SATURATION-LEVEL CASES
# =============================================================================

CASES = {
    "NS": {
        "label": "NS",
        "theta_total_upper": 0.468,
        "theta_total_bottom": 0.225,
        "omega": OMEGA_REFERENCE["NS"],
        "dispersivity": DISPERSIVITY_REFERENCE["NS"],
        "dt_initial": 0.004,
        "dt_min": 0.0004,
        "dt_max": 0.5,
    },
    "60": {
        "label": "60%",
        "theta_total_upper": 0.265,
        "theta_total_bottom": 0.225,
        "omega": OMEGA_REFERENCE["60"],
        "dispersivity": DISPERSIVITY_REFERENCE["60"],
        "dt_initial": 0.04,
        "dt_min": 0.004,
        "dt_max": 0.5,
    },
    "40": {
        "label": "40%",
        "theta_total_upper": 0.174,
        "theta_total_bottom": 0.225,
        "omega": OMEGA_REFERENCE["40"],
        "dispersivity": DISPERSIVITY_REFERENCE["40"],
        "dt_initial": 0.4,
        "dt_min": 0.04,
        "dt_max": 0.5,
    },
}

if selected_case not in CASES:
    raise ValueError(
        f"selected_case must be one of {tuple(CASES)}, "
        f"not {selected_case!r}."
    )

case = CASES[selected_case]


# =============================================================================
# 6. SPATIAL AND TEMPORAL DISCRETIZATION
# =============================================================================

L = 9.0                  # Column length [cm]
dx = 0.25                # Node spacing [cm]
nodes = int(round(L / dx)) + 1
depth = np.linspace(0.0, L, nodes)

t_end = 450.0            # Total simulation time [min]
dt_initial = case["dt_initial"]
dt_max = case["dt_max"]
dt_min = case["dt_min"]
dt_output = 1.0          # Output interval [min]

diagnostic_depth = 4.0   # Diagnostic depth [cm]
diagnostic_node = int(round(diagnostic_depth / dx))

if diagnostic_node < 0 or diagnostic_node >= nodes:
    raise ValueError(
        f"Diagnostic depth {diagnostic_depth} cm is outside the model domain."
    )

actual_diagnostic_depth = depth[diagnostic_node]


# =============================================================================
# 7. INITIAL AND BOUNDARY CONDITIONS
# =============================================================================

theta_total_upper = case["theta_total_upper"]
theta_total_bottom = case["theta_total_bottom"]

# Constant upper water flux. Positive means downward.
q_top = 0.006  # [cm/min]


# =============================================================================
# 8. SOLUTE-TRANSPORT SETTINGS
# =============================================================================

C0 = 0.0005                 # Injected concentration [mg/cm3]
tracer_duration = 90.0      # Tracer pulse duration [min]
initial_concentration = 0.0

# Mechanical dispersion only; molecular diffusion is neglected.
SOLUTE_TIME_SCHEME = "crank_nicolson"
SOLUTE_SPACE_SCHEME = "galerkin_finite_elements"
GALERKIN_STORAGE_MATRIX = "lumped"

NEGATIVE_CONCENTRATION_ABSOLUTE_TOLERANCE = 1.0e-12
NEGATIVE_CONCENTRATION_RELATIVE_TOLERANCE = 1.0e-6

# Low-dispersivity stability controls retained from the working code.
ENABLE_LOW_DISPERSIVITY_DT_CONTROL = True
SOLUTE_PE_CR_LIMIT = 2.0
SOLUTE_DT_SAFETY_FACTOR = 0.90
ABSOLUTE_SOLUTE_DT_MIN = 1.0e-8
ALLOW_IMPLICIT_FALLBACK_FOR_LOW_DISPERSIVITY = True

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
# 8. VAN GENUCHTEN AND HYDRAULIC FUNCTIONS
# =============================================================================

def vg_m(n_value):
    """Calculate the van Genuchten m parameter."""
    return 1.0 - 1.0 / n_value


def effective_saturation_from_h(h, alpha_value, n_value):
    """Calculate effective saturation from pressure head."""
    h = np.asarray(h, dtype=float)
    m_value = vg_m(n_value)

    Se = np.ones_like(h)
    unsaturated = h < 0.0
    Se[unsaturated] = (
        1.0 + (alpha_value * np.abs(h[unsaturated])) ** n_value
    ) ** (-m_value)

    return np.clip(Se, 0.0, 1.0)


def theta_from_h(h, theta_r, theta_s, alpha_value, n_value):
    """Calculate volumetric water content from pressure head."""
    Se = effective_saturation_from_h(h, alpha_value, n_value)
    return theta_r + (theta_s - theta_r) * Se


def effective_saturation_from_theta(theta, theta_r, theta_s):
    """Calculate effective saturation directly from water content."""
    theta = np.asarray(theta, dtype=float)
    Se = (theta - theta_r) / (theta_s - theta_r)
    return np.clip(Se, 0.0, 1.0)


def immobile_effective_saturation(theta_im):
    """Current immobile-domain effective saturation at every node."""
    return effective_saturation_from_theta(theta_im, WCIR, WCIM)


def mobile_h_from_theta(theta):
    """Invert the mobile van Genuchten retention relation."""
    theta = np.asarray(theta, dtype=float)
    m_value = vg_m(n)
    Se = effective_saturation_from_theta(theta, WCR, WCS)
    Se = np.clip(Se, 1.0e-10, 1.0 - 1.0e-10)
    h_absolute = ((Se ** (-1.0 / m_value) - 1.0) ** (1.0 / n)) / alpha
    return -h_absolute


def hydraulic_conductivity(h):
    """Mobile-domain van Genuchten-Mualem hydraulic conductivity."""
    m_value = vg_m(n)
    Se = effective_saturation_from_h(h, alpha, n)

    conductivity_term = (
        1.0 - (1.0 - Se ** (1.0 / m_value)) ** m_value
    ) ** 2

    K = Ks * Se ** l_mualem * conductivity_term
    return np.maximum(K, 1.0e-20)


def K_face_value(K_up, K_down):
    """Arithmetic hydraulic-conductivity averaging at a node interface."""
    return 0.5 * (
        np.asarray(K_up, dtype=float)
        + np.asarray(K_down, dtype=float)
    )


# =============================================================================
# 9. SIGMOID OMEGA AND THREE-REGIME TOTAL-WATER-CONTENT DISPERSIVITY
# =============================================================================

def normalized_state_coordinate(
    driver,
    driver_start,
    driver_end,
    parameter_name="parameter",
):
    """Map a state variable between its modeled start and end anchors to [0, 1]."""
    driver_array = np.asarray(driver, dtype=float)
    driver_start = float(driver_start)
    driver_end = float(driver_end)

    if not np.isfinite(driver_start) or not np.isfinite(driver_end):
        raise ValueError(
            f"{parameter_name} start/end driver anchors must be finite."
        )

    denominator = driver_end - driver_start

    if abs(denominator) <= 1.0e-12:
        raise ValueError(
            f"{parameter_name} start and end driver anchors are too close."
        )

    x = (
        driver_array - driver_start
    ) / denominator

    return np.clip(x, 0.0, 1.0)


def logistic(value):
    """Numerically stable logistic function."""
    value = np.asarray(value, dtype=float)
    value = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-value))


def normalized_logistic(
    x,
    transition_fraction,
    steepness,
):
    """Logistic step normalized so f(0)=0 and f(1)=1 exactly."""
    x = np.asarray(x, dtype=float)
    transition_fraction = float(transition_fraction)
    steepness = float(steepness)

    if not 0.0 < transition_fraction < 1.0:
        raise ValueError(
            "transition_fraction must lie strictly between 0 and 1."
        )

    if not np.isfinite(steepness) or steepness <= 0.0:
        raise ValueError("steepness must be positive and finite.")

    raw = logistic(
        steepness * (x - transition_fraction)
    )
    raw_at_zero = float(
        logistic(-steepness * transition_fraction)
    )
    raw_at_one = float(
        logistic(steepness * (1.0 - transition_fraction))
    )

    denominator = raw_at_one - raw_at_zero

    if abs(denominator) <= 1.0e-15:
        raise RuntimeError(
            "The normalized logistic denominator is too small."
        )

    normalized = (
        raw - raw_at_zero
    ) / denominator

    return np.clip(normalized, 0.0, 1.0)


def resolved_omega_start_value():
    """Return the early/low omega value."""
    value = float(OMEGA_VALUE_START)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("OMEGA_VALUE_START must be positive and finite.")
    return value


def resolved_omega_end_value():
    """Return the late/high omega value."""
    value = float(OMEGA_VALUE_END)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("OMEGA_VALUE_END must be positive and finite.")
    return value


def omega_from_saturation(S_im, parameter_mode):
    """Return nodal omega from a late sigmoid increase with S_im."""
    if parameter_mode not in ALL_PARAMETER_MODES:
        raise ValueError(f"Unknown parameter_mode: {parameter_mode!r}")

    S_im_array = np.asarray(S_im, dtype=float)

    if parameter_mode in ("omega_variable", "both_variable"):
        if OMEGA_SATURATION_START is None or OMEGA_SATURATION_END is None:
            raise RuntimeError(
                "OMEGA_SATURATION_START/END have not yet been assigned."
            )

        x_omega = normalized_state_coordinate(
            driver=S_im_array,
            driver_start=OMEGA_SATURATION_START,
            driver_end=OMEGA_SATURATION_END,
            parameter_name="omega",
        )

        transition = normalized_logistic(
            x=x_omega,
            transition_fraction=OMEGA_TRANSITION_FRACTION,
            steepness=OMEGA_STEEPNESS,
        )

        omega_nodes = (
            resolved_omega_start_value()
            + (
                resolved_omega_end_value()
                - resolved_omega_start_value()
            ) * transition
        )

        if np.any(~np.isfinite(omega_nodes)) or np.any(omega_nodes <= 0.0):
            raise RuntimeError(
                "Variable omega became non-positive or non-finite."
            )

        return omega_nodes

    return np.full_like(
        S_im_array,
        float(case["omega"]),
        dtype=float,
    )


def dispersivity_from_total_water_content(theta_total, parameter_mode):
    """Return nodal dispersivity from a smooth three-regime theta_total function.

    Variable dispersivity follows:

        early high -> middle low -> late high

    using two logistic transitions in normalized total water content.
    """
    if parameter_mode not in ALL_PARAMETER_MODES:
        raise ValueError(f"Unknown parameter_mode: {parameter_mode!r}")

    theta_total_array = np.asarray(theta_total, dtype=float)

    if parameter_mode in ("dispersivity_variable", "both_variable"):
        if (
            DISPERSIVITY_THETA_TOTAL_START is None
            or DISPERSIVITY_THETA_TOTAL_END is None
        ):
            raise RuntimeError(
                "DISPERSIVITY_THETA_TOTAL_START/END have not yet been assigned."
            )

        values = (
            DISPERSIVITY_VALUE_EARLY,
            DISPERSIVITY_VALUE_MIDDLE,
            DISPERSIVITY_VALUE_LATE,
            DISPERSIVITY_TRANSITION_1,
            DISPERSIVITY_TRANSITION_2,
            DISPERSIVITY_STEEPNESS_1,
            DISPERSIVITY_STEEPNESS_2,
        )

        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError(
                "All variable-dispersivity settings must be finite."
            )

        if (
            DISPERSIVITY_VALUE_EARLY <= 0.0
            or DISPERSIVITY_VALUE_MIDDLE <= 0.0
            or DISPERSIVITY_VALUE_LATE <= 0.0
        ):
            raise ValueError(
                "All dispersivity regime values must be strictly positive."
            )

        if not (
            0.0 < DISPERSIVITY_TRANSITION_1
            < DISPERSIVITY_TRANSITION_2
            < 1.0
        ):
            raise ValueError(
                "Dispersivity transitions must satisfy "
                "0 < transition_1 < transition_2 < 1."
            )

        if (
            DISPERSIVITY_STEEPNESS_1 <= 0.0
            or DISPERSIVITY_STEEPNESS_2 <= 0.0
        ):
            raise ValueError(
                "Dispersivity steepness values must be strictly positive."
            )

        x_lambda = normalized_state_coordinate(
            driver=theta_total_array,
            driver_start=DISPERSIVITY_THETA_TOTAL_START,
            driver_end=DISPERSIVITY_THETA_TOTAL_END,
            parameter_name="dispersivity",
        )

        # Smooth loss of the early-high contribution.
        early_to_middle = logistic(
            DISPERSIVITY_STEEPNESS_1
            * (
                x_lambda
                - DISPERSIVITY_TRANSITION_1
            )
        )

        # Smooth gain of the late-high contribution.
        middle_to_late = logistic(
            DISPERSIVITY_STEEPNESS_2
            * (
                x_lambda
                - DISPERSIVITY_TRANSITION_2
            )
        )

        dispersivity = (
            DISPERSIVITY_VALUE_MIDDLE
            + (
                DISPERSIVITY_VALUE_EARLY
                - DISPERSIVITY_VALUE_MIDDLE
            ) * (1.0 - early_to_middle)
            + (
                DISPERSIVITY_VALUE_LATE
                - DISPERSIVITY_VALUE_MIDDLE
            ) * middle_to_late
        )

        if np.any(~np.isfinite(dispersivity)) or np.any(dispersivity <= 0.0):
            raise RuntimeError(
                "Three-regime dispersivity became non-positive or non-finite."
            )

        return dispersivity

    return np.full_like(
        theta_total_array,
        float(case["dispersivity"]),
        dtype=float,
    )


def parameter_profiles(theta_mo, theta_im, parameter_mode):
    """Return S_im, omega, and dispersivity at the current nodal state."""
    theta_mo = np.asarray(theta_mo, dtype=float)
    theta_im = np.asarray(theta_im, dtype=float)

    if theta_mo.shape != theta_im.shape:
        raise ValueError("theta_mo and theta_im must have identical shapes.")

    theta_total = theta_mo + theta_im
    S_im = immobile_effective_saturation(theta_im)

    omega_nodes = omega_from_saturation(
        S_im,
        parameter_mode,
    )

    dispersivity_nodes = dispersivity_from_total_water_content(
        theta_total,
        parameter_mode,
    )

    return S_im, omega_nodes, dispersivity_nodes


def gamma_w_exchange(h_mo, theta_im, parameter_mode):
    """Calculate signed water exchange without changing its formulation.

    Gamma_w = omega(S_im) * (Se_mo - Se_im)

    Omega may vary with S_im through its sigmoid function.
    Dispersivity may vary independently with local total water content.
    """
    theta_mo = theta_from_h(h_mo, WCR, WCS, alpha, n)
    Se_mo = effective_saturation_from_theta(theta_mo, WCR, WCS)

    Se_im, omega_nodes, dispersivity_nodes = parameter_profiles(
        theta_mo,
        theta_im,
        parameter_mode,
    )

    Gamma_w = omega_nodes * (Se_mo - Se_im)

    return (
        Gamma_w,
        Se_mo,
        Se_im,
        omega_nodes,
        dispersivity_nodes,
    )


# =============================================================================
# 11. HYDRUS-CONSISTENT INITIAL MOBILE/IMMOBILE SPLIT
# =============================================================================

def hydrus_split_total_theta(theta_total):
    """Split total water content using the HYDRUS capacity ratio."""
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

    theta_mo = np.clip(theta_mo, WCR, WCS)
    theta_im = np.clip(theta_im, WCIR, WCIM)
    return float(theta_mo), float(theta_im)


def create_initial_state():
    """Create independent initial arrays for one simulation run."""
    theta_mo_upper_initial, theta_im_upper_initial = hydrus_split_total_theta(
        theta_total_upper
    )
    h_upper_initial = float(mobile_h_from_theta(theta_mo_upper_initial))

    theta_mo_bottom, theta_im_bottom = hydrus_split_total_theta(
        theta_total_bottom
    )
    h_bottom = float(mobile_h_from_theta(theta_mo_bottom))

    h_mo_initial = np.full(nodes, h_upper_initial, dtype=float)
    h_mo_initial[-1] = h_bottom

    theta_im_initial = np.full(nodes, theta_im_upper_initial, dtype=float)
    theta_im_initial[-1] = theta_im_bottom

    theta_mo_initial = theta_from_h(h_mo_initial, WCR, WCS, alpha, n)
    theta_total_initial = theta_mo_initial + theta_im_initial

    c_mo_initial = np.full(nodes, initial_concentration, dtype=float)
    c_im_initial = np.full(nodes, initial_concentration, dtype=float)

    # The first-type upper concentration boundary is active from t=0. This
    # old-time boundary value is required by the Crank-Nicolson operator.
    c_mo_initial[0] = inlet_concentration(0.0)

    return {
        "h_mo": h_mo_initial,
        "theta_im": theta_im_initial,
        "theta_mo": theta_mo_initial,
        "theta_total": theta_total_initial,
        "c_mo": c_mo_initial,
        "c_im": c_im_initial,
        "theta_mo_upper_initial": theta_mo_upper_initial,
        "theta_im_upper_initial": theta_im_upper_initial,
        "h_upper_initial": h_upper_initial,
        "theta_mo_bottom_initial": theta_mo_bottom,
        "theta_im_bottom_initial": theta_im_bottom,
        "h_bottom": h_bottom,
    }


# =============================================================================
# 12. DARCY FLUX FUNCTION
# =============================================================================

def downward_flux(h_up, h_down, K_up, K_down):
    """Calculate downward-positive Darcy flux across a node interface."""
    K_face = K_face_value(K_up, K_down)
    dh_dz = (h_down - h_up) / dx
    return K_face * (1.0 - dh_dz)


# =============================================================================
# 13. NONLINEAR SOLVER BOUNDS
# =============================================================================

h_lower = np.full(nodes, -1.0e7, dtype=float)
h_upper = np.full(nodes, 100.0, dtype=float)

numerical_theta_tolerance = 1.0e-12

theta_im_lower = np.full(
    nodes,
    WCIR - numerical_theta_tolerance,
    dtype=float,
)

theta_im_upper = np.full(
    nodes,
    WCIM + numerical_theta_tolerance,
    dtype=float,
)

lower_bounds = np.concatenate([h_lower, theta_im_lower])
upper_bounds = np.concatenate([h_upper, theta_im_upper])


# =============================================================================
# 14. IMPLICIT NONLINEAR WATER RESIDUAL SYSTEM
# =============================================================================

def residual_system(
    unknown_vector,
    h_mo_old,
    theta_im_old,
    current_dt,
    h_bottom,
    parameter_mode,
):
    """Residual equations for one fully implicit water-flow time step."""
    h_mo_new = unknown_vector[:nodes]
    theta_im_new = unknown_vector[nodes:]

    residual = np.zeros(2 * nodes, dtype=float)

    theta_mo_old = theta_from_h(h_mo_old, WCR, WCS, alpha, n)
    theta_mo_new = theta_from_h(h_mo_new, WCR, WCS, alpha, n)
    K_new = hydraulic_conductivity(h_mo_new)

    Gamma_w, _, _, _, _ = gamma_w_exchange(
        h_mo_new,
        theta_im_new,
        parameter_mode,
    )

    q_faces = np.zeros(nodes - 1, dtype=float)
    for face in range(nodes - 1):
        q_faces[face] = downward_flux(
            h_up=h_mo_new[face],
            h_down=h_mo_new[face + 1],
            K_up=K_new[face],
            K_down=K_new[face + 1],
        )

    # Upper mobile node: prescribed incoming water flux q_top.
    residual[0] = (
        (theta_mo_new[0] - theta_mo_old[0]) / current_dt
        + (q_faces[0] - q_top) / dx
        + Gamma_w[0]
    )

    # Internal mobile nodes.
    for i in range(1, nodes - 1):
        q_in = q_faces[i - 1]
        q_out = q_faces[i]
        residual[i] = (
            (theta_mo_new[i] - theta_mo_old[i]) / current_dt
            + (q_out - q_in) / dx
            + Gamma_w[i]
        )

    # Fixed mobile pressure head at the bottom.
    residual[nodes - 1] = h_mo_new[-1] - h_bottom

    # Immobile storage at every node.
    residual[nodes:] = (
        (theta_im_new - theta_im_old) / current_dt
        - Gamma_w
    )

    return residual


# =============================================================================
# 15. CRANK-NICOLSON + GALERKIN FINITE-ELEMENT SOLUTE SYSTEM
# =============================================================================

def inlet_concentration(time_value):
    """Pulse input: C0 through 90 min, followed by solute-free water."""
    return C0 if time_value <= tracer_duration + 1.0e-12 else 0.0


def assemble_weighted_consistent_mass_matrix(coefficient_nodes):
    r"""Assemble \int N_i * coefficient * N_j dz for linear elements.

    For one element with endpoint coefficient values a and b, exact integration
    gives

        dx/12 * [[3a+b, a+b],
                 [a+b, a+3b]].
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

        matrix[left, left] += local[0, 0]
        matrix[left, right] += local[0, 1]
        matrix[right, left] += local[1, 0]
        matrix[right, right] += local[1, 1]

    return matrix


def low_dispersivity_transport_dt_limit(
    theta_mo_nodes,
    q_faces,
    dispersivity_nodes,
):
    """Return a Pe*Cr-based maximum transport time step [min].

    For mechanical dispersion,

        D = lambda * |v|

    and therefore

        Pe * Cr
        = (dx/lambda) * (|v|*dt/dx)
        = |v|*dt/lambda.

    Requiring Pe*Cr <= SOLUTE_PE_CR_LIMIT gives

        dt <= SOLUTE_PE_CR_LIMIT * lambda / |v|.

    The smallest element limit is used. Zero-flow elements do not constrain dt.
    """
    if not ENABLE_LOW_DISPERSIVITY_DT_CONTROL:
        return np.inf

    theta_mo_nodes = np.asarray(theta_mo_nodes, dtype=float)
    q_faces = np.asarray(q_faces, dtype=float)
    dispersivity_nodes = np.asarray(dispersivity_nodes, dtype=float)

    if theta_mo_nodes.shape != (nodes,):
        raise ValueError("theta_mo_nodes must contain one value per node.")

    if q_faces.shape != (nodes - 1,):
        raise ValueError("q_faces must contain one value per element.")

    if dispersivity_nodes.shape != (nodes,):
        raise ValueError("dispersivity_nodes must contain one value per node.")

    theta_elements = np.maximum(
        0.5 * (
            theta_mo_nodes[:-1]
            + theta_mo_nodes[1:]
        ),
        1.0e-15,
    )

    lambda_elements = 0.5 * (
        dispersivity_nodes[:-1]
        + dispersivity_nodes[1:]
    )

    if np.any(~np.isfinite(lambda_elements)) or np.any(lambda_elements <= 0.0):
        raise RuntimeError(
            "Dispersivity must remain strictly positive and finite."
        )

    pore_velocity = np.abs(q_faces) / theta_elements

    moving = pore_velocity > 1.0e-15

    if not np.any(moving):
        return np.inf

    element_dt_limits = (
        SOLUTE_PE_CR_LIMIT
        * lambda_elements[moving]
        / pore_velocity[moving]
    )

    stable_dt = float(np.min(element_dt_limits))

    if not np.isfinite(stable_dt) or stable_dt <= 0.0:
        raise RuntimeError(
            "Could not determine a positive low-dispersivity transport dt."
        )

    return SOLUTE_DT_SAFETY_FACTOR * stable_dt


def assemble_galerkin_advection_dispersion_operator(
    theta_mo_nodes,
    q_faces,
    dispersivity_nodes,
):
    """Assemble standard Galerkin advection-dispersion element matrices.

    The saturation-dependent dispersivity is evaluated at the nodes. The value
    used in each finite element is the arithmetic mean of its two nodal values:

        lambda_e = 0.5 * (lambda_left + lambda_right)

    Mechanical dispersion is

        D_e = lambda_e * abs(q_e / theta_e).
    """
    theta_mo_nodes = np.asarray(theta_mo_nodes, dtype=float)
    q_faces = np.asarray(q_faces, dtype=float)
    dispersivity_nodes = np.asarray(dispersivity_nodes, dtype=float)

    if theta_mo_nodes.shape != (nodes,):
        raise ValueError("theta_mo_nodes must contain one value per node.")
    if q_faces.shape != (nodes - 1,):
        raise ValueError("q_faces must contain one value per element.")
    if dispersivity_nodes.shape != (nodes,):
        raise ValueError("dispersivity_nodes must contain one value per node.")

    operator = np.zeros((nodes, nodes), dtype=float)

    for element in range(nodes - 1):
        left = element
        right = element + 1
        q_element = q_faces[element]

        theta_element = max(
            0.5 * (theta_mo_nodes[left] + theta_mo_nodes[right]),
            1.0e-15,
        )
        dispersivity_element = 0.5 * (
            dispersivity_nodes[left] + dispersivity_nodes[right]
        )

        pore_velocity = q_element / theta_element
        D_element = dispersivity_element * abs(pore_velocity)
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

        operator[left, left] += local[0, 0]
        operator[left, right] += local[0, 1]
        operator[right, left] += local[1, 0]
        operator[right, right] += local[1, 1]

    # Natural lower boundary: zero dispersive gradient plus advective outflow.
    operator[-1, -1] += q_faces[-1]
    return operator


def assemble_coupled_transport_matrices(
    theta_mo,
    theta_im,
    q_faces,
    Gamma_w,
    dispersivity_nodes,
):
    """Assemble conservative mobile-immobile storage and transport matrices."""
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
        assemble_galerkin_advection_dispersion_operator(
            theta_mo_nodes=theta_mo,
            q_faces=q_faces,
            dispersivity_nodes=dispersivity_nodes,
        )
    )

    # Water-driven solute exchange uses the donor concentration.
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
    dispersivity_nodes_old,
    dispersivity_nodes_new,
    current_dt,
    time_old,
    time_new,
    solute_time_weight=None,
):
    """Solve transport with Crank-Nicolson and standard Galerkin FE.

    For temporal weight beta,

      [M_new + dt*beta*K_new] C_new
        = [M_old - dt*(1-beta)*K_old] C_old.

    Omega may vary with current S_im, while dispersivity may vary through its three-regime total-water-content function.
    Consequently, the old and new transport matrices use their corresponding
    old- and new-time parameter fields.
    """
    old_concentrations = np.concatenate([
        np.asarray(c_mo_old, dtype=float),
        np.asarray(c_im_old, dtype=float),
    ])

    # Enforce the old first-type boundary before evaluating the old operator.
    old_concentrations[0] = inlet_concentration(time_old)

    storage_old, transport_old = assemble_coupled_transport_matrices(
        theta_mo=theta_mo_old,
        theta_im=theta_im_old,
        q_faces=q_faces_old,
        Gamma_w=Gamma_w_old,
        dispersivity_nodes=dispersivity_nodes_old,
    )

    storage_new, transport_new = assemble_coupled_transport_matrices(
        theta_mo=theta_mo_new,
        theta_im=theta_im_new,
        q_faces=q_faces_new,
        Gamma_w=Gamma_w_new,
        dispersivity_nodes=dispersivity_nodes_new,
    )

    beta = (
        SOLUTE_TIME_WEIGHT
        if solute_time_weight is None
        else float(solute_time_weight)
    )

    if not (0.5 <= beta <= 1.0):
        raise ValueError(
            "solute_time_weight must lie between 0.5 and 1.0."
        )

    A = storage_new + current_dt * beta * transport_new
    b = (
        storage_old - current_dt * (1.0 - beta) * transport_old
    ) @ old_concentrations

    # First-type upper concentration boundary. Apply row replacement only after
    # full matrix assembly; interior equations retain their coupling to node 0.
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = inlet_concentration(time_new)

    concentrations = np.linalg.solve(A, b)

    if not np.all(np.isfinite(concentrations)):
        raise RuntimeError(
            "The Crank-Nicolson/Galerkin solution contains non-finite values."
        )

    minimum_concentration = float(np.min(concentrations))
    negative_tolerance = max(
        NEGATIVE_CONCENTRATION_ABSOLUTE_TOLERANCE,
        NEGATIVE_CONCENTRATION_RELATIVE_TOLERANCE * abs(C0),
    )

    if minimum_concentration < -negative_tolerance:
        raise RuntimeError(
            "The Crank-Nicolson/Galerkin solution produced a materially "
            f"negative concentration: {minimum_concentration:.6e}; "
            f"allowed tolerance: {-negative_tolerance:.6e}."
        )

    concentrations = np.maximum(concentrations, 0.0)
    return concentrations[:nodes], concentrations[nodes:]


# =============================================================================
# 16. COUPLED WATER-SOLUTE SIMULATION FUNCTION
# =============================================================================

def run_simulation(parameter_mode):
    """Run one complete coupled simulation and return all result arrays."""
    if parameter_mode not in ALL_PARAMETER_MODES:
        raise ValueError(
            f"parameter_mode must be one of {ALL_PARAMETER_MODES}, "
            f"not {parameter_mode!r}."
        )

    initial = create_initial_state()

    h_mo = initial["h_mo"].copy()
    theta_im = initial["theta_im"].copy()
    c_mo = initial["c_mo"].copy()
    c_im = initial["c_im"].copy()
    h_bottom = initial["h_bottom"]

    theta_mo_initial = initial["theta_mo"].copy()
    theta_im_initial = initial["theta_im"].copy()
    theta_total_initial = initial["theta_total"].copy()

    S_im_initial, omega_initial_nodes, dispersivity_initial_nodes = (
        parameter_profiles(
            theta_mo_initial,
            theta_im_initial,
            parameter_mode,
        )
    )

    K_initial = hydraulic_conductivity(h_mo)
    q_bottom_initial = downward_flux(
        h_up=h_mo[-2],
        h_down=h_mo[-1],
        K_up=K_initial[-2],
        K_down=K_initial[-1],
    )

    time_current = 0.0
    dt = dt_initial
    cumulative_bottom_flux = 0.0

    tracked_time = [0.0]
    tracked_bottom_flux = [q_bottom_initial]
    tracked_cumulative_flux = [0.0]
    tracked_bottom_btc = [0.0]
    tracked_bottom_concentration = [0.0]
    tracked_average_Gamma = []

    # Full immobile-saturation profile at every accepted time level.
    # This is used to reconstruct S_im(depth, time) at regular output times.
    tracked_S_im_profiles = [S_im_initial.copy()]
    tracked_theta_total_profiles = [theta_total_initial.copy()]

    tracked_theta_mo_4cm = [float(theta_mo_initial[diagnostic_node])]
    tracked_theta_im_4cm = [float(theta_im_initial[diagnostic_node])]
    tracked_theta_total_4cm = [
        float(theta_mo_initial[diagnostic_node] + theta_im_initial[diagnostic_node])
    ]

    tracked_theta_mo_bottom = [float(theta_mo_initial[-1])]
    tracked_theta_im_bottom = [float(theta_im_initial[-1])]
    tracked_theta_total_bottom = [
        float(theta_mo_initial[-1] + theta_im_initial[-1])
    ]

    PRINT_EVERY_ACCEPTED_STEPS = 50
    DT_GROWTH_FACTOR = 1.5

    accepted_steps = 0
    rejected_water_steps = 0
    rejected_solute_steps = 0
    wall_start = time.perf_counter()

    print("\n" + "=" * 84, flush=True)
    print(
        f"Starting coupled water-solute simulation: {case['label']} | "
        f"{MODE_LABELS[parameter_mode]}",
        flush=True,
    )
    print("=" * 84, flush=True)
    print(
        f"Grid nodes={nodes}, t_end={t_end} min, "
        f"dt_initial={dt_initial:g} min, dt_max={dt_max:g} min",
        flush=True,
    )
    print(
        f"Initial upper state: theta_total={theta_total_upper:.6f}, "
        f"theta_mo_upper={initial['theta_mo_upper_initial']:.6f}, "
        f"theta_im_upper={initial['theta_im_upper_initial']:.6f}, "
        f"h_upper={initial['h_upper_initial']:.3f} cm",
        flush=True,
    )
    print(
        f"Initial bottom state: theta_total={theta_total_bottom:.6f}, "
        f"theta_mo_bottom={initial['theta_mo_bottom_initial']:.6f}, "
        f"theta_im_bottom={initial['theta_im_bottom_initial']:.6f}, "
        f"h_bottom={h_bottom:.3f} cm",
        flush=True,
    )
    print(
        "Exchange formulation retained: "
        "Gamma_w = omega * (Se_mo - Se_im)",
        flush=True,
    )
    print(
        "Mechanical dispersion only; molecular diffusion is neglected.",
        flush=True,
    )
    print(
        "Low-dispersivity control: "
        f"Pe*Cr <= {SOLUTE_PE_CR_LIMIT:g}; "
        f"absolute solute dt floor={ABSOLUTE_SOLUTE_DT_MIN:.1e} min; "
        f"implicit fallback={ALLOW_IMPLICIT_FALLBACK_FOR_LOW_DISPERSIVITY}.",
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
        "Upper solute boundary: prescribed concentration "
        "C(0,t)=C_in(t) (first type).",
        flush=True,
    )

    while time_current < t_end - 1.0e-12:
        current_dt = min(dt, t_end - time_current)

        # A Crank-Nicolson step must not cross the abrupt end of the tracer
        # pulse. Force a time level exactly at tracer_duration.
        if (
            time_current < tracer_duration - 1.0e-12
            and time_current + current_dt > tracer_duration
        ):
            current_dt = tracer_duration - time_current

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

        # Old-time hydraulic and saturation-dependent transport quantities are
        # required by Crank-Nicolson.
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
        (
            Gamma_old_step,
            _,
            _,
            _,
            dispersivity_old_nodes,
        ) = gamma_w_exchange(
            h_mo_old_step,
            theta_im_old_step,
            parameter_mode,
        )

        # -------------------------------------------------------------
        # Low-dispersivity stability control.
        #
        # Do this BEFORE the water solve so water and solute advance over
        # exactly the same accepted time interval.
        # -------------------------------------------------------------
        transport_dt_limit = low_dispersivity_transport_dt_limit(
            theta_mo_nodes=theta_mo_old_step,
            q_faces=q_faces_old_step,
            dispersivity_nodes=dispersivity_old_nodes,
        )

        if (
            np.isfinite(transport_dt_limit)
            and current_dt > transport_dt_limit
        ):
            current_dt = max(
                transport_dt_limit,
                ABSOLUTE_SOLUTE_DT_MIN,
            )

            # Still respect the exact tracer shutoff time.
            if (
                time_current < tracer_duration - 1.0e-12
                and time_current + current_dt > tracer_duration
            ):
                current_dt = tracer_duration - time_current

        water_guess = np.concatenate([h_mo_old_step, theta_im_old_step])

        water_solution = least_squares(
            residual_system,
            water_guess,
            bounds=(lower_bounds, upper_bounds),
            args=(
                h_mo_old_step,
                theta_im_old_step,
                current_dt,
                h_bottom,
                parameter_mode,
            ),
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
                f"residual={water_maximum_residual:.3e}; "
                f"new dt={dt:.3e} min.",
                flush=True,
            )
            # A low physical dispersivity may force the coupled step below the
            # original hydraulic dt_min. Permit that when required by transport.
            minimum_allowed_dt = min(
                dt_min,
                ABSOLUTE_SOLUTE_DT_MIN,
            )

            if dt < minimum_allowed_dt:
                raise RuntimeError(
                    f"Water solver failed at t={time_current:.8f} min; "
                    f"maximum residual={water_maximum_residual:.4e}."
                )
            continue

        h_mo_candidate = water_solution.x[:nodes]
        theta_im_candidate = water_solution.x[nodes:]
        theta_mo_candidate = theta_from_h(
            h_mo_candidate,
            WCR,
            WCS,
            alpha,
            n,
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

        (
            Gamma_candidate,
            Se_mo_candidate,
            Se_im_candidate,
            omega_candidate_nodes,
            dispersivity_candidate_nodes,
        ) = gamma_w_exchange(
            h_mo_candidate,
            theta_im_candidate,
            parameter_mode,
        )

        used_implicit_fallback = False

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
                dispersivity_nodes_old=dispersivity_old_nodes,
                dispersivity_nodes_new=dispersivity_candidate_nodes,
                current_dt=current_dt,
                time_old=time_current,
                time_new=time_current + current_dt,
            )

        except (np.linalg.LinAlgError, RuntimeError) as cn_error:
            # ---------------------------------------------------------
            # Low-dispersivity fallback:
            # keep the same physical lambda and same dt, but retry the
            # transport solve with backward Euler (beta=1). This adds
            # temporal damping only; it does NOT change dispersivity.
            # ---------------------------------------------------------
            if (
                ALLOW_IMPLICIT_FALLBACK_FOR_LOW_DISPERSIVITY
                and SOLUTE_TIME_WEIGHT < 1.0
            ):
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
                        dispersivity_nodes_old=dispersivity_old_nodes,
                        dispersivity_nodes_new=dispersivity_candidate_nodes,
                        current_dt=current_dt,
                        time_old=time_current,
                        time_new=time_current + current_dt,
                        solute_time_weight=1.0,
                    )

                    used_implicit_fallback = True

                    print(
                        f"Low-dispersivity transport fallback at "
                        f"t={time_current:.6f} min: "
                        "Crank-Nicolson rejected; fully implicit step accepted.",
                        flush=True,
                    )

                except (np.linalg.LinAlgError, RuntimeError) as implicit_error:
                    rejected_solute_steps += 1
                    dt = current_dt / 2.0

                    print(
                        f"Solute step rejected at t={time_current:.6f} min; "
                        f"CN reason={cn_error}; "
                        f"implicit fallback reason={implicit_error}; "
                        f"new dt={dt:.3e} min.",
                        flush=True,
                    )

                    if dt < ABSOLUTE_SOLUTE_DT_MIN:
                        raise RuntimeError(
                            f"Solute solution failed at "
                            f"t={time_current:.8f} min even after "
                            "low-dispersivity stabilization."
                        ) from implicit_error

                    continue

            else:
                rejected_solute_steps += 1
                dt = current_dt / 2.0

                print(
                    f"Solute step rejected at t={time_current:.6f} min; "
                    f"reason={cn_error}; new dt={dt:.3e} min.",
                    flush=True,
                )

                if dt < ABSOLUTE_SOLUTE_DT_MIN:
                    raise RuntimeError(
                        f"Solute solution failed at t={time_current:.8f} min."
                    ) from cn_error

                continue

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
        tracked_average_Gamma.append(float(np.mean(Gamma_candidate)))
        tracked_S_im_profiles.append(Se_im_candidate.copy())
        tracked_theta_total_profiles.append(
            (theta_mo_candidate + theta_im_candidate).copy()
        )

        tracked_theta_mo_4cm.append(
            float(theta_mo_candidate[diagnostic_node])
        )
        tracked_theta_im_4cm.append(
            float(theta_im_candidate[diagnostic_node])
        )
        tracked_theta_total_4cm.append(
            float(
                theta_mo_candidate[diagnostic_node]
                + theta_im_candidate[diagnostic_node]
            )
        )

        tracked_theta_mo_bottom.append(float(theta_mo_candidate[-1]))
        tracked_theta_im_bottom.append(float(theta_im_candidate[-1]))
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
                f"dt={current_dt:.3f} min, "
                f"water_nfev={water_solution.nfev}, "
                f"BTC={bottom_relative_concentration:.4e}, "
                f"remaining~{estimated_remaining/60.0:.1f} min",
                flush=True,
            )

    wall_time_minutes = (time.perf_counter() - wall_start) / 60.0

    print("\nCoupled simulation finished successfully.", flush=True)
    print(
        f"Accepted steps={accepted_steps}, "
        f"water rejections={rejected_water_steps}, "
        f"solute rejections={rejected_solute_steps}, "
        f"wall time={wall_time_minutes:.2f} min",
        flush=True,
    )

    # -------------------------------------------------------------------------
    # Interpolate accepted states to regular output times.
    # -------------------------------------------------------------------------
    output_time = np.arange(0.0, t_end + dt_output, dt_output)
    output_time = output_time[output_time <= t_end + 1.0e-10]

    bottom_flux_output = np.interp(
        output_time,
        tracked_time,
        tracked_bottom_flux,
    )
    cumulative_flux_output = np.interp(
        output_time,
        tracked_time,
        tracked_cumulative_flux,
    )
    bottom_concentration_output = np.interp(
        output_time,
        tracked_time,
        tracked_bottom_concentration,
    )
    BTC_output = np.interp(
        output_time,
        tracked_time,
        tracked_bottom_btc,
    )
    Time_simulated = output_time.copy()

    theta_mo_4cm_output = np.interp(
        output_time,
        tracked_time,
        tracked_theta_mo_4cm,
    )
    theta_im_4cm_output = np.interp(
        output_time,
        tracked_time,
        tracked_theta_im_4cm,
    )
    theta_total_4cm_output = np.interp(
        output_time,
        tracked_time,
        tracked_theta_total_4cm,
    )

    theta_mo_bottom_output = np.interp(
        output_time,
        tracked_time,
        tracked_theta_mo_bottom,
    )
    theta_im_bottom_output = np.interp(
        output_time,
        tracked_time,
        tracked_theta_im_bottom,
    )
    theta_total_bottom_output = np.interp(
        output_time,
        tracked_time,
        tracked_theta_total_bottom,
    )

    # -------------------------------------------------------------------------
    # Reconstruct the complete state evolution through time.
    # -------------------------------------------------------------------------
    tracked_time_array = np.asarray(tracked_time, dtype=float)

    tracked_S_im_profiles_array = np.asarray(
        tracked_S_im_profiles,
        dtype=float,
    )
    tracked_theta_total_profiles_array = np.asarray(
        tracked_theta_total_profiles,
        dtype=float,
    )

    expected_shape = (tracked_time_array.size, nodes)

    if tracked_S_im_profiles_array.shape != expected_shape:
        raise RuntimeError(
            "Tracked immobile-saturation profiles have an unexpected shape."
        )

    if tracked_theta_total_profiles_array.shape != expected_shape:
        raise RuntimeError(
            "Tracked total-water-content profiles have an unexpected shape."
        )

    S_im_profile_output = np.empty(
        (output_time.size, nodes),
        dtype=float,
    )
    theta_total_profile_output = np.empty(
        (output_time.size, nodes),
        dtype=float,
    )

    for node_index in range(nodes):
        S_im_profile_output[:, node_index] = np.interp(
            output_time,
            tracked_time_array,
            tracked_S_im_profiles_array[:, node_index],
        )
        theta_total_profile_output[:, node_index] = np.interp(
            output_time,
            tracked_time_array,
            tracked_theta_total_profiles_array[:, node_index],
        )

    # Depth-averaged state values use trapezoidal integration over the column.
    S_im_mean_output = (
        np.trapezoid(
            S_im_profile_output,
            depth,
            axis=1,
        )
        / L
    )
    S_im_min_output = np.min(S_im_profile_output, axis=1)
    S_im_max_output = np.max(S_im_profile_output, axis=1)
    S_im_4cm_output = S_im_profile_output[:, diagnostic_node]
    S_im_bottom_output = S_im_profile_output[:, -1]

    theta_im_profile_output = (
        WCIR
        + S_im_profile_output * (WCIM - WCIR)
    )

    theta_im_mean_output = (
        np.trapezoid(
            theta_im_profile_output,
            depth,
            axis=1,
        )
        / L
    )
    theta_im_min_output = np.min(theta_im_profile_output, axis=1)
    theta_im_max_output = np.max(theta_im_profile_output, axis=1)

    theta_total_mean_output = (
        np.trapezoid(
            theta_total_profile_output,
            depth,
            axis=1,
        )
        / L
    )
    theta_total_min_output = np.min(theta_total_profile_output, axis=1)
    theta_total_max_output = np.max(theta_total_profile_output, axis=1)

    # Reconstruct parameter fields from their actual controlling variables.
    omega_profile_output = omega_from_saturation(
        S_im_profile_output,
        parameter_mode,
    )

    dispersivity_profile_output = dispersivity_from_total_water_content(
        theta_total_profile_output,
        parameter_mode,
    )

    omega_mean_output = (
        np.trapezoid(
            omega_profile_output,
            depth,
            axis=1,
        )
        / L
    )
    dispersivity_mean_output = (
        np.trapezoid(
            dispersivity_profile_output,
            depth,
            axis=1,
        )
        / L
    )

    theta_mo_final = theta_from_h(h_mo, WCR, WCS, alpha, n)
    theta_im_final = theta_im.copy()
    theta_total_final = theta_mo_final + theta_im_final

    (
        Gamma_final,
        Se_mo_final,
        Se_im_final,
        omega_final_nodes,
        dispersivity_final_nodes,
    ) = gamma_w_exchange(h_mo, theta_im_final, parameter_mode)

    # Water mass balance.
    initial_storage = np.trapezoid(theta_total_initial, depth)
    final_storage = np.trapezoid(theta_total_final, depth)
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

    return {
        "parameter_mode": parameter_mode,
        "mode_label": MODE_LABELS[parameter_mode],
        "h_bottom": h_bottom,
        "h_mo_final": h_mo.copy(),
        "theta_mo_initial": theta_mo_initial,
        "theta_im_initial": theta_im_initial,
        "theta_total_initial": theta_total_initial,
        "theta_mo_final": theta_mo_final,
        "theta_im_final": theta_im_final,
        "theta_total_final": theta_total_final,
        "S_im_initial": S_im_initial,
        "S_im_final": Se_im_final,
        "Se_mo_final": Se_mo_final,
        "omega_initial_nodes": omega_initial_nodes,
        "omega_final_nodes": omega_final_nodes,
        "dispersivity_initial_nodes": dispersivity_initial_nodes,
        "dispersivity_final_nodes": dispersivity_final_nodes,
        "Gamma_final": Gamma_final,
        "output_time": output_time,
        "Time_simulated": Time_simulated,
        "bottom_flux_output": bottom_flux_output,
        "cumulative_flux_output": cumulative_flux_output,
        "bottom_concentration_output": bottom_concentration_output,
        "BTC_output": BTC_output,
        "theta_mo_4cm_output": theta_mo_4cm_output,
        "theta_im_4cm_output": theta_im_4cm_output,
        "theta_total_4cm_output": theta_total_4cm_output,
        "theta_mo_bottom_output": theta_mo_bottom_output,
        "theta_im_bottom_output": theta_im_bottom_output,
        "theta_total_bottom_output": theta_total_bottom_output,
        "S_im_profile_output": S_im_profile_output,
        "S_im_mean_output": S_im_mean_output,
        "S_im_min_output": S_im_min_output,
        "S_im_max_output": S_im_max_output,
        "S_im_4cm_output": S_im_4cm_output,
        "S_im_bottom_output": S_im_bottom_output,
        "theta_im_profile_output": theta_im_profile_output,
        "theta_im_mean_output": theta_im_mean_output,
        "theta_im_min_output": theta_im_min_output,
        "theta_im_max_output": theta_im_max_output,
        "theta_total_profile_output": theta_total_profile_output,
        "theta_total_mean_output": theta_total_mean_output,
        "theta_total_min_output": theta_total_min_output,
        "theta_total_max_output": theta_total_max_output,
        "omega_profile_output": omega_profile_output,
        "omega_mean_output": omega_mean_output,
        "dispersivity_profile_output": dispersivity_profile_output,
        "dispersivity_mean_output": dispersivity_mean_output,
        "initial_storage": initial_storage,
        "final_storage": final_storage,
        "cumulative_top_inflow": cumulative_top_inflow,
        "cumulative_bottom_outflow": cumulative_bottom_outflow,
        "mass_balance_error": mass_balance_error,
        "relative_mass_balance_error": relative_mass_balance_error,
        "accepted_steps": accepted_steps,
        "rejected_water_steps": rejected_water_steps,
        "rejected_solute_steps": rejected_solute_steps,
        "wall_time_minutes": wall_time_minutes,
        "tracked_average_Gamma": np.asarray(tracked_average_Gamma),
    }





# =============================================================================
# RUN CONSTANT REFERENCE AND SATURATION-DEPENDENT SIMULATION
# =============================================================================

# The constant reference run is part of the saturation-dependent formulation:
# its modeled state evolution defines the start/end coordinates used below.
constant_results = run_simulation(parameter_mode="constant")

OMEGA_SATURATION_START = float(
    constant_results["S_im_mean_output"][0]
)
OMEGA_SATURATION_END = float(
    constant_results["S_im_mean_output"][-1]
)

DISPERSIVITY_THETA_TOTAL_START = float(
    constant_results["theta_total_mean_output"][0]
)
DISPERSIVITY_THETA_TOTAL_END = float(
    constant_results["theta_total_mean_output"][-1]
)

if abs(OMEGA_SATURATION_END - OMEGA_SATURATION_START) <= 1.0e-12:
    raise RuntimeError(
        "Initial and final depth-averaged S_im are too close to define "
        "the variable-omega relationship."
    )

if abs(
    DISPERSIVITY_THETA_TOTAL_END
    - DISPERSIVITY_THETA_TOTAL_START
) <= 1.0e-12:
    raise RuntimeError(
        "Initial and final depth-averaged total water contents are too close "
        "to define the variable-dispersivity relationship."
    )

# Verify the normalized omega relationship at its two anchors.
omega_check_start = float(
    omega_from_saturation(
        np.array([OMEGA_SATURATION_START]),
        parameter_mode="omega_variable",
    )[0]
)
omega_check_end = float(
    omega_from_saturation(
        np.array([OMEGA_SATURATION_END]),
        parameter_mode="omega_variable",
    )[0]
)

if not np.isclose(
    omega_check_start,
    OMEGA_VALUE_START,
    atol=1.0e-15,
    rtol=0.0,
):
    raise RuntimeError(
        "Omega normalized-sigmoid start verification failed."
    )

if not np.isclose(
    omega_check_end,
    OMEGA_VALUE_END,
    atol=1.0e-15,
    rtol=0.0,
):
    raise RuntimeError(
        "Omega normalized-sigmoid end verification failed."
    )

# Verify that the variable-dispersivity function remains positive and finite.
lambda_check = dispersivity_from_total_water_content(
    np.array([
        DISPERSIVITY_THETA_TOTAL_START,
        0.5 * (
            DISPERSIVITY_THETA_TOTAL_START
            + DISPERSIVITY_THETA_TOTAL_END
        ),
        DISPERSIVITY_THETA_TOTAL_END,
    ]),
    parameter_mode="dispersivity_variable",
)

if np.any(~np.isfinite(lambda_check)) or np.any(lambda_check <= 0.0):
    raise RuntimeError(
        "Three-regime dispersivity verification failed."
    )

print("\n" + "=" * 84)
print("SATURATION-DEPENDENT PARAMETER ANCHORS")
print("=" * 84)
print(
    f"Omega S_im anchors:          "
    f"{OMEGA_SATURATION_START:.8f} -> {OMEGA_SATURATION_END:.8f}"
)
print(
    f"Total-theta anchors:         "
    f"{DISPERSIVITY_THETA_TOTAL_START:.8f} -> "
    f"{DISPERSIVITY_THETA_TOTAL_END:.8f}"
)
print(
    f"Omega values:                "
    f"{OMEGA_VALUE_START:.8e} -> {OMEGA_VALUE_END:.8e} 1/min"
)
print(
    f"Dispersivity values:         "
    f"{DISPERSIVITY_VALUE_EARLY:g} -> "
    f"{DISPERSIVITY_VALUE_MIDDLE:g} -> "
    f"{DISPERSIVITY_VALUE_LATE:g} cm"
)
print("=" * 84)

# Run the selected saturation-dependent formulation.
variable_results = run_simulation(parameter_mode=comparison_mode)

SIMULATION_RESULTS = {
    "constant": constant_results,
    comparison_mode: variable_results,
}


# =============================================================================
# SAVE NUMERICAL OUTPUTS
# =============================================================================

def safe_mode_name(parameter_mode):
    """Return a filesystem-safe mode label."""
    return parameter_mode.replace(" ", "_")


def save_simulation_outputs(simulation):
    """Save the principal numerical outputs for one simulation."""
    parameter_mode = simulation["parameter_mode"]
    mode_suffix = safe_mode_name(parameter_mode)
    prefix = f"{selected_case}_{mode_suffix}"

    # 1. Main regularly sampled model time series.
    time_series = pd.DataFrame({
        "time_min": simulation["output_time"],
        "bottom_flux_cm_per_min": simulation["bottom_flux_output"],
        "cumulative_bottom_outflow_cm": simulation["cumulative_flux_output"],
        "bottom_concentration": simulation["bottom_concentration_output"],
        "bottom_C_over_C0": simulation["BTC_output"],
        "theta_mobile_at_diagnostic_depth": simulation["theta_mo_4cm_output"],
        "theta_immobile_at_diagnostic_depth": simulation["theta_im_4cm_output"],
        "theta_total_at_diagnostic_depth": simulation["theta_total_4cm_output"],
        "theta_mobile_bottom": simulation["theta_mo_bottom_output"],
        "theta_immobile_bottom": simulation["theta_im_bottom_output"],
        "theta_total_bottom": simulation["theta_total_bottom_output"],
        "S_im_depth_average": simulation["S_im_mean_output"],
        "theta_total_depth_average": simulation["theta_total_mean_output"],
        "omega_depth_average_1_per_min": simulation["omega_mean_output"],
        "dispersivity_depth_average_cm": simulation[
            "dispersivity_mean_output"
        ],
    })
    time_series_file = OUTPUT_DIRECTORY / f"time_series_{prefix}.txt"
    time_series.to_csv(time_series_file, sep="\t", index=False)

    # 2. Bottom breakthrough curve.
    btc = pd.DataFrame({
        "time_min": simulation["output_time"],
        "bottom_concentration": simulation["bottom_concentration_output"],
        "C_over_C0": simulation["BTC_output"],
    })
    btc_file = OUTPUT_DIRECTORY / f"BTC_{prefix}.txt"
    btc.to_csv(btc_file, sep="\t", index=False)

    # 3. Bottom flux and cumulative water outflow.
    flux = pd.DataFrame({
        "time_min": simulation["output_time"],
        "bottom_flux_cm_per_min": simulation["bottom_flux_output"],
        "cumulative_bottom_outflow_cm": simulation["cumulative_flux_output"],
    })
    flux_file = OUTPUT_DIRECTORY / f"flux_{prefix}.txt"
    flux.to_csv(flux_file, sep="\t", index=False)

    # 4. State and parameter evolution through time.
    state_summary = pd.DataFrame({
        "time_min": simulation["output_time"],
        "S_im_min": simulation["S_im_min_output"],
        "S_im_depth_average": simulation["S_im_mean_output"],
        "S_im_max": simulation["S_im_max_output"],
        "theta_total_min": simulation["theta_total_min_output"],
        "theta_total_depth_average": simulation["theta_total_mean_output"],
        "theta_total_max": simulation["theta_total_max_output"],
        "omega_depth_average_1_per_min": simulation["omega_mean_output"],
        "dispersivity_depth_average_cm": simulation[
            "dispersivity_mean_output"
        ],
    })
    state_summary_file = OUTPUT_DIRECTORY / f"state_evolution_{prefix}.txt"
    state_summary.to_csv(state_summary_file, sep="\t", index=False)

    # 5. Full time-depth fields controlling the variable relationships.
    time_grid, depth_grid = np.meshgrid(
        simulation["output_time"],
        depth,
        indexing="ij",
    )
    state_fields = pd.DataFrame({
        "time_min": time_grid.ravel(),
        "depth_cm": depth_grid.ravel(),
        "S_im": simulation["S_im_profile_output"].ravel(),
        "theta_immobile": simulation["theta_im_profile_output"].ravel(),
        "theta_total": simulation["theta_total_profile_output"].ravel(),
        "omega_1_per_min": simulation["omega_profile_output"].ravel(),
        "dispersivity_cm": simulation[
            "dispersivity_profile_output"
        ].ravel(),
    })
    state_fields_file = OUTPUT_DIRECTORY / f"time_depth_fields_{prefix}.txt"
    state_fields.to_csv(state_fields_file, sep="\t", index=False)

    # 6. Initial/final depth profiles.
    final_profile = pd.DataFrame({
        "depth_cm": depth,
        "theta_mobile_initial": simulation["theta_mo_initial"],
        "theta_mobile_final": simulation["theta_mo_final"],
        "theta_immobile_initial": simulation["theta_im_initial"],
        "theta_immobile_final": simulation["theta_im_final"],
        "theta_total_initial": simulation["theta_total_initial"],
        "theta_total_final": simulation["theta_total_final"],
        "S_im_initial": simulation["S_im_initial"],
        "S_im_final": simulation["S_im_final"],
        "Se_mobile_final": simulation["Se_mo_final"],
        "omega_initial_1_per_min": simulation["omega_initial_nodes"],
        "omega_final_1_per_min": simulation["omega_final_nodes"],
        "dispersivity_initial_cm": simulation["dispersivity_initial_nodes"],
        "dispersivity_final_cm": simulation["dispersivity_final_nodes"],
        "Gamma_w_final_per_min": simulation["Gamma_final"],
    })
    profile_file = OUTPUT_DIRECTORY / f"final_profile_{prefix}.txt"
    final_profile.to_csv(profile_file, sep="\t", index=False)

    # 7. Compact run summary for reproducibility.
    run_summary = pd.DataFrame([{
        "case": selected_case,
        "case_label": case["label"],
        "parameter_mode": parameter_mode,
        "omega_reference_1_per_min": case["omega"],
        "dispersivity_reference_cm": case["dispersivity"],
        "omega_anchor_start_S_im": OMEGA_SATURATION_START,
        "omega_anchor_end_S_im": OMEGA_SATURATION_END,
        "dispersivity_anchor_start_theta_total": (
            DISPERSIVITY_THETA_TOTAL_START
        ),
        "dispersivity_anchor_end_theta_total": (
            DISPERSIVITY_THETA_TOTAL_END
        ),
        "column_length_cm": L,
        "dx_cm": dx,
        "simulation_time_min": t_end,
        "dt_initial_min": dt_initial,
        "dt_min_min": dt_min,
        "dt_max_min": dt_max,
        "accepted_steps": simulation["accepted_steps"],
        "rejected_water_steps": simulation["rejected_water_steps"],
        "rejected_solute_steps": simulation["rejected_solute_steps"],
        "wall_time_minutes": simulation["wall_time_minutes"],
        "initial_storage_cm": simulation["initial_storage"],
        "cumulative_top_inflow_cm": simulation["cumulative_top_inflow"],
        "cumulative_bottom_outflow_cm": simulation[
            "cumulative_bottom_outflow"
        ],
        "final_storage_cm": simulation["final_storage"],
        "mass_balance_error_cm": simulation["mass_balance_error"],
        "relative_mass_balance_error_percent": simulation[
            "relative_mass_balance_error"
        ],
    }])
    summary_file = OUTPUT_DIRECTORY / f"run_summary_{prefix}.txt"
    run_summary.to_csv(summary_file, sep="\t", index=False)

    return (
        time_series_file,
        btc_file,
        flux_file,
        state_summary_file,
        state_fields_file,
        profile_file,
        summary_file,
    )


saved_output_files = {
    mode_name: save_simulation_outputs(simulation)
    for mode_name, simulation in SIMULATION_RESULTS.items()
}


# =============================================================================
# CONCISE FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 84)
print("MODEL RUN COMPLETE")
print("=" * 84)

for mode_name, simulation in SIMULATION_RESULTS.items():
    print(
        f"{MODE_LABELS[mode_name]}: "
        f"max C/C0={np.max(simulation['BTC_output']):.6f}, "
        f"water-balance error="
        f"{simulation['relative_mass_balance_error']:.6f}%"
    )

print(f"\nOutputs written to:\n{OUTPUT_DIRECTORY}")
