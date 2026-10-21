"""Direction-fit residual operators with symmetric fit/evaluation transforms."""

from .operators import (
    GeometryResidualizer,
    OperatorError,
    ResidualOperator,
    fit_geometry_residualizer,
    fit_residual_operator,
)

__all__ = [
    "GeometryResidualizer",
    "OperatorError",
    "ResidualOperator",
    "fit_geometry_residualizer",
    "fit_residual_operator",
]
