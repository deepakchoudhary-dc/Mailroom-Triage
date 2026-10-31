"""Decision metrics and calibration, vendored from Open-Jev.

Keeping this code in-repo makes calibration, evaluation and tests work from a
clean clone without requiring the Open-Jev repository as a sibling directory.
See THIRD_PARTY_NOTICES.md and LICENSES/Open-Jev-MIT.txt.
"""

from ._vendor_metrics import evaluate_probabilities, fit_temperature

__all__ = ["evaluate_probabilities", "fit_temperature"]
