"""FOCUS: Fused Ordinal-CORN with Uncertainty-weighted Scoring.

A frozen-backbone architecture for point-wise terrain traversability scoring,
trained on sparse pairwise comparisons under the TOLRIZZ loss. See the
top-level README and chapters/05-method.tex (in the thesis this repo
accompanies) for the full description.
"""

from focus.loss import tolrizz_loss
from focus.model import FocusHead, FusionEncoder
from focus.train import focus_factory

__all__ = ["FocusHead", "FusionEncoder", "tolrizz_loss", "focus_factory"]
