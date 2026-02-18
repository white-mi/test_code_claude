"""Explainability module: SHAP, permutation importance, PDP, model cards."""

from dynamic_refitting.explainability.shap_explainer import ShapExplainer
from dynamic_refitting.explainability.importance import PermutationImportance
from dynamic_refitting.explainability.pdp import PartialDependence
from dynamic_refitting.explainability.counterfactual import CounterfactualGenerator
from dynamic_refitting.explainability.model_card import ModelCardGenerator

__all__ = [
    "ShapExplainer",
    "PermutationImportance",
    "PartialDependence",
    "CounterfactualGenerator",
    "ModelCardGenerator",
]
