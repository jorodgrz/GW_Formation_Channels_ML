"""
Multi-Code Ensemble Generation Module

This module provides a unified interface for generating population synthesis
ensembles across multiple codes (COMPAS, COSMIC, SEVN) to quantify epistemic
uncertainty from stellar evolution model systematics.
"""

from .unified_generator import UnifiedEnsembleGenerator, PopSynthCode

__all__ = ['UnifiedEnsembleGenerator', 'PopSynthCode']

