"""ML-based financial reconciliation, after Chew (ICAIF 2020).

Stages: unique-amount matching -> QC -> cross-language SVD matching -> QC -> manual,
with every validated match fed back into the training corpus.
"""
from .data import bank_frame, id_ground_truth, load_bank, load_register, register_frame
from .matcher import MatcherConfig, MLMatcher, TrainingPair, pairs_from_frames
from .metrics import Scores, evaluate
from .svd_model import AlignedSVDModel, SVDConfig
from .unique import UniqueConfig, unique_amount_matches
from .workflow import (InteractiveReviewer, OracleReviewer, ReconciliationSession, ThresholdReviewer,
                       TrainingStore, no_resolver, oracle_resolver)

__all__ = [
    "AlignedSVDModel", "InteractiveReviewer", "MLMatcher", "MatcherConfig", "OracleReviewer",
    "ReconciliationSession", "SVDConfig", "Scores", "ThresholdReviewer", "TrainingPair", "TrainingStore",
    "UniqueConfig", "bank_frame", "evaluate", "id_ground_truth", "load_bank", "load_register",
    "no_resolver", "oracle_resolver", "pairs_from_frames", "register_frame", "unique_amount_matches",
]
__version__ = "0.1.0"
