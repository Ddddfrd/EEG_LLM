"""Temporal alarm-policy research over frozen EEG probability timelines."""

from .artifacts import (
    PREDICTION_ARTIFACT_SCHEMA_VERSION,
    PredictionArtifact,
    load_prediction_artifact,
    save_content_addressed_json,
    save_prediction_artifact,
)
from .cohort import SelectionObjective, evaluate_cohort
from .contracts import EventInterval, ProbabilityTimeline
from .evaluator import (
    AlarmConfig,
    actions_from_probabilities,
    evaluate_alarm_actions,
    evaluate_probability_policy,
    voted_actions,
)
from .features import (
    EnrollmentStatistics,
    build_policy_observation,
    causal_probability_histories,
    compute_enrollment_statistics,
)
from .rules import (
    FixedRuleGrid,
    TemporalRule,
    apply_rule,
    default_threshold_grid,
    pareto_frontier,
    search_fixed_rules,
    select_operating_point,
)
from .splits import (
    DevelopmentGate,
    HeldOutAccessError,
    PartitionError,
    PolicySplit,
    base_model_role,
    scheme_c_model_split,
    validate_strict_subject_partition,
)
from .supervised import (
    SUPERVISED_FEATURE_NAMES,
    SupervisedControl,
    build_supervised_features,
    fit_logistic_control,
    fit_mlp_control,
)

__all__ = [
    "AlarmConfig",
    "DevelopmentGate",
    "EnrollmentStatistics",
    "EventInterval",
    "FixedRuleGrid",
    "HeldOutAccessError",
    "PREDICTION_ARTIFACT_SCHEMA_VERSION",
    "PartitionError",
    "PolicySplit",
    "PredictionArtifact",
    "ProbabilityTimeline",
    "SelectionObjective",
    "SUPERVISED_FEATURE_NAMES",
    "SupervisedControl",
    "TemporalRule",
    "actions_from_probabilities",
    "apply_rule",
    "base_model_role",
    "build_policy_observation",
    "build_supervised_features",
    "causal_probability_histories",
    "compute_enrollment_statistics",
    "default_threshold_grid",
    "evaluate_alarm_actions",
    "evaluate_cohort",
    "evaluate_probability_policy",
    "fit_logistic_control",
    "fit_mlp_control",
    "load_prediction_artifact",
    "pareto_frontier",
    "save_content_addressed_json",
    "save_prediction_artifact",
    "scheme_c_model_split",
    "search_fixed_rules",
    "select_operating_point",
    "validate_strict_subject_partition",
    "voted_actions",
]
