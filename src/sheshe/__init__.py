from .sheshe import ModalBoundaryClustering, ClusterRegion
from .subspace_scout import SubspaceScout
from .modal_scout_ensemble import ModalScoutEnsemble
from .region_interpretability import RegionInterpreter
from .meta_optimization import random_search
from .combiantions import find_comb_dim_spaces, plot_rule_metrics, select_ruleset_or_greedy
from .inside_forest import (
    InsideForest,
    InsideForestClassifier,
    InsideForestRegressor,
    Region,
    Rule,
)
from .shushu import ShuShu
from .cheche import CheChe
from .chuchu import (
    Chuchu,
    ChuchuClassifier,
    ChuchuConfig,
    ChuchuRegressor,
    ChangePointConfig,
    DeltaRecord,
)

# ``OpenAIRegionInterpreter`` relies on the optional ``openai`` dependency.  In
# environments where that dependency (or the module itself) is missing we still
# want the base package to be importable.  Import lazily and fall back to a
# ``None`` placeholder so that ``from sheshe import OpenAIRegionInterpreter``
# works even when the optional components are unavailable.
try:  # pragma: no cover - exercised via import side effect
    from .openai_text import OpenAIRegionInterpreter  # type: ignore
except Exception:  # pragma: no cover - optional dependency not installed
    OpenAIRegionInterpreter = None  # type: ignore

__all__ = [
    "ModalBoundaryClustering",
    "ClusterRegion",
    "SubspaceScout",
    "ModalScoutEnsemble",
    "RegionInterpreter",
    "random_search",
    "OpenAIRegionInterpreter",
    "ShuShu",
    "CheChe",
    "Chuchu",
    "ChuchuClassifier",
    "ChuchuConfig",
    "ChuchuRegressor",
    "ChangePointConfig",
    "DeltaRecord",
    "InsideForest",
    "InsideForestClassifier",
    "InsideForestRegressor",
    "Region",
    "Rule",
    "find_comb_dim_spaces",
    "plot_rule_metrics",
    "select_ruleset_or_greedy",
]

__version__ = "0.1.3"
