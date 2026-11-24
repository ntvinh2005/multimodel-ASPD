from collections import Counter

from aspd.multimodel.cli.measure_induction_query import (
    CONDITIONS,
    FAMILIES,
    MODEL_ORDER,
    SITES,
    _select,
)
from aspd.multimodel.posthoc.ids import GateId


def test_discovery_shortlist_balances_sites_families_and_control_profiles():
    rows = {}
    # Four features contrast clean/counterfactual with controls, two fire everywhere.
    for family in FAMILIES:
        for index in range(8):
            group = f"{family}/{index}"
            for condition in CONDITIONS:
                positive = condition in {"induction", "counterfactual"}
                gates = list(range(7)) if positive else [4, 5]
                values = [1.0 + gate / 10 for gate in gates]
                rows[f"{group}/{condition}"] = {
                    "group_id": group,
                    "family": family,
                    "condition": condition,
                    "pool_id": family,
                    **{site: {"active_gate_ids": gates, "values": values} for site in SITES},
                }
    reps = {
        (GateId(site, gate), model): (object(),)
        for site in SITES
        for gate in range(6)
        for model in MODEL_ORDER
    }
    selection, _ = _select(rows, reps)
    reverse, _ = _select(dict(reversed(list(rows.items()))), reps)
    assert selection == reverse
    assert Counter(gate.site for gate in selection) == {"pre": 6, "mid": 6}
    assert Counter(value["kind"] for value in selection.values()) == {
        "induction_contrast": 8,
        "fires_across_conditions": 4,
    }
    for gate, record in selection.items():
        assert gate.feature in range(6)
        assert Counter(group.split("/")[0] for group in record["groups"]) == {
            family: 4 for family in FAMILIES
        }
