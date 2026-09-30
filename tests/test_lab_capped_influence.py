"""A single observer must not be able to move another agent's reputation
without limit.

reputation.v1 sums every receipt outcome it is handed, so influence is
proportional to volume: one agent that never trades can drive a
counterparty's score arbitrarily low by writing receipts in a loop, and
reputation_consistent approves every step, because the arithmetic really
is internally consistent. Consistency was never the property under
attack.

reputation.capped.v1 caps any one observer at a single point and gives an
observer with no settled trade of its own no weight at all. Both halves
matter: the cap alone still lets a crowd of fresh identities vote, and the
weight alone still lets one established agent shout.
"""

import pytest

from nandatown.sim.api import TownAPI
from nandatown.sim.engine import Engine
from nandatown.sim.scenario import load_bundled
from nandatown.sim.validators import (
    Trace, reputation_capped, reputation_consistent,
)

CAPPED = "reputation.capped.v1"


def slander(trust_plugin: str, times: int = 5):
    """One observer with no settled trade rates one seller bad, repeatedly."""
    spec = load_bundled("marketplace")
    spec.layers["trust"] = trust_plugin
    engine = Engine(spec)
    engine.layers["identity"].create("slanderer")
    api = TownAPI(engine, "slanderer")
    for _ in range(times):
        api.rate("seller-a", "bad")
    return engine, api


@pytest.mark.parametrize("trust_plugin,want", [
    (CAPPED, 0),           # no settled trade: the reports carry no weight
    ("reputation.v1", -5),  # the gap this change addresses, kept pinned
])
def test_one_observer_with_no_trade_history(trust_plugin, want):
    """Five reports from one voice. Bounded moves nothing; v1 moves five."""
    _, api = slander(trust_plugin)
    assert api.reputation("seller-a") == want


def test_unbounded_slander_still_satisfies_the_reference_arithmetic_check():
    """The existing check approves the attack, so it cannot detect it.

    reputation_consistent replays the +1/-1 formula and finds it correct.
    That is a true statement about arithmetic and says nothing about
    whether one observer should have had that much influence.
    """
    engine, _ = slander("reputation.v1")
    assert reputation_consistent(Trace(engine.events)).status == "passed"


def test_an_established_observer_is_still_capped_at_one_point():
    """The weight rule alone would let one trading agent shout."""
    spec = load_bundled("marketplace")
    spec.layers["trust"] = CAPPED
    engine = Engine(spec)
    engine.layers["identity"].create("buyer-real")
    engine.emit("town", "payment_settled", "order-x",
                {"from": "buyer-real", "to": "seller-a", "cents": 10,
                 "via": "escrow"})
    api = TownAPI(engine, "buyer-real")
    for _ in range(4):
        api.rate("seller-a", "good")
    assert api.reputation("seller-a") == 1
    deltas = [e.detail["delta"] for e in engine.events
              if e.kind == "reputation_updated"]
    assert deltas == [1, 0, 0, 0], "only the first report may move the score"


def test_distinct_observers_still_accumulate():
    """The cap is per observer, not a cap on the score itself."""
    spec = load_bundled("marketplace")
    spec.layers["trust"] = CAPPED
    engine = Engine(spec)
    for who in ("buyer-x", "buyer-y"):
        engine.layers["identity"].create(who)
        engine.emit("town", "payment_settled", f"order-{who}",
                    {"from": who, "to": "seller-a", "cents": 10,
                     "via": "escrow"})
        TownAPI(engine, who).rate("seller-a", "good")
    assert engine.layers["trust"].score("seller-a") == 2


def test_a_discarded_report_is_still_recorded():
    """A report that did not count must not vanish from the trace."""
    engine, _ = slander(CAPPED, times=2)
    updates = [e for e in engine.events if e.kind == "reputation_updated"]
    assert len(updates) == 2
    for event in updates:
        assert event.detail["delta"] == 0
        assert event.detail["observer_weight"] == 0
        assert event.detail["reason"] == "observer_has_no_settled_trade"


# -- the companion validator --------------------------------------------


def mixed_trace(trust_plugin: str):
    """One observer that settled a trade, one that never did."""
    spec = load_bundled("marketplace")
    spec.layers["trust"] = trust_plugin
    engine = Engine(spec)
    for who in ("buyer-real", "slanderer"):
        engine.layers["identity"].create(who)
    engine.emit("town", "payment_settled", "order-x",
                {"from": "buyer-real", "to": "seller-a", "cents": 10,
                 "via": "escrow"})
    TownAPI(engine, "buyer-real").rate("seller-a", "good")
    api = TownAPI(engine, "slanderer")
    for _ in range(5):
        api.rate("seller-a", "bad")
    return engine


def test_capped_check_passes_a_capped_trace():
    engine = mixed_trace(CAPPED)
    stage = reputation_capped(Trace(engine.events))
    assert stage.status == "passed", stage.note
    assert engine.layers["trust"].score("seller-a") == 1


def test_capped_check_names_the_full_movement_it_rejects():
    """The note must carry the magnitude, not just the first point of it."""
    stage = reputation_capped(Trace(mixed_trace("reputation.v1").events))
    assert stage.status == "failed"
    assert "-5" in stage.note, stage.note
    assert "slanderer" in stage.note


def test_empty_capped_check_is_missing_not_success():
    assert reputation_capped(Trace([])).status == "not_enough_evidence"


def test_capped_check_does_not_judge_the_reference_formula():
    """The two checks stay separate: neither validates the other's rules."""
    events = mixed_trace(CAPPED).events
    # A capped trace fails the unbounded arithmetic check by construction,
    # which is exactly why reputation_consistent was left alone.
    assert reputation_consistent(Trace(events)).status == "failed"
    assert reputation_capped(Trace(events)).status == "passed"
