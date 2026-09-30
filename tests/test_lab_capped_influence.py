"""A single observer must not be able to move another agent's reputation
without limit.

reputation.v1 sums every receipt outcome it is handed, so influence is
proportional to volume: one agent that never trades can drive a
counterparty's score arbitrarily low by writing receipts in a loop, and
reputation_consistent approves every step, because the arithmetic really
is internally consistent. Consistency was never the property under
attack.

The first test states the bound and fails against reputation.v1. The
second pins why the existing check cannot catch it.
"""

from nandatown.sim.api import TownAPI
from nandatown.sim.engine import Engine
from nandatown.sim.scenario import load_bundled
from nandatown.sim.validators import Trace, reputation_consistent


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


def test_one_observer_cannot_drive_reputation_unbounded():
    """The invariant: any one observer moves a score by at most 1."""
    _, api = slander("reputation.v1")
    assert api.reputation("seller-a") >= -1


def test_unbounded_slander_still_satisfies_the_reference_arithmetic_check():
    """The existing check approves the attack, so it cannot detect it.

    reputation_consistent replays the +1/-1 formula and finds it correct.
    That is a true statement about arithmetic and says nothing about
    whether one observer should have had that much influence.
    """
    engine, api = slander("reputation.v1")
    assert api.reputation("seller-a") == -5
    assert reputation_consistent(Trace(engine.events)).status == "passed"
