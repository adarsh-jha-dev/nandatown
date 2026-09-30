"""Trust layer: reputation with a per-observer influence cap.

reputation.v1 sums every receipt outcome it is handed, so influence is
proportional to how many receipts an agent writes. One agent that never
trades can drive a counterparty's score arbitrarily low by writing
receipts in a loop, and the arithmetic stays internally consistent the
whole way down. Consistency was never the property under attack.

The formula, recomputed from scratch on every update:

    score(subject) = sum over distinct observers of
                     weight(observer) * clamp(net(observer, subject), -1, +1)

    net(observer, subject) = goods - bads that observer reported about
                             subject
    weight(observer)       = 1 if observer was party to a settled trade,
                             else 0

So one observer is worth at most +/-1 no matter how many receipts it
files, and an observer with no trading history of its own is worth 0.
Both halves matter: the cap alone still lets a crowd of fresh identities
vote, and the weight alone still lets one established agent shout.

WHY THE WEIGHT READS payment_settled. The observer's own trade history
has to be evidence a replaying third party can see, and it must not be
self-certifiable. Receipts about the observer were the other candidate,
but in this town nobody ever attests about a buyer, so that definition
silences the honest buyer along with the slanderer and every score stays
0. A settled payment names both parties, which makes an honest buyer
count and a pure reviewer not count. This layer only reads those events;
it never writes to payments and does not change how money moves.

WHY THIS NEEDS A COMPANION VALIDATOR, NOT reputation_consistent.
reputation_consistent recomputes an unbounded running sum and requires
detail["score"] to equal it exactly, with delta in {+1, -1}. A clamped
score fails it by construction, and so does the delta of 0 this layer
emits for a discarded report. The tradeoff was: relax that check so both
formulas pass it, or leave it alone and add a second one. Relaxing it
would weaken a check that currently catches nineteen regressions in
tests/test_lab_reputation.py for every scenario that uses it, to
accommodate a formula those scenarios do not use, and it would silently
re-verify old marketplace bundles under looser rules. So the reference
check keeps its exact meaning and reputation_capped states the bounded
invariant separately. The cost is two checks to maintain, and a bundle
is only judged by whichever one its scenario selects: neither validates
the other layer's formula.

The event shape stays reputation_updated for every report, including
discarded ones, which then carry delta 0 and a reason. An "updated"
event that changed nothing is a little awkward, but dropping the event
would hide an attributed report from the trace, and the evidence rule
here is that a report that was made is recorded whether or not it
counted.
"""

from __future__ import annotations

from . import register


def clamp(value: int, low: int = -1, high: int = 1) -> int:
    return max(low, min(high, value))


@register("trust", "reputation.capped.v1")
class CappedReputation:
    """Reputation capped at +/-1 per observer, zero without trade history."""

    def __init__(self, engine):
        self.engine = engine
        # observer -> subject -> net goods minus bads, unclamped
        self.reports: dict[str, dict[str, int]] = {}

    # -- the formula ----------------------------------------------------

    def traded(self, observer: str) -> bool:
        """True if a settled payment in this run names the observer.

        Read-only over the event log, so a replaying validator reaches the
        same answer from events.jsonl alone.
        """
        for event in self.engine.events:
            if event.kind != "payment_settled":
                continue
            detail = event.detail if isinstance(event.detail, dict) else {}
            if observer in (detail.get("from"), detail.get("to")):
                return True
        return False

    def weight(self, observer: str) -> int:
        return 1 if self.traded(observer) else 0

    def score(self, name: str) -> int:
        total = 0
        for observer, subjects in self.reports.items():
            if name in subjects:
                total += self.weight(observer) * clamp(subjects[name])
        return total

    # -- the layer surface ----------------------------------------------

    def update(self, observer: str, subject: str, outcome: str,
               receipt_id: str) -> int:
        before = self.score(subject)
        net = self.reports.setdefault(observer, {})
        net[subject] = net.get(subject, 0) + (1 if outcome == "good" else -1)
        after = self.score(subject)
        delta = after - before

        weight = self.weight(observer)
        if weight == 0:
            reason = "observer_has_no_settled_trade"
        elif delta == 0:
            reason = "observer_cap_reached"
        else:
            reason = "counted"

        self.engine.emit(observer, "reputation_updated", subject,
                         {"outcome": outcome, "delta": delta,
                          "score": after, "receipt": receipt_id,
                          "formula": "reputation.capped.v1",
                          "observer_weight": weight,
                          "observer_net": clamp(net[subject]),
                          "reason": reason})
        return after
