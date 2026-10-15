"""Routing - each severity goes to a DIFFERENT recipient.

The same person should not receive everything. If every decision lands on
the field officer who took the photographs, the system has only produced a
longer to-do list for the one person with the least authority to act on it.
So the recipient changes with what the decision actually requires:

    INFORMATIONAL  -> the monument's record.        Nobody is interrupted.
    WATCH          -> the capture schedule.         The next visit moves earlier.
    ALERT          -> the regional office's priority list.  A conservator is sent.
    CRITICAL       -> a structural engineer, with a recommendation that the
                      custodian restrict access.

AUTHORITY
---------
KEYSTONE measures and recommends. It has no legal authority to close a
monument or any part of one - that decision belongs to the custodian (for
centrally protected monuments, the Archaeological Survey of India). Every
action below is therefore worded as a recommendation, and pipeline.py puts
that wording into the Alert's `reason`.
"""
from __future__ import annotations

from dataclasses import dataclass

from contracts.schema import SEVERITIES

__all__ = ["Route", "ROUTES", "route_for"]


@dataclass(frozen=True)
class Route:
    severity: str
    recipient: str
    action: str           # what KEYSTONE recommends, in words a reader can act on
    notify: bool          # does a human get interrupted?


ROUTES: dict[str, Route] = {
    "INFORMATIONAL": Route(
        "INFORMATIONAL",
        "Monument record",
        "Recorded in the monument's file. No action recommended.",
        notify=False,
    ),
    "WATCH": Route(
        "WATCH",
        "Capture schedule",
        "Recommend bringing the next capture of this region forward (about half the usual "
        "interval) so that persistence can be confirmed or ruled out. No site visit is "
        "requested beyond the capture itself.",
        notify=False,
    ),
    "ALERT": Route(
        "ALERT",
        "Regional office priority list",
        "Recommend adding this region to the regional office's priority inspection list for an "
        "on-site assessment by a conservator who was not involved in the captures.",
        notify=True,
    ),
    "CRITICAL": Route(
        "CRITICAL",
        "Structural engineer + site closure recommendation",
        "Recommend an urgent assessment by a structural engineer. KEYSTONE recommends that the "
        "custodian consider restricting public access to the area around this region until it "
        "has been assessed. This is a recommendation only: KEYSTONE has no authority to close "
        "a monument, and the decision rests with the custodian.",
        notify=True,
    ),
}

assert tuple(ROUTES) == SEVERITIES, "every contract severity needs exactly one route"
assert len({r.recipient for r in ROUTES.values()}) == len(ROUTES), "recipients must differ"


def route_for(severity: str) -> Route:
    try:
        return ROUTES[severity]
    except KeyError:
        raise ValueError(f"unknown severity {severity!r}; expected one of {SEVERITIES}") from None


if __name__ == "__main__":
    print("ROUTING\n")
    for r in ROUTES.values():
        print(f"  {r.severity:<14} -> {r.recipient}{'   [notifies a person]' if r.notify else ''}")
        print(f"  {'':<14}    {r.action}\n")
