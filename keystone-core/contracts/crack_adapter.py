"""Bridge from the crack-seg team's output into the KEYSTONE contract.

The crack-seg project produces a CrackDelta. We do not import their code:
we accept the plain dict from CrackDelta.as_dict(), so the two codebases
stay decoupled and either can change internally without breaking the other.
The dict is the whole agreement.

Their dict carries: d_max_width, d_mean_width, d_total_length, d_area,
new_crack_area, units, earlier, later.

Their width estimate has +/- 1 px quantisation error (documented in their
keystone/interface.py), so the adapter converts that into millimetres and
attaches it. Decision must ignore widening smaller than that uncertainty.
"""
from __future__ import annotations

from contracts.schema import ContractError, CrackChange

__all__ = ["from_crack_delta"]


def from_crack_delta(delta: dict, region_id: str, mm_per_px: float) -> CrackChange:
    """Convert one crack-seg CrackDelta.as_dict() into a CrackChange.

    Args:
        delta: output of CrackDelta.as_dict(). Must be in millimetres; a
            delta computed without mm_per_px is in pixels and is rejected,
            because pixel deltas are not comparable between captures.
        region_id: which KEYSTONE region the photographed surface belongs to.
        mm_per_px: the image scale used for that capture, which sets the
            size of the +/- 1 px uncertainty in millimetres.
    """
    required = ("d_max_width", "new_crack_area", "units")
    missing = [k for k in required if k not in delta]
    if missing:
        raise ContractError(f"crack delta missing keys {missing}")
    if delta["units"] != "mm":
        raise ContractError(
            f"crack delta is in {delta['units']!r}, not 'mm'. The crack team must "
            "call compare_captures(..., mm_per_px=...) so results are physical."
        )
    if not mm_per_px > 0:
        raise ContractError("mm_per_px must be positive")

    return CrackChange(
        region_id=region_id,
        d_max_width_mm=float(delta["d_max_width"]),
        new_crack_area_mm2=max(0.0, float(delta["new_crack_area"])),
        # One pixel of quantisation, expressed in millimetres.
        width_uncertainty_mm=float(mm_per_px),
    )


if __name__ == "__main__":
    sample = {"d_max_width": 2.0, "d_mean_width": 1.99, "d_total_length": 29.5,
              "d_area": 300.0, "new_crack_area": 292.5, "units": "mm",
              "earlier": {}, "later": {}}
    c = from_crack_delta(sample, "north_upper", mm_per_px=0.5)
    print(c)
    try:
        from_crack_delta({**sample, "units": "px"}, "north_upper", 0.5)
        print("MISSED: pixel delta accepted")
    except ContractError:
        print("rejected: pixel delta")
