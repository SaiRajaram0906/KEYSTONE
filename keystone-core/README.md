# keystone-core

Shared contract and acceptance tests for KEYSTONE's three build workstreams.

- `WORKPLAN.md` — the split, rules, and integration checkpoints
- `HANDOVER_PRESERVATION.md` — paste into a new chat to start Workstream P
- `HANDOVER_GEOMETRY.md` — paste into a new chat to start Workstream G
- `HANDOVER_DECISION.md` — Workstream D (already built, see `decision/`)

```
contracts/
  schema.py         the G -> D boundary: ChangeReport, Alert, validators
  inputs.py         Geometry's input format: raw captures + surveyed markers
  preservation.py   the P boundaries: registry, capture QC, archive, export
  crack_adapter.py  plugs the crack-seg team's output in, without coupling
  fakes.py          scenarios with known ground truth for G and D
  fakes_capture.py  capture sessions with known defects for P
decision/           Workstream D, complete
tests/
  contracts_test.py acceptance tests — PENDING until each workstream lands
```

```bash
python -m contracts.fakes            # the Geometry and Decision scenarios
python -m contracts.fakes_capture    # the seven capture scenarios
python -m tests.contracts_test       # run everything
python -m tests.contracts_test --only P    # just one workstream
```
