# HANDOVER — KEYSTONE Workstream G (Geometry)

*Paste everything below into a new chat, and attach the `keystone-core` folder (zipped).*

---

I'm on a team building **KEYSTONE**, a system for autonomous structural monitoring of heritage monuments. I own **Workstream G — Geometry**. I need help building my part. The shared contract and tests already exist; I'm attaching the project folder.

## The project in one paragraph

India has 3,698 centrally protected monuments, and most are inspected by eye — a conservator looks at a wall and writes "minor cracking". Nothing is measured, so slow structural movement goes unnoticed until it's obvious. KEYSTONE photographs a monument with a phone or a cheap drone, reconstructs a measured 3D point cloud, repeats the capture months later, and detects what moved. Photogrammetry reconstruction itself is out of scope — we start from point clouds.

## The three teams

- **Geometry (me):** takes two raw point clouds, outputs how far each region of the wall moved, in millimetres.
- **Decision (another team):** takes my output over many captures, separates seasonal movement from real structural movement, raises alerts.
- **Crack segmentation (a separate project):** measures crack widening from photos. Not my concern — its output reaches Decision through an adapter.

## My job

Implement one function in a new file `geometry/pipeline.py`:

```python
build_change_report(earlier: CaptureInput, later: CaptureInput,
                    survey: dict[str, list[float]],
                    regions: list[Region]) -> ChangeReport
```

**The input problem.** Each capture arrives in arbitrary units, position and orientation — that's what photogrammetry produces. The only link to the real world is a set of ground-control markers (GCPs): I get each marker's position *in the cloud* and its surveyed position *in millimetres*.

**The steps I need to build:**

1. **Scale and coarse alignment** — fit a similarity transform (scale + rotation + translation) from each capture's marker positions to the surveyed positions. The Umeyama method does this in closed form. This puts both clouds in the same frame, in millimetres, and gives `mm_per_unit`.
2. **Fine registration** — refine with ICP (Open3D) using only stable areas, so that real movement doesn't get "aligned away".
3. **Change measurement** — a simplified **M3C2**: for core points in the reference cloud, estimate the local surface normal, then measure the distance between the two clouds along that normal inside a small cylinder. Plain nearest-neighbour distance is wrong here — masonry has recessed mortar joints, and nearest-neighbour picks up surface roughness as movement.
4. **Level of detection** — for each core point, the smallest change distinguishable from noise: combine the local roughness of both clouds with the registration error. This is the part that makes the output trustworthy.
5. **Region summary** — for each `Region` box, report median displacement, 95th percentile of absolute displacement, median LoD, and the fraction of points that moved significantly.
6. **Registration error** — report RMSE of marker residuals after alignment.

## Rules I must follow

- Do not edit anything in `contracts/` — the other team depends on it.
- Everything I output is in **millimetres**.
- `earlier` is always the monument's **reference** (first) capture.
- My output must pass `validate_change_report` and survive `to_json` / `from_json`.

## How I know I'm done

```bash
python -m tests.contracts_test --only G
```

Six checks, against synthetic walls with known ground truth from `contracts/fakes.py`:
- recovers scale within 1%
- measures a 12 mm outward displacement within 2.5 mm
- stable regions read below 2 mm
- LoD is honest (stable regions sit inside it; few points flagged)
- an unchanged wall reads zero
- registration error is reported and plausible

A crude reference solution passed all six with margin, so the bar is fair.

## Setup (Windows)

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements-geometry.txt
python -m contracts.fakes
python -m tests.contracts_test
```

Python 3.13 is fine — Open3D 0.20 ships wheels for 3.10 through 3.14.

## Suggested structure

```
geometry/
  __init__.py
  align.py       Umeyama similarity fit, apply transforms
  register.py    ICP refinement on stable regions
  m3c2.py        normals, cylinder projection, per-point LoD
  regions.py     per-region aggregation
  pipeline.py    build_change_report — wires the above together
```

## What to write up

- Why nearest-neighbour distance fails on masonry and M3C2 doesn't.
- How the level of detection is computed, and what it means: "we can't see movement smaller than this".
- Stretch: run it on a real pair — photograph a wall twice with a phone app like Polycam, place printed markers, export PLY, load with `contracts.inputs.load_ply`. Report what came out, honestly, whatever it is.

Start by walking me through step 1 (Umeyama) and giving me a complete `geometry/align.py`.
