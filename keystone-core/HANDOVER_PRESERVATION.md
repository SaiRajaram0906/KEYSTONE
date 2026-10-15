# HANDOVER — KEYSTONE Workstream P (Capture & Preservation)

*Paste everything below into a new chat, and attach `keystone-core.zip`.*

---

I'm building one part of a project called KEYSTONE. I own the CAPTURE & PRESERVATION part. The shared contract, fake data and acceptance tests already exist — I'm attaching the project folder. I'm on Windows using VS Code.

Please guide me step by step starting from unzipping the folder, through setup, then build my part ONE COMPLETE FILE AT A TIME (no snippets, no TODO placeholders). Give me one terminal command at a time and tell me what output I should see before moving on.

=== WHAT KEYSTONE IS ===
India has 3,698 centrally protected monuments, and most are inspected by eye — a conservator looks at a wall and writes "minor cracking". Nothing is measured, so slow structural movement goes unnoticed until it's obvious and expensive.

KEYSTONE photographs a monument with a phone or a cheap drone, turns the photos into a measured 3D point cloud, repeats the capture months later, and automatically detects what moved, decides whether it matters, and tells the right person. It also keeps every capture forever, so if the monument is ever lost, a measured record of it survives.

=== THE FOUR PARTS ===
1. Crack segmentation — a separate team. Finds cracks in photos and measures widening.
2. Geometry — another person. Two 3D scans in, "how many mm did each region move" out.
3. Decision — another person (already finished). Separates seasonal movement from real structural damage and raises alerts.
4. CAPTURE & PRESERVATION — me. The front door and the back door of the whole system.

I don't wait for anyone. I build against synthetic data in contracts/fakes_capture.py.

=== MY JOB, AND WHY IT MATTERS ===

FRONT DOOR. Before a monument can be measured, someone has to describe it: where its surveyed markers are, which regions matter, how big the face is. And when a conservator photographs it, someone has to check the photo set BEFORE they leave the site. A capture that fails quality checks back at the office means another trip to a remote monument — which in practice means it never gets re-measured at all. That check is the single most valuable thing I build.

BACK DOOR. Every capture, report and alert is kept, because the archive is half of what KEYSTONE promises. That archive only ever grows, so it needs a retention policy. And a record is only worth keeping if something can be got back out of it — plans, elevations and sections.

MIDDLE. I also own the handoff to Geometry. Running the photogrammetry itself is an off-the-shelf command (COLMAP, or a phone app like Polycam) and is not the interesting part. But the cloud that comes out is in arbitrary units and arbitrary orientation, and the ONLY thing tying it to the real world is where the surveyed markers sit inside it. Finding and identifying those markers is not off-the-shelf, and nobody else in the project does it. If I get it wrong, every millimetre Geometry reports afterwards is wrong too.

=== WHAT I BUILD ===
Create preservation/pipeline.py exposing exactly four things:

1. check_capture(session: CaptureSession, monument: MonumentRecord) -> CaptureQC
2. locate_markers(points, monument: MonumentRecord) -> dict[marker_id, ndarray]
       Find the surveyed targets inside a raw reconstructed cloud and say
       which is which. See "Finding the markers" below.
3. class Archive(root) with methods:
       store(data: bytes, *, monument_id, kind, created_on, capture_id=None, is_reference=False) -> ArchiveEntry
       fetch(artefact_id) -> bytes          # raises if the bytes changed
       verify(artefact_id) -> bool
       path_for(artefact_id) -> Path        # where the bytes live
       entries(monument_id) -> list[ArchiveEntry]
       stats(monument_id) -> ArchiveStats
       apply_retention(monument_id, keep_full_days=365) -> int   # bytes reclaimed
4. extract_section(points, monument, capture_id, kind, at_mm, thickness_mm) -> Section
       kind is "plan" | "elevation" | "section"

=== THE QUALITY GATE — THE HARD PART ===
Six things go wrong with a field capture, and each needs a DIFFERENT check. No one or two checks catch them all:

  too_few         shot from too far back. Coverage and overlap both fine, but
                  every pixel spans 1.5 mm of stone, so a 3 mm crack is
                  barely more than a pixel. Caught by GROUND SAMPLING
                  DISTANCE = footprint width in mm / image width in px.
  thin_overlap    THE TRAP. Every part of the wall appears in some photo, so
                  a "is it covered?" check passes. But almost no point
                  appears in 3+ photos, and photogrammetry triangulates each
                  point from multiple views. Caught by REDUNDANCY — the
                  fraction of the face seen from three or more positions.
  blurry          handheld in low light, a third of frames shaken. Caught by
                  SHARPNESS (variance of the Laplacian). The absolute value
                  depends on the subject, so compare within a session
                  (e.g. against the session median), not against a constant.
  overexposed     shot into the sun; highlights blown, no texture left for
                  feature matching. Caught by EXPOSURE CLIPPING.
  markers_missing framed tight on the stonework, so fewer than 3 surveyed
                  markers ever appear. The cloud can never be scaled or
                  placed. Caught by the UNION of markers across all photos.
  dark_gap        the lower east face was never photographed. This one is
                  SALVAGEABLE — most of the wall is fine — so it must be
                  PASS_WITH_GAPS naming the uncovered region, not RESHOOT.
                  Rejecting it wastes a visit; passing it silently makes
                  Geometry report a region it cannot see.

Three verdicts: PASS, PASS_WITH_GAPS, RESHOOT.

RESHOOT must come with specific guidance, because the operator is still standing at the monument. "Poor quality" sends them home none the wiser. "8 frames blurred from camera shake; re-shoot C001-P003, C001-P006..." does not.

=== FINDING THE MARKERS — THE OTHER HARD PART ===
A printed target on the wall reconstructs as a small dense disc standing slightly proud of the face. Two steps:

  LOCATE   The disc is far denser than the surrounding masonry, so local
           density finds it. Careful: the cloud is at an UNKNOWN scale, so
           a fixed search radius in millimetres is meaningless. Use
           something scale-free, like the distance to each point's k-th
           nearest neighbour, then cluster the densest points.

  IDENTIFY This is where it gets interesting. My six markers sit in an
           arrangement that is symmetric under a 180-degree rotation, so
           position alone CANNOT tell GCP-1 from GCP-4 — the constellation
           fits equally well upside down, and every measurement then comes
           out mirrored. I verified this: matching by position alone gave a
           convincing 2 mm fit and marker positions 2.9 METRES wrong.

           Real surveys solve it with coded or differently sized targets.
           Mine are sized 30, 36, 42, 48, 54 and 60 mm (MarkerSpec.
           target_radius_mm). Radii are unknown in raw units, but their
           ORDER is scale-free, so sorting both lists and pairing them up
           works.

=== DATA FORMAT (contracts/preservation.py — I must NOT edit anything in contracts/) ===
MarkerSpec: marker_id, position_mm [x,y,z], accuracy_mm, description, target_radius_mm
MonumentRecord: monument_id, name, markers, regions, face_size_mm, reference_capture_id, lat, lon
    .survey() returns the {marker_id: [x,y,z]} dict Geometry expects
PhotoMeta: photo_id, width, height, sharpness, exposure_clip, footprint_mm [x0,z0,x1,z1],
    markers_visible, iso, path
CaptureSession: capture_id, monument_id, captured_on, photos, device, temperature_c,
    rainfall_mm_7d, operator
CheckResult: name, passed, detail, value, threshold
CaptureQC: capture_id, monument_id, verdict, checks, coverage_fraction, weak_region_ids,
    reshoot_guidance
ArchiveEntry: artefact_id, monument_id, kind, created_on, size_bytes, content_hash,
    density, is_reference, capture_id
ArchiveStats: monument_id, n_entries, total_bytes, reference_bytes, reclaimable_bytes,
    oldest, newest
Section: monument_id, capture_id, kind, at_mm, thickness_mm, polylines, extent_mm, n_points

Frame: x along the face, y outward, z up. Millimetres everywhere.

=== RULES ===
- Do not edit anything in contracts/ — three other people build against it.
- The REFERENCE capture is never decimated. Every future comparison is made
  against it, so if it degrades, the whole measurement history is worthless.
  validate_archive_entry() enforces this.
- The archive must be able to PROVE the bytes read back are the bytes written.
  An archive that can't is a folder, not a record. Use a content hash.
- PASS cannot list weak regions; PASS_WITH_GAPS must. The validator rejects both mistakes.
- Everything must survive to_json() / from_json().

=== TEST DATA ===
contracts/fakes_capture.py
  make_monument()                     the 6 m x 3 m wall, 6 markers, 6 regions
  make_session(scenario, seed)        one capture session + the truth about it
  make_wall_cloud(seed)               the wall as a point cloud in survey mm,
                                      with a 400 mm buttress between x=2000-2400
  make_capture_cloud(monument, seed)  a RAW reconstructed cloud: random scale
                                      and pose, with the six targets embedded
                                      as dense discs. Returns the truth too.
  coverage_grid(session)              how many photos see each 100 mm cell —
                                      a reference implementation of the idea
                                      the gate is built on
  write_sample_images(dir)            REAL JPEGs (sharp / blurry / blown) so I
                                      can write the actual sharpness and
                                      exposure extractors against real pixels

Run `python -m contracts.fakes_capture` first. It prints a table of all seven
scenarios. Look at the thin_overlap row: coverage 1.00, redundancy 0.04.

=== HOW I KNOW I'M DONE ===
python -m tests.contracts_test --only P    → 13 of 13 pass:
- registry is valid and gives Geometry its survey
- markers are found and identified in a raw cloud (5 seeds, within 25 mm)
- located markers recover the true scale within 1%
- QC output valid for every scenario, JSON-safe
- good captures are never sent back (12 seeds) — the false-positive test
- every defect type is caught (5 defects x 6 seeds)
- a salvageable capture is flagged, not rejected
- reshoot guidance says what to re-shoot
- archive round-trips and verifies
- archive detects a single flipped byte
- retention reclaims space but spares the reference
- elevation comes out at the wall's real size (6000 x 3000 mm)
- plan view shows depth an elevation cannot (the 400 mm buttress)

A crude reference implementation passed all 13, so the bar is reachable
(its worst marker error was 9 mm against the 25 mm the test allows).
A naive gate that checks only photo count and coverage let 4 of 5 defect
types through AND rejected the salvageable capture.

=== MY SETUP ===
Windows, VS Code, PowerShell. Only Python 3.13 installed (use py -3.13).
  py -3.13 -m venv .venv
  .venv\Scripts\activate
  (if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned, then activate again)
  pip install -r requirements-preservation.txt
  python -m contracts.fakes_capture
  python -m tests.contracts_test        → should show P pending, others passing

=== FILES TO BUILD, IN ORDER ===
preservation/__init__.py
preservation/photo_metrics.py   sharpness + exposure from real image files
preservation/markers.py         locate and identify targets in a raw cloud
preservation/coverage.py        coverage, redundancy, ground sampling distance,
                                per-region coverage
preservation/gate.py            the six checks, the verdict, the guidance
preservation/registry.py        create / load / save monument records, and the
                                write-once rule for reference_capture_id
preservation/archive.py         content-addressed store, integrity, retention
preservation/export.py          slice a point cloud into plan / elevation / section
preservation/reconstruct.py     run the off-the-shelf photogrammetry tool and
                                load its output, or load a phone app's PLY
preservation/pipeline.py        check_capture, locate_markers, Archive,
                                extract_section — what the tests import
preservation/cli.py             optional: a field tool that runs the gate on a
                                folder of photos and prints the verdict

=== WHAT I WRITE UP AT THE END ===
- Why each of the six checks exists, and which defect it catches
- The naive-gate comparison: 4 of 5 defects missed, 1 good capture wasted
- The retention policy, with actual numbers: what the archive costs per
  monument per year, and what tiering saves. This is the storage question
  the project has never answered.
- Honest limitation: footprint_mm is given in the fakes. A real build gets it
  from the capture app's pose estimate or a coarse match, and that estimate
  has its own error which I have not modelled.
- Honest limitation: the sharpness threshold is relative to the session
  median, which fails if the WHOLE session is blurred. Real deployments need
  an absolute floor calibrated per device.
- Honest limitation: my targets are found by density in a clean synthetic
  cloud. A real reconstruction has noise, vegetation and holes, and the
  densest clusters are not always the targets.
- Worth reporting: the symmetry result above. Matching markers by position
  alone gave a 2 mm fit that was 2.9 m wrong, which is a good argument for
  why target size or coding is not optional.

Start with Step 1 of setup.
