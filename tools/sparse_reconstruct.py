"""Photos -> sparse 3D point cloud (PLY), CPU only, no COLMAP binary needed.

    pip install pycolmap
    python tools/sparse_reconstruct.py PHOTO_FOLDER OUTPUT_FOLDER

Writes OUTPUT_FOLDER/sparse_points.ply and prints how many photos registered.

What this is: structure-from-motion. It finds matching features across the
photos, solves for camera positions, and triangulates a SPARSE cloud (hundreds
to a few thousand points, a skeleton of the scene). Dense, solid surfaces need
COLMAP's patch-match stereo, which needs an NVIDIA GPU.

What this is not: a measurement. The cloud is in arbitrary units until printed
scale markers are in the photos and keystone-core's Geometry stage fits them
to the surveyed positions. Load the result with contracts.inputs.load_ply.

Rules of thumb for the photos: 20-40 per wall, 60-80% overlap between
neighbours, move the camera between shots (do not just rotate on the spot),
even light, no blur, and a textured surface (plain painted walls fail).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("images", help="folder of photos (.jpg / .png)")
    ap.add_argument("out", help="output folder")
    ap.add_argument("--tries", type=int, default=5,
                    help="mapping attempts; the best is kept (default 5)")
    ap.add_argument("--max-image-size", type=int, default=2000,
                    help="max image dimension for feature extraction (default 2000; avoids CPU RAM exhaustion)")
    ap.add_argument("--threads", type=int, default=0,
                    help="threads for feature extraction/matching (default 0 = auto-limit to min(4, CPU count))")
    ap.add_argument("--first-octave", type=int, default=0,
                    help="first octave for SIFT (default 0; -1 doubles resolution and consumes 4x RAM)")
    args = ap.parse_args()

    try:
        import pycolmap
    except ImportError:
        raise SystemExit("pycolmap is not installed:  pip install pycolmap")

    import os
    num_threads = args.threads if args.threads > 0 else min(4, os.cpu_count() or 1)

    images, out = Path(args.images), Path(args.out)
    if not images.is_dir():
        raise SystemExit(f"{images} is not a folder")
    out.mkdir(parents=True, exist_ok=True)
    db, sparse = out / "database.db", out / "sparse"
    sparse.mkdir(exist_ok=True)
    if db.exists():
        db.unlink()

    print(f"1/3 extracting features (max_size={args.max_image_size}px, threads={num_threads}, octave={args.first_octave}) ...")
    try:
        extract_opt = pycolmap.FeatureExtractionOptions()
        if args.max_image_size > 0 and hasattr(extract_opt, "max_image_size"):
            extract_opt.max_image_size = args.max_image_size
        if hasattr(extract_opt, "num_threads"):
            extract_opt.num_threads = num_threads
        if hasattr(extract_opt, "sift") and hasattr(extract_opt.sift, "first_octave"):
            extract_opt.sift.first_octave = args.first_octave
        extract_kwargs = {"extraction_options": extract_opt}
    except Exception:
        extract_kwargs = {}

    pycolmap.extract_features(db, images, **extract_kwargs)

    print(f"2/3 matching every photo against every other (threads={num_threads}) ...")
    try:
        match_opt = pycolmap.FeatureMatchingOptions()
        if hasattr(match_opt, "num_threads"):
            match_opt.num_threads = num_threads
        match_kwargs = {"matching_options": match_opt}
    except Exception:
        match_kwargs = {}

    pycolmap.match_exhaustive(db, **match_kwargs)

    # The mapping step starts from a randomly chosen pair of photos, so on a
    # marginal scene (a flat wall with little texture) two runs can differ a
    # lot. Matching is done once; mapping is cheap, so try several times and
    # keep the reconstruction that links the most photos, then the most points.
    print(f"3/3 solving cameras and triangulating points ({args.tries} tries, threads={num_threads}) ...")
    try:
        pipe_opt = pycolmap.IncrementalPipelineOptions()
        if hasattr(pipe_opt, "num_threads"):
            pipe_opt.num_threads = num_threads
        pipe_kwargs = {"options": pipe_opt}
    except Exception:
        pipe_kwargs = {}

    candidates = []
    for i in range(args.tries):
        tdir = sparse / f"try{i}"
        tdir.mkdir(exist_ok=True)
        maps = pycolmap.incremental_mapping(db, images, tdir, **pipe_kwargs)
        for r in maps.values():
            candidates.append(r)
        best = max((r.num_reg_images(), r.num_points3D()) for r in maps.values()) if maps else (0, 0)
        print(f"   try {i + 1}: {best[0]} photos, {best[1]} points")
    if not candidates:
        raise SystemExit("reconstruction failed: no photos could be linked. "
                         "Need more overlap and a textured surface.")
    rec = max(candidates, key=lambda r: (r.num_reg_images(), r.num_points3D()))
    maps = candidates
    n_photos = sum(1 for p in images.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    ply = out / "sparse_points.ply"
    rec.export_PLY(str(ply))

    print(f"\nregistered {rec.num_reg_images()} of {n_photos} photos "
          f"(best of {args.tries} tries)")
    print(f"{rec.num_points3D()} 3D points")
    try:
        print(f"mean reprojection error {rec.compute_mean_reprojection_error():.2f} px")
    except Exception:
        pass
    print(f"-> {ply}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
