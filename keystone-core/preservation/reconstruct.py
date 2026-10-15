"""Thin adapters around off-the-shelf photogrammetry and PLY exports."""
from __future__ import annotations
import subprocess
from pathlib import Path
from contracts.inputs import load_ply

__all__=["run_colmap","load_point_cloud"]

def run_colmap(image_dir, output_dir, executable="colmap", extra_args=None):
    """Run COLMAP as an external process; P only owns this adapter, not SfM."""
    image_dir=Path(image_dir); output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    args=[executable,"automatic_reconstructor","--image_path",str(image_dir),"--workspace_path",str(output_dir)]
    if extra_args: args.extend(map(str,extra_args))
    return subprocess.run(args,check=True,capture_output=True,text=True)

def load_point_cloud(path): return load_ply(str(path))
