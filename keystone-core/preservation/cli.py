from __future__ import annotations
import argparse
from pathlib import Path
from contracts.fakes_capture import make_monument
from contracts.preservation import CaptureSession, PhotoMeta
from .gate import check_capture
from .photo_metrics import image_metrics

def main():
    ap=argparse.ArgumentParser(description="Run the KEYSTONE P capture gate on image metadata.")
    ap.add_argument("folder"); ap.add_argument("--capture-id",default="FIELD-001"); ap.add_argument("--monument-id",default="M-01")
    args=ap.parse_args(); monument=make_monument(args.monument_id); photos=[]
    files=sorted(p for p in Path(args.folder).iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"})
    if not files: raise SystemExit("No supported images found.")
    # A real capture app supplies footprint/marker visibility. CLI defaults to
    # the full face so this command is primarily a metrics demonstration.
    for i,path in enumerate(files):
        m=image_metrics(path); photos.append(PhotoMeta(f"{args.capture_id}-P{i:03d}",m["width"],m["height"],m["sharpness"],m["exposure_clip"],[0.,0.,*monument.face_size_mm]))
    session=CaptureSession(args.capture_id,monument.monument_id,"2026-01-01",photos)
    qc=check_capture(session,monument); print(qc.to_json())
if __name__=="__main__": main()
