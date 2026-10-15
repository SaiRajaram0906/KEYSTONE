from __future__ import annotations
from pathlib import Path
from contracts.preservation import MonumentRecord, PreservationError, validate_monument_record
__all__=["create_monument","load_monument","save_monument","set_reference_capture"]

def create_monument(*,monument_id,name,markers,regions,face_size_mm=(6000.,3000.),latitude=None,longitude=None):
    r=MonumentRecord(monument_id=monument_id,name=name,markers=list(markers),regions=list(regions),face_size_mm=list(face_size_mm),latitude=latitude,longitude=longitude); validate_monument_record(r); return r

def load_monument(path): return MonumentRecord.from_json(Path(path).read_text(encoding="utf-8"))

def save_monument(record,path):
    validate_monument_record(record); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=load_monument(path)
        if old.reference_capture_id is not None and old.reference_capture_id != record.reference_capture_id:
            raise PreservationError(f"reference_capture_id is write-once: existing {old.reference_capture_id!r}, attempted {record.reference_capture_id!r}")
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(record.to_json(),encoding="utf-8"); tmp.replace(path); return record

def set_reference_capture(record,capture_id):
    if not capture_id: raise PreservationError("capture_id is required")
    if record.reference_capture_id is not None and record.reference_capture_id != capture_id: raise PreservationError(f"reference_capture_id is write-once: already {record.reference_capture_id!r}")
    record.reference_capture_id=capture_id; validate_monument_record(record); return record
