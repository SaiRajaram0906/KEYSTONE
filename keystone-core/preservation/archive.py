from __future__ import annotations
import hashlib, json, math
from datetime import date,timedelta
from pathlib import Path
from contracts.preservation import ArchiveEntry,ArchiveStats,PreservationError,validate_archive_entry
__all__=["Archive"]

class Archive:
    INDEX_NAME="index.json"
    def __init__(self,root):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self._data=self.root/"objects"; self._data.mkdir(exist_ok=True); self._index_path=self.root/self.INDEX_NAME; self._entries={}; self._load()
    def _load(self):
        if not self._index_path.exists(): return
        raw=json.loads(self._index_path.read_text(encoding="utf-8")); self._entries={k:ArchiveEntry(**v) for k,v in raw.items()}
        for e in self._entries.values(): validate_archive_entry(e)
    def _save(self):
        tmp=self._index_path.with_suffix(".tmp"); tmp.write_text(json.dumps({k:vars(v) for k,v in self._entries.items()},indent=2),encoding="utf-8"); tmp.replace(self._index_path)
    def _object_path(self,aid): return self._data/str(aid)[:2]/str(aid)
    def store(self,data:bytes,*,monument_id,kind,created_on,capture_id=None,is_reference=False):
        if not isinstance(data,(bytes,bytearray,memoryview)): raise TypeError("archive data must be bytes-like")
        payload=bytes(data); digest=hashlib.sha256(payload).hexdigest(); existing=self._entries.get(digest)
        if existing is not None:
            if is_reference and not existing.is_reference: raise PreservationError("the same bytes are already archived as a non-reference artefact")
            return existing
        e=ArchiveEntry(artefact_id=digest,monument_id=str(monument_id),kind=str(kind),created_on=str(created_on),size_bytes=len(payload),content_hash=digest,density=1.0,is_reference=bool(is_reference),capture_id=None if capture_id is None else str(capture_id)); validate_archive_entry(e)
        path=self._object_path(digest); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".tmp"); tmp.write_bytes(payload); tmp.replace(path); self._entries[digest]=e; self._save(); return e
    def path_for(self,artefact_id):
        path=self._object_path(artefact_id)
        if not path.exists(): raise FileNotFoundError(str(artefact_id))
        return path
    def verify(self,artefact_id):
        e=self._entries.get(str(artefact_id))
        if e is None: raise KeyError(str(artefact_id))
        p=self._object_path(e.artefact_id)
        if not p.exists(): return False
        b=p.read_bytes(); return len(b)==e.size_bytes and hashlib.sha256(b).hexdigest()==e.content_hash
    def fetch(self,artefact_id):
        e=self._entries.get(str(artefact_id))
        if e is None: raise KeyError(str(artefact_id))
        p=self.path_for(e.artefact_id); b=p.read_bytes()
        if len(b)!=e.size_bytes or hashlib.sha256(b).hexdigest()!=e.content_hash: raise PreservationError(f"archive integrity check failed for {e.artefact_id}")
        return b
    def entries(self,monument_id): return sorted([e for e in self._entries.values() if e.monument_id==monument_id],key=lambda e:(e.created_on,e.artefact_id))
    def stats(self,monument_id):
        items=self.entries(monument_id); total=sum(e.size_bytes for e in items); ref=sum(e.size_bytes for e in items if e.is_reference); reclaim=sum(e.size_bytes for e in items if not e.is_reference and e.density>=1.0 and e.kind in ("capture","pointcloud")); dates=[e.created_on for e in items]
        return ArchiveStats(str(monument_id),len(items),total,ref,reclaim,min(dates) if dates else None,max(dates) if dates else None)
    @staticmethod
    def _decimate(data,density=.10):
        if not data: return data
        target=max(1,int(math.ceil(len(data)*density)))
        if target>=len(data): return data
        # Deterministic evenly spaced byte sampling; reference bytes are never passed here.
        return bytes(data[(i*(len(data)-1))//(target-1)] for i in range(target)) if target>1 else data[:1]
    def apply_retention(self,monument_id,keep_full_days=365):
        if keep_full_days<0: raise ValueError("keep_full_days must be >= 0")
        cutoff=date.today()-timedelta(days=int(keep_full_days)); reclaimed=0; changed=False
        for aid,e in list(self._entries.items()):
            if e.monument_id!=monument_id or e.is_reference or e.density<1.0 or e.kind not in ("capture","pointcloud"): continue
            try: created=date.fromisoformat(e.created_on)
            except ValueError: continue
            if created>=cutoff: continue
            old=self.fetch(aid); new=self._decimate(old)
            if len(new)>=len(old): continue
            self._object_path(aid).write_bytes(new)
            u=ArchiveEntry(e.artefact_id,e.monument_id,e.kind,e.created_on,len(new),hashlib.sha256(new).hexdigest(),.10,False,e.capture_id); validate_archive_entry(u); self._entries[aid]=u; reclaimed+=len(old)-len(new); changed=True
        if changed: self._save()
        return reclaimed
