"""Hard guard: make it impossible for this pipeline to read the source dataset.

The 22 August clarification says training on the source data - the published dataset the
challenge was carved out of - is not in the spirit of the event.  Rather than rely on
review, this intercepts h5py.File and refuses to open that file at all.  Importing this
module is the guarantee; any code path that tried would raise immediately.

Outside data (SNI_merged_0531.h5ad, a different experiment on different animals) is
explicitly allowed and passes through untouched.
"""
from __future__ import annotations
import h5py

SOURCE_BASENAME = "MERFISH_spinal_cord_0531.h5ad"
_original = h5py.File


class SourceDataAccess(RuntimeError):
    pass


def _guarded(name, *args, **kwargs):
    if SOURCE_BASENAME in str(name):
        raise SourceDataAccess(
            f"refusing to open {SOURCE_BASENAME}: the source dataset may not be used to "
            f"train this model (rule clarification, 22 August)")
    return _original(name, *args, **kwargs)


if h5py.File is _original:
    h5py.File = _guarded
