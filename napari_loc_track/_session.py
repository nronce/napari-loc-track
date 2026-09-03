"""Saving a whole working session, and finding its way back to the data.

A session file is a manifest, not an archive. It records where the data lives,
every parameter that was set, and what had been computed - and on load the
pipeline is run again from those parameters to arrive back at the same state.
That keeps a session in the kilobytes: the raw stack is already on disk and is
never copied, and trajectories, diffusion coefficients and reconstructions are
all reproducible from the localizations and the settings that produced them.

The one thing that cannot be cheaply recomputed is a set of localizations that
came from fitting inside the plugin and was never written out. Re-fitting ten
thousand frames costs minutes; the table gzips to a fraction of the raw stack,
so that one is saved beside the session rather than recomputed.

Nothing here imports Qt, napari or pandas, so the format can be read, written
and reasoned about without any of them.
"""
from __future__ import annotations

import json
from pathlib import Path

# Bumped only for a change that older readers could misinterpret. A file from a
# newer format is refused rather than half-applied: a session that silently
# dropped the half it did not understand would restore the wrong state while
# looking like it had worked.
SESSION_FORMAT = 1
SESSION_KEY = "napari_loc_track_session"

SESSION_SUFFIX = ".loctrack-session.json"
# Localizations saved beside the session, when they exist nowhere else.
SESSION_LOCS_SUFFIX = "_localizations.csv.gz"

SESSION_FILTER = "napari-loc-track session (*.loctrack-session.json);;JSON files (*.json)"


def session_base(path):
    """The stem of a session path, with the full compound suffix removed."""
    path = Path(path)
    name = path.name
    if name.endswith(SESSION_SUFFIX):
        return path.with_name(name[: -len(SESSION_SUFFIX)])
    return path.with_suffix("")


def session_path_for(path):
    """`path` with the session suffix, adding it only if it is not already there."""
    path = Path(path)
    if path.name.endswith(SESSION_SUFFIX):
        return path
    return session_base(path).with_name(session_base(path).name + SESSION_SUFFIX)


def locs_path_for(session_path):
    base = session_base(session_path)
    return base.with_name(base.name + SESSION_LOCS_SUFFIX)


def source_record(path, session_dir):
    """Where a file is, recorded both absolutely and relative to the session.

    Both, because each survives a different accident. The absolute path
    survives the session file being moved on its own; the relative one survives
    the whole tree being moved, renamed or copied to another machine - which is
    what actually happens to an analysis folder on a shared drive.
    """
    if not path:
        return None
    path = Path(path)
    record = {"path": str(path)}
    try:
        record["relative"] = str(path.resolve().relative_to(Path(session_dir).resolve()))
    except (ValueError, OSError):
        # Not under the session folder, or unresolvable - the absolute path is
        # the only thing that can be said about it.
        pass
    return record


def resolve_source(record, session_dir):
    """The first place a recorded file actually exists, or None.

    Relative first: if the tree has been moved, the absolute path may still
    point at a stale copy of the same file somewhere else, and the neighbour
    that travelled with the session is the one that belongs to it.
    """
    if not record:
        return None
    if isinstance(record, str):          # tolerated: an older or hand-written file
        record = {"path": record}
    candidates = []
    relative = record.get("relative")
    if relative:
        candidates.append(Path(session_dir) / relative)
    if record.get("path"):
        candidates.append(Path(record["path"]))
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def missing_sources(manifest, session_dir):
    """Recorded files that are not where the session left them."""
    missing = []
    for name, record in (manifest.get("sources") or {}).items():
        if not isinstance(record, dict) or "path" not in record:
            continue
        if resolve_source(record, session_dir) is None:
            missing.append((name, record["path"]))
    return missing


def read_session(path):
    """Parse a session file. Raises ValueError with a reason it cannot be used."""
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except OSError as exc:
        raise ValueError(f"could not be read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict) or SESSION_KEY not in manifest:
        raise ValueError("is not a napari-loc-track session file")
    version = manifest.get(SESSION_KEY)
    if not isinstance(version, int) or version > SESSION_FORMAT:
        raise ValueError(
            f"was written in session format {version}, and this version of the "
            f"plugin understands up to {SESSION_FORMAT}"
        )
    return manifest


def write_session(path, manifest):
    """Write a manifest, and return how many bytes it took."""
    path = Path(path)
    text = json.dumps(manifest, indent=2, default=str)
    path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def describe_session(manifest):
    """A one-line summary of what a session holds, for the log."""
    parts = []
    saved = manifest.get("saved_at")
    if saved:
        parts.append(f"saved {saved}")
    expected = manifest.get("expected") or {}
    if expected.get("localizations"):
        parts.append(f"{expected['localizations']} localizations")
    if expected.get("trajectories"):
        parts.append(f"{expected['trajectories']} trajectories")
    rebuild = [name for name, wanted in (manifest.get("rebuild") or {}).items() if wanted]
    if rebuild:
        parts.append("rebuilds " + ", ".join(sorted(rebuild)))
    return "; ".join(parts) if parts else "no recorded contents"
