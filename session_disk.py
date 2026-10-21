"""Atomic local last-session snapshots, authenticated before deserialization.

Only changed entries are written, so adjusting a control does not rewrite every
FITS image. This is a local single-user workspace recovery file, not an upload API.
"""
import hashlib
import hmac
import os
import pickle
import secrets
import shutil
import sqlite3
from pathlib import Path

import workspace_settings


def directory():
    return workspace_settings.SETTINGS_FILE.parent / 'session-recovery'


def secret(root):
    path = root / 'key'
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_bytes()
    with os.fdopen(fd, 'wb') as handle:
        value = secrets.token_bytes(32)
        handle.write(value)
    return value


def stamp(value):
    from imageProcessing import Frame
    if isinstance(value, Frame):
        if not hasattr(value, '_recovery_identity'):
            value._recovery_identity = id(value)
        return ('frame', value._recovery_identity)
    if isinstance(value, dict):
        return ('dict', tuple((stamp(k), stamp(v)) for k, v in value.items()))
    if isinstance(value, (tuple, list, set)):
        return (type(value).__name__, tuple(stamp(v) for v in value))
    if isinstance(value, (str, int, float, bool, type(None), Path)):
        return value
    return (type(value).__name__, id(value))


def signature(key, payload, signing_key):
    return hmac.new(signing_key, key.encode() + b'\0' + payload, hashlib.sha256).digest()


def write_snapshot(token, snapshot, previous_stamps):
    root = directory()
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    signing_key = secret(root)
    snapshot = dict(snapshot)
    snapshot['_recovery_files'] = preserve_files(snapshot, root)
    stamps = {key: stamp(value) for key, value in snapshot.items()}
    path = root / 'latest.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('PRAGMA auto_vacuum=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, payload BLOB, signature BLOB)')
        db.execute('CREATE TABLE IF NOT EXISTS metadata (name TEXT PRIMARY KEY, value TEXT)')
        old_token = db.execute("SELECT value FROM metadata WHERE name='token'").fetchone()
        force = not old_token or old_token[0] != token
        existing = {row[0] for row in db.execute('SELECT key FROM state')}
        for key, value in snapshot.items():
            if force or key not in existing or previous_stamps.get(key) != stamps[key]:
                payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                db.execute('INSERT OR REPLACE INTO state VALUES (?,?,?)',
                           (key, payload, signature(key, payload, signing_key)))
        db.executemany('DELETE FROM state WHERE key=?', [(k,) for k in existing - snapshot.keys()])
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('token',?)", (token,))
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('version','1')")
    path.chmod(0o600)
    # Only the latest committed snapshot is retained on disk.
    assets = root / 'files'
    keep = {entry[1] for entry in snapshot['_recovery_files']}
    if assets.exists():
        for asset in assets.iterdir():
            if asset.is_file() and asset.name not in keep:
                asset.unlink()
    return stamps


def read_snapshot():
    root = directory()
    path = root / 'latest.sqlite3'
    if not path.exists():
        return None, {}
    signing_key = (root / 'key').read_bytes()
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        version = db.execute("SELECT value FROM metadata WHERE name='version'").fetchone()
        if version != ('1',):
            raise ValueError('Versão da sessão guardada incompatível.')
        token = db.execute("SELECT value FROM metadata WHERE name='token'").fetchone()[0]
        rows = db.execute('SELECT key,payload,signature FROM state').fetchall()
    # Validate every entry before loading any object. Never load unsigned files.
    for key, payload, digest in rows:
        if not hmac.compare_digest(signature(key, payload, signing_key), digest):
            raise ValueError('A sessão guardada está danificada ou foi alterada.')
    snapshot = {key: pickle.loads(payload) for key, payload, _ in rows}
    restore_files(snapshot.pop('_recovery_files', []), root)
    return token, snapshot


def preserve_files(snapshot, root):
    """Retain worker FITS files that TemporaryDirectory would delete on exit.

    Restoring their original paths and mtimes preserves recorded path fingerprints.
    User input files are never copied or overwritten by this mechanism.
    """
    paths = set()
    def visit(value):
        if isinstance(value, Path) and any(part.startswith('fits-job-') for part in value.parts):
            paths.add(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)
    visit(snapshot)
    assets = root / 'files'
    manifest = []
    for path in sorted(paths):
        info = path.stat()
        name = hashlib.sha256(repr((str(path), info.st_mtime_ns, info.st_size)).encode()).hexdigest()
        destination = assets / name
        if not destination.exists():
            assets.mkdir(exist_ok=True, mode=0o700)
            temporary = destination.with_suffix('.tmp')
            shutil.copy2(path, temporary)
            temporary.replace(destination)
        manifest.append((str(path), name, info.st_mtime_ns, info.st_size))
    return manifest


def restore_files(manifest, root):
    for original, name, mtime, size in manifest:
        path = Path(original)
        if path.exists():
            continue
        asset = root / 'files' / name
        if not asset.is_file() or asset.stat().st_size != size:
            raise ValueError('Falta uma imagem na sessão guardada: ' + path.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(asset, path)
        os.utime(path, ns=(mtime, mtime))
