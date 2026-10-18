"""Remember the working folder across browser sessions and app restarts."""
import json
from pathlib import Path


SETTINGS_FILE = Path(__file__).resolve().parent / '.streamlit' / 'working_folder.json'
DEFAULT_SCIENCE_TARGETS = ('hd226868', 'deneb')


def load_working_folder(default):
    try:
        folder = Path(json.loads(SETTINGS_FILE.read_text())['folder']).expanduser()
        if folder.is_dir():
            return folder
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return Path(default)


def save_working_folder(folder):
    folder = Path(folder).expanduser()
    if folder.is_dir():
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = SETTINGS_FILE.with_suffix('.tmp')
        temporary.write_text(json.dumps({'folder': str(folder.resolve())}) + '\n')
        temporary.replace(SETTINGS_FILE)


def load_science_targets():
    targets = list(DEFAULT_SCIENCE_TARGETS)
    try:
        saved = json.loads(SETTINGS_FILE.with_name('science_targets.json').read_text())
        if isinstance(saved, list):
            known = {target.casefold() for target in targets}
            for target in saved:
                if isinstance(target, str) and target.strip() and target.strip().casefold() not in known:
                    targets.append(target.strip())
                    known.add(target.strip().casefold())
    except (OSError, ValueError, TypeError):
        pass
    return targets


def save_science_targets(targets):
    destination = SETTINGS_FILE.with_name('science_targets.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(list(targets), ensure_ascii=False) + '\n')
    temporary.replace(destination)
