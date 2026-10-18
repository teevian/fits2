"""Regex selection of local frames, preserving manual removals between edits."""
from pathlib import Path
import re

import streamlit as st


def select_matching_frames(pattern_key, frames_key, catalog):
    pattern = st.session_state[pattern_key].strip()
    error_key = f'{pattern_key}_error'
    try:
        expression = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        st.session_state[error_key] = f'Invalid regex: {exc}'
        return
    st.session_state.pop(error_key, None)
    matches = lambda label: expression.match(label) is not None
    st.session_state[frames_key] = [
        label for label, value in catalog.items()
        if pattern and isinstance(value, Path) and matches(label)]


def initialize_matching_frames(pattern_key, frames_key, default_pattern, catalog, folder):
    """Preselect matching local files once per folder, preserving manual edits."""
    if pattern_key not in st.session_state:
        st.session_state[pattern_key] = default_pattern
    folder_key = f'{frames_key}_folder'
    if (frames_key not in st.session_state or
            st.session_state.get(folder_key) != str(folder)):
        select_matching_frames(pattern_key, frames_key, catalog)
        st.session_state[folder_key] = str(folder)



# Obtém o número final do nome FITS para selecionar um intervalo inclusivo de imagens.
def science_file_number(name):
    match = re.search(r'_(\d+)\.(?:fits|fit|fts)(?:\.(?:gz|fz))?$', name, re.IGNORECASE)
    return int(match[1]) if match else None


# Seleciona os ficheiros cujo número final está entre os dois limites, incluindo-os.
def science_file_range(labels, first, last):
    return [name for name in labels
            if (number := science_file_number(name)) is not None and first <= number <= last]
