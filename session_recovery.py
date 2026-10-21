"""Recover the latest local workspace across reloads and server restarts."""
from functools import wraps
from threading import RLock
from uuid import uuid4, UUID

import streamlit as st
from streamlit.runtime.state.session_state_proxy import get_session_state
from session_disk import read_snapshot, write_snapshot, directory


@st.cache_resource(show_spinner=False)
def recovery_store():
    return {}, RLock()


def copy_containers(value):
    # Scientific products are immutable after generation; copy mutable selections
    # without duplicating potentially gigabytes of FITS arrays on every interaction.
    if isinstance(value, dict):
        return {k: copy_containers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [copy_containers(v) for v in value]
    if isinstance(value, tuple):
        return tuple(copy_containers(v) for v in value)
    if isinstance(value, set):
        return set(value)
    return value


def restore_missing_widgets(previous):
    for key in previous.get('_recovery_widget_keys', []):
        if key not in st.session_state and key in previous:
            st.session_state[key] = copy_containers(previous[key])


def begin_session():
    requested = st.query_params.get('session', '')
    store, lock = recovery_store()
    with lock:
        # Opening the base URL means resume the latest workspace, not start empty.
        if not store.get('__disk_loaded'):
            try:
                saved_token, saved = read_snapshot()
                if saved_token:
                    store[saved_token] = saved
                    store['__latest'] = saved_token
            except Exception as exc:
                st.warning(f'Não foi possível recuperar a sessão guardada: {exc}')
            store['__disk_loaded'] = True
        token = requested or store.get('__latest') or str(uuid4())
        try:
            token = str(UUID(token))
        except (ValueError, TypeError, AttributeError):
            token = store.get('__latest') or str(uuid4())
        if token not in store and store.get('__latest'):
            token = store['__latest']
        previous = store.get(token, {})
        if st.session_state.get('_recovery_token') != token:
            for key, value in previous.items():
                if not key.startswith('_recovery_'):
                    st.session_state[key] = copy_containers(value)
        else:
            restore_missing_widgets(previous)
    if requested != token:
        st.query_params['session'] = token
    st.session_state['_recovery_token'] = token


def save_session():
    token = st.session_state.get('_recovery_token')
    if not token:
        return
    # Buttons/upload widgets cannot legally be restored via session_state. Use
    # widget metadata rather than guessing from names (checkboxes are booleans too).
    state = get_session_state()._state
    excluded, widgets = set(), set()
    for widget_id, meta in state._new_widget_state.widget_metadata.items():
        key = state._key_id_mapper.id_key_mapping.get(widget_id)
        if key:
            widgets.add(key)
        if meta.value_type in ('trigger_value', 'string_trigger_value', 'chat_input_value', 'file_uploader_state_value'):
            key = state._key_id_mapper.id_key_mapping.get(widget_id)
            if key:
                excluded.add(key)
    snapshot = {key: copy_containers(value) for key, value in st.session_state.items()
                if key not in excluded and not key.startswith('_recovery_')
                and not key.endswith(('_pending', '_workspace'))
                and key not in ('refresh_image_view', 'workflow_science_combine_requested')}
    store, lock = recovery_store()
    with lock:
        previous = store.get(token, {})
        retained_widgets = (widgets | set(previous.get('_recovery_widget_keys', []))) - excluded
        for key in retained_widgets:
            if key not in snapshot and key in previous:
                snapshot[key] = previous[key]
        snapshot['_recovery_widget_keys'] = sorted(retained_widgets)
        store[token] = snapshot
        store['__latest'] = token
        try:
            stamps_key = '__stamps_' + str(directory())
            store[stamps_key] = write_snapshot(token, snapshot, store.get(stamps_key, {}))
            st.session_state.pop('_recovery_save_error', None)
        except Exception as exc:
            message = f'Não foi possível guardar a sessão em disco: {exc}'
            if st.session_state.get('_recovery_save_error') != message:
                st.warning(message)
            st.session_state['_recovery_save_error'] = message


def checkpoint(function):
    """Save on normal completion, st.stop and st.rerun, including fragment reruns."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = st.session_state.get('_recovery_token')
        if token:
            store, lock = recovery_store()
            with lock:
                restore_missing_widgets(store.get(token, {}))
        try:
            return function(*args, **kwargs)
        finally:
            save_session()
    return wrapped
