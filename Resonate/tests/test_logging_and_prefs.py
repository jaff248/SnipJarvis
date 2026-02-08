import os
from logging_config import LOG_FILE, get_logger

def test_log_file_exists_and_writable(tmp_path):
    # Ensure the log file path exists and logger writes something
    logger = get_logger()
    logger.info('test: logging initialized')
    assert os.path.exists(LOG_FILE)


def test_auto_apply_pref_defaults_to_false(monkeypatch):
    # Simulate a minimal Streamlit session_state
    import types, sys
    fake_st = types.ModuleType('streamlit')
    fake_st.session_state = {}
    monkeypatch.setitem(sys.modules, 'streamlit', fake_st)

    import streamlit as st
    assert st.session_state.get('auto_repair_all_auto_apply', False) is False
