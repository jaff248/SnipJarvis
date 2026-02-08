import importlib.util
from pathlib import Path


def test_style_description_uses_text_area():
    # Ensure UI uses a multiline text area for style description to avoid Enter submitting the form
    ui_path = Path(__file__).parents[1] / 'ui' / 'app.py'
    text = ui_path.read_text()
    assert 'st.text_area' in text
    assert 'Style Description (optional)' in text


def test_quick_remaster_button_present():
    ui_path = Path(__file__).parents[1] / 'ui' / 'app.py'
    text = ui_path.read_text()
    assert 'Quick Remaster (one-click)' in text


def test_whole_stem_confirmation_logic():
    # Logic mirrors the UI condition that requires confirmation when no regions exist
    def should_execute_whole(regen_whole: bool, target_choice: str, has_regions: bool, whole_confirm: bool) -> bool:
        return regen_whole or (target_choice == 'Whole Stem' and (has_regions or whole_confirm))

    # Case: no regions, not forced, not confirmed -> should not execute
    assert should_execute_whole(False, 'Whole Stem', False, False) is False

    # Case: no regions, confirmed -> should execute
    assert should_execute_whole(False, 'Whole Stem', False, True) is True

    # Case: has regions -> should execute regardless of confirmation
    assert should_execute_whole(False, 'Whole Stem', True, False) is True

    # Case: force regen checkbox checked -> should execute
    assert should_execute_whole(True, 'Anything', False, False) is True
