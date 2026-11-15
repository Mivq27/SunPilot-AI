"""Basic, headless release guards. Actual visual QA is done on Windows."""
from pathlib import Path
import ast

SOURCE = Path(__file__).resolve().parents[1] / "sunpilot_desktop_v1.py"
CODE = SOURCE.read_text(encoding="utf-8")

def test_release_ui_parses():
    ast.parse(CODE)

def test_tabs_are_plain_and_unnumbered():
    for tab in ("Plan", "Automate", "Simulator", "Research"):
        assert f'"{tab}"' in CODE
    for old in ('"1  Plan"', '"2  Automate"', '"3  Simulator"', '"4  Research"'):
        assert old not in CODE

def test_method_picker_is_prominent_and_reachable():
    assert 'class MethodPicker(QWidget):' in CODE
    assert 'self.plan_controls.body.insertWidget(1, self.method_panel)' in CODE
    assert 'WEATHER FORECAST' in CODE
    assert 'SUNPILOT AI' in CODE
    assert 'Recommended' in CODE and 'Experimental' in CODE
    assert 'self.method.currentIndexChanged.connect(self._method_changed)' in CODE
    assert 'self.advanced_toggle' not in CODE

def test_wheel_safe_and_opt_in_scheduler_retained():
    assert 'class NoWheelSpinBox(QSpinBox):' in CODE
    assert 'event.ignore()' in CODE
    assert 'Confirm executable launch' in CODE
    assert 'register_launch' in CODE
