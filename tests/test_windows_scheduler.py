"""Tests that do not create or execute any real Windows tasks."""
from datetime import datetime, timedelta, timezone
import base64
from pathlib import Path
import pytest
from src import windows_scheduler as ws

NOW = datetime(2026, 11, 15, 13, 0, tzinfo=timezone.utc)


def test_reject_non_executable(tmp_path):
    f = tmp_path / 'something.bat'
    f.write_text('echo testing')
    with pytest.raises(ValueError, match=r'\.exe'):
        ws.validate_request(f, NOW + timedelta(hours=2), now_utc=NOW)


def test_reject_missing_executable(tmp_path):
    with pytest.raises(FileNotFoundError):
        ws.validate_request(tmp_path / 'missing.exe', NOW + timedelta(hours=2), now_utc=NOW)


def test_reject_naive_timestamp(tmp_path):
    f = tmp_path / 'test.exe'
    f.write_bytes(b'test')
    with pytest.raises(ValueError, match='timezone'):
        ws.validate_request(f, datetime(2026, 11, 16, 10, 0), now_utc=NOW)


def test_reject_past_or_too_close(tmp_path):
    f = tmp_path / 'test.exe'
    f.write_bytes(b'test')
    with pytest.raises(ValueError, match='2 minutes'):
        ws.validate_request(f, NOW + timedelta(seconds=30), now_utc=NOW)


def test_accept_valid_future_program(tmp_path):
    f = tmp_path / 'my app.EXE'
    f.write_bytes(b'test')
    path, when = ws.validate_request(f, NOW + timedelta(hours=2), now_utc=NOW)
    assert path == f.resolve()
    assert when == NOW + timedelta(hours=2)


def test_encoded_ps_script_keeps_user_paths_as_data():
    dangerous = r'C:\Users\Bob\hello $evil; Remove-Item C:\.exe'
    script = ws._ps_with_payload("Write-Output 'ok'", {"exe":dangerous})
    assert dangerous not in script
    assert 'FromBase64String' in script
    assert '$evil' not in script
    import json
    assert json.loads(base64.b64decode(script.split("FromBase64String('")[1].split("')")[0]).decode())["exe"] == dangerous


def test_only_prefixed_tasks_can_be_cancelled(monkeypatch):
    monkeypatch.setattr(ws, '_invoke', lambda script: {'removed':'SunPilotAI_012345abcdef'})
    with pytest.raises(ValueError, match='Invalid'):
        ws.cancel_launch('OtherProgram')
    ws.cancel_launch('SunPilotAI_012345abcdef')


def test_register_handles_malicious_filename_without_interpolation(tmp_path, monkeypatch):
    f = tmp_path / 'a$(dangerous).exe'
    f.write_bytes(b'test')
    capture = {}
    monkeypatch.setattr(ws.uuid, 'uuid4', lambda: type('FakeUUID', (), {'hex':'1234567890abcdefabcdef'})())
    def fake_run(script):
        capture['script']=script
        return {'name':'SunPilotAI_1234567890ab','run_local':'2026-11-16T10:00:00','exe':str(f.resolve())}
    monkeypatch.setattr(ws, '_invoke', fake_run)
    result = ws.register_launch(f, NOW + timedelta(hours=5), now_utc=NOW)
    assert result['name']=='SunPilotAI_1234567890ab'
    assert '$(dangerous)' not in capture['script']


def test_list_handles_empty_and_single_task(monkeypatch):
    monkeypatch.setattr(ws,'_invoke', lambda script: [])
    assert ws.list_launches() == []
    monkeypatch.setattr(ws,'_invoke', lambda script: {'name':'SunPilotAI_012345abcdef','exe':'X.exe'})
    assert len(ws.list_launches())==1


def test_no_powershell_on_nonwindows():
    import sys
    if sys.platform!='win32':
        with pytest.raises(ws.ScheduleError,match='only on Windows'):
            ws._invoke('Write-Output a')
