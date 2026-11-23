import pytest
from weft.omni_console import OmniFleetConsole

def test_omni_console_fleet_detection():
    console = OmniFleetConsole()
    assert console.status.loaded_models == 9
    assert console.status.total_models == 9
    assert console.status.total_ram_mb < 1400.0  # < 1.4 GB

def test_omni_console_routing():
    console = OmniFleetConsole()
    res_audio = console.route_query("speech.wav")
    assert res_audio["selected_engine"] == "whisper-large-v3"
    
    res_code = console.route_query("def binary_search():")
    assert res_code["selected_engine"] == "speculative-coder-14b-32b"
    assert res_code["speedup"] > 2.0

    res_music = console.route_query("compose theme")
    assert res_music["selected_engine"] == "musicgen-small"
