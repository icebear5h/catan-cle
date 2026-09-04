from playground.game_viewer.app import server_run_options


def test_stateful_viewer_disables_debug_and_reloader_by_default(monkeypatch):
    monkeypatch.delenv("CATAN_VIEWER_DEBUG", raising=False)
    monkeypatch.delenv("CATAN_VIEWER_RELOAD", raising=False)

    assert server_run_options() == {
        "debug": False,
        "use_reloader": False,
    }


def test_viewer_debug_and_reloader_require_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("CATAN_VIEWER_DEBUG", "true")
    monkeypatch.setenv("CATAN_VIEWER_RELOAD", "1")

    assert server_run_options() == {
        "debug": True,
        "use_reloader": True,
    }
