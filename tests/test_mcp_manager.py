from harness.mcp_manager import MCPServerManager


def test_empty_and_malformed_manager_inputs_are_safe():
    assert MCPServerManager(None).report() == []
    assert MCPServerManager({"servers": []}).report() == []


def test_server_validation_statuses_and_secret_redaction():
    manager = MCPServerManager(
        {
            "servers": {
                "bad.name": {"builtin": "filesystem"},
                "wrong": {"builtin": "filesystem", "timeout": "SECRET"},
                "ready": {"builtin": "filesystem"},
            }
        }
    )

    report = manager.report()

    assert [item["state"] for item in report] == [
        "invalid_config",
        "invalid_config",
        "not_started",
    ]
    assert "SECRET" not in str(report)
    assert manager.settings("bad.name") is None
    assert manager.settings("wrong") is None
    assert manager.settings("ready") is not None
    assert manager.names() == ("bad.name", "wrong", "ready")


def test_mark_updates_one_server_and_ignores_unknown_names():
    manager = MCPServerManager(
        {"servers": {"one": {"builtin": "filesystem"}, "two": {"builtin": "obsidian"}}}
    )
    manager.mark("unknown", "unavailable", "RuntimeError")
    manager.mark("one", "unavailable", "OSError")

    one, two = manager.report()

    assert one["state"] == "unavailable"
    assert one["error"] == "OSError"
    assert one["checked_at"]
    assert two["state"] == "not_started"
    assert two["checked_at"] is None


def test_report_probes_enabled_servers_without_blocking_disabled_or_peers():
    def loader(name):
        if name == "offline":
            raise OSError("private endpoint")

    manager = MCPServerManager(
        {
            "servers": {
                "offline": {"builtin": "filesystem"},
                "online": {"builtin": "obsidian"},
                "disabled": {"builtin": "filesystem", "enabled": False},
            }
        },
        loader=loader,
    )
    offline = manager.start("offline", force=True)
    online = manager.start("online", force=True)
    disabled = manager.start("disabled", force=True)

    assert offline["state"] == "unavailable"
    assert offline["error"] == "OSError"
    assert online["state"] == "available"
    assert online["error"] is None
    assert disabled["state"] == "disabled"
    assert "private endpoint" not in str((offline, online, disabled))


def test_start_without_loader_and_unknown_name_are_read_only():
    manager = MCPServerManager({"servers": {"offline": {"builtin": "filesystem"}}})

    assert manager.start("offline")["state"] == "not_started"
    assert manager.start("missing") is None
