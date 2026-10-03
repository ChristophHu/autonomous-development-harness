"""Secret-safe root-cause labels for isolated child-process failures."""


def process_failure_category(result):
    """Distinguish an outer macOS sandbox denial from ordinary child failures."""
    stderr = getattr(result, "stderr", "") or ""
    if isinstance(stderr, bytes):
        stderr = stderr.decode("utf-8", errors="replace")
    if (
        isinstance(stderr, str)
        and "sandbox_apply" in stderr.casefold()
        and "operation not permitted" in stderr.casefold()
    ):
        return "host_sandbox_blocked"
    return "child_process_failed"
