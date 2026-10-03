# Native macOS acceptance

The Codex desktop process can run inside an outer sandbox that refuses nested
`sandbox-exec` profiles. That host-level refusal cannot be fixed by widening the
Harness profile: doing so would weaken the boundary without granting the outer
process the missing macOS capability. Run this procedure from the user's macOS
Terminal in the normal project checkout.

## Distinguish outer-host denial from a Harness profile issue

From the normal macOS Terminal, run this harmless control first. It creates no
files and only asks macOS to launch `/usr/bin/true` with the default sandbox
policy:

```sh
/usr/bin/sandbox-exec -p '(version 1)(allow default)' /usr/bin/true
printf 'sandbox-control-exit=%s\n' "$?"
```

If this prints `sandbox-control-exit=0`, the OS accepts sandbox activation in
that Terminal session; continue with `harness isolation doctor` below. If it
returns 71 with `sandbox_apply: Operation not permitted`, the denial occurs
before any Harness policy is applied and must be resolved in the execution
environment/Terminal context. Do not loosen the Harness profile to compensate.
In the current Codex execution environment, this baseline control itself
returns 71; the same result was reproduced for the Harness process, Git, and
orchestrator probes.

```sh
cd ~/Desktop/autonomous-development-harness
.venv/bin/harness isolation doctor --output data/isolation-capability.json
```

The probe is read-only outside its temporary directory. Continue with the tests
even if it reports `blocked_by_host`; that result is diagnostic, not a skip
instruction. A `supported` result is a prerequisite for interpreting native
isolation/Git results as an acceptance run.

```sh
.venv/bin/pytest --no-cov -q \
  tests/test_process_isolation.py \
  tests/test_git_broker.py \
  tests/test_git_http.py \
  tests/test_git_ssh.py \
  tests/test_git_integration.py \
  tests/test_git_resume.py
```

Then run the authoritative complete verification. It re-probes the host,
collects JUnit and coverage, runs Ruff/format checks on success, and preserves
the pytest failure exit code while writing a failure-cluster report on failure.

```sh
sh scripts/verify.sh
```

Return these outputs with the result: `data/isolation-capability.json`, the
focused pytest summary, the complete verify summary/exit code, and
`data/verification-failure-clusters.json` if generated. Do not edit sandbox
rules, disable tests, or interpret a partial run as full acceptance. Only a
successful complete verify can refresh the hash-bound verification evidence.
