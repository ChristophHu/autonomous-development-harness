# Qdrant operations

The Harness never starts, upgrades, or stops the operator-owned Qdrant service as
part of memory operations. Its collection is configured in `config.yaml`; the
optional API key is resolved from the normal secret sources and is never meant to
be copied into a command transcript.

## Read-only checks

- `harness memory qdrant-status` checks HTTP health and the configured collection's
  status, vector dimension, distance metric, and point counts. It does not start
  the MCP registry.
- `harness memory qdrant-acceptance --confirm` additionally sends one generic
  embedding query and performs a Qdrant search after checking the configured
  model ID and vector dimensions. The report distinguishes raw candidates from
  hits backed by a current Vault source hash; stale or missing source documents
  do not count as matches. At least one current source is required before the
  command stores successful embedding evidence. It does not create, update, or delete points. The
  explicit flag is required because the embedding call may use a metered provider.
  This command also works without starting MCP servers.
- `harness doctor` reports the same service and collection health as a startup
  diagnostic; it does not repair the service.
- `harness memory qdrant-config-audit PATH --config PATH` statically reviews an
  operator Compose file and optional Qdrant YAML without starting containers or
  printing configuration values. Finding codes include `image_not_pinned`,
  `qdrant_port_unbound`, `literal_api_key`, `config_mount_writable`, and
  `cors_enabled_review`. Exit codes are 0 for no findings, 1 for findings, and
  2 for an unavailable/invalid input. The audit is a heuristic, not a replacement
  for checking routing, TLS, or provider firewall rules.

Run status checks as part of the operator's monitoring routine. Run the embedding
acceptance probe after changing the embedding model, endpoint, API key, dimensions,
or Qdrant version—not as a frequent heartbeat.

## Backup and recovery boundary

Backing up and restoring the Qdrant server is infrastructure-owned and is not a
Harness feature. Use the configured Proxmox or VM-provider backup/restore process
for the VM that hosts Qdrant. Backup scheduling, retention, off-host copies,
restore drills, and recovery approval belong to that infrastructure procedure;
the Harness neither creates collection snapshots nor uploads/restores them.

After an infrastructure restore, use the read-only `qdrant-status` and explicit
`qdrant-acceptance --confirm` checks to verify service/collection health and the
embedding-search contract. Never point a test at the live collection if it would
mutate data.

## Upgrade and rollback

Keep the image pinned; do not use `latest`. For an upgrade, follow the
infrastructure provider's backup and rollback procedure for the VM. Test the
candidate image in an isolated Compose project and verify health, collection
readability, a representative persisted point, and restart persistence before
changing the operator-managed Compose file. Do not assume a data volume can be
opened safely by an older Qdrant binary.

The opt-in rehearsal is:

```sh
harness qdrant-upgrade-smoke \
  --baseline-image qdrant/qdrant:<exact-current-version> \
  --candidate-image qdrant/qdrant:<tested-candidate-version> \
  --confirm
```

Both references must be pinned official version tags or immutable SHA-256 image
digests, and must differ. The command creates a unique disposable Compose project
and volume, writes a marker point under the baseline, starts the candidate against
that same test volume, verifies collection/point readability after the transition
and after a candidate restart, then removes only that isolated project and volume.
It never reads or mounts the operator's data volume. `--confirm` is required
because Docker may pull images and create/remove disposable resources. This is a
compatibility smoke test, not a production rollback rehearsal or a substitute for
the VM provider's recovery plan.

Qdrant image changes, VM backup/restore, retention, and alert delivery remain
operator responsibilities; Harness commands do not silently perform these
mutations.

## References
