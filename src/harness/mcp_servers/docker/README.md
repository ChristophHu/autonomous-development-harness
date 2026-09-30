# Docker MCP Server

This standalone server lives beside its tests so its implementation can be
developed and verified independently from the Harness Docker Compose broker.

Run only this server's unit tests from the repository root:

```sh
.venv/bin/pytest -o addopts= src/harness/mcp_servers/docker/tests
```

The tests stub the MCP/ dotenv libraries and mock every Docker subprocess; they
do not need a daemon and do not create or remove containers. `sh scripts/verify.sh`
discovers and runs each MCP server's local `tests/` directory separately.

The `docker_mcp_server.py` runtime itself requires the `mcp` and `python-dotenv`
packages. It invokes the Docker CLI with the current user permissions; only use
it with a Docker daemon and workspace whose privileges are understood.
