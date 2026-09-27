# Autonomous Development Harness

Lokales, macOS-orientiertes Harness für zustandsbehaftete Entwicklungsaufgaben.

## Schnellstart

```zsh
cd autonomous-development-harness
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
harness doctor
harness tasks create "Beispielaufgabe"
harness start
```

Die API läuft standardmäßig auf `127.0.0.1:8080`; Swagger ist unter `/docs` verfügbar.

```zsh
curl http://127.0.0.1:8080/health
curl -N http://127.0.0.1:8080/events/stream
```

Secrets werden aus Defaults, `config.yaml`, `.env` und (auf macOS) der Keychain aufgelöst. Niemals Secrets in `config.yaml` eintragen.

## Modell- und Taskvertrag

Exakte Modell-IDs müssen über `LMSTUDIO_MODEL`, `OPENAI_MODEL` oder `OPENROUTER_MODEL` konfiguriert werden. Die Beispielprofile sind Local-first. Es gibt keinen künstlichen Produktionsfallback; unbekannte Provider/Profile, fehlende Modelle und ungültige Pläne scheitern.

Ein Task mit lediglich einem Titel wird zur Anforderungsklärung pausiert. Für den Entwicklungsablauf sind `goal`, `requirements`, `acceptance_criteria`, `test_commands` und `coverage_command` notwendig; sie können über REST angelegt/geändert werden. Kommandos sind Argumentlisten, keine impliziten Shellstrings. Coverage wird aus einem frisch erzeugten `coverage.json` gelesen. Pflichtfragen lassen sich mit passenden JSON-Taskfeldern beantworten und setzen den Task fort.

Die Beispielkonfiguration erlaubt Dateiänderungen und Shellprozesse für das Codingprofil. H1c erzwingt auf macOS `sandbox-exec`: Shell-, Test-, Lint- und Dockerprozesse samt Nachkommen dürfen nur im CWD schreiben, keine Git-Metadaten verändern und keine Netzwerk-/Daemonverbindung öffnen. Kontrolliertes Git darf nur sein kanonisches Binary starten; Hooks und externe Filter-/Transporthelfer sind gesperrt. Es gibt keinen Fallback ohne Isolation. Leserechte für lokale Laufzeitbibliotheken bleiben breit; dies ist keine vollständige Geheimnis-/Lesepfadisolation. Destruktive Dateilöschung bleibt deaktiviert. Live-Provider und Qdrant sind nicht live abgenommen.

## Verifikation und Umfang

```sh
sh scripts/verify.sh
```

Prüft Tests, 100 % Statements/Zeilen/Branches, Funktionsabdeckung je Modul und Ruff. Die reale H1c-Abnahme benötigt eine macOS-Sitzung, die Sandboxaktivierung erlaubt. Eine bereits eingeschränkte übergeordnete Sandbox kann `sandbox_apply: Operation not permitted` erzeugen; dann scheitert die Ausführung geschlossen. Kontrollierte Provider-Fixtures treiben reale SQLite-/Dateisystem-/Subprozess-E2Es mit Testfehler, Korrektur, Coverage und Harness-Prozess-Neustart an. [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) beschreibt die umgesetzten Pakete; [GAP_MATRIX.md](GAP_MATRIX.md) hält sämtliche 109 Anforderungen und verbleibende Lücken fest. 100 % Codecoverage bedeutet nicht, dass alle Spezifikationspunkte abgeschlossen sind.

Agent-/Modell-/Tool-Runs sind inzwischen per Task und Profil korreliert; Migration v3 erhält vorhandene Daten. MemoryService indiziert Vaultnoten in überlappenden Chunks mit echten Embeddings und lädt beim Retrieval die Originalquelle. Qdrant ist kein Ersatz für Markdown: fehlende/veraltete Quellen werden verworfen, falsche Vektordimensionen und Dienstfehler sichtbar gemacht. Hashvektoren werden nicht als semantische Embeddings verwendet.

Unterbrochene Tasks können nach Ablauf der exklusiven Lease erneut gestartet werden. Recovery liest Events, Git, Dateien und frische Tests; ein unabhängiger Review bestimmt offene Requirements. Der neue Plan muss alle Restziele über `recovery_targets` abdecken und genaue relative `write_paths` deklarieren. Bestätigte Dateien ohne offene Kriterien sind geschützt. Recovery-Executor-Mutationen erfolgen ausschließlich über `filesystem.write/create`; unbeschränkte Shell-/Git-/Docker-Mutationen werden abgewiesen. Testkommandos laufen ebenfalls unter Kernelisolation. Alte Subtask-Erfolgsflags sind keine Resume-Checkpoints.

## Git-Workflows (opt-in)

`git.enabled` ist standardmäßig false. Aktivierung erfordert ein sauberes Repository mit vorhandenen `main`-/`dev`-Branches und `tools.permissions.git: write`. Taskfelder: `workflow` (feature/bugfix/hotfix/release/other), für Releases zusätzlich `release_version`. Ohne Workflowfeld gilt die Titelheuristik. `git.remote` bleibt standardmäßig null; ein explizit konfiguriertes Remote wird ff-only gepullt. Es gibt keinen automatischen Push.

Normale lokale Abläufe führen nach echten Tests/Validatorprüfung Commit/Merge, für Hotfix/Release dev-Synchronisierung und für Release ein annotiertes semantisches Tag aus. Branchlöschung bleibt eine separate optionale approve/deny-Frage. Freigaben binden Task, Repository, exakte Argumente und tatsächliches Pushziel; SQLite verhindert mehrfaches Ausführen auch kopierter Grants. Bei aktiviertem Git werden Starts im gemeinsamen Store gegen andere aktive Taskleases serialisiert. H1c schützt auch Python-Kindprozesse, kopierte Git-Binaries, Bare-Repositories und Worktree-Metadaten. Remote-Transporte einschließlich lokaler Pushes über `git-receive-pack`, Docker-Daemonoperationen und Git-Schreibzugriffe auf Worktree-Metadaten außerhalb des CWD sind geschlossen, bis sichere Broker implementiert sind. Für die lokalen Workflows `git.remote` leer lassen.

Gitfehler pausieren mit Pflichtfrage. Vor einem Task-Commit speichert der Harness dessen Parent-ID, erlaubte Dateipfade und erwartete Commitnachricht. Nach einem Prozessabbruch wird ein vorhandener Commit nur dann zugeordnet, wenn Parent, Nachricht und geänderte Pfade dazu passen. Nach jedem Merge laufen Tests und die unabhängige Validation erneut auf dem Zielbranch; Hotfix und Release prüfen zusätzlich den synchronisierten `dev`-Branch. Ein fehlgeschlagener Zielbranch bleibt wartend und wird nicht als abgeschlossen gemeldet. Bereits erfolgte Merges werden über Git-Abstammung erkannt, Release-Tags auf Zielcommit und annotierte Form geprüft. Kein automatischer Reset, Stash, Konfliktentscheid oder Tag-Overwrite. Veränderte Branches/Tags bleiben gesperrt. Autorisierte Repair-Operationen binden Task, Zielcommit und erlaubte Dateipfade; nach Reparatur erfolgen erneut Tests und unabhängige Abnahme.
