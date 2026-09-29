# Autonomous Development Harness

Lokales, macOS-orientiertes Harness für zustandsbehaftete Entwicklungsaufgaben.

## Schnellstart

```zsh
cd autonomous-development-harness
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[dev]'
harness doctor
harness config validate
harness config show
harness config resolved
harness tasks create "Beispielaufgabe"
harness start
```

Die API läuft standardmäßig auf `127.0.0.1:8080`; Swagger ist unter `/docs` verfügbar.

```zsh
curl http://127.0.0.1:8080/health
curl -N http://127.0.0.1:8080/events/stream
```

Konfiguration wird mit der Priorität Defaults < `config.yaml` < `.env` < Environment < macOS Keychain aufgelöst. `config show` zeigt gesetzte Werte, `config resolved` zusätzlich wirksame Defaults; sensible Werte werden in beiden Ausgaben maskiert. `harness config validate` prüft Schema und lokale Betriebsgrenzen. Niemals Secrets in `config.yaml` eintragen.

## Modell- und Taskvertrag

Exakte Modell-IDs müssen über `LMSTUDIO_MODEL`, `OPENAI_MODEL` oder `OPENROUTER_MODEL` konfiguriert werden. Die Beispielprofile sind Local-first. Es gibt keinen künstlichen Produktionsfallback; unbekannte Provider/Profile, fehlende Modelle und ungültige Pläne scheitern.

Persistierte Modellaufrufe lassen sich mit `harness models usage --task-id 42 --provider lmstudio --limit 50` auswerten; Filter umfassen Task, Provider, Modell, Zeitraum und Pagination. Der gleichartige REST-Report liegt unter `GET /models/usage` und `GET /api/models/usage`. Reports enthalten Aufrufmetadaten und bekannte Token-/Kostenwerte, aber keine Prompts oder Secrets. Fehlende Messwerte werden ausdrücklich als unbekannt gezählt und nicht als Nullkosten ausgegeben.

Ein Task mit lediglich einem Titel wird zur Anforderungsklärung pausiert. Für den Entwicklungsablauf sind `goal`, `requirements`, `acceptance_criteria`, `test_commands` und `coverage_command` notwendig; sie können über REST angelegt/geändert werden. Kommandos sind Argumentlisten, keine impliziten Shellstrings. Coverage wird aus einem frisch erzeugten `coverage.json` gelesen. Pflichtfragen lassen sich mit passenden JSON-Taskfeldern beantworten und setzen den Task fort.

Die Beispielkonfiguration erlaubt Dateiänderungen und Shellprozesse für das Codingprofil. H1c erzwingt auf macOS `sandbox-exec`: Shell-, Test-, Lint- und Dockerprozesse samt Nachkommen dürfen nur im CWD schreiben, keine Git-Metadaten verändern und keine Netzwerk-/Daemonverbindung öffnen. Kontrolliertes Git darf nur sein kanonisches Binary starten; Hooks und externe Filter-/Transporthelfer sind gesperrt. Es gibt keinen Fallback ohne Isolation. Leserechte für lokale Laufzeitbibliotheken bleiben breit; dies ist keine vollständige Geheimnis-/Lesepfadisolation. Destruktive Dateilöschung bleibt deaktiviert. Live-Provider und Qdrant sind nicht live abgenommen.

Für Workspace-Recherchen stehen `search.files`, `search.text` und `search.symbols` mit Unterverzeichnis und Ergebnislimit bereit. Die Symbolsuche erkennt gängige Deklarationsschlüsselwörter in Python, JavaScript/TypeScript, Go, Rust und Java; sie ist kein vollständiger Compilerindex. Test- und Qualitätsaktionen heißen `test.run_tests`, `test.run_file`, `test.run_coverage`, `quality.lint`, `quality.format` und `quality.typecheck`. Sie erwarten explizite Argumentlisten; `test.run_file` hängt einen geprüften Workspace-Dateipfad an. Alle Prozessaktionen erfordern `shell: write` und laufen unter derselben Isolation wie `shell.execute`.

## Verifikation und Umfang

```sh
sh scripts/verify.sh
```

Prüft Tests, 100 % Statements/Zeilen/Branches, Funktionsabdeckung je Modul und Ruff. Die reale H1c-Abnahme benötigt eine macOS-Sitzung, die Sandboxaktivierung erlaubt. Eine bereits eingeschränkte übergeordnete Sandbox kann `sandbox_apply: Operation not permitted` erzeugen; dann scheitert die Ausführung geschlossen. Kontrollierte Provider-Fixtures treiben reale SQLite-/Dateisystem-/Subprozess-E2Es mit Testfehler, Korrektur, Coverage und Harness-Prozess-Neustart an. [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) beschreibt die umgesetzten Pakete; [GAP_MATRIX.md](GAP_MATRIX.md) hält sämtliche 109 Anforderungen und verbleibende Lücken fest. 100 % Codecoverage bedeutet nicht, dass alle Spezifikationspunkte abgeschlossen sind.

Agent-/Modell-/Tool-Runs sind inzwischen per Task und Profil korreliert; Migration v3 erhält vorhandene Daten. MemoryService indiziert Vaultnoten in überlappenden Chunks mit echten Embeddings und lädt beim Retrieval die Originalquelle. Qdrant ist kein Ersatz für Markdown: fehlende/veraltete Quellen werden verworfen, falsche Vektordimensionen und Dienstfehler sichtbar gemacht. Hashvektoren werden nicht als semantische Embeddings verwendet.

Unterbrochene Tasks können nach Ablauf der exklusiven Lease erneut gestartet werden. Recovery liest Events, Git, Dateien und frische Tests; ein unabhängiger Review bestimmt offene Requirements. Der neue Plan muss alle Restziele über `recovery_targets` abdecken und genaue relative `write_paths` deklarieren. Bestätigte Dateien ohne offene Kriterien sind geschützt. Recovery-Executor-Mutationen erfolgen ausschließlich über `filesystem.write/create`; unbeschränkte Shell-/Git-/Docker-Mutationen werden abgewiesen. Testkommandos laufen ebenfalls unter Kernelisolation. Alte Subtask-Erfolgsflags sind keine Resume-Checkpoints.

## Git-Workflows (opt-in)

`git.enabled` ist standardmäßig false. Aktivierung erfordert ein sauberes Repository mit vorhandenen `main`-/`dev`-Branches und `tools.permissions.git: write`. Taskfelder: `workflow` (feature/bugfix/hotfix/release/other), für Releases zusätzlich `release_version`. Ohne Workflowfeld gilt die Titelheuristik. `git.remote` bleibt standardmäßig null; ein explizit konfiguriertes Remote wird ff-only gepullt. Es gibt keinen automatischen Push.

Normale lokale Abläufe führen nach echten Tests/Validatorprüfung Commit/Merge, für Hotfix/Release dev-Synchronisierung und für Release ein annotiertes semantisches Tag aus. Branchlöschung bleibt eine separate optionale approve/deny-Frage. Freigaben binden Task, Repository, exakte Argumente und tatsächliches Pushziel; SQLite verhindert mehrfaches Ausführen auch kopierter Grants. Lokale Pushziele binden zusätzlich Gerät/Inode. Bei aktiviertem Git werden Starts im gemeinsamen Store gegen andere aktive Taskleases serialisiert. H1c schützt auch Python-Kindprozesse, kopierte Git-Binaries, Bare-Repositories und Worktree-Metadaten.

Der Git-Broker unterstützt Clone, Fetch und `pull --ff-only <remote> <branch>` von lokalen Repositories, HTTPS- und SSH-Remotes. Für HTTPS muss `tools.git.allowed_hosts` mindestens den exakten Zielhost enthalten; TLS-Zertifikate werden geprüft. Das Git-Sandboxprofil erlaubt Netzwerk nur zum kurzlebigen Loopback-Proxy. Dieser prüft den Zielhost und verbindet sich mit einer zuvor geprüften öffentlichen IP. Credentials werden hostgenau über `tools.git.credentials` mit Benutzername und Secret-Namen aus SecretResolver/macOS-Keychain (oder gleichnamiger Umgebungsvariable) zugeordnet. Der Basic-Auth-Header existiert nur in der Laufzeitumgebung des Git-Prozesses und wird aus Ergebnis-/Audittexten redigiert; URL-Credentials, lokale Credential-Helper/Headers und Redirects bleiben gesperrt. SSH benötigt eine exakte `tools.git.ssh.allowed_hosts`- und `host_keys`-Konfiguration sowie pro Host `credentials` mit Username und SHA256-Fingerprint eines im gleichnamigen Benutzer-Agent geladenen Schlüssels. Es werden ausschließlich öffentliche Schlüssel gelistet; die private Identität verbleibt im SSH-Agent. DNS wird vorab auf öffentliche IPs aufgelöst und die SSH-Verbindung über einen kurzlebigen Tunnel auf die ausgewählte IP gepinnt. Strict host-key checking ist erzwungen; Agent-/Port-/Weiterleitungs- und globale/user SSH-Konfiguration werden nicht übernommen. SSH unterstützt hier denselben begrenzten Einzel-Branch-Pushvertrag; weitergehende SSH-Zertifikate/Agent-Lifecycle sind nicht implementiert.

HTTPS- und SSH-Push/Branchlöschung brauchen persistierte, zielgebundene Freigaben sowie konfigurierte Credentials; unterstützt wird genau ein expliziter Branch-Refspec. Nach erfolgreichem Push wird genau der betroffene lokale Trackingref aktualisiert oder gelöscht. Lokaler Push und Branchlöschung benötigen ebenfalls eine persistierte Freigabe; lokale Pushziele müssen Bare-Repositories sein. Worktree-Schreibrechte außerhalb des CWD werden nur nach Prüfung von Gitdir, Commondir und Backlink gewährt. Fehlgeschlagene Vorprüfungen verbrauchen keinen Ausführungsgrant. Nach Prozessstart bleibt ein Grant einmalig, auch bei ungewissem Ausgang oder Abbruch.

`push -u`/`--set-upstream` ist für genau einen expliziten lokalen Quellbranch und ein konkretes Remote-Ziel freigegeben. Der Broker entfernt das Flag vor dem FD-/URL-Transport und schreibt den Upstream erst nach erfolgreichem Push und Trackingabgleich als `remote/branch`; Deletes, `HEAD` und mehrdeutige Quellen werden abgewiesen. Der Fetch-Broker unterstützt nun transportübergreifend mehrere explizite Branch-Refs, Tags und Prune nur im Remote-Tracking-Namespace; mehrere Push-Refs und weitere Git-Optionen bleiben offen. HTTPS unterstützt derzeit Basic-Auth über das Secret-Broker-Mapping, aber noch keine Token-Erneuerung oder andere Auth-Schemata. Bei Task-Abort oder Leaseverlust beendet RunControl taskeigene Prozessgruppen erst mit SIGTERM und eskaliert nötigenfalls zu SIGKILL. Taskeigene Provider-, Tool-HTTP-, Embedding- und Qdrant-Requests nutzen abbrechbare AsyncClient-I/O mit konfigurierbaren Gesamt-/Connect-/Read-Grenzen (`timeout: {total: 30, connect: 5, read: 30}` im jeweiligen Abschnitt). Eigene synchrone Client-Injektionen werden nur vor und nach dem Aufruf geprüft; serverseitig bereits angenommene Requests können dort weiterlaufen. Der SSH-Agent-/Host-Key-Vertrag und die lokale SSH-Server-/Agent-Clone-/Push-E2E bestehen nativ.

## Begrenzter Docker-/Compose-Broker

`docker.execute` akzeptiert nur `status`, `logs`, `start` und `stop` für den Harness-eigenen Qdrant-Dienst. Beliebige Docker-Argumente, Container-Exec, Build, Compose-Overrides, fremde Images, Bind-Mounts, TCP-Daemonendpunkte und `down`/Volume-Löschung sind nicht verfügbar. Der Broker validiert die Compose-Datei auf exakt das erlaubte Service-/Volume-/Portschema, schreibt für jeden Lauf einen unveränderlichen temporären Snapshot, nutzt einen lokalen Unix-Socket und begrenzte Prozesszeit/-ausgabe. Qdrant ist auf `v1.19.0` festgelegt, bindet HTTP nur an Loopback und verwendet ein persistentes Named Volume. Docker ist in der Beispielkonfiguration standardmäßig verweigert; erst `tools.permissions.docker: write` aktiviert auch Start/Stop. Die CLI-/Daemonintegration ist nicht live abgenommen; der Broker ist durch gemockte Prozessverträge getestet.

Gitfehler pausieren mit Pflichtfrage. Vor einem Task-Commit speichert der Harness dessen Parent-ID, erlaubte Dateipfade und erwartete Commitnachricht. Nach einem Prozessabbruch wird ein vorhandener Commit nur dann zugeordnet, wenn Parent, Nachricht und geänderte Pfade dazu passen. Nach jedem Merge laufen Tests und die unabhängige Validation erneut auf dem Zielbranch; Hotfix und Release prüfen zusätzlich den synchronisierten `dev`-Branch. Ein fehlgeschlagener Zielbranch bleibt wartend und wird nicht als abgeschlossen gemeldet. Bereits erfolgte Merges werden über Git-Abstammung erkannt, Release-Tags auf Zielcommit und annotierte Form geprüft. Kein automatischer Reset, Stash, Konfliktentscheid oder Tag-Overwrite. Veränderte Branches/Tags bleiben gesperrt. Autorisierte Repair-Operationen binden Task, Zielcommit und erlaubte Dateipfade; nach Reparatur erfolgen erneut Tests und unabhängige Abnahme.
