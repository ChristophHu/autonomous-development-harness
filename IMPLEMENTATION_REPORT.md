# Implementierungsstand und Nachweise

Stand: 2026-09-27. Die Matrix wurde mit dem tatsächlichen Code synchronisiert, nicht pauschal als vollständig bewertet.

## Ergebnis dieser Runde

**Aktuelle Runde H1c – verbindliche OS-Prozessisolation:** `isolation.py` baut macOS-Seatbelt-Profile direkt als Argumente für `sandbox-exec`, ohne veränderbare Profildatei. ToolExecutor verwendet die Grenze für Shell, Test, Lint, Docker und Git einschließlich Remote-Zielauflösung; `git_status` delegiert an denselben kontrollierten Pfad. Kindprozesse erben die Kernelrechte. Untrusted Prozesse dürfen nur im kanonischen CWD schreiben, nicht in Git-Metadaten; Netzwerk, Unix-Daemon-Sockets und Mach-Zugriffe bleiben geschlossen. Der Scanner schützt `.git`, Gitdir-/Commondir-Pointer, Symlinks, vorhandene Bare-Repositories und vorhandene Hardlink-Aliase anhand Gerät/Inode. Derselbe Metadatenschutz gilt für mutierende Filesystemtools, inklusive Ziel- und Elternpfaden. Die Erzeugung neuer Hardlinks und Metadaten-Rename wurden real als blockiert geprüft.

Native Gitaufrufe erhalten einen separaten Ausführungsvertrag: ausschließlich ein kanonisches Git-Binary aus `/opt/homebrew/bin:/usr/bin:/bin`, außerhalb des schreibbaren CWD, darf starten. Task-/Profil-PATH kann dieses Binary nicht ersetzen. Externe Filter-/Shell-/Transporthelfer dürfen nicht starten; Hooks sind mit `core.hooksPath=/dev/null` deaktiviert. DYLD-/LD-/GIT-Environmentinjektionen werden unabhängig von der Env-Allowlist entfernt. Als Workspace wird die Dateisystemwurzel abgewiesen. Unverfügbare Sandbox, andere Plattformen oder fehlendes Git scheitern geschlossen; kein Prozess wird unisoliert erneut gestartet. Die frühere `sandbox_apply`-Ablehnung stammte von der verschachtelten Codex-Sandbox; die echte macOS-Abnahme außerhalb dieser Sandbox ist erfolgreich.

**Funktionale Grenze:** Remote-Clone/Fetch/Pull/Push benötigen gesonderte Transporthelfer, lokale Pushes starten beispielsweise `git-receive-pack` über eine Shell. Diese Pfade sowie Docker-Daemonoperationen und Worktree-Schreibzugriffe außerhalb des CWD sind bis zu sicheren Brokern nicht operativ. Ein passend genehmigter Remote-Delete scheitert ohne Referenzänderung; der verbrauchte Grant bleibt einmalig. Punkt 62 ist damit erfüllt, Punkt 56 wegen dieser Einschränkung wieder teilweise; 63/108 bleiben teilweise. Lokale Feature/Bugfix/Hotfix/Release- und Repair-/Recoveryabläufe bestehen unter Isolation. Leserechte für lokale Bibliotheken bleiben breit; aktive Prozessgruppen-Unterbrechung bei Abort/Leaseverlust ist ein separates Paket.

TDD: Die neue Modulschnittstelle wurde zuerst als fehlend nachgewiesen. Ein späterer echter Test bewies, dass ein schon vorhandener Hardlink den bloßen Pfadschutz umgeht; nach dem Inode-Alias-Schutz besteht derselbe Test für OS- und Filesystemtools. `test_process_isolation.py` enthält 28 Tests einschließlich gültiger Bare-Referenz und kopiertem Git, Python-Kindprozessen, Symlink-/Worktree-/Common-Pointern, Filter-/Hookschutz, Netzwerk-/Unix-Sockets, Sandboxfehlern und PATH-/Environmentinjektion. Keine Coverage-Ausnahmen oder unisolierten Fallbacks wurden eingeführt.

**Vorherige Runde H1b3 – autorisierte Zielbranchreparatur:** Nach fehlgeschlagener Zielbranchprüfung wird eine Pflichtfreigabe erzeugt, deren Grundhash Task-ID, kanonisches Repository, Zielbranch, konkreten fehlgeschlagenen Commit, Reparaturbranch und exakte Pfade bindet. Vor `approve` wird kein Reparaturbranch erzeugt. Die Ablehnung bleibt BLOCKED. Bei Zustimmung prüft der Harness unveränderten Ziel-HEAD, sauberen Workspace und freigegebene Frage, verbraucht den SQLite-Grant atomar und erstellt den Reparaturbranch vom exakt geprüften Zielobjekt. Crashfenster zwischen Antwort, Grant-Verbrauch und Branchwechsel werden reconciled; nach Reparatur commitet/merget der Harness nur genehmigte Pfade und führt Tests plus unabhängige Validation erneut auf dem Zielbranch aus. Integrationstests beweisen erfolgreiche Korrektur, Ablehnung ohne Mutation, fremde/veraltete Zielzustände und Freigabe-Wiederaufnahme.

**H1b4 – Operationsgrenzen und Klassifikationsaudit:** Git-Risiko/Approval/Remote sind als unveränderliches `GitOperation` modelliert und bilden die dynamische Tool-Risikostufe. `init`/`clone` erlauben nur dokumentierte begrenzte Optionen, erlaubte Protokolle und Ziele unterhalb des konfigurierten Workspace; Pfadtraversal, Symlink-Ausbruch, remote-helper-Protokolle, Inline-Credentials und unbekannte Optionen werden vor Prozessstart abgelehnt. Der ToolExecutor prüft dieselbe Workspacegrenze nochmals. Workflow-Routing erzeugt strukturierte, persistierte Entscheidungsereignisse für explizite Taskvorgaben bzw. deterministische Titelklassifikation.

Historischer Stand H1b: Punkte 56–61 waren abgenommen; Punkt 62 blieb wegen fehlender OS-Isolation teilweise. Der eingeschränkte lokale Aufruf meldete `sandbox_apply: Operation not permitted`. H1c löst die Isolation mit realer Abnahme außerhalb der verschachtelten Sandbox; aktuelle Funktionsgrenzen stehen oben.

Grenze: Konfliktzustände werden nicht automatisch destruktiv zurückgesetzt; sie bleiben menschlich prüfbar und können nach manueller Auflösung reconciled werden. Ein fehlgeschlagener Release-main wird als neuer Release-Task angelegt, statt einen freigegebenen Releaseablauf unbemerkt umzuschreiben.

Nach jedem Merge führt der Orchestrator Tests und unabhängige Validation im jeweiligen Zielbranch aus, bevor er fortfährt. Hotfix und Release prüfen main und danach den synchronisierten dev-Branch; Release setzt den Tag erst nach erfolgreicher Prüfung von main. Ein fehlgeschlagener Zielbranch erzeugt eine Pflichtfrage und verhindert completed. H1b3 ergänzt die erneute autorisierte Reparatur.

Drei neue Regressionstests prüfen Commit nach Git-Schreibvorgang/unterbrochenem SQLite-Update, Zurückweisung eines fremden Commits und reale Zielbranchfehler durch unabhängige dev-Änderungen (Feature sowie Hotfix-Synchronisierung). Lokale isolierte Gitrepositories, keine externen Remotes.

**Vorherige Runde H1b1 – objektgestützte Git-Wiederaufnahme:** Nach persistiertem Commit wird dessen echte Git-ID gespeichert. retry startet über die gemeinsame exklusive TaskService-Lease, prüft Repository, unveränderten Taskbranch und sauberen Arbeitsbaum, wechselt zum Taskbranch und führt erneut Tests/Validator aus. Primärmerge und Hotfix-/Release-Synchronisierung werden anhand tatsächlicher Git-Abstammung erkannt statt doppelt ausgeführt. Bereits vorhandene Release-Tags müssen annotiert sein und auf den richtigen Primary-Commit zeigen. Veränderte primäre Branches werden vor weiteren Merge-Mutationen abgewiesen. Bestehende Cleanupfragen bleiben erhalten.

18 lokale Git-Tests aus H1b1: sechs zunächst rote Unterbrechungsfälle vor/nach Merge, anschließend grün; weitere Negativfälle für Repository/Source/Dirty-Tree/fehlenden Branch, konkurrierende Änderungen vor Finish, falschen Tag, fehlende Annotation und veränderten Primärbranch. Ein echter Mergekonflikt bleibt unverändert bis zur manuellen Auflösung; danach kann die Arbeit fortgesetzt werden. Ein Retry mit frischen fehlgeschlagenen Tests erreicht keinen Merge. Keine externen Repositories oder kostenpflichtigen Modelle verwendet.

Historische Grenze nach H1b2: Repair und auditierte Klassifikation waren damals noch offen. H1b3/H1b4 schließen diese Verträge; H1c ergänzt die OS-Isolation. Aktuelle Grenzen und Matrixzählung stehen oben bzw. unten.

**Vorherige Runde H1a – Git-Freigaben und normale Workflowsteuerung:** Frozen GitApprovalTarget bindet Task-ID, kanonischen Workspace/Repositorypfad, exakte Argumentliste einschließlich Branch/Forceflag und bei Push die tatsächliche einzelne Push-URL. Die sichtbare HITL-Frage und ihr persistierter Reasonhash binden dieselben Daten. Globale Gitoptions, breite/mehrdeutige Pushes, mehrere tatsächliche Pushziele und Credential-URLs werden abgewiesen. Jeder Push benötigt Freigabe; es gibt keinen automatischen Push.

Gitpolicy trennt READ/WRITE: Ein Leserecht kann nicht committen. Branch-Deleteflags einschließlich Standardform `push origin --delete branch`, kombinierter Kurzflags und leerer Push-Refspecs benötigen eine passende Freigabe. Grant-Ausgabe reserviert die beantwortete Frage; unmittelbar vor Ausführung beansprucht ein atomarer SQLite-Update genau diese Frage/Reason als executed. Auch kopierte/parallel konsumierte Grants können sie nur einmal verwenden. Fehlversuche gegen andere Tasks, Branches, Repositories, Argumente oder geänderte Push-URLs verbrauchen die Freigabe nicht. Git-Umgebungen sind gefiltert; GIT_DIR-Repointing wird standardmäßig nicht übernommen.

GitWorkflowService ist standardmäßig deaktiviert. Bei ausdrücklicher Aktivierung werden Tasks über eine strukturierte und auditierte Workflowentscheidung (explizites Taskfeld oder deterministische Titelklassifikation) geroutet, auf sauberen Workspace geprüft und aus dev (Feature/Bugfix/Release) bzw. main (Hotfix) verzweigt. Ein optional ausdrücklich konfiguriertes Remote wird vorab ff-only gepullt. Gitaktivierte Starts beanspruchen den gemeinsamen Store exklusiv gegen andere aktive Taskleases. Interne Gitmetadaten können weder bei TaskService.create eingeschleust noch über patch geändert werden.

Nach realen Tests, Coverage und unabhängiger Validation: nur explizite einzelne Taskdateien mit literal Pathspecs committen (`--only`), Feature/Bugfix nach dev, Hotfix nach main und dann dev, Release nach main mit annotiertem semantischem Tag und anschließender dev-Synchronisierung. Feature/Bugfix/Hotfix erzeugen eine optionale separate Cleanupfrage. Der fachlich fertige Task bleibt Completed; approve/deny laufen über dieselbe TaskService-Antwortgrenze wie API/CLI. Approval löscht genau den freigegebenen Branch, Deny bewahrt ihn. Gitaufrufe sind als korrelierte STARTED/COMPLETED/FAILED-Calls auditiert; Nonzero-Ergebnisse sind Fehler, keine Erfolge.

Gitfehler, Dirty-Tree und falsche Branch-/Repo-/Phasenzustände pausieren mit Pflichtfrage. Es gibt keinen Auto-Stash, Reset, Konflikt-Auto-Merge oder automatischen Remote-Push. H1b1/H1b2 ergänzen kontrollierte Commit-/Teilmerge-Wiederaufnahme und Zielbranchabnahme; H1b3 genehmigte Repair-Verträge; H1b4 typisierte Operations-/Pfadverträge und auditierte Workflowklassifikation. H1c ergänzt Kernelisolation; Punkt 62 ist erfüllt, die Transportfunktionalität in Punkt 56 bleibt teilweise.

TDD-Nachweise (jeweils RED vor Fix): Standard-Remote-Delete umgeht Guard; Leserecht erlaubt Commit; mehrere Pushziele nicht gebunden; leere Testevidenz erreicht Git; Grant-Kopien doppelt verwendbar; zweiter Git-Task wechselt fremd geleasten Workspace. Danach GREEN. Insgesamt 63 neue Tests: 41 Safety-/Policytests, 14 Workflowvertragsprüfungen und 8 echte lokale Git-/Bare-Integrationstests. Keine Änderung eines externen Remote-Repositories und keine bezahlten Modellaufrufe.

Aktuelle Matrix: 17/109 erfüllt (15,60 %), 90 teilweise, 2 offen. Die folgenden Blöcke dokumentieren die vorherigen Pakete.

**Vorherige Runde F1b – kontrollierte Restarbeit und Prozess-Neustart:** Die bestehende ReconciliationService prüft Events, Git, Dateien und frische Tests. Ein separater auditierter Validator-Review klassifiziert jedes exakte Requirement strukturiert als completed/remaining/uncertain mit Kriterienreferenzen und Begründung. Nicht bestätigte, unbekannte oder fehlende Evidenz wird abgewiesen. Test-/Gitfehler halten Erfolgsclaims unsicher.

Der persistierte `recovery.scope`-Vertrag enthält alle offenen Kriterien/Requirements, Testprobleme und eine obligatorische erneute Prüfung. Jeder Planschritt muss explizite recovery_targets angeben; ausschließlich offene Ziele sind erlaubt, ihre Vereinigung muss den gesamten Scope abdecken. Reine Verify-Schritte dürfen nicht schreiben. Schreibpfade müssen workspace-relativ sein. Bestätigte Dateien ohne offene Kriterien auf derselben Datei sind unveränderlich; Dateien mit noch offenen Kriterien können gezielt erweitert werden. Abschluss bleibt von erneuten Tests und unabhängiger Gesamtvalidation abhängig.

ContextVar-basierte Toolkontrolle beschränkt Recovery-Mutationen auf filesystem.write/create und genau deklarierte Dateien. Unbeschränkte Shell-/Git-/Docker-Mutationen sind im Executor-Recovery-Scope verboten. Hash-/Symlinkschutz prüft bestätigte Artefakte vor Ausführung, nach Tests und unmittelbar vor Completion. Dies ist keine OS-Sandbox; vertrauenswürdige Testkommandos laufen weiterhin außerhalb des Executor-Scope. Unzulässige Pläne und Evidenz brechen fehlersicher ab und müssen korrigiert werden; es erfolgt kein stiller Fallback auf vollständige Wiederholung.

Durchgängiger E2E: Der erste echte Harness-Prozess implementiert Addition über den realen Executor/ToolRegistry und wird während des zweiten Executor-Schritts tatsächlich per SIGKILL beendet. SQLite hält ersten Schritt completed, zweiten running und die aktive Lease. Ein konkurrierender Claim scheitert. Nach deterministisch simulierter Leaseexpiry startet ein separater Harness-Prozess, liest echtes lokales Git/Dateien, führt frische zunächst fehlschlagende Tests aus, plant nur den fehlenden Multiplikationsschritt und erreicht mit echten Schreibtools, Tests, 100-%-Coverage und unabhängiger Validation Completed. Die bestätigte Datei bleibt hashgleich und die neue Lease wird freigegeben. Der lokale Git-Test nutzt vorhandenes Homebrew-Git und isolierte Repositorykonfiguration, keine Systemänderung oder Remote-Push.

RED-Vertrag vor Implementierung: Restarbeits-Scope fehlte; danach 24 Vertrag-/Negativtests und ein Crash-to-Completed-Prozesstest ergänzt (25 neue Tests). Punkte 52 und 98 jetzt erfüllt. Modellantworten stammen ausdrücklich aus kontrollierten Testadaptern; keine kostenpflichtigen Live-Modellaufrufe. Gesamt-DoD, OS-Prozessisolation und spezialisierte Workflows sind dadurch nicht automatisch erfüllt.

Die folgenden D1/I- und G1-Blöcke dokumentieren die vorherige Runde.

Priorität 1: **D1/I – nachvollziehbare Ausführung.** Migration v3 erweitert Tool-/Modellruns um Correlation und Metriken. AuditRecorder nutzt ContextVar-Kontexte für Task, Agent-Run, Agent, Profil und Komplexität; asyncio.to_thread übernimmt diese Kontexte. Requirements, Planner, Executor, Tester und Validator nutzen eine gemeinsame Agentinvokation. Parallel laufende Tasks sind voneinander isoliert.

Toolstart und Toolereignis werden atomar gespeichert; Abschluss/Fehler aktualisieren dieselbe Call-ID. Nonzero-Prozess-Exitcodes und HTTP-Fehlerstatus sind keine Erfolge. Bekannte Secrets und Credentialfelder werden in Auditpayloads redigiert. Große Prompts werden nicht in Modellspans oder Standardlogs gespeichert. Der Schutz sämtlicher Taskmetadaten und aller Loggingpfade bleibt weiter offen.

Modellspans enthalten Provider/Modell, Task/Agent/Profile/Komplexität, Start/Ende, Status, monotone Latenz, verfügbare Tokenwerte, konfigurierte Schätzkosten, Fallbackindex und Fehlertyp. Fehlende Werte bleiben NULL; unbekannte Preise bedeuten keine erfundenen Nullkosten. Cache-/Reasoningtokenfelder folgen der [offiziellen OpenAI-Dokumentation](https://developers.openai.com/api/docs/guides/predicted-outputs). Es wurden weder API-Schlüssel beschafft noch bezahlte Live-Modellaufrufe durchgeführt.

Priorität 2: **G1 – wahrheitsgetreuer Memory-Pfad.** ObsidianMemory validiert relative Notenpfade; absolute Pfade, Traversal und Symlinks außerhalb des Vaults werden abgewiesen. Append unterstützt neue verschachtelte Notizen; Search liest keine externen Symlinkziele.

MemoryService verbindet Laden, konfigurierbare überlappende Chunks und Indexierung mit echten Embeddings. Qdrant akzeptiert keine künstlichen Hashvektoren mehr. Vektoren müssen nichtleer, numerisch, endlich und dimensionskonform sein. Bestehende Collections werden gegen Größe/Cosine geprüft; HTTP-Fehler propagieren. Der Index referenziert Originalnoten und deren Contenthash. Retrieval lädt die aktuelle Markdownquelle, dedupliziert Dokumente und verwirft fehlende/veraltete Quellen. Vektortext ist keine Wissens-Source-of-Truth.

Priorität 3 einer früheren Runde: **Matrixkonsistenz.** 61 Einzelzeilen wurden damals aktualisiert. Die Matrixregression prüft vollständige Nummerierung, Statuszählung und wichtige Code-/Statusverträge. Nach H1a aktuell: Erfüllt 15/109 (13,76 %), Teilweise 92/109, Offen 2/109. Teilweise Punkte werden nicht künstlich gewichtet.

## Weiterhin vorhandener Kern aus der vorherigen Runde

Strikte Provider-/Profilauflösung ohne DeterministicProvider oder Ersatzplan; typisierter Task mit allen Mindestfeldern; Migrationen mit Legacy-Datenerhalt; atomare Start-Leases/Heartbeat und Konkurrenzschutz; gemeinsame TaskService-Mutationen; Lifecycle-/Pflichtfrage-/Terminalschutz.

Requirements prüfen Task/Parent/Answers/Decisions vor Obsidian/Qdrant/Repository und Human. Pläne haben validierte Abhängigkeitsgraphen, Resultate, Kriterien, Agent/Profile und Toolverträge. Der echte Executor hält Toolcalls und zugehörige Beobachtungen zusammen.

Echte Test-/Lint-/Coverage-Kommandos laufen im Workspace. Frische Berichte, executable Acceptance Criteria, beobachtete Dateien und separates Validatorprofil steuern Completion. Korrekturen führen zu erneuter Ausführung und Prüfung. Historische Subtaskausgaben sind Audit-Artefakte, keine Resume-Checkpoints.

## Tests und technische Schwellen

```sh
sh scripts/verify.sh
```

Letzter erfolgreicher Lauf nach H1c: **362 Tests**; **100 % Statements/Zeilen/Branches** (2651 Statements, 836 Branches), **23 Laufzeitmodule**, **262 vollständig geprüfte Funktionen**, keine ausgeschlossenen Coveragezeilen. `sh scripts/verify.sh` außerhalb der verschachtelten Sandbox bestanden; Ruff und Formatcheck bestanden. Eine upstream Starlette-TestClient-Deprecation-Warnung bleibt.

RED zuerst nachgewiesen: fehlende Tool-/Task-Korrelation, Vault-Traversal und Hashembeddings; anschließend GREEN. Die Matrixzählungsregression wurde ebenfalls zuerst rot ausgeführt.

Tests: zusätzlich test_recovery_contract.py und test_recovery_process.py; daneben test_reconciliation.py, test_next_packages.py, test_provenance_memory.py, test_gap_matrix.py und bestehende Gesamtsuite. Sie prüfen reale SQLite-/Datei-/Prozessressourcen und lokales Git sowie Scope-/Replay-/Symlink-/Completion-Schutz. Kontrollierte HTTP-/Providerfixtures ersetzen kostenpflichtige/externe Dienste; sie sind ausdrücklich keine Live-Modellabnahme.

## Nächste priorisierte Umsetzungen

1. **Sichere Transport-/Container-Broker (56/63):** Remote- und Worktreeoperationen sowie Docker erhalten begrenzte Ausführungsrechte, ohne allgemeine Shell-/Host-Daemonumgehungen zu öffnen.
2. **G2/I – Memorylifecycle, vollständige Services und Betrieb:** Index-Refresh/Cleanup, strukturierte Decisions, Config-/Model-/Readservices, HTTP-Fehlerverträge, CLI, Logrotation, Compose und separat autorisierte Live-Abnahmen.
3. **Aktive Prozesssteuerung:** Abort/Leaseverlust muss laufende Worker/Subprozesse tatsächlich stoppen; die implementierte Kernelisolation ersetzt diese Lebenszyklusfunktion nicht.

Gesamt-DoD (100) und Abschlussprüfung (105) bleiben offen. Coverage ersetzt weder diese Anforderungen noch vollständige Prozessisolation oder Live-Verifikation.
