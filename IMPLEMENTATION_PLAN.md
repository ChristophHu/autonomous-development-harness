# Weiterer Implementierungsplan zur GAP_MATRIX

## Abgeschlossenes Paket OPS1 – CLI-Service-Lifecycle und Diagnostik (Punkte 80–83)

Der CLI-Lifecycle ist auf `ServiceLifecycle` vereinheitlicht. PID-Datensätze sind schema-validiert, atomar geschrieben und enthalten nur einen SHA-256-Fingerprint aus Prozessstartzeit/Kommando statt Klartext-Prozessdaten. Ein exklusiver Lock verhindert parallele Starts. `start` meldet READY erst nach Uvicorn-Startup und erfolgreichem Harness-Healthcheck und entfernt seinen PID-Eintrag beim Ende oder Startfehler. `stop` verifiziert vor SIGTERM die Prozessidentität, wartet begrenzt auf Ende und löscht keine zwischenzeitlich ersetzte PID-Datei. `status` liest SQLite ausschließlich über `mode=ro` und meldet Task-/Provider-/Servicezustände. `doctor` liefert fail-safe Prüfungen für Runtime, Konfiguration, lokale Verzeichnisrechte, optionale Docker-/Qdrant-Voraussetzungen, Port und Provider. TDD deckt PID-Reuse, ungültige/stale Einträge, Lock, atomare Dateirechte, Readiness, Timeout und Diagnosefehler ab. Native `sh scripts/verify.sh`: **922 passed**, 100 % Statements/Branches/Funktionen (5792 Statements/2048 Branches, 34 Module/445 Funktionen), Ruff und Formatcheck.

Matrixstatus: 81–83 erfüllt; 80 teilweise, da eine explizite Provider-/Abhängigkeitsübersicht als `start`-Preflight fehlt. Punkt 79 (restliche CLI-Funktionsbreite) bleibt teilweise.

## Abgeschlossenes Paket E2 – Plan-/Diff-/Git-Abgleich

Validator gleicht Plan und Executor-Ergebnisse ab, verifiziert Mutationsclaims gegen Tool-Evidenz und read-only Git-Statusdelta und bindet Pfade an Workspace sowie deklarierte `write_paths`. Git-Root/Branch/HEAD und relevante Statuspfade werden vor/nach Execute gelesen; unerwartete Änderungen und Identitätswechsel blockieren Completion. Native Abnahme: **863 Tests**, 100 % Statements/Branches/Funktionen (33 Module/414 Funktionen), Ruff und Formatcheck. Noch offen bleiben der semantische Diff-Hunk-/Requirementsabgleich sowie persistente Correction-Workitems.

## Abgeschlossenes Paket E3 – Workspace-Mutationsnachweis

Vorher-/Nachher-Snapshots erfassen Inhalte, Typen, Modi, Symlinkziele und Verzeichnisse des Workspace zwischen Executor und Testphase. Das schließt bereits dirty Pfade, Shell-Mutationen und Git-ignorierte Dateien ein. Symlinks werden nicht verfolgt; unlesbare, instabile oder spezielle Einträge sowie fehlende Snapshots blockieren Completion. Konfigurierte Datenbank-, Vault- und Logpfade sowie `.git` sind ausgeschlossen; Git-Identität/-Status laufen getrennt über den Broker. Native Abnahme: **876 Tests**, 100 % Statements/Branches/Funktionen (33 Module/416 Funktionen), Ruff und Formatcheck.

## Abgeschlossenes Paket E4a – Persistenter Correction-Store

SQLite-Schema v5 und `CorrectionRepository` persistieren Findings mit stabiler taskbezogener SHA-256-ID, Plan-/Subtaskbezug, Evidenz und Status. Wiederholte Findings sind dedupliziert; atomare Events begleiten echte Anlage/Änderung und Statuswechsel. Migration, FK-Verhalten, Parallelzugriffe und Rollbacks sind getestet. Native Abnahme: **886 Tests**, 100 % Statements/Branches/Funktionen (33 Module/426 Funktionen), Ruff und Formatcheck. Die Orchestrator-Synchronisierung bleibt absichtlich außen vor.

## Abgeschlossenes Paket E4b – Correction-Loop-Anbindung

1. `CorrectionFinding` als typisierten Vertrag ergänzt; EvidenceValidator erzeugt kategorisierte/rule-stabile Findings direkt aus Prüfpfaden. Legacy-`errors` bleibt für Anzeige/Kompatibilität erhalten; keine Freitextklassifikation.
2. Findings werden mit Task-/Planbezug idempotent gespeichert. Vor Execute werden offene Items geladen und atomar auf `in_progress` gesetzt; deren Kategorie/Regel/Evidenz geht als JSON-Korrekturkontext an den Executor.
3. Fehlerhafte Revalidierung setzt aktive Items zurück auf `open`; bei erfolgreicher Revalidierung werden sie geschlossen. Attempts und Audit-Events laufen über das bestehende Repository. Neustartlogik öffnet bei Prozessabbruch zurückgebliebene `in_progress`-Items wieder.
4. `TaskRepository.transition(..., completed)` blockiert atomar bei offenen oder `in_progress`-Korrekturen.
5. TDD deckt strukturiertes Finding, erfolgreichen Retry, ausgeschöpftes Retrybudget, Resume nach Abbruch, atomaren DoD-Gate und Ablehnung nachträglicher Findings nach Taskabschluss ab. Native `sh scripts/verify.sh`: **893 passed**, 100 % Statements/Branches (5586 Statements/1988 Branches) und Funktionen (33 Module/427 Funktionen), Ruff und Formatcheck bestanden.

## Abgeschlossenes Paket E1 – unabhängige Validator- und Task-DoD-Absicherung

Coverage-/Reviewbelege werden streng geprüft; Shellbasierte Test-, Lint- und Coverageverifikation läuft über auditiert registrierte ToolRegistry-Aktionen mit echten Exitcodes. Pflichtfragen blockieren den Validator sowie atomar den SQLite-Übergang zu `completed`. TDD und vollständige native Abnahme: **848 Tests bestanden**, 100 % Statements/Branches/Funktionen (33 Module/409 Funktionen), Ruff und Formatcheck. Noch offene Vertiefungen: Git-Diff-/Planabgleich und persistente, präzise Correction-Workitems; Matrixpunkte 50/51/99 bleiben teilweise.

## D2 – Live-Smoke vom Nutzer bestätigt

Der Nutzer meldete `Isolated Qdrant smoke test passed: healthy=True persisted_after_restart=True cleaned=True`. Damit sind Live-Health, Punktpersistenz über Service-Restart und Cleanup des isolierten Compose-Projekts bestätigt. Die Gap-Matrix bleibt bei 21/63/64 teilweise, weil Upgrade-/Backup-/Monitoring-/Langzeitbetriebsnachweise darüber hinausgehen.

## Abgeschlossenes Paket PR1 – begrenzte Provider-Retries (Punkte 29/30/37 vertieft)

Provider wiederholen konfigurierte transiente HTTP-Statuscodes begrenzt mit exponentiellem Backoff; numerisches `Retry-After` wird gedeckelt berücksichtigt. Mehrdeutige Transportfehler, normale 4xx-Antworten und ungültige Erfolgsantworten werden nicht wiederholt. Task-/Lease-Cancellation stoppt den Backoff und unterbindet weitere Versuche sowie Router-Fallback. TDD deckt Erfolg nach transientem Fehler, Budgetende, Statusfilter, ungültige Retry-After-Werte, Delay-Cap, Cancellation und Konfigurationsgrenzen ab. `sh scripts/verify.sh`: **826 Tests bestanden**, 100 % Statements/Branches/Funktionen, Ruff und Formatcheck bestanden.

## Abgeschlossenes Paket G3 – Embedding-Konfiguration und Batching

Default-Dimensionen und konfigurierbare Batchgröße durch Config→Orchestrator→Provider→Qdrant vereinheitlicht; indexsortierte/validierte Embedding-Antworten, gebündelter Qdrant-Upsert und begrenzte MemoryService-Batches ergänzt. Alte Punkte werden bei einem Fehler weiterhin nicht bereinigt. Fokussierte Memory-Abdeckung: 100 % Statements/Branches. Der volle Testlauf bleibt wegen sandbox-exec-/native-Isolation-Fehlern unbestätigt; Live-Embedding/Qdrant-Abnahme ist offen. Details: [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md).

## Abgeschlossenes Paket U1 – Model-Usage-Read-Service und Reports

Repositorybasierte, validierte Usage-Auswertung mit gemeinsamen REST-/CLI-Verträgen, Filterung und Pagination. Berichte erhalten fehlende Token-/Kostenwerte als unbekannt und aggregieren nur tatsächlich gemeldete Werte; Prompts und Secrets werden nicht ausgegeben. `usage.py` und die gesamte CLI erreichen im fokussierten Nachweis 100 % Statements/Branches; Details und Grenzen: [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md).

## Umgesetzt: D2-Isolation für die Qdrant-Live-Abnahme (Live-Smoke bestätigt)

Der rein interne `live_smoke_test` nutzt eine zufällig erzeugte Compose-Projekt-ID, ein damit namespacetes eigenes Named Volume und einen dynamischen Loopback-Port. Er prüft Health, legt eine einmalige Collection und einen Prüfpunkteintrag an, startet ausschließlich den isolierten Service neu, liest den Eintrag erneut und entfernt abschließend exakt das Testprojekt samt Volume. Der bestehende Produktions-/Agentpfad kann weder `restart` noch Cleanup aufrufen. Die CLI verlangt `harness qdrant-smoke --confirm`.

Initial war der Docker-Daemon nicht verfügbar. Danach bestätigte der Nutzer die reale Ausgabe `healthy=True persisted_after_restart=True cleaned=True`; Health, Neustartpersistenz und Cleanup sind live verifiziert. Vollständige E1-Abnahme danach: **848 Tests bestanden**, 100 % Statements/Branches/Funktionen (33 Module/409 Funktionen), Ruff und Formatcheck.

## Aktuelle priorisierte Umsetzung und Fortsetzung

Implementiert: D1/I, G1, F1a/F1b, H1a–H1b4, H1c, lokaler Git-Broker, lokales Push-Tracking, HTTPS-/SSH-Transporte, native SSH-Server-E2E, hostgebundene SSH-Identität und HTTPS Basic/Bearer-Credentials mit Secretauflösung/Redaktion, genehmigter Einzel-/Multi-Ref-Push, `push -u`-Upstreamabgleich sowie H1d-Fetch-Verträge für Multi-Branch-Refs, Tags und begrenztes Prune. Außerdem: Qdrant-spezifischer Docker-Compose-Broker, aktive RunControl-Prozessgruppenunterbrechung bei Task-Abort/Leaseverlust (Punkt 66 erfüllt) sowie kooperativer HTTP-Abbruch für Provider, Tool, Embedding und Qdrant mit konfigurierbaren Gesamt-/Connect-/Read-Grenzen. Such-/Test-/Qualitätswerkzeuge 67–69, typisierte REST-/Task-/Event-/SSE-/OpenAPI-Verträge 73–77 und Config-CLI mit tiefer Defaultauflösung, Quellpriorität, rekursiver Secret-Redaction und Schema-Validierung (86 erfüllt) sind vorhanden. Dockeraktionen sind status/log/start/stop für ein exakt validiertes Compose-Manifest; keine beliebigen Daemonbefehle. Punkt 62 bleibt erfüllt; Punkt 56 bleibt wegen OAuth-Erneuerung und weiterer Git-Optionen teilweise; Punkt 63/64 bleiben bis zu Live-Daemon-, Health- und Betriebsabnahme teilweise. Punkte 5/108 bleiben wegen weiterer Kontroll-/Betriebsanforderungen teilweise. Ergebnisse: [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md).

Aktuelle Runde abgeschlossen: **G2 – Qdrant-Index-Lifecycle und stale-chunk-Bereinigung** (Punkte 72/107 vertieft, weiterhin teilweise). Qdrant-Scroll paginiert mit exaktem Quellenfilter; Reindex lädt aktuelle Chunks vor jeder Bereinigung hoch, `unindex` entfernt nur die Note und `reconcile` synchronisiert den Vault und löscht nur Harness-markierte Obsidian-Punkte. Fremde/unmarkierte Quellen bleiben erhalten. Vollständige native Abnahme: **768 Tests bestanden**, 100 % Statements/Branches/Funktionen über 32 Module/388 Funktionen; Ruff und Formatcheck bestanden.

Der isolierte D2-Smoke ist live bestätigt. Nächster Ausbau für Punkte 21/63/64 wäre ein separat abgegrenzter Upgrade-/Backup-/Monitoring-Vertrag; kein solcher Betriebstest wird aus dem Smoke-Ergebnis abgeleitet.

Vorige Runde abgeschlossen: M2 – SQLite→Obsidian-Entscheidungsprojektion (Punkte 18/107 vertieft, weiterhin teilweise). `harness memory sync` erzeugt atomar wiederaufbaubare YAML-Frontmatter-Notizen aus SQLite und fasst nur manifestgeführte Harness-Dateien an. Native Abnahme: 757 Tests, 100 % Coverage auf 32 Modulen/381 Funktionen.

Abgeschlossenes G2-Paket:
1. Qdrant bietet validiertes, paginiertes Scrollen mit exakt gefilterter Quellenabfrage und Wiederholungsschutz für Cursor.
2. Reindex liest bestehende Quellpunkte, upsertet alle aktuellen Chunks inklusive Source-Marker und entfernt veraltete IDs erst nach erfolgreichem Upload; Embedding-/Upsertfehler lösen keine vorzeitige Bereinigung aus.
3. `unindex` entfernt markierte Punkte einer validierten Quelle. `reconcile` indexiert sichere Markdown-Dateien, bereinigt nicht mehr vorhandene Harness-Quellen und lässt fremde, unmarkierte oder unsichere Quellen unangetastet.
4. TDD deckt Kürzung, Failure-before-prune, gelöschte Dateien, Pagination, Cursorfehler, Fremdpunkte, Symlinks und Idempotenz ab; Memory-Module erreichen separat 100 % Statements/Branches.
5. Punkte 72/107 wurden vertieft, nicht pauschal geschlossen; Live-Dienst, Monitoring/Backup und Betriebsintegration bleiben ausstehend.

D2 – Docker-Live-Daemon-/Health-/Persistenzabnahme (Punkte 21/63/64): zuerst Read-only Doctor-Prüfung von lokalem Unix-Socket, Daemonversion, Compose-Plugin und exakt erwartetem Qdrant-Service; isolierter kurzlebiger Daemon und temporäres Volume, keine bestehende Instanz, keine Volume-Löschung und kein Start ohne explizite Zustimmung. Bei fehlendem isoliertem Daemon wird der Live-Betriebsteil nicht durch Mocks ersetzt. Anschließend verbleibende Memory-Health-/Betriebsintegration.

Danach G2/I – Memorylifecycle, vollständige Services und Betrieb. Jeder Block benötigt RED/GREEN und `sh scripts/verify.sh` mit unveränderter 100-%-Schwelle; kein unisolierter Git-Fallback.

Die folgenden Abschnitte dokumentieren den ursprünglichen Gesamtfahrplan und dessen frühere Befunde; für aktuelle Statuswerte gilt die Gap-Matrix.

Aktualisierung nach Umsetzung am 2026-09-27: Die drei nächsten Schritte A1/B1/C1-D1-E1 sind im zentralen Runtimepfad implementiert; siehe [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) und die aktualisierte [GAP_MATRIX.md](GAP_MATRIX.md). Tests und Coverage werden durch `sh scripts/verify.sh` geprüft. Die folgenden Befunde und Arbeitspakete dokumentieren den ursprünglichen Planungsstand vor diesen Änderungen; verbleibende Vertiefungen sind in der Matrix aktuell ausgewiesen.

Ursprünglicher Planungsstand: 2026-09-27. Grundlage: Gap-Matrix, Quellcodeprüfung und Master-Spezifikation.

## Priorisierungsentscheidung

Zuerst muss der Harness echte Ausführung und belegten Abschluss unterscheiden. Danach werden Taskmodell, Lifecycle und gemeinsame Services vervollständigt. Darauf bauen Requirement Completion, Toolausführung, unabhängige Validation, Recovery und Git-Workflows auf. Memory, vollständige Schnittstellen und Betriebsabnahme schließen die Folge ab.

Vier konkrete Befunde bestimmen diese Reihenfolge:

- `ModelRegistry.get()` fällt bei unbekannten Namen auf `DeterministicProvider` zurück. Dessen konstante Antwort kann vom Executor als erfolgreich gewertet werden. Dieser Testersatz gehört außerhalb des Produktionspfads.
- `Validator.validate()` prüft lediglich die Erfolgsflags der Executor-Ausgaben. Requirements, Acceptance Criteria, tatsächliche Tests und Coverage tragen noch nicht zur Abschlussentscheidung bei.
- Ein auf den Workspace begrenztes Prozess-CWD verhindert allein keinen Dateizugriff außerhalb dieses Verzeichnisses. Git-Löschfreigaben sind bislang an die Aktion, aber nicht an Repository und konkreten Branch gebunden; etwa `git push origin --delete branch` wird vom aktuellen Guard nicht erkannt.
- Spezifikationspunkte 52–53 verlangen `Inspect → Reconcile → Replan → Execute` und schließen Step-Level-Checkpoints ausdrücklich aus. Die aktuelle Wiederverwendung gespeicherter Schritt-Erfolge muss durch Prüfung des tatsächlichen Zustands ersetzt werden. Subtask-Ergebnisse bleiben Audit-Artefakte, keine Wiederanlaufgarantie.

## Reihenfolge und Abhängigkeiten

| Paket | Vorrang | Umfang | Zugeordnete Gap-Punkte | Abhängigkeit | Abnahmekriterium |
|---|---|---|---|---|---|
| A | P0 | Echte Provider-/Profilauflösung, korrekter Toolcall-Vertrag, keine künstlichen Produktions-Erfolge | 24–42, 48, 95, 104, 106 | keine | Unbekannte/ausgefallene Provider und ungültige Antworten erzeugen nachvollziehbare Fehler; ein realer Toolcall hat eine zuordenbare Beobachtung. |
| B | P0 | Taskmodell, versionierte SQLite-Migrationen, Lifecycle, Questions/Decisions und gemeinsame Application Services | 5, 14–17, 19, 45, 49, 71, 87, 108 | minimaler Fehlervertrag aus A | API und CLI benutzen dieselben Services; genau ein Ausführer besitzt einen Task; Pflichtfragen und ungültige Übergänge blockieren Ausführung/Abschluss. |
| C | P1 | Anforderungsanalyse, Context-Auflösung, ausführbare Pläne und Dependency-Scheduling | 23, 43–47, 102 | A, B; Memory-Verträge aus G | Retrieval-Reihenfolge aus Punkt 44 ist eingehalten; beantwortete Fragen werden nicht wiederholt; nur freigegebene, vollständige Pläne werden ausgeführt. |
| D | P0/P1 | Tool-Schemas, Rechte, Prozessausführung, Suche, HTTP, Test-/Lint-Werkzeuge und Audit-Korrelation | 54–55, 63, 65–70, 93–94 | B; Provider-Toolcall-Vertrag aus A | Unerlaubte/ungültige Aufrufe werden vor Ausführung abgewiesen; Exitcode, Streams und Timeout werden ausgewertet; Calls sind Task, Agent und Profil zugeordnet. |
| E | P1 | Unabhängige Validation, echte Test-/Coverage-Reports, Korrekturaufträge und Task-DoD | 50–51, 68–69, 91–92, 97, 99 | B, C, D | Ein Task erreicht completed nur bei erfüllten Requirements/Acceptance Criteria, bestandenen Tests/Coverage und ohne offene Pflichtfragen oder Korrekturen. |
| F | P1 | Recovery anhand Events, Git, Dateisystem und Tests; Replan verbleibender Arbeit | 52–53, 98 | B, C, D, E | Ein neu gestarteter Prozess ermittelt tatsächliche Arbeit und Restarbeit; ein gespeichertes Erfolgsflag allein löst weder Skip noch Retry aus. |
| G | P1 | Memory Service, Obsidian, echte Embeddings, Qdrant und Compose | 18, 20–23, 64, 72, 107 | B, A-Konfiguration | Decisions und Taskwissen sind strukturiert abrufbar; Vektordimensionen stimmen; Fehler sind sichtbar; SQLite ist weiterhin die operative Wahrheit. |
| H | P1 | Feature-, Bugfix-, Hotfix- und Release-Workflows samt zielgebundener Löschfreigabe | 56–62 | B, D, E, F | Branchbasis, Tests, Commit, Merge und Synchronisation folgen der Spezifikation; jede Branch-Löschung benötigt eine passende persistierte Freigabe. |
| I | P2 | Vollständige Konfiguration/Secrets, REST/SSE, CLI/Doctor, Logging und Observability | 2, 6–13, 21, 73–86, 88–90, 96 | jeweilige Services aus A–H | Alle spezifizierten Bedienwege sind dokumentiert und geprüft; Secret-Priorität und Redaction gelten durchgehend; Live-Diagnosen unterscheiden verfügbar/deaktiviert/fehlerhaft. |
| J | Abschluss | Gesamtabnahme und Belege pro Matrixzeile | 1, 3–4, 100–101, 103, 105, 109 | A–I | Jede nummerierte Anforderung hat Implementierungs-, Verifikations- und Betriebsbelege; keine offenen oder teilweise erfüllten Anforderungen bleiben. |

Überlappende Punktnummern sind beabsichtigt: beispielsweise entstehen Testwerkzeuge in D, ihre verbindliche DoD-Wirkung in E. Ein Matrixpunkt wird erst nach Abschluss aller zugehörigen Teile als erfüllt bewertet. Punkt 78 bleibt bereits erfüllte Grundlage und wird bei der CLI-Regression mitgeprüft.

Konkrete Arbeitsfolge: **A1 → B1 → D1 + G1 → C1 → E1 → F1 → H1 → G2/I → J**. G1 liefert zunächst die Retrieval-Verträge und minimale Integration, G2 die vollständigen Live-/Betriebsnachweise. D1 und G1 können nach B1 unabhängig bearbeitet werden. Die für A1/B1 nötigen typisierten Konfigurationsfelder aus I werden bereits dort eingeführt; die gesamte Konfigurations-/CLI-Abnahme folgt später. Dadurch wartet kein Paket auf einen erst nachgelagerten Vertrag.

## Unmittelbar nächste Umsetzungspakete

### A1 – Produktionsfehler und echte Modellantworten

1. Verhaltenstests zuerst: unbekannter Provider, fehlendes/ungeeignetes Modell, ungültiges Profil, alle Fallbacks ausgefallen, ungültiger Plan und bloße Textantwort ohne Arbeitsnachweis.
2. Testprovider in Test-Fixtures verlagern; Produktionsregistry kennt nur explizit konfigurierte Provider/Modelle. Modellnamen, Fähigkeiten und lokale Priorität werden aus Konfiguration aufgelöst.
3. Providerantworten vereinheitlichen: Text, Toolcalls, Usage und Fehler. Toolcall-IDs und Gesprächsverlauf einschließlich zugehöriger Tool-Ergebnisse erhalten.
4. Executor-Ergebnis vom abschließenden Validatorurteil trennen. Eine Providertextantwort allein ist kein Nachweis erledigter Implementierung.
5. Bestehende Tests auf echte Verhaltensverträge umstellen und einen vollständigen HTTP-Toolcall-Dialog mit einem kontrollierten Provider-Testserver prüfen.

Abnahme: Ein nicht konfiguriertes oder fehlgeschlagenes Modell führt niemals zu einer erfolgreichen Task-Abnahme. Failover und Toolcall-Verlauf sind reproduzierbar belegt. Betroffene Module: agents.py, providers.py, core.py, config.yaml und deren Tests.

### B1 – Task-Vertrag, Zustandsautomat und Service-Grenze

1. Tests für Migrationsupgrade ohne Datenverlust, Parent/Child-Tasks, Requirements/Constraints/Acceptance Criteria und Planversionen schreiben.
2. Task-/Subtaskmodelle nach Punkt 14 vervollständigen; zulässige Zustandsübergänge und deren Events zentral definieren.
3. Application Services für Erstellen/Ändern/Starten/Beantworten/Abbrechen einführen. API und CLI delegieren an dieselben Services.
4. Gleichzeitige Starts durch atomare Besitzübernahme mit begrenzter Laufzeit verhindern. Abbruch und Neustart lassen historische Ergebnisse bestehen.
5. Jede offene Pflichtfrage blockiert Ausführung und Abschluss unabhängig vom aktuellen Taskstatus. Antworten setzen automatisch den passenden fachlichen Schritt fort, einschließlich Requirements-, Recovery- und Aktionsfragen.

Abnahme: Zwei konkurrierende Startanfragen führen zu genau einer Ausführung; Antworten verändern weder den falschen Task noch eine fremde Freigabe. Neue Pflichtfragen verhindern completed. Betroffene Module: database.py, core.py, neuer Application-Service-Bereich, api.py, cli.py.

### C1/D1/E1 – Ein belastbarer vertikaler Entwicklungsablauf

1. Requirement Completion prüft Task → Parent → Answers → Decisions → Obsidian → Qdrant → Repository → Human. Fehlende Dienste und fehlende Informationen werden unterschieden.
2. Pläne enthalten erwartete Ergebnisse, Akzeptanzkriterien, Agent/Profile, Tools, Test-/Validationstrategie und einen validierten Abhängigkeitsgraphen. Zyklen, fehlende Referenzen und doppelte IDs werden abgewiesen.
3. Toolschemas vollständig validieren. Globale Rechte und Profilrechte gemeinsam durchsetzen. Prozess-Argumentlisten korrekt ausführen; tatsächliche Zugriffsgrenzen und Shell-Modus ausdrücklich festlegen und testen.
4. Test-, Coverage-, Lint- und Typecheck-Läufe mit Exitcodes, Reports und Korrelation persistieren. Formatierung bleibt eine verändernde Werkzeugaktion und wird vor der finalen Prüfung ausgeführt.
5. Validator unabhängig aus Originaltask, Plan, Diff, Testbelegen und Acceptance Criteria entscheiden lassen. Findings erzeugen zuordenbare Korrekturaufträge mit erneuter Prüfung.
6. Einen E2E-Test in einem isolierten Beispielprojekt durchführen: echter Dateiedit → absichtlich fehlschlagender Test → Korrektur → bestandene Tests → dokumentierter Abschluss.

Abnahme: Erfolgreiche Agentenmeldungen bei fehlschlagenden Tests oder unerfüllten Kriterien führen zu FAILED/Korrektur. Ein bestandener Task besitzt nachvollziehbare Belege für sämtliche DoD-Bedingungen.

## Danach konkret abzuarbeiten

- **F:** Recovery liest Events, Git-Status/Diff, relevante Dateien und aktuelle Testreports. Daraus entsteht ein eigener Reconciliation-Report mit erledigter Arbeit, Unsicherheit und Restarbeit sowie ein neuer Plan. Die bisherigen Checkpoint-Abkürzungen werden entfernt. E2E: tatsächlicher Prozessabbruch, neuer Harness-Prozess, Teiländerung, Replan und erfolgreicher Abschluss. Menschen werden für nicht ableitbare Entscheidungen oder erforderliche Aktionsfreigaben einbezogen.
- **G:** Obsidian-Pfadschutz und strukturierte Task-/Decision-Notizen; Memory-Service statt direktem Agenten-Qdrant; Embeddingmodell und Dimensionen prüfen; keine Hashvektoren als semantisches Produktions-Retrieval; Compose-Health/Persistenz und Restart-Live-Test.
- **H:** Feature/Bugfix basieren auf aktualisiertem dev; Hotfix auf main mit Synchronisation nach dev; Release auf dev, finale Validation, Merge nach main, Tag und Synchronisation. Dirty-Tree und Konflikte benötigen nachvollziehbare Zustände. Freigaben binden Task, Repository, konkreten Branch/Remote und Operation; alle Delete-Argumentvarianten und Umgehungswege werden geprüft.
- **I:** Typisierte Defaults und Merge-Priorität nach Spezifikation, sichere Keychain-Bedienung, redigierte Ausgabe; vollständige Task-/Model-/Config-CLI, Start/Stop/Status/Doctor; typisierte HTTP-Fehler, SSE-Reconnect/Mehrclient-Tests; rotierende Logs und Model-Usage-/Latenz-/Kostenzuordnung.
- **J:** Matrixzeilen einzeln abnehmen, sämtliche fehlenden spezifizierten Operationen prüfen, lokale Installation/Start/Stop und realen Coding- sowie Recovery-Workflow verifizieren. Aggregate wie Punkt 1/100 werden erst zuletzt geschlossen.

## TDD- und Nachweisregeln

Für jedes fachliche Verhalten: zuerst einen relevanten fehlschlagenden Test nachweisen, minimale Implementierung, Refactoring und betroffene Regressionstests. Nach einem fertiggestellten Paket folgen die gesamte Suite und die statischen Checks. Coverage darf weder durch künstliche Erfolgspfade noch durch unbegründete Ausschlüsse erreicht werden.

Jede Matrixzeile dokumentiert: konkrete Implementierung/Servicepfade, zugeordnete Verhaltenstests, Integrationstest-Ergebnis, verbleibende Tiefe und Datum der letzten Prüfung. Ein dokumentierter Plan verändert keinen Erfüllungsstatus.

Unit-/Vertragstests verwenden kontrollierte Fixtures. Integrationen mit Git, SQLite und lokalen Dateien laufen gegen echte isolierte Ressourcen. Qdrant, Embeddings und Modellprovider erhalten separat ausgewiesene Live-Tests; fehlende Dienste werden als nicht verifiziert ausgewiesen.

Statements, Branches, Funktionen und Zeilen brauchen einen prüfbaren 100-%-Nachweis mit technischer Schwelle. Die aktuelle pytest-cov-Konfiguration erzwingt Gesamtcoverage mit Branchmessung; ein gesonderter Funktionsnachweis ist im bisherigen Fahrplan nicht belegt und wird ergänzt.

Letzter protokollierter Lauf vor dieser Planung: 76 bestandene Tests und 100 % Gesamtcoverage; Ruff bestanden. In diesem Planungsschritt wurde die Testsuite nicht erneut ausgeführt.
