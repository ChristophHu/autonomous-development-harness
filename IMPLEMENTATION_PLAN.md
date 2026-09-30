# Weiterer Implementierungsplan zur GAP_MATRIX

## Umgesetztes Paket PROVIDER-HEALTH1

OpenAI-kompatible Provider liefern einen gemeinsamen `ProviderHealth`-Bericht statt bloßem HTTP-Status: Erreichbarkeit, API-/Inventargültigkeit, konfigurierte Modellverfügbarkeit und diskrete redigierte Zustände. Ein explizites `kind: lmstudio` aktiviert ausschließlich für LM Studio den zusätzlichen nativen `/api/v1/models`-Read mit geladener LLM-Instanzprüfung; `/v1/models` allein gilt nicht als Loaded-Beleg. Geladene Instanzen müssen zugleich in der kompatiblen API sichtbar sein. Alle Health-Requests sind zeitlich begrenzt. CLI-Inventar, `status`, `doctor` und Start-Preflight nutzen die Berichte; Discovery bleibt im Laufzeit-Snapshot. RED/GREEN-Tests umfassen remote/nativ, geladen/nur heruntergeladen, Embeddings, defekte API-/Native-Antworten, Transportfehler, Konfigurationsvalidierung und CLI-/Preflight-Grenzen. Native `sh scripts/verify.sh`: **1126 Tests bestanden**, 100 % Statements/Branches/Funktionen (7127 Statements/2552 Branches, 39 Module/522 Funktionen), Ruff und Formatcheck bestanden. Punkt 39 wird nach dem Ursprungsprompt abgeschlossen; reale LM-Studio-Hardwareabnahme gehört weiterhin zu Punkt 38. Matrix: 43 erfüllt, 64 teilweise, 2 offen.

## Umgesetztes Paket MODEL-DISC1

Provider-Discovery liefert strukturvalidierte Modell-IDs an einen erneuerbaren Snapshot der `ModelRegistry`; die CLI nutzt diesen Snapshot statt einer eigenen Abfrage. Konfigurierte Modell-IDs bleiben vorrangig. Ein direkt per Profil gewählter Provider ohne konfigurierte ID kann genau ein entdecktes Modell verwenden; bei mehreren IDs, ungültigem Inventar oder unbekannten Tool-Fähigkeiten wird nicht geraten. Ein fehlgeschlagener Refresh verwirft den alten Snapshot. RED/GREEN-Tests decken Einzelauswahl, Mehrdeutigkeit, Konfigurationsvorrang, fehlerhafte Antworten, Snapshot-Austausch und CLI-Anzeige ab. Native `sh scripts/verify.sh`: **1105 Tests bestanden**, 100 % Statements/Branches/Funktionen (7033 Statements/2508 Branches, 39 Module/514 Funktionen), Ruff und Formatcheck bestanden. Punkt 40 schließt nach dem Wortlaut der Ursprungsspezifikation; Punkt 39/36 bleiben getrennt. Matrix: 42 erfüllt, 65 teilweise, 2 offen.

## Umgesetzte Pakete MCP-SEC1 und MCP-OBS1

MCP-SEC1: Konfiguration und Registry verlangen für externe lokale MCP-Kommandos eine explizite `trusted_local: true`-Bestätigung. Builtin-Server erhalten eine native macOS-Sandbox mit selektiven Dateiinhalt-Leseverboten unter `/Users`, `/private/var/folders`, `/private/tmp` und `/Volumes` und Wiedergenehmigung für Workspace, Harness-Code, Python-Laufzeit und optionalen Vault. Native Canaries belegen Zugriff innerhalb und Verbot außerhalb der Wurzeln. Wegen notwendiger Laufzeitleserechte und breiter Rechte außerhalb der genannten Pfadbereiche ist dies ausdrücklich keine vollständige OS-weite Leseisolation; die Matrixpunkte 93/94 bleiben teilweise.

MCP-OBS1: Ein eigenständiger stdio-Server im `mcp_servers`-Paket stellt ausschließlich `read_note`, `list_notes` und `search_notes` für sichtbare Markdown-Notizen bereit. Er nutzt die sichere dirfd-/`O_NOFOLLOW`-Dateizugriffsschicht, ignoriert Hidden-Pfade/Symlinks, begrenzt Scans/Treffer und wird durch `obsidian: read` sowie Profil-Toolfreigaben gegatet. Unerwartete vom Builtin beworbene Tools scheitern. Direkte, Protokoll-, Registry-, Profil- und native Prozess-E2Es decken Positiv- und Negativfälle ab. Punkte 54/107 werden vertieft, aber nicht vollständig geschlossen. Finale native `sh scripts/verify.sh`: **1097 Tests bestanden**, 100 % Statements/Branches/Funktionen (6994 Statements/2486 Branches, 39 Module/513 Funktionen), Ruff und Formatcheck bestanden. Matrix: 41 erfüllt, 66 teilweise, 2 offen.

## Strukturpaket MCP-SERVER-ORDNER

Der integrierte Filesystem-MCP-Server wurde ohne Verhaltensänderung nach `src/harness/mcp_servers/filesystem.py` verschoben. Das neue Python-Paket enthält künftig die integrierten MCP-Server; `mcp.py` bleibt Client und `tools.py` Registry. Builtin-Startkommando, Imports, Modul-Tests und `harness-filesystem-mcp`-Entry-Point referenzieren nur noch den neuen Paketpfad. Native `sh scripts/verify.sh`: **1088 Tests bestanden**, 100 % Statements/Branches/Funktionen (38 Module/504 Funktionen), Ruff und Formatcheck. Die Matrixzählung ändert sich durch diesen Strukturumbau nicht.

## Umgesetzte Pakete MODELCLI2 und FSMCP4

MODELCLI2: `harness models list/status` verbinden Provider-Health, dynamische Discovery und konfigurierte Registry zu einer deterministischen Übersicht. Je Modell werden ID, Provider, Alias, Tier und Verfügbarkeit angezeigt. Konfigurierte, aber nicht entdeckte Modelle sowie fehlende Provider bleiben sichtbar; Discoveryfehler geben keine Exceptiondetails aus. Alle acht mindestens geforderten CLI-Top-Level-Befehle/-Gruppen sind per CLI-Vertrag geprüft. Punkte 79 und 85 sind nach Abgleich mit den Mindestanforderungen des Ursprungsprompts erfüllt; persistente Discovery gehört zu Punkt 40.

FSMCP4: Der Filesystem-MCP-Referenzserver ergänzt `copy_file`, `exists` und `glob`. Copy nutzt die bestehenden sicheren dirfd-Read-/exklusiven Create-Pfade, überschreibt kein Ziel und verändert die Quelle nicht; `exists` folgt keinen Symlinks. Glob traversiert nur reguläre Workspace-Dateien, ignoriert Symlinks und `.git` und begrenzt Ergebnisse. Registry-Rechte klassifizieren Exists/Glob als Read und Copy als Write; Delete bleibt separat. Punkt 65 ist damit erfüllt; allgemeine Prozess-Leseisolation bleibt Punkt 93/94.

RED/GREEN-Tests umfassen Modellstatus, CLI-Gruppen, Pfadausbrüche, Symlinks, vorhandene Copy-Ziele, Rechte-/Limitgrenzen und eine native macOS-MCP-Ausführung. Finale `sh scripts/verify.sh`: **1087 Tests bestanden**, 100 % Statements/Branches/Funktionen (6890 Statements/2438 Branches, 37 Module/504 Funktionen), Ruff und Formatcheck bestanden. Matrix: 41 erfüllt, 66 teilweise, 2 offen.

## Umgesetzte Pakete CFG3, KEY2 und TASKCLI1

CFG3: Eine getrennte, sichere `config.example.yaml` deckt die im Prompt vorgeschlagenen fünf Modelltiers, sieben Profile, Git-Workflowpräfixe und Betriebsabschnitte ab. Modell-IDs sind austauschbare Platzhalter, keine im Code festgeschriebenen Annahmen. Dynamische Modellrates sowie HTTPS-Basic-/Bearer- und SSH-Credentialobjekte sind strikt typisiert; fehlerhafte Werte und Moduskombinationen scheitern bei der Validierung. Punkt 8 ist erfüllt, Punkt 7 bleibt wegen absichtlich erweiterbarer Felder und nicht durchgängig live abgenommener optionaler Betriebskonfiguration teilweise.

KEY2: Der Keychain-Provider akzeptiert einen expliziten absoluten Test-Schlüsselbundpfad für sämtliche Operationen. `set` sendet einen hexkodierten Wert über stdin an `security -i` und prüft per Readback, ohne Secrets in argv oder Fehlertext zu legen. Ein temporärer macOS-Schlüsselbund mit Leerzeichen im Pfad belegt Set/Get/Update/List/Exists/Delete, CLI-Eingabe und Environment-Fallback; er wird danach gezielt gelöscht. Der persönliche Schlüsselbund bleibt unverändert. Punkte 10 und 13 sind erfüllt; allgemeine Secret-Safety bleibt Punkt 11.

TASKCLI1: `tasks create` nimmt neben den bisherigen Positionsargumenten `--file` mit validiertem JSON/YAML-Taskvertrag entgegen; geschützte Lifecycle-/Ergebnisfelder, unbekannte Felder und übergroße Eingaben werden zurückgewiesen. `tasks watch` liest lokale Task-SSE-Ereignisse, nutzt `Last-Event-ID` für Wiederaufnahme, ignoriert wiederholte IDs und beendet sich bei terminalen Zuständen. Verbindungsfehler werden ohne Serverdetails angezeigt. Punkt 84 ist erfüllt; Punkt 79 bleibt wegen weiterer CLI-Gesamtbreite teilweise.

Abnahme der drei Pakete zusammen mit MCP1–MCP3: Native `sh scripts/verify.sh` bestand mit **1082 Tests**, 100 % Statements/Branches/Funktionen (6838 Statements/2410 Branches, 37 Module/501 Funktionen), Ruff und Formatcheck. Die Matrix enthält nun 38 erfüllte, 69 teilweise erfüllte und 2 offene Punkte von 109.

## Umgesetzte Pakete MCP1–MCP3 – lokale MCP-Toolschicht und Filesystem-Referenzserver

MCP1: Typisierte, standardmäßig deaktivierte `tools.mcp.servers`-Konfiguration; für ausdrücklich freigegebene lokale `stdio`-Server werden Protokollversion und `tools/list` geprüft, Input-/Outputschemas validiert und erlaubte Namen unter `mcp.<server>.<tool>` registriert. Externe Servertools sind konservativ destruktiv und benötigen globale `mcp.<server>: write`- sowie Profilfreigabe. Die bestehenden Tool-Audit-/Abbruchpfade bleiben verbindlich; MCP-Audit speichert keine ungeprüften Argumentwerte oder Inhalte. Nachrichten, Laufzeit und Prozessgruppenlebensdauer sind begrenzt.

MCP2: `harness-filesystem-mcp` ist ein separat startbarer Referenzserver für Workspace-relative Datei-/Verzeichnisoperationen. Read/List/Search, Create/Write/Mkdir/Delete/Move sind mit separaten Schreib-/Löschrechten, `dir_fd`/`O_NOFOLLOW`, `.git`-Sperre und Größenlimits umgesetzt. `write_file` ersetzt atomar, `create_file` räumt Fehlversuche auf. Move kopiert exklusiv und löscht danach die Quelle; bei Abbruch zwischen beiden Schritten kann eine zusätzliche Kopie verbleiben, aber keine stille Überschreibung des Ziels.

MCP3: Echte `stdio`-E2Es einschließlich nativer `sandbox-exec`-Abnahme für Read/Create/Move/Delete, Registry-/Profil-/Recovery-Gates, fehlerhafte und bösartige Serverantworten, Schemawechsel, Timeouts, Prozessgruppenkill, Pfadausbrüche, Symlinks, Audit-Canary und Konfigurationsfehler wurden per RED/GREEN getestet. Native `sh scripts/verify.sh`: **1043 Tests bestanden**, 100 % Statements/Branches/Funktionen (6721 Statements/2364 Branches, 37 Module/496 Funktionen), Ruff und Formatcheck bestanden. Die Matrixpunkte 54/65/93 werden vertieft, aber nicht voreilig geschlossen; Punkt 55 bleibt erfüllt. Streamable HTTP und allgemeine Read-Isolation externer Server bleiben eigenständige Folgepakete.

## Abgeschlossenes Paket SS1 – durchgängige Secret-Redaktion (Punkt 11 vertieft)

`AuditRecorder.sanitize()` erkennt sensible Feldnamen rekursiv und unabhängig von Groß-/Kleinschreibung bzw. Trennzeichen (API-Key, Token, Passwort, Secret, Credential, Authorization, Private Key, Cookie) und ersetzt konfigurierte Secretwerte in Freitexten. Die effektive Runtime-Konfiguration wird nicht verändert.

`Store` redigiert Tasks, Metadaten, Resultate, Fragen/Antworten, Events, Pläne, Subtasks, Validierungen und Korrekturfindings an den Schreibgrenzen. Interne Reads, API und CLI redigieren zusätzlich historische Task-/Event-/Frage-/Plan-/Validation-Daten; Requirement- und Recovery-Kontexte nutzen diese Read-Fassade ebenfalls. Fehlertext wird vor Persistenz und erneutem Propagieren bereinigt. Bekannte Legacydaten werden nicht rückwirkend in SQLite umgeschrieben. Nicht konfigurierte beliebige Geheimnisse in Freitexten sowie externe Plugin-/Bibliothekslogs lassen sich nicht allgemein erkennen; Punkt 11 bleibt daher teilweise.

Canary-TDD deckt Persistenz und Readback für Task/Metadaten, Frage/Antwort, Events, Plan/Subtask, Validierung/Korrektur, Fehlertext, API/CLI und Legacyzeilen ab. Native `sh scripts/verify.sh`: **996 passed**, 100 % Statements/Branches (6289 Statements/2182 Branches), 100 % Funktionen (35 Module/476 Funktionen), Ruff und Formatcheck bestanden.

## Abgeschlossenes Paket KEY1 – Secret-Provider und CLI (Punkte 10/13/79 vertieft)

`SecretProvider` definiert den injizierbaren Read-Vertrag; `KeychainSecretProvider` kapselt macOS-`security`, `EnvironmentSecretProvider` liest die Prozessumgebung, und `SecretResolver` behält Keychain-vor-Environment-Priorität sowie die bestehenden Aufrufer bei. Secret-Namen sind auf `[A-Z0-9_]{1,128}` begrenzt. Beim Setzen wird der Secretwert über stdin statt argv an `security` übergeben. Das CLI bietet `secrets list/set/delete/exists`; `list` filtert Accountnamen auf den exakten Harness-Service, und Fehlertexte geben keine Exceptiondetails oder Werte preis.

TDD prüft Providerinjektion, Priorität/Fallback, Namenvalidierung, stdin statt argv, servicegenaues Listing, Fehlerfälle und CLI-Exit-/Ausgabeverträge. KEY1-fokussiert: **53 Tests**, 100 % Statements/Branches für `security.py` und `cli.py`. Vollständige native `sh scripts/verify.sh`: **989 passed**, 100 % Statements/Branches (6229 Statements/2176 Branches), 100 % Funktionen (35 Module/462 Funktionen), Ruff und Formatcheck. Keine reale Keychain wurde verändert; ein isolierter Live-Keychain-Abnahmetest bleibt offen. Punkt 10/13/79 wird vertieft, aber nicht pauschal geschlossen; Punkt 11 bleibt ein separates Secret-Safety-Paket.

## Abgeschlossenes Paket CFG2 – typisierte Config-Schemas und atomarer Reload (Punkt 12 erfüllt)

`src/harness/configuration.py` definiert strikte Pydantic-Modelle für die bekannten Settings aus Harness, Pfaden, SQLite, Memory/Qdrant/Embeddings, Git/Docker, API, Logging, Provider/Modellregistry/Strategien, Profile, Testing/Coverage und Tools. Bekannt definierte Felder verwenden Strict-Typen und Wertebereiche; dynamische Registries und nicht spezifizierte Erweiterungen bleiben erhalten. Timeout- und Retry-Objekte sind selbst strikt, damit unbekannte Felder in diesen begrenzten Verträgen nicht stillschweigend akzeptiert werden.

`ConfigurationService.settings` liefert eine frisch aus dem aktuellen kompatiblen `.data`-Mapping validierte typed view. `reload()` baut YAML, dotenv, Prozessumgebung und Secretauflösung erneut auf, validiert den vollständigen Kandidaten und tauscht erst danach `configured`/`data` aus; bei einer Validierungsstörung bleibt der vorherige gültige Zustand erhalten. `Config.path` nutzt die typisierte Pfadsektion einschließlich erhaltener Extra-Pfade; `build()` liest die Loggingstufe typed. Alte `.data`-Consumer bleiben für den schrittweisen Umbau kompatibel.

TDD deckt promptnahe vollständige Konfiguration, bestehende Runtimekonfiguration, strikte Typen/Wertebereiche, Extra-Felder an Root und verschachtelten Sektionen, sichere feldpfadbezogene Diagnose, Runtime-Pfade und Reload samt Rollback ab. Native `sh scripts/verify.sh`: **976 passed**, 100 % Statements/Branches/Funktionen (6163 Statements/2162 Branches; 35 Module/453 Funktionen), Ruff und Formatcheck bestanden. Punkt 12 ist erfüllt. Punkte 7/8 bleiben teilweise: dynamische Rates/Credentials/Extensions sind bewusst nicht vollständig ausmodelliert, und die aktive Local-first-Beispielkonfiguration enthält nicht alle vorgeschlagenen Modelltiers/-Profile als konkrete Einträge.

## Abgeschlossenes Paket CFG1 – zentrale Konfigurationsauflösung (Punkt 6 erfüllt)

`ConfigurationService` ist die zentrale Implementierung; `Config` bleibt als rückwärtskompatibler Name für bestehende Runtime-/CLI-Aufrufer erhalten. Die Auflösung führt Defaults, YAML, `.env` und Prozessumgebung zusammen. Unterstützte dotenv-Schreibweisen und dokumentierte Variablen für Workspace/Vault, Qdrant, LLM, Git und API werden deterministisch ausgewertet; Prozessumgebung überschreibt `.env`. LLM-Endpoint-Overrides folgen dem in YAML oder Umgebung gewählten Provider. Secrets werden Keychain-first aufgelöst und fallen bei fehlendem Keychainwert auf `.env`/Environment bzw. YAML zurück. Secretwerte werden rekursiv maskiert.

Die pfadbezogene Validierung deckt bekannte Harness-, Pfad-, Datenbank-, Git-/Docker-, API-, Memory-/Embedding-, Modell-/Provider-/Registry-/Strategie-, Profil-, Logging-, Testing-/Coverage-, Tool- und Secretstrukturen ab. Fehler bei YAML oder dotenv geben keine Eingabewerte aus. `.env.example` dokumentiert lokale Overrides; `.env` ist bereits in `.gitignore` ausgeschlossen. CLI show/resolved/validate und Runtime nutzen dieselbe Config-Fassade.

TDD ergänzt Prioritätskonflikte, dotenv-Quotes/Kommentare/`export`, ungültige Variablen/Quotes/Ports, YAML-Fehler ohne Werte-Leak, Provider-Endpoint-Zuordnung, Secretpriorität, Bereichsvalidierung und Ladbarkeit der `.env.example`. Native `sh scripts/verify.sh`: **964 passed**, 100 % Statements/Branches/Funktionen (5995 Statements/2158 Branches; 34 Module/451 Funktionen), Ruff und Formatcheck. Punkt 6 ist erfüllt; Punkte 7/9/12 bleiben teilweise (nicht alle optionalen/erweiterbaren Felder sind als strikte Datenobjekte modelliert), 10 ohne Live-Keychain-Abnahme, 11 mit verbleibenden nicht-Config-Ausgabepfaden und 13 mit fehlenden Secret-CLI-Funktionen ebenfalls teilweise.

## Abgeschlossenes Paket OPS3 – SQLite-Verbindungslebenszyklus (Punkt 17 vertieft)

Warnungszuordnung bestätigte, dass `status`/`doctor` bei `_sqlite_health` offene SQLite-Verbindungen hinterließen: Der Connection-Context-Manager führt Transaktionsabschluss aus, schließt aber die Verbindung nicht. `_sqlite_health` nutzt jetzt explizites `try/finally`-Schließen, auch wenn `quick_check` scheitert. Die read-only URI (`mode=ro`), der Quick-Check und fail-safe Statusvertrag bleiben erhalten. Der zentrale `Database.connect`-Wrapper hatte bereits einen expliziten Close im `finally`; dort war keine Änderung erforderlich.

TDD deckt Close bei `available`, `unhealthy` und SQLite-Exception ab. Die gesamte CLI-Suite besteht mit `-W error::ResourceWarning` (**35 Tests**) und `cli.py` mit 100 % Statements/Branches. Native `sh scripts/verify.sh`: **930 passed**, 100 % Statements/Branches/Funktionen (5862 Statements/2074 Branches; 34 Module/448 Funktionen), Ruff und Formatcheck; im vollständigen Lauf keine ResourceWarnings. Punkt 17 bleibt insgesamt teilweise.

## Abgeschlossenes Paket OPS2 – Start-Preflight und Bereitschaftsdiagnose (Punkt 80)

`harness start` baut nun denselben validierten Runtimekontext wie die API und prüft vor PID-Registrierung Config sowie SQLite. Fehler in diesen kritischen Gates stoppen mit Exitcode 1, bevor Lifecycle-Datei oder API-Prozess gestartet werden. Providerrollen werden aus Profilen und Modellaliases ermittelt; aktivierte Provider mit Rolle werden parallel geprüft (maximal acht Worker, Gesamtwartefrist fünf Sekunden). Aktivierte, aber nicht verwendete, deaktivierte und referenzierte, jedoch nicht verfügbare Provider werden getrennt angezeigt. Provider- und Qdrant-Ausfälle sind optionale Warnungen und verhindern den API-Start nicht. Aus Buildfehlern werden keine rohen Exceptiontexte ausgegeben. Danach bleibt der bisherige echte Uvicorn-Startup- und `/health`-READY-Handshake bestehen.

TDD: kritische Sperre vor PID-/Serverstart, optionale Providerwarnung bei fortgesetztem Start, Primary-/Fallback-/Aliasauflösung, `unused`/`disabled`/`not_configured`, Health-Ausnahme-/Timeoutbehandlung und secret-freie Ausgabe. `cli.py`: 100 % Statements/Branches in der fokussierten CLI-/Lifecycle-Suite. Matrixpunkt 80 nun erfüllt; Punkte 81–83 bleiben erfüllt. Native `sh scripts/verify.sh`: **927 passed**, 100 % Statements/Branches/Funktionen (5860 Statements/2074 Branches; 34 Module/448 Funktionen), Ruff und Formatcheck bestanden.

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
