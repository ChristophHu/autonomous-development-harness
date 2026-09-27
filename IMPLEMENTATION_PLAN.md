# Weiterer Implementierungsplan zur GAP_MATRIX

## Aktuelle priorisierte Umsetzung und Fortsetzung

Implementiert: D1/I, G1, F1a/F1b, H1a–H1b4 und H1c. H1c ergänzt verbindliche macOS-Kernelisolation für Toolprozesse und deren Nachkommen; Punkt 62 ist erfüllt. Die Isolation schließt unsichere Transport-/Docker-Helfer und externe Worktree-Schreibpfade, deshalb ist Punkt 56 wieder teilweise. Ergebnisse und genaue Funktionsgrenzen: [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md).

Nächster Vorrang: sichere Broker für Git-Remote-/Worktree- und Dockeroperationen (56/63), danach G2/I – Memorylifecycle, vollständige Services und Betrieb. Aktive Prozessgruppen-Unterbrechung bei Abort/Leaseverlust bleibt separat offen (5/108). Jeder Block benötigt RED/GREEN und `sh scripts/verify.sh` mit unveränderter 100-%-Schwelle. Die H1c-Abnahme muss außerhalb einer verschachtelten Sandbox erfolgen, die macOS `sandbox_apply` verbietet; im Harness gibt es keinen unisolierten Fallback.

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
