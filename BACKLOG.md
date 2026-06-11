# Backlog — Single-Camera Dashboard (Additional Tasks)

Scrum-lite backlog for the 4-week additional tasks (Karim & Anas).

## Work split

- **Karim** — dashboard core, ship panel, occlusion log, click-to-edit ID,
  integration, documentation, unit tests.
- **Anas** — manual testing & QA (vessel clicking, ID corrections, bug
  reports), demo prep, team report contribution.

## Sprint goals

- **Week 1** — Project setup + Feature 1 (Ship Panel).
- **Week 2** — Feature 2 (Occlusion Log).
- **Week 3** — Feature 3 (Click-to-edit ID).
- **Week 4** — Polish, docs, tests, demo prep.

## User stories

### Feature 1 — Ship Panel
- [x] List vessels visible at the current frame (Karim)
- [x] Show each vessel's current position (Karim)
- [x] Show how long each vessel has been tracked (Karim)

### Feature 2 — Occlusion Log
- [x] Detect disappear/reappear gaps per track (Karim)
- [x] Display occlusions newest-first with timecode (Karim)

### Feature 3 — Click-to-Edit ID
- [x] Capture clicks on the video frame (Karim)
- [x] Map a click to the vessel under it (Karim)
- [x] Reassign a track ID from the clicked frame onward (Karim)
- [x] Persist corrections without touching the original MOT file (Karim)

### Cross-cutting
- [x] Input selection: existing runs + manual upload (Karim)
- [x] Unit tests for data layer (Karim, validated by Anas)
- [x] Manual QA: vessel clicking, ID corrections, bug reports (Anas)
- [x] README + usage guide (Karim)
- [ ] Live demo script + screencast fallback (Karim & Anas)
- [ ] Contribution to team report (due 5 June)
