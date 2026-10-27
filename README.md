# openarm-humanoid-intel
Catalog + collector for public OpenArm / humanoid / bimanual robot-learning sources (compiled 2026-10-27).

- `data/sources.csv` — seed list (name, url, category, priority, source_kind)
- `scripts/collect.py` — fetches public metadata (license, pushed_at, README topics) -> `data/records.jsonl` (full 29-field schema; unfilled = null = unknown). Resumable; set `GITHUB_TOKEN` to avoid the 60 req/h limit.
- `scripts/filter.py [topic...]` — topics: openarm, humanoid, bimanual, vla, world_model -> `data/shortlist.md`
- `data/raw/` — raw GitHub API responses

Rules: missing = unknown, not 0; reported benchmark ≠ local result; check OpenArm version (1.0/2.0) and license before use; no private VinRobotics data into public services.
Technical fields (action_dim, control_rate_hz, loss...) are NOT auto-filled; they need per-source reading (paper/code) and must cite the source.
