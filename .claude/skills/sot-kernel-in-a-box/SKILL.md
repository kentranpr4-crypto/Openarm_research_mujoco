---
name: sot-kernel-in-a-box
description: Build and operate a file-based Source of Truth (SOT) with provenance labels, source-file fingerprints, append-only verification logs, fail-loud checks, adversarial bites tests, and a digest-bound publish gate. Use when an organization has conflicting facts across files, needs an auditable registry, must detect source drift, or needs verified exports. v1.3-transfer adds an optional Jev/System One seam for candidate triage only; it never verifies facts, resolves conflicts, or replaces deterministic gates.
---

# SOT Kernel-in-a-Box v1.3-transfer

Create an auditable chain from canonical source files to a machine-readable registry and verified outputs:

```text
source_canonical → adapter → registry → baseline/check → publish gate → exports
```

Keep the gates deterministic. Use AI to inspect, propose, draft, or write adapters; never use an LLM as the verifier.

## Non-negotiable invariants

1. Treat each canonical source file as the only editable authority for its facts. Treat registry entries and output documents as derivatives.
2. Give each canonical source one human owner.
3. Preserve provenance. Map evidence-backed facts to `verified_primary`; keep unverified numbers as `claim` with `verified_value: null`; use `flagged` for known conflicts.
4. Fail loudly on content violations. Do not downgrade errors merely to obtain a clean run.
5. Trust a gate only after `sot_bites.py` proves that it catches injected faults.
6. Require a current publish token tied to the registry digest before releasing an output.
7. Keep project data outside this skill. Do not modify the kernel scripts for one project's data model; write or adapt a project adapter instead.

Read [LY_THUYET_CHUYEN_GIAO.md](LY_THUYET_CHUYEN_GIAO.md) for the model and its limits; read [README.md](README.md) for reproducible quick-start commands.
Read [JEV_DECISION_SEAM.md](JEV_DECISION_SEAM.md) only when the user asks to evaluate or integrate Jev/System One.

## Optional Jev/System One decision seam

The seam defaults to `OFF` and sits outside the canonical chain. It may propose
`new / duplicate / conflict / legacy / unsure`, rank review priority, or suggest an owner
queue. It may not edit canonical sources or registry, change provenance, resolve conflict,
accept baseline drift, create a publish token, or mark a fact verified.

`SHADOW` only logs predictions. `SELECTIVE` may sort a review queue but cannot disposition
a record. Timeout, malformed, `unsure` or low confidence returns to the maintainer without
changing the old workflow.

## Choose the operation

- **Create a new SOT:** follow the full setup workflow below.
- **Audit an existing SOT:** inspect `PROJECT.yaml`, `sot.yaml`, `sot.hygiene.yaml`, canonical sources, adapters, registry, hashes, verification log, and publish token; then run `sot_check.py` and `sot_bites.py`.
- **Accept an authorized source change:** rerun the adapter, review the registry diff, run baseline `accept`, then run the full check. Never accept drift before confirming the owner and intended change.
- **Publish an artifact:** run the gate, verify its token is current, then publish through `sot_publish.py`.
- **Clean provenance:** use `hygiene.py` operations only after inspecting the target files and configuration. Back up or work on a copy for mutating operations.

## Set up a new SOT

1. Copy `project_template/` into a user-approved target directory.
2. Fill `PROJECT.yaml` with project identity and ownership.
3. Configure canonical paths, registry paths, hashes, logs, and export rules in `sot.yaml`.
4. Translate the organization's status vocabulary conservatively in `sot.hygiene.yaml`. Never map an ambiguous status to verified merely to pass the guard.
5. Start with one to three high-value canonical sources and roughly twenty important facts.
6. Create a deterministic adapter in the project's `adapters/` directory. Use the bundled CSV adapter as a pattern. Require stable unique IDs, explicit provenance, deterministic ordering, and no invented values.
7. Run the adapter and review the registry diff against the canonical source.
8. Initialize the source baseline.
9. Run the full check until it passes for substantive reasons.
10. Run all bites tests. Do not declare the SOT live unless every expected fault is detected.
11. Establish a recurring check and require all external artifacts to pass the publish gate.

Use `examples/demo_org/` as a disposable worked example. Never treat its fictional data as project data.

## Run the tools

Set `SKILL_DIR` to this skill folder and `PROJECT_DIR` to the target SOT project. Prefer an isolated Python environment. The tools require Python 3.8+, `pyyaml`, and `ruamel.yaml`.

```bash
python3 "$SKILL_DIR/tools/sot_baseline.py" "$PROJECT_DIR" init
python3 "$SKILL_DIR/tools/sot_check.py" "$PROJECT_DIR"
python3 "$SKILL_DIR/tools/sot_bites.py" "$PROJECT_DIR"
python3 "$SKILL_DIR/tools/sot_publish.py" "$PROJECT_DIR" status
```

Inspect each tool's CLI usage before a mutating or publish operation. `sot_bites.py` copies the target to a temporary directory and injects faults only into that copy; verify this remains true if the script is ever changed.

## Authorized source-change ritual

1. Confirm the canonical source owner authorized the change.
2. Rerun the relevant adapter.
3. Inspect the registry diff, including provenance and stable IDs.
4. Run:

```bash
python3 "$SKILL_DIR/tools/sot_baseline.py" "$PROJECT_DIR" accept
python3 "$SKILL_DIR/tools/sot_check.py" "$PROJECT_DIR"
```

5. Recreate the publish token through the gate. A registry change must invalidate the old token.

## Completion criteria

Report the SOT ready only when:

- canonical source ownership is recorded;
- adapters are deterministic and registry diffs were reviewed;
- provenance guard passes without dishonest mappings;
- baseline check passes with no unexplained changed or missing sources;
- verification log records the run;
- every bites test detects its injected failure;
- publish status is valid for the current registry digest;
- remaining claims and flagged conflicts are listed explicitly.

If a dependency is unavailable, a source owner is unknown, or an evidence conflict cannot be resolved, report the exact blocker. Do not label the system clean.

## Transfer edition limits

The publish token binds sources, configuration, adapter code, baseline and registry bytes. It does not prove that an arbitrary output document uses registry facts correctly. Review the artifact against evidence before release. These local files provide integrity checks, not tamper-proof access control. Human authorization remains necessary for source ownership and external release.
