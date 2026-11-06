#!/usr/bin/env bash
# Static + API + frontend checks for OpenArm Research Studio (no simulator rollout, no model loading).  Run from the repository root.
set -u; PY=${PY:-python3}; rc=0
echo "== python syntax"; $PY -m py_compile app/backend/*.py app/tests/test_backend.py || rc=1
echo "== backend + API tests"; $PY -m unittest discover -s app/tests -v 2>&1 | tail -n 5; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "== frontend syntax"; for f in app/frontend/js/*.js; do node --check "$f" || rc=1; done
echo "== frontend tests"; node --test app/tests/test_frontend.mjs 2>&1 | grep -E "^# (pass|fail)"; [ ${PIPESTATUS[0]} -eq 0 ] || rc=1
echo "== eslint (if installed)"; (cd app/frontend && npx --no-install eslint --config eslint.config.js js) && echo "eslint clean" || echo "eslint unavailable or reported problems"
exit $rc
