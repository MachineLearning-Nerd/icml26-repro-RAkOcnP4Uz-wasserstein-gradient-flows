# Verification run

## Current run status

The historical run associated with this logbook was produced by the
superseded <code>repro/src/verify_mwgrad.py</code> toy/proxy verifier. It is
not a run of the root <code>verify_all.py</code> suite and is not a publication
gate.

No fresh full-suite run was performed during the documentation audit. To
produce auditable evidence, run:

    pip install uv
    uv sync
    uv run python verify_all.py

Then record the source commit, environment, parameters, dataset provenance,
raw outputs, and independent checks before updating the gate.
