# Productionization Changes

This copy converts the original exploratory workspace into a maintainable package.

## Structural Changes

- Source code lives under `src/qxg_platform`.
- Runtime configs live under `configs`.
- Tests live under `tests`.
- Large model and data artifacts are excluded from Git.
- The network API avoids pickle and uses JSON-safe payloads.

## Operational Expectations

- Keep model weights in an external artifact directory.
- Validate configs before deployment.
- Add model cards and metric reports for each production model.
- Pin deployment environments separately from research notebooks and scripts.

## Next Hardening Steps

- Add DVC or Git LFS if model artifact versioning is required.
- Add structured logging and metrics around frame latency, detection count, and graph size.
- Add an integration fixture with a tiny recorded clip.
- Add a real model registry entry for relevance and action prediction artifacts.
