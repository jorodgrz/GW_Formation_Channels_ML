# POSYDON Integration Status

**Status:** WIRED — generator integrated into UnifiedEnsembleGenerator  
**Last Updated:** December 4, 2025

---

## Why POSYDON?

POSYDON (Fragos et al. 2023) provides self-consistent binary evolution tracks
that couple MESA stellar models with interpolation engines and population
synthesis samplers. It offers:

1. Detailed post-interaction binary grids that complement COMPAS and COSMIC.
2. Modern prescriptions for common-envelope and mass transfer phases.
3. Consistent interfaces for Monte Carlo sampling plus catalog-level HDF5
   outputs that fit directly into the existing ensemble pipelines.

Adding POSYDON improves model diversity while avoiding the maintenance burden
of legacy session-based APIs.

---

## Recent Progress

- [DONE] Exported `configs/infrastructure/environment-posydon.yml` and documented
  activation steps.
- [DONE] POSYDON v2.2.0 installed in editable mode under `simulators/posydon/posydon/`.
- [DONE] CLI wrapper at `pipelines/ensemble_generation/posydon/generate_ensemble.py`
  that interfaces with `posydon-run-grid`.
- [DONE] Wired into `UnifiedEnsembleGenerator` - can be invoked via multi-code interface.
- [PLANNED] Requires POSYDON interpolation grids (HMS-HMS, CO-HeMS, etc.) downloaded via
  `posydon-setup-pipeline` before production runs.

---

## Usage

POSYDON is now accessible via the `UnifiedEnsembleGenerator`:

```python
from pipelines.ensemble_generation.multi_code.unified_generator import (
    UnifiedEnsembleGenerator,
    PopSynthCode,
)

# POSYDON-only ensemble
gen = UnifiedEnsembleGenerator(
    codes_to_run=[PopSynthCode.POSYDON],
    output_base="experiments/runs/posydon_ensemble",
    n_systems_per_run=10000,
)

# Multi-code ensemble (COMPAS + COSMIC + POSYDON)
gen = UnifiedEnsembleGenerator(
    codes_to_run=[PopSynthCode.COMPAS, PopSynthCode.COSMIC, PopSynthCode.POSYDON],
    output_base="experiments/runs/multi_code_ensemble",
    n_systems_per_run=10000,
)

# Run ensemble (requires POSYDON grids configured)
gen.run(n_alpha_points=10, test_run=True)
```

**Note:** Production runs require POSYDON interpolation grids downloaded via
`posydon-setup-pipeline`. See [Next Steps](#next-steps) below.

---

## Integration Plan

| Phase | Description | Effort | Owner |
| --- | --- | --- | --- |
| 1 | Bootstrap POSYDON environment (conda env + compiled interpolators) | 0.5 day | [DONE] Done |
| 2 | Mirror COMPAS/COSMIC CLI flags in `pipelines/ensemble_generation/posydon/generate_ensemble.py` | 1 day | [DONE] Done |
| 3 | Harmonize parameter grids (α_CE, λ_CE, kicks, metallicity) and metadata schema | 0.5 day | [DONE] Done |
| 4 | Validate 3×100-system smoke tests locally; profile 10k system timing | 0.5 day | [PLANNED] Requires grids |
| 5 | Wire into `UnifiedEnsembleGenerator` (add POSYDON code enum branch) and downstream loaders | 0.5 day | [DONE] Done |

Target completion: **mid-December 2025**, aligned with AWS COMPAS delivery so
multi-code priors (COMPAS + COSMIC + POSYDON) are available before the first SBI
training run.

---

## Dependencies

- POSYDON GitHub repo cloned under `simulators/posydon/` (see README stub).
- MESA-compatible toolchain (gcc ≥ 11, OpenMP, HDF5).
- Python packages: `posydon`, `mesa_reader`, `h5py`, `numpy`, `scipy`.
- Storage budget: ~25 GB for POSYDON grids + intermediate HDF5 files.

---

## Open Tasks

1. **Environment provisioning**
   - [x] Export `environment-posydon.yml` under `configs/infrastructure/`.
   - [x] Document build steps in `docs/simulator_notes/POSYDON_INTEGRATION.md`.

2. **Generator implementation**
   - [x] Create CLI + driver stub (`generate_ensemble.py`).
   - [ ] Port the `POSYDONPopulation` sampler into a reusable helper.
   - [ ] Implement `generate_parameter_grid()` mirroring COMPAS/COSMIC flags.
   - [ ] Support sparse-grid slicing (`--start-index/--end-index`) for AWS runs.

3. **Metadata + Outputs**
   - [ ] Emit HDF5 tables using the shared schema (`/dcos`, `/metadata`).
   - [ ] Write per-run JSON metadata shards compatible with
         `merge_metadata.py`.

4. **Testing**
   - [ ] Add unit tests under `tests/integration/posydon/` for parameter grid
         generation and metadata validation.
   - [ ] Extend the multi-code smoke test to include POSYDON.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Large POSYDON grids slow local development | Medium | Use down-sampled grids for smoke tests; run full grid on AWS Batch. |
| API drift between POSYDON releases | Medium | Pin git commit hash in `simulators/posydon/README.md` and re-vendor if needed. |
| Parameter mismatch with COMPAS/COSMIC | High | Maintain a single source of truth for priors inside `pydantic` config objects shared across generators. |

---

## Next Steps

1. Create `simulators/posydon/README.md` with clone/build instructions.
2. Draft `generate_ensemble.py` stub with CLI + argument parsing (even if it
   raises `NotImplementedError` initially).
3. Update AWS playbooks once runtime benchmarks are available.

Track progress by updating this file at the end of each phase. Once Phase 5
finishes, promote POSYDON to “operational” status in the README and architecture
docs.

