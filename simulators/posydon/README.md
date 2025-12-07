# POSYDON Placeholder

POSYDON (Population Synthesis with DYnamics and stellar evOlutioN) is the next
population-synthesis code scheduled for integration. The full source checkout
and build instructions will live in this directory once the code is vendored.
Until then, this folder acts as a marker so documentation and tooling can
reference a canonical path.

## Planned Layout

```
simulators/posydon/
├── README.md               # This file
├── env/                    # Conda or virtualenv files
├── posydon/                # Upstream source checkout (submodule or vendored)
├── configs/                # Run cards used by ensemble generators
└── notebooks/              # Validation notebooks (optional)
```

## Next Steps

1. Clone upstream POSYDON (recommended commit: `main@2025-11-20`) into
   `simulators/posydon/posydon`.
2. Create the conda environment via
   `conda env create -f configs/infrastructure/environment-posydon.yml`.
3. Install the vendored repo with `pip install -e simulators/posydon/posydon`.
4. Document build + smoke-test instructions in
   `docs/simulator_notes/POSYDON_INTEGRATION.md`.

Once the generator is implemented, remember to update `README.md` and
`docs/overview/ARCHITECTURE.md` to mark POSYDON as operational.

