# Data Processing Policy

This document outlines the policies for handling datasets and generated artifacts within this project.

## Dataset Handling

- **Confidentiality**: Original CSV and Excel exports should be kept in restricted areas (e.g., internal drives or private storage). Only anonymized sample files should be stored in the repository.
- **Samples/Fixtures**: Public examples or test fixtures should be placed in the `tests/fixtures/` directory to remain traceable and easy to test.
- **Cleanup**: Before committing, ensure that sensitive fields in CSV and Excel input files have been desensitized or replaced with dummy data.

## Generated Artifacts

- **Output Directories**: Automatically generated content such as `runs/`, `__pycache__/`, and `.pyc` files must be ignored via `.gitignore` to avoid polluting the repository.
- **Run Tracking**: Each execution creates a `runs/<timestamp_uuid>/` directory containing the outputs and a `summary.txt` (or similar) report.
