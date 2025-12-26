# Reorganization Plan (Proposed Only)

No files were moved, deleted, or renamed. This plan is derived from the current inventory and the classifications in docs/classification_report.md.

## Recommended target directory structure
- scripts/entry/ (CLI entrypoints such as main.py)
- scripts/gui/ (GUI entrypoints such as gradio_toolbox.py)
- scripts/build/ (build helpers like build_xi_li.py and convert_logo.py)
- xili/ (core reusable algorithms and CLI)
- common/ (shared helpers)
- docs/ (active documentation)
- docs/legacy/ (legacy docs kept for reference)
- docs/archive/ (unused or archived notes)
- assets/ (logos and images used by builds)
- tests/ (pytest tests)
- tests/fixtures/ (trimmed datasets used by tests)
- tests/fixtures/manual/ (manual datasets currently under test/)
- runs/ (runtime outputs from GUI/CLI runs)
- dist/ (build outputs)
- build/ (build temp outputs)
- outputs/ (runtime outputs from scripts)
- archive/ (optional holding area for old copies if still needed)

## Suggested path mapping (no execution)
| Current path | Suggested target | Evidence in classification_report.md | Reason |
| --- | --- | --- | --- |
| `main.py` | `scripts/entry/main.py` | ENTRY/active (main.py entry) | Keep entrypoints grouped |
| `gradio_toolbox.py` | `scripts/gui/gradio_toolbox.py` | ENTRY/active (GUI entry) | Separate GUI entry |
| `xili/` | `xili/` | CORE/active | Core package already in place |
| `common/` | `common/` | CORE/supported | Shared helpers already in place |
| `build_xi_li.py` | `scripts/build/build_xi_li.py` | BUILD/supported | Build tooling grouping |
| `convert_logo.py` | `scripts/build/convert_logo.py` | BUILD/supported | Build tooling grouping |
| `XiLiToolbox.spec` | `build/XiLiToolbox.spec` | BUILD/supported | Build config isolation |
| `logo.png`, `logo.ico` | `assets/` | ASSET/supported | Centralize assets |
| `docs/*.md` | `docs/` | DOC/supported | Keep documentation together |
| `Unused Documents/*` | `docs/archive/unused_documents/` | DOC/legacy | Archive unused docs |
| `archive/docs/unused/*` | `docs/archive/` | DOC/legacy | Consolidate archives |
| `Legacy Scripts/*.py` | `legacy/` | ENTRY/legacy | Preserve legacy wrappers |
| Root legacy CN scripts (e.g., `K-means.py`) | `legacy/` | ENTRY/legacy | Keep backwards compatibility, isolate legacy |
| Active EN scripts (e.g., `kmeans.py`) | `scripts/` | ENTRY/active | Group runnable scripts |
| `tests/` | `tests/` | TEST/supported | Keep tests in place |
| `tests/fixtures/*` | `tests/fixtures/` | TEST/supported | Fixtures remain under tests |
| `test/*` | `tests/fixtures/manual/` | TEST/supported | Manual datasets separated |
| `runs/*` | `runs/` | ARTIFACT/generated | Generated runtime outputs |
| `dist/*`, `hello.dist/*` | `dist/` | ARTIFACT/generated | Build outputs grouped |
| `build/*` | `build/` | ARTIFACT/generated | Build temp outputs grouped |
| `outputs/*` | `outputs/` | ARTIFACT/generated | Script outputs grouped |
| `__pycache__/*`, `*.pyc` | `(ignore)` | ARTIFACT/generated | Cache artifacts excluded |
| `*.log`, `*.zip` | `runs/` | ARTIFACT/generated | Generated logs/archives |
| `.claude/settings.local.json` | `config/local/` | DEVOPS/wip keep_in_git=N | Local-only settings |

## .gitignore suggestions (grouped)

### generated
- `runs/`
- `outputs/`
- `dist/`
- `build/`
- `test_run/`
- `*.log`
- `*.zip`

### packaging
- `hello.dist/`
- `*.exe`
- `*.dll`
- `*.pyd`

### cache
- `__pycache__/`
- `*.pyc`
- `.pytest_cache/`
- `.uv-cache/`
- `.venv/`

### local
- `.claude/settings.local.json`

## Notes
- All suggestions are based on the classifications and evidence in docs/classification_report.md.
- No changes are applied in this plan.
