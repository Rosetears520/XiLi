import importlib as _importlib

_mod = _importlib.import_module("scripts.tools.linear_interpolation")
for _k in dir(_mod):
    if not _k.startswith("_"):
        globals()[_k] = getattr(_mod, _k)


def cli(argv=None) -> int:
    return _mod.cli(argv)


if __name__ == "__main__":
    raise SystemExit(cli())
