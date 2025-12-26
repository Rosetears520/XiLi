import importlib as _importlib

_mod = _importlib.import_module("scripts.tools.word_cloud")
for _k in dir(_mod):
    if not _k.startswith("_"):
        globals()[_k] = getattr(_mod, _k)

if __name__ == "__main__":
    raise SystemExit(_mod.cli())
