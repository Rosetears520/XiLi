import pytest
import sys

if __name__ == "__main__":
    # Run pytest with quiet output
    retcode = pytest.main(["-q"])
    sys.exit(retcode)
