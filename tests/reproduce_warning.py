import pandas as pd
import numpy as np
import sys
import warnings
from pathlib import Path

# Add parent directory to path to import xili module
sys.path.insert(0, str(Path(__file__).parent.parent))

from xili.entropy_topsis import run_entropy_topsis

def test_reproduce_warning():
    """Test case that should trigger RuntimeWarning: divide by zero encountered in divide"""
    print("\n=== Reproducing Divide-by-Zero Warning ===")
    
    # Create test data with identical rows
    # All rows are the same, so d_plus and d_minus will be 0
    data = {
        '地区': ['A', 'A', 'A'],
        '指标1': [10, 10, 10],
        '指标2': [20, 20, 20]
    }
    df = pd.DataFrame(data)
    print("Input data:")
    print(df)

    # Catch warnings to see if we get the RuntimeWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        # This should trigger the warning in _topsis_scores
        result = run_entropy_topsis(
            df,
            id_cols=['地区']
        )
        
        warning_found = False
        for warning in w:
            print(f"Caught warning: {warning.message}")
            if "divide by zero" in str(warning.message).lower():
                warning_found = True
        
        if warning_found:
            print("SUCCESS: Reproduced the divide-by-zero warning!")
        else:
            print("FAILED: Could not reproduce the warning.")

if __name__ == "__main__":
    test_reproduce_warning()
