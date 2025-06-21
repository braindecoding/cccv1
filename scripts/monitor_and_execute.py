#!/usr/bin/env python3
"""
Monitor and Execute Fair Comparison
==================================

Monitor SOTA training completion and auto-execute fair comparison.
"""

import time
import subprocess
import json
from pathlib import Path
from datetime import datetime

def check_sota_completion():
    """Check if SOTA training has completed."""

    # Check for completion indicators
    results_dir = Path("sota_comparison/comparison_results")

    if results_dir.exists():
        recent_files = list(results_dir.glob("academic_evaluation_*.json"))

        if recent_files:
            # Check if files contain complete CV results
            for file in recent_files:
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)

                    if 'results' in data:
                        complete_results = 0
                        for result in data['results']:
                            if (result['status'] == 'success' and
                                'cv_scores' in result and
                                len(result['cv_scores']) == 10):
                                complete_results += 1

                        # If we have complete results for multiple methods/datasets
                        if complete_results >= 4:  # At least 2 methods x 2 datasets
                            return True

                except Exception as e:
                    continue

    return False

def execute_fair_comparison():
    """Execute fair comparison with real SOTA results."""

    print("Executing fair comparison with real SOTA results...")

    try:
        result = subprocess.run([
            'python', 'scripts/fair_comparison_framework.py'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("Fair comparison completed successfully!")
            return True
        else:
            print(f"Error in fair comparison: {result.stderr}")
            return False

    except Exception as e:
        print(f"Error executing fair comparison: {e}")
        return False

def main():
    """Monitor and execute fair comparison."""

    print("Auto Fair Comparison Monitor")
    print("Monitoring SOTA training completion...")

    check_interval = 300  # Check every 5 minutes
    max_checks = 144      # Maximum 12 hours of monitoring

    for check_count in range(max_checks):
        if check_sota_completion():
            print("SOTA training completed! Executing fair comparison...")

            if execute_fair_comparison():
                print("Fair comparison completed successfully!")
                break
            else:
                print("Error in fair comparison execution")
                break
        else:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f"[{current_time}] Still waiting for SOTA completion... (Check {check_count + 1}/{max_checks})")
            time.sleep(check_interval)

    else:
        print("Maximum monitoring time reached. Please check SOTA training status manually.")

if __name__ == "__main__":
    main()