#!/usr/bin/env python3
"""
Monitor COMPAS Ensemble Generation Progress

This script monitors the progress of the ongoing COMPAS ensemble generation
and provides real-time status updates.
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

def get_ensemble_status(ensemble_dir):
    """Get current status of ensemble generation"""
    ensemble_path = Path(ensemble_dir)
    
    if not ensemble_path.exists():
        return None
    
    # Load metadata if it exists
    metadata_file = ensemble_path / 'ensemble_metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        n_total = len([r for r in metadata.get('runs', [])])
        n_success = len([r for r in metadata.get('runs', []) if r['status'] == 'success'])
        n_failed = len([r for r in metadata.get('runs', []) if r['status'] == 'failed'])
    else:
        n_total = 0
        n_success = 0
        n_failed = 0
    
    # Count directories
    run_dirs = list(ensemble_path.glob('alpha*'))
    n_dirs = len(run_dirs)
    
    # Check HDF5 files
    h5_files = list(ensemble_path.glob('*/COMPAS_Output/COMPAS_Output.h5'))
    h5_sizes = {}
    for h5_file in h5_files:
        if h5_file.stat().st_size > 1000:  # More than 1KB means data written
            h5_sizes[h5_file.parent.parent.name] = h5_file.stat().st_size
    
    return {
        'n_dirs': n_dirs,
        'n_success': n_success,
        'n_failed': n_failed,
        'n_total': n_total,
        'completed_runs': len(h5_sizes),
        'h5_sizes': h5_sizes
    }

def format_size(size_bytes):
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def monitor_ensemble(ensemble_dir, interval=60, max_iterations=None):
    """
    Monitor ensemble generation progress
    
    Args:
        ensemble_dir: Path to ensemble output directory
        interval: Update interval in seconds
        max_iterations: Maximum number of iterations (None = infinite)
    """
    print("="*70)
    print("COMPAS Ensemble Generation Monitor")
    print("="*70)
    print(f"Monitoring: {ensemble_dir}")
    print(f"Update interval: {interval} seconds")
    print(f"Press Ctrl+C to stop monitoring")
    print("="*70)
    print()
    
    iteration = 0
    start_time = datetime.now()
    
    try:
        while True:
            status = get_ensemble_status(ensemble_dir)
            
            if status is None:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for ensemble directory to be created...")
            else:
                elapsed = datetime.now() - start_time
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Update (Elapsed: {str(elapsed).split('.')[0]})")
                print("-"*70)
                print(f"  Run directories created: {status['n_dirs']}")
                print(f"  Completed runs (HDF5 > 1KB): {status['completed_runs']}")
                print(f"  Successful (in metadata): {status['n_success']}")
                print(f"  Failed (in metadata): {status['n_failed']}")
                
                if status['h5_sizes']:
                    print(f"\n  Recent completed runs:")
                    for run_id, size in list(status['h5_sizes'].items())[-3:]:
                        print(f"    - {run_id}: {format_size(size)}")
                
                if status['completed_runs'] > 0 and elapsed.total_seconds() > 0:
                    rate = status['completed_runs'] / (elapsed.total_seconds() / 3600)  # runs per hour
                    if status['n_dirs'] > 0:
                        remaining = status['n_dirs'] - status['completed_runs']
                        if rate > 0:
                            eta_hours = remaining / rate
                            eta = datetime.now() + timedelta(hours=eta_hours)
                            print(f"\n  Rate: {rate:.2f} runs/hour")
                            print(f"  Estimated completion: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
                            print(f"  Estimated time remaining: {timedelta(hours=eta_hours)}")
            
            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                break
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print("="*70)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Monitor COMPAS ensemble generation progress'
    )
    parser.add_argument(
        '--ensemble-dir',
        default='./compas_ensemble_sparse',
        help='Path to ensemble output directory'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Update interval in seconds'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run once and exit (no continuous monitoring)'
    )
    
    args = parser.parse_args()
    
    if args.once:
        status = get_ensemble_status(args.ensemble_dir)
        if status:
            print(f"Ensemble Status for {args.ensemble_dir}:")
            print(f"  Directories: {status['n_dirs']}")
            print(f"  Completed: {status['completed_runs']}")
            print(f"  Successful: {status['n_success']}")
            print(f"  Failed: {status['n_failed']}")
        else:
            print(f"Ensemble directory not found: {args.ensemble_dir}")
    else:
        monitor_ensemble(args.ensemble_dir, args.interval)

if __name__ == "__main__":
    main()

