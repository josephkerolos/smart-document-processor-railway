#!/usr/bin/env python3
"""
Cleanup script for debug files
Removes old debug files to prevent disk space issues
"""

import os
import time
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_debug_files(days_to_keep=7):
    """Remove debug files older than specified days"""
    debug_dir = os.path.join(os.path.dirname(__file__), "debug")
    
    if not os.path.exists(debug_dir):
        logger.info("Debug directory does not exist, nothing to clean")
        return
    
    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
    removed_count = 0
    total_size = 0
    
    for filename in os.listdir(debug_dir):
        filepath = os.path.join(debug_dir, filename)
        
        try:
            if os.path.isfile(filepath):
                file_stat = os.stat(filepath)
                if file_stat.st_mtime < cutoff_time:
                    file_size = file_stat.st_size
                    os.remove(filepath)
                    removed_count += 1
                    total_size += file_size
                    logger.info(f"Removed: {filename} (age: {(time.time() - file_stat.st_mtime) / 86400:.1f} days)")
        except Exception as e:
            logger.error(f"Error removing {filename}: {e}")
    
    if removed_count > 0:
        logger.info(f"Cleanup complete: removed {removed_count} files, freed {total_size / 1024 / 1024:.2f} MB")
    else:
        logger.info("No old debug files to remove")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean up old debug files")
    parser.add_argument("--days", type=int, default=7, help="Keep files newer than this many days (default: 7)")
    args = parser.parse_args()
    
    cleanup_debug_files(args.days)