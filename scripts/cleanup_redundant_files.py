#!/usr/bin/env python3
"""
Cleanup Redundant Files
=======================

Remove redundant and incorrect files to prevent future confusion.
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime

def create_backup_list():
    """Create list of files to backup before deletion"""
    
    backup_files = []
    
    # Files to backup (important but will be removed)
    important_redundant = [
        "results/reconstruction_metrics_FIXED_20250621_080616.json",
        "scripts/visualize_reconstruction_fixed.py",
        "scripts/calculate_fixed_metrics.py"
    ]
    
    for file_path in important_redundant:
        if os.path.exists(file_path):
            backup_files.append(file_path)
    
    return backup_files

def create_backup(backup_files):
    """Create backup of important files before deletion"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup/cleanup_backup_{timestamp}"
    
    os.makedirs(backup_dir, exist_ok=True)
    
    print(f"📦 CREATING BACKUP: {backup_dir}")
    print("-" * 60)
    
    for file_path in backup_files:
        if os.path.exists(file_path):
            # Create subdirectory structure in backup
            backup_path = os.path.join(backup_dir, file_path)
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Copy file
            shutil.copy2(file_path, backup_path)
            print(f"✅ Backed up: {file_path}")
    
    print(f"💾 Backup completed: {backup_dir}")
    return backup_dir

def cleanup_publication_packages():
    """Remove publication package backups (safe)"""
    
    print(f"\n🗑️ REMOVING PUBLICATION PACKAGES")
    print("-" * 60)
    
    pub_dir = "publication_packages"
    if os.path.exists(pub_dir):
        # Calculate size before removal
        total_size = 0
        file_count = 0
        for root, dirs, files in os.walk(pub_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except:
                    pass
        
        size_mb = total_size / (1024 * 1024)
        
        # Remove directory
        shutil.rmtree(pub_dir)
        print(f"✅ Removed: {pub_dir}")
        print(f"📊 Files removed: {file_count}")
        print(f"💾 Space saved: {size_mb:.1f} MB")
        
        return size_mb, file_count
    else:
        print(f"⚠️ Directory not found: {pub_dir}")
        return 0, 0

def cleanup_python_cache():
    """Remove Python cache files (safe)"""
    
    print(f"\n🗑️ REMOVING PYTHON CACHE FILES")
    print("-" * 60)
    
    cache_dirs = []
    pyc_files = []
    
    # Find __pycache__ directories
    for root, dirs, files in os.walk("."):
        if "__pycache__" in dirs:
            cache_dirs.append(os.path.join(root, "__pycache__"))
        
        for file in files:
            if file.endswith(".pyc"):
                pyc_files.append(os.path.join(root, file))
    
    # Remove cache directories
    total_size = 0
    for cache_dir in cache_dirs:
        try:
            # Calculate size
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        total_size += os.path.getsize(file_path)
                    except:
                        pass
            
            shutil.rmtree(cache_dir)
            print(f"✅ Removed: {cache_dir}")
        except Exception as e:
            print(f"❌ Error removing {cache_dir}: {e}")
    
    # Remove .pyc files
    for pyc_file in pyc_files:
        try:
            total_size += os.path.getsize(pyc_file)
            os.remove(pyc_file)
            print(f"✅ Removed: {pyc_file}")
        except Exception as e:
            print(f"❌ Error removing {pyc_file}: {e}")
    
    size_mb = total_size / (1024 * 1024)
    total_files = len(cache_dirs) + len(pyc_files)
    
    print(f"📊 Cache items removed: {total_files}")
    print(f"💾 Space saved: {size_mb:.1f} MB")
    
    return size_mb, total_files

def cleanup_incorrect_visualization_results():
    """Remove incorrect visualization results"""
    
    print(f"\n🗑️ REMOVING INCORRECT VISUALIZATION RESULTS")
    print("-" * 60)
    
    # Files with incorrect CortexFlow results
    incorrect_files = [
        "results/reconstruction_metrics_FIXED_20250621_080616.json",
        "results/reconstruction_visualization_FIXED_20250621_080301",
        "results/reconstruction_visualization_FIXED_20250621_080422"
    ]
    
    total_size = 0
    removed_count = 0
    
    for file_path in incorrect_files:
        if os.path.exists(file_path):
            try:
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
                    os.remove(file_path)
                    print(f"✅ Removed file: {file_path}")
                    removed_count += 1
                elif os.path.isdir(file_path):
                    # Calculate directory size
                    for root, dirs, files in os.walk(file_path):
                        for file in files:
                            full_path = os.path.join(root, file)
                            try:
                                total_size += os.path.getsize(full_path)
                            except:
                                pass
                    
                    shutil.rmtree(file_path)
                    print(f"✅ Removed directory: {file_path}")
                    removed_count += 1
            except Exception as e:
                print(f"❌ Error removing {file_path}: {e}")
        else:
            print(f"⚠️ Not found: {file_path}")
    
    size_mb = total_size / (1024 * 1024)
    
    print(f"📊 Items removed: {removed_count}")
    print(f"💾 Space saved: {size_mb:.1f} MB")
    
    return size_mb, removed_count

def cleanup_redundant_scripts():
    """Remove redundant/incorrect scripts"""
    
    print(f"\n🗑️ REMOVING REDUNDANT SCRIPTS")
    print("-" * 60)
    
    # Scripts that are redundant or incorrect
    redundant_scripts = [
        "scripts/visualize_reconstruction_fixed.py",
        "scripts/calculate_fixed_metrics.py",
        "scripts/test_correct_cortexflow_loading.py",
        "scripts/debug_cortexflow_loading.py"
    ]
    
    total_size = 0
    removed_count = 0
    
    for script_path in redundant_scripts:
        if os.path.exists(script_path):
            try:
                total_size += os.path.getsize(script_path)
                os.remove(script_path)
                print(f"✅ Removed: {script_path}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Error removing {script_path}: {e}")
        else:
            print(f"⚠️ Not found: {script_path}")
    
    size_mb = total_size / (1024 * 1024)
    
    print(f"📊 Scripts removed: {removed_count}")
    print(f"💾 Space saved: {size_mb:.1f} MB")
    
    return size_mb, removed_count

def cleanup_old_visualization_results():
    """Remove old/superseded visualization results"""
    
    print(f"\n🗑️ REMOVING OLD VISUALIZATION RESULTS")
    print("-" * 60)
    
    # Keep only the latest correct results
    old_results = [
        "results/reconstruction_visualization_20250621_075654",
        "results/reconstruction_visualization_20250621_075751"
    ]
    
    total_size = 0
    removed_count = 0
    
    for result_dir in old_results:
        if os.path.exists(result_dir):
            try:
                # Calculate directory size
                for root, dirs, files in os.walk(result_dir):
                    for file in files:
                        full_path = os.path.join(root, file)
                        try:
                            total_size += os.path.getsize(full_path)
                        except:
                            pass
                
                shutil.rmtree(result_dir)
                print(f"✅ Removed: {result_dir}")
                removed_count += 1
            except Exception as e:
                print(f"❌ Error removing {result_dir}: {e}")
        else:
            print(f"⚠️ Not found: {result_dir}")
    
    size_mb = total_size / (1024 * 1024)
    
    print(f"📊 Result directories removed: {removed_count}")
    print(f"💾 Space saved: {size_mb:.1f} MB")
    
    return size_mb, removed_count

def create_cleanup_summary(backup_dir, cleanup_stats):
    """Create cleanup summary report"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"results/cleanup_summary_{timestamp}.json"
    
    summary = {
        "cleanup_timestamp": timestamp,
        "backup_directory": backup_dir,
        "cleanup_statistics": cleanup_stats,
        "total_space_saved_mb": sum(stat["space_saved_mb"] for stat in cleanup_stats.values()),
        "total_items_removed": sum(stat["items_removed"] for stat in cleanup_stats.values()),
        "remaining_files": {
            "correct_cortexflow_metrics": "results/correct_cortexflow_metrics_20250621_082448.json",
            "correct_visualization_script": "scripts/visualize_with_correct_cortexflow.py",
            "fair_comparison_results": "results/final_fair_comparison_20250621_073025/",
            "model_checkpoints": "models/*_cv_best.pth"
        },
        "cleanup_actions": [
            "Removed publication package backups",
            "Cleaned Python cache files", 
            "Removed incorrect visualization results",
            "Removed redundant scripts",
            "Removed old visualization results"
        ]
    }
    
    os.makedirs("results", exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary_file

def main():
    """Execute cleanup process"""
    
    print("🧹 REDUNDANT FILES CLEANUP")
    print("=" * 80)
    print("🎯 Goal: Remove redundant files to prevent future confusion")
    print("⚠️ Important files will be backed up before removal")
    print("=" * 80)
    
    # Create backup of important files
    backup_files = create_backup_list()
    backup_dir = create_backup(backup_files) if backup_files else None
    
    # Perform cleanup operations
    cleanup_stats = {}
    
    # 1. Publication packages (safe)
    size1, count1 = cleanup_publication_packages()
    cleanup_stats["publication_packages"] = {"space_saved_mb": size1, "items_removed": count1}
    
    # 2. Python cache (safe)
    size2, count2 = cleanup_python_cache()
    cleanup_stats["python_cache"] = {"space_saved_mb": size2, "items_removed": count2}
    
    # 3. Incorrect visualization results
    size3, count3 = cleanup_incorrect_visualization_results()
    cleanup_stats["incorrect_visualizations"] = {"space_saved_mb": size3, "items_removed": count3}
    
    # 4. Redundant scripts
    size4, count4 = cleanup_redundant_scripts()
    cleanup_stats["redundant_scripts"] = {"space_saved_mb": size4, "items_removed": count4}
    
    # 5. Old visualization results
    size5, count5 = cleanup_old_visualization_results()
    cleanup_stats["old_visualizations"] = {"space_saved_mb": size5, "items_removed": count5}
    
    # Create summary
    summary_file = create_cleanup_summary(backup_dir, cleanup_stats)
    
    # Final summary
    total_space = sum(stat["space_saved_mb"] for stat in cleanup_stats.values())
    total_items = sum(stat["items_removed"] for stat in cleanup_stats.values())
    
    print("\n" + "=" * 80)
    print("🏆 CLEANUP COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"📦 Backup created: {backup_dir}")
    print(f"📊 Total items removed: {total_items}")
    print(f"💾 Total space saved: {total_space:.1f} MB")
    print(f"📋 Cleanup summary: {summary_file}")
    print("\n✅ REMAINING CORRECT FILES:")
    print("   📊 results/correct_cortexflow_metrics_20250621_082448.json")
    print("   🔧 scripts/visualize_with_correct_cortexflow.py")
    print("   📈 results/final_fair_comparison_20250621_073025/")
    print("   💾 models/*_cv_best.pth")
    print("\n🎯 Future visualizations will use only correct CortexFlow results!")
    print("=" * 80)

if __name__ == "__main__":
    main()
