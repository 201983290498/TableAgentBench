"""
Environment Manager
Responsible for recording and restoring the file environment to prevent tool calls from contaminating original data
"""
import os
from typing import Set, Optional


class EnvManager:
    """
    Environment Manager - Record and restore the file environment
    
    Usage:
    - Record environment snapshot before TableAgent runs
    - Restore environment after TableAgent runs (delete newly generated files)
    """
    
    def __init__(self, base_path: str = None):
        """
        Args:
            base_path: Base path to be managed
        """
        self.base_path = base_path
        self.initial_files: Set[str] = set()
        self._snapshot_taken = False
    
    def snapshot(self, path: str = None) -> int:
        """
        Record all files in the current environment
        
        Args:
            path: Target path, uses base_path if not specified
            
        Returns:
            Number of recorded files
        """
        target_path = path or self.base_path
        if not target_path or not os.path.exists(target_path):
            return 0
        self.base_path = target_path  # Update base_path
        self.initial_files.clear()
        for root, dirs, files in os.walk(target_path):
            for file in files:
                if file.startswith(".") or file.startswith("_"):
                    continue
                file_path = os.path.abspath(os.path.join(root, file))
                self.initial_files.add(file_path)
        
        self._snapshot_taken = True
        return len(self.initial_files)
    
    def restore(self, path: str = None, verbose: bool = False) -> int:
        """
        Restore environment - Delete added files and empty directories
        
        Args:
            path: Target path, uses base_path if not specified
            verbose: Whether to print deletion information
            
        Returns:
            Number of deleted files
        """
        target_path = path or self.base_path
        if not target_path or not self._snapshot_taken:
            return 0
        
        if not os.path.exists(target_path):
            return 0
        
        # Collect all current files
        current_files = set()
        for root, dirs, files in os.walk(target_path):
            for file in files:
                file_path = os.path.abspath(os.path.join(root, file))
                current_files.add(file_path)
        
        # Find newly added files
        new_files = current_files - self.initial_files
        deleted_count = 0
        
        # Delete newly added files
        for file_path in new_files:
            try:
                os.remove(file_path)
                deleted_count += 1
                if verbose:
                    print(f"[EnvManager] Deleting file: {file_path}")
            except Exception as e:
                if verbose:
                    print(f"[EnvManager] Deletion failed: {file_path}, Error: {e}")
        
        # Delete empty directories (from deep to shallow)
        for root, dirs, files in os.walk(target_path, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                try:
                    if os.path.isdir(dir_path) and not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        if verbose:
                            print(f"[EnvManager] Deleting empty directory: {dir_path}")
                except Exception:
                    pass
        
        return deleted_count
    
    def get_new_files(self, path: str = None) -> Set[str]:
        """
        Get the list of added files (without deleting them)
        
        Args:
            path: Target path
            
        Returns:
            Set of added files
        """
        target_path = path or self.base_path
        if not target_path or not self._snapshot_taken:
            return set()
        
        current_files = set()
        for root, dirs, files in os.walk(target_path):
            for file in files:
                file_path = os.path.abspath(os.path.join(root, file))
                current_files.add(file_path)
        
        return current_files - self.initial_files
    
    def reset(self):
        """Reset the manager state"""
        self.initial_files.clear()
        self._snapshot_taken = False
