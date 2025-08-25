#!/usr/bin/env python3
"""
Utility script to clean up problematic Unicode filenames in the data directory
and fix file encoding issues for the face detection system.
"""

import os
import re
import shutil
import unicodedata
from pathlib import Path

def sanitize_filename(filename):
    """
    Sanitize filename to remove problematic Unicode characters
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for file system operations
    """
    # Normalize Unicode characters
    filename = unicodedata.normalize('NFKD', filename)
    
    # Remove or replace problematic characters
    # Keep only alphanumeric, spaces, hyphens, underscores, and dots
    filename = re.sub(r'[^\w\s\-_.]', '', filename)
    
    # Replace multiple spaces/hyphens with single underscore
    filename = re.sub(r'[-\s]+', '_', filename)
    
    # Remove leading/trailing underscores and dots
    filename = filename.strip('_.')
    
    # Ensure filename is not empty
    if not filename:
        filename = 'unknown_user'
    
    return filename

def cleanup_data_directory(data_dir='data'):
    """
    Clean up the data directory by renaming files with problematic Unicode characters
    
    Args:
        data_dir: Path to the data directory
    """
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' does not exist.")
        return 0, 0
    
    print(f"Cleaning up data directory: {data_dir}")
    
    renamed_files = []
    problem_files = []
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        try:
            # Try to read the file to check if it's accessible
            with open(file_path, 'rb') as f:
                f.read(1)  # Try to read first byte
            
            # Check if filename needs sanitization
            sanitized_name = sanitize_filename(filename)
            
            if sanitized_name != filename:
                # Create new filename with timestamp if needed
                name_part, ext = os.path.splitext(sanitized_name)
                if not ext:
                    ext = '.jpg'  # Default extension
                
                new_filename = f"{name_part}{ext}"
                new_file_path = os.path.join(data_dir, new_filename)
                
                # Handle filename conflicts
                counter = 1
                while os.path.exists(new_file_path):
                    new_filename = f"{name_part}_{counter}{ext}"
                    new_file_path = os.path.join(data_dir, new_filename)
                    counter += 1
                
                # Rename the file
                try:
                    shutil.move(file_path, new_file_path)
                    renamed_files.append((filename, new_filename))
                    print(f"Renamed: '{filename}' -> '{new_filename}'")
                except Exception as rename_error:
                    print(f"Failed to rename '{filename}': {rename_error}")
                    problem_files.append(filename)
            else:
                print(f"File OK: '{filename}'")
                
        except Exception as e:
            print(f"Problem with file '{filename}': {e}")
            problem_files.append(filename)
    
    # Summary
    print(f"\nCleanup Summary:")
    print(f"Files renamed: {len(renamed_files)}")
    print(f"Problem files: {len(problem_files)}")
    
    if renamed_files:
        print(f"\nRenamed files:")
        for old_name, new_name in renamed_files:
            print(f"  '{old_name}' -> '{new_name}'")
    
    if problem_files:
        print(f"\nProblem files (consider manual cleanup):")
        for filename in problem_files:
            print(f"  '{filename}'")
    
    return len(renamed_files), len(problem_files)

def test_file_access(data_dir='data'):
    """
    Test file access for all files in the data directory
    
    Args:
        data_dir: Path to the data directory
    """
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' does not exist.")
        return [], []
    
    print(f"Testing file access in: {data_dir}")
    
    accessible_files = []
    inaccessible_files = []
    
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        
        # Skip directories
        if os.path.isdir(file_path):
            continue
        
        try:
            # Test OpenCV imread
            import cv2
            image = cv2.imread(file_path)
            
            if image is not None:
                accessible_files.append(filename)
                print(f"Accessible: '{filename}' ({image.shape})")
            else:
                # Try binary read method
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                
                import numpy as np
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is not None:
                    accessible_files.append(filename)
                    print(f"Accessible (binary): '{filename}' ({image.shape})")
                else:
                    inaccessible_files.append(filename)
                    print(f"Inaccessible: '{filename}'")
                    
        except Exception as e:
            inaccessible_files.append(filename)
            print(f"Error accessing '{filename}': {e}")
    
    print(f"\nAccess Test Summary:")
    print(f"Accessible files: {len(accessible_files)}")
    print(f"Inaccessible files: {len(inaccessible_files)}")
    
    return accessible_files, inaccessible_files

def main():
    """
    Main function to run cleanup and testing
    """
    print("Data Directory Cleanup Utility")
    print("=" * 40)
    
    # Cleanup filenames
    renamed, problems = cleanup_data_directory()
    
    print("\n" + "=" * 40)
    
    # Test file access
    accessible, inaccessible = test_file_access()
    
    print("\nCleanup completed!")
    
    if problems == 0 and len(inaccessible) == 0:
        print("All files should now work with the face detection system.")
    else:
        print("Some files may still have issues. Consider manual cleanup.")

if __name__ == "__main__":
    main()