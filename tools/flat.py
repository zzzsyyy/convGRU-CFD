import os
import shutil

parent_folder = '.'

for root, dirs, files in os.walk(parent_folder):
    if root == parent_folder:
        continue
    
    for file in files:
        file_path = os.path.join(root, file)
        dest_path = os.path.join(parent_folder, file)
        if os.path.exists(dest_path):
            base, extension = os.path.splitext(dest_path)
            counter = 1
            while os.path.exists(dest_path):
                dest_path = f"{base}_{counter}{extension}"
                counter += 1
        shutil.move(file_path, dest_path)
    if not os.listdir(root):
        os.rmdir(root)

