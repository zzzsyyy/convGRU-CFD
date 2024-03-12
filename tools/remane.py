import os
import re

directory_path = '../data/naca_new'

def extract_numerical_part(filename):
    match = re.search(r'MultiBlock_(\d+),', filename)
    if match:
        return int(match.group(1))
    else:
        return None

files = [f for f in os.listdir(directory_path) if f.endswith('.dat')]
files.sort(key=extract_numerical_part)


for i, filename in enumerate(files, start=1):
    new_filename = f"{i}.dat"
    old_file = os.path.join(directory_path, filename)
    new_file = os.path.join(directory_path, new_filename)
    os.rename(old_file, new_file)
    print(f"Renamed '{filename}' to '{new_filename}'")
