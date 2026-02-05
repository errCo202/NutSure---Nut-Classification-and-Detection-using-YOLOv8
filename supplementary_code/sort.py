import os
import re

labels_dir = "finetune/labels"
keyword = "pistachio"
start_number = 200


def extract_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')


files = sorted(
    [f for f in os.listdir(labels_dir) if f.endswith('.txt') and keyword in f],
    key=extract_number
)

counter = start_number

for file_name in files:
    file_path = os.path.join(labels_dir, file_name)

    new_file_name = f"{keyword}{counter}.txt"
    new_file_path = os.path.join(labels_dir, new_file_name)

    os.rename(file_path, new_file_path)
    print(f"Renamed: {file_name} -> {new_file_name}")
    counter += 1

print("Renaming completed!")
