import os

labels_dir = "finetune/labels"  # labels dir
keyword = "walnut"  # prefix
correct_class_number = 4  # class number for this class (e.g., peanut = 3)

for file_name in os.listdir(labels_dir):
    if file_name.startswith(keyword) and file_name.endswith(".jpg"):
        file_path = os.path.join(labels_dir, file_name)

        with open(file_path, "r") as file:
            lines = file.readlines()

        updated_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                parts[0] = str(correct_class_number)  # change prefix of file
                updated_lines.append(" ".join(parts))

        with open(file_path, "w") as file:
            file.write("\n".join(updated_lines) + "\n")

        print(f"Updated class numbers in: {file_name}")

print("Class number updates completed!")