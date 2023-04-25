import os
import tarfile
import lzma

# Path to folder containing tar files
source_folder = ""

# Path to folder where extracted files will be saved
target_folder = ""

# Extract tar files
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".tar"):
            tar_path = os.path.join(root, file)
            with tarfile.open(tar_path, "r:") as tar:
                tar.extractall(path=target_folder)

# Extract xz files
for root, dirs, files in os.walk(target_folder):
    for file in files:
        if file.endswith(".xz"):
            xz_path = os.path.join(root, file)
            with lzma.open(xz_path, "rb") as f:
                data = f.read()
                target_path = os.path.join(root, os.path.splitext(file)[0])
                with open(target_path, "wb") as out_file:
                    out_file.write(data)
            os.remove(xz_path)
