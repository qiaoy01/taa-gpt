import os

target_folder = 'D:/pywork/taa/data/openwebtext'

for file in os.listdir(target_folder):
    if os.path.isfile(os.path.join(target_folder, file)):
        os.rename(os.path.join(target_folder, file), os.path.join(target_folder, file + '.txt'))