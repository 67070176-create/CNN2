import os

dataset_path = r"E:/move work/year3/deeplearn/ThaiCharacter Dataset"
total_files = sum(len(files) for _, _, files in os.walk(dataset_path))
print(f"Total files extracted: {total_files}")