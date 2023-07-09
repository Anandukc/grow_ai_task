import os
import shutil

# Specified the path to the directory containing the captured facial images
captured_images_folder = "D:\\computer_vission\\task_growai\\anandu_kc"

# Create the parent folder to hold the train and test folders
parent_folder = os.path.join(captured_images_folder, "split_data")
os.makedirs(parent_folder, exist_ok=True)

# Create the train and test folders within the parent folder
train_folder = os.path.join(parent_folder, "train")
test_folder = os.path.join(parent_folder, "test")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Get a list of all image files in the captured images folder
image_files = [file for file in os.listdir(captured_images_folder) if file.endswith(".jpg")]

# Calculate the number of images for train and test sets
num_images = len(image_files)
num_train = num_images // 2

# Move images to the train folder
for i, image_file in enumerate(image_files[:num_train]):
    src = os.path.join(captured_images_folder, image_file)
    dst = os.path.join(train_folder, image_file)
    shutil.move(src, dst)

# Move images to the test folder
for i, image_file in enumerate(image_files[num_train:]):
    src = os.path.join(captured_images_folder, image_file)
    dst = os.path.join(test_folder, image_file)
    shutil.move(src, dst)
