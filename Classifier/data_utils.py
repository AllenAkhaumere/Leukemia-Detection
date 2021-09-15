from Classifier.Classes.utils import image_to_csv

file_dir = "C:/Peter Moss AML Leukemia Research\Dataset/all_test"
save_dir = "C:\Peter Moss AML Leukemia Research\Dataset"

image_to_csv(file_path=file_dir, save_path=save_dir, mode="test")