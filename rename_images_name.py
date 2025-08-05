import os

# Set your target directory path
folder_path = r'C:\\Users\\yuval\\Desktop\\Final-Project-CS\\DATASETS\\From the internet\\June 2025\\Sensitive\\mix'  
# folder_path = r'C:\\Users\\yuval\\Desktop\\FinetuningJuly\\tyuta'  

file_name = "mix_of_Personal_Identifiable_Information"  

# List all files in the folder
files = os.listdir(folder_path)

# Filter image files (you can expand this list as needed)
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp', '.tif']
image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

# Rename images
for index, filename in enumerate(sorted(image_files), start=1):
    ext = os.path.splitext(filename)[1]
    new_name = f"{file_name}_{index}{ext}"
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    os.rename(src, dst)
    print(f"Renamed '{filename}' to '{new_name}'")

print("Renaming completed.")
