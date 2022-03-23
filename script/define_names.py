import os
import is_global_folder
def main(path):
    folders = os.listdir(path)
    folders = filter(is_global_folder.main, folders)
    return folders