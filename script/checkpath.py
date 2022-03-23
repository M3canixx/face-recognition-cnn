import os
def main(path):
    if not os.path.exists(path):
        os.makedirs(path)
