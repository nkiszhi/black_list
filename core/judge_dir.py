import os

def dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

