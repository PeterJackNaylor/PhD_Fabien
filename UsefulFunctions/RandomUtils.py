import os


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)
