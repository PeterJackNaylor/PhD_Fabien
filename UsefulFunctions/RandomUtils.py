import os


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)

        
def CheckExistants(path):
    assert os.path.isdir(path)


def CheckFile(path):
    assert os.path.isfile(path)
