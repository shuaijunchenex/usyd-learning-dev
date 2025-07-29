from __future__ import annotations
import sys

sys.path.append("..")
sys.path.append("../ml_utils")
print("\n".join(sys.path))

from ml_utils.filename_generator import FileNameGenerator
from ml_utils.filename_maker import FileNameMaker


##############################################

dict_pair = {"a": "1", "b": "2", "c" : "3" }

def test_fileName_generator():
    print("Test FileNameGenerator(filename_generator.py) class:")
    file_name = FileNameGenerator(dict_pair).generate("abc")
    print(file_name)
    print("\n")
    return


def test_fileName_maker():
    print("Test FileNameMaker(filename_make.py) class:")
    file_names = FileNameMaker.make(dict_pair, "abc")
    print(file_names)
    print("name: " + file_names.name)          #name - only origin name
    print("path: " + file_names.path)          #path - only path
    print("file name: " + file_names.filename)      #filename - only file name
    print("full name: " + file_names.fullname)      #fullname - combine of path and file name
    print("\n")

    file_names_1 = FileNameMaker.with_args("./results-1/").make(dict_pair, "abc")
    print(file_names_1)
    print("name: " + file_names_1.name)
    print("path: " + file_names_1.path)
    print("file name: " + file_names_1.filename)
    print("full name: " + file_names.fullname)
    return


def main():
    test_fileName_generator()
    test_fileName_maker()

    return

if __name__ == "__main__":
    main()
