from capstone import *
import subprocess
import os
import shutil

GHIDRA_ANALYZEHEADLESS_PATH = "path/to/ghidra"
BINARY_PATH = "path/to/binary"


def ghidra(binary):
    if os.path.exists(f"ghidra{os.sep}tmp"):
        shutil.rmtree(f"ghidra{os.sep}tmp")
    os.mkdir(f"ghidra{os.sep}tmp")
    cmd = []
    if not os.path.exists(binary):
        print("[-] input path does not exist")
        return

    cmd = [
        GHIDRA_ANALYZEHEADLESS_PATH,
        f"ghidra{os.sep}tmp",
        "analyze",
        "-prescript",
        "ghidra/pre.py",
        "-postscript",
        "ghidra/post.py",
        "-import",
        binary,
    ]
    subprocess.run(cmd)
    return


if __name__ == "__main__":
    bins = os.listdir(BINARY_PATH)
    for binary in bins:
        binary = BINARY_PATH + binary
        ghidra(binary)
