from capstone import *
import subprocess
import os
import shutil

GHIDRA_ANALYZEHEADLESS_PATH = "/home/hohyeon/ghidra/ghidra_10.0.3_PUBLIC_20210908/ghidra_10.0.3_PUBLIC/support/analyzeHeadless"
BINARY_PATH = "/home/nozima/projects/AsmDepictor/dataset/bins/mips/"


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
        if binary == "binutils-2.39_gcc-6.4.0_mips_64_O2_readelf":
            binary = BINARY_PATH + binary
            ghidra(binary)
            break
