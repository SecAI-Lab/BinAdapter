import subprocess
import re
import os
from capstone import *


class Instruction:
    def __init__(self, arch) -> None:
        if arch == "x86":
            self.md = Cs(CS_ARCH_X86, CS_MODE_64)
        elif arch == "arm":
            self.md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
        elif arch == "mips":
            self.md = Cs(CS_ARCH_MIPS, CS_MODE_MIPS64)
        else:
            print(arch + " architecture is not supported!")
            exit(0)

    def clean(self, instr):
        norm_instrs = (
            instr.replace(" + ", "+")
            .replace(" - ", "-")
            .replace(" ", "_")
            .replace(",_", ", ")
        )
        return norm_instrs

    def byte_to_str(self, instr_line):
        code = instr_line.split("\t")[-1].strip()
        norm_instrs = ""
        for bytes in code.split(","):
            byte = b""
            byte += bytearray.fromhex(bytes)
            for i in self.md.disasm(byte, 0x1000):
                opcode = i.mnemonic
                oprnd = i.op_str
                operands = [x.strip() for x in oprnd.strip().split(",")]
                norm_instr = (
                    opcode
                    if len(oprnd) == 0
                    else str(opcode + "_" + "_".join(operands))
                )
                norm_instrs += norm_instr + ", "
        norm_instrs = self.clean(norm_instrs)
        return norm_instrs


class Preprocess:
    def __init__(self, prefix, inp_dir, src_file, trg_file, bpe_file, read_bytes=False):
        self.prefix = prefix
        self.input_dir = inp_dir
        self.src_file = src_file
        self.trg_file = trg_file
        self.bpe_file = bpe_file
        self.read_bytes = read_bytes
        self.instr_obj = Instruction(arch="x86")

    def split_src_trg(self):
        inp_files = os.listdir(self.input_dir)
        out_data = {"trg": [], "src": []}
        c = 0
        for inp_file in inp_files:
            with open(self.input_dir + inp_file, "r") as f:
                for line in f.readlines():
                    if self.read_bytes:
                        src = self.instr_obj.byte_to_str(line)
                    else:
                        src = line.split("\t")[-1].strip()
                    trg = line.split("\t")[0].strip()
                    out_data["src"].append(src)
                    out_data["trg"].append(trg)
            c += 1
        print(len(inp_files), c)
        print("before", len(out_data["src"]), len(out_data["trg"]))
        srcs, trgs = self.refine_data(out_data["src"], out_data["trg"])
        print("after", len(srcs), len(trgs))
        self.tokenize_names(srcs, trgs)

    def refine_data(self, srcs, trgs):
        unique_data = dict()
        unique_data[srcs[0]] = trgs[0]
        srcs.remove(srcs[0])
        trgs.remove(trgs[0])

        # Removing <<same body different function name>> lines
        for src, trg in zip(srcs, trgs):
            if src in unique_data:
                if trg == unique_data[src]:
                    unique_data[src] = trg
            else:
                unique_data[src] = trg

        srcs, trgs = [], []
        for src, trg in unique_data.items():
            srcs.append(src)
            trgs.append(trg)

        return srcs, trgs

    def tokenize_names(self, codes=None, names=None):
        clean_names, clean_codes = [], []
        if names and codes:
            print("Cleaning function names....")

            for name, code in zip(names, codes):
                p = re.compile("[^a-zA-Z]")
                p = p.sub(" ", name).strip()
                camel_letters = re.findall(r"[A-Za-z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", p)
                if not camel_letters:
                    p = p.replace("_", " ").replace("  ", " ").strip().lower()
                else:
                    p = [i.lower() for i in camel_letters]

                if not isinstance(p, list):
                    p = p.split(" ")

                p = " ".join(p)
                clean_names.append(p)
                clean_codes.append(code)

            with open(self.src_file, "w") as f:
                for code in clean_codes:
                    f.write(code.strip() + "\n")

            with open(self.trg_file, "w") as f:
                for name in clean_names:
                    f.write(name.strip() + "\n")

    def apply_bpe(self, num_operations, voc_file):
        print("Applying BPE....")
        bpe_learn_cmd = (
            f"subword-nmt learn-bpe -s {num_operations} < {self.src_file} > {voc_file}"
        )
        bpe_apply_cmd = f"subword-nmt apply-bpe --codes {voc_file} --input {self.src_file} --output {self.bpe_file}"
        subprocess.call(bpe_learn_cmd, shell=True)
        subprocess.call(bpe_apply_cmd, shell=True)

    def split_train_test(self):
        data = {"src": [], "trg": []}
        with open(self.bpe_file, "r") as f:
            for line in f.readlines():
                data["src"].append(line.strip())

        with open(self.trg_file, "r") as f:
            for line in f.readlines():
                data["trg"].append(line.strip())

        train_size = int(len(data["src"]) * 0.85) + 1

        train_src = data["src"][:train_size]
        test_src = data["src"][train_size:]

        train_trg = data["trg"][:train_size]
        test_trg = data["trg"][train_size:]

        out_path = "../dataset/" + self.prefix
        validate_path(out_path)

        with open(out_path + "/train_source.txt", "w") as f:
            for src in train_src:
                f.write(src + "\n")

        with open(out_path + "/train_target.txt", "w") as f:
            for trg in train_trg:
                f.write(trg + "\n")

        with open(out_path + "/test_source.txt", "w") as f:
            for src in test_src:
                f.write(src + "\n")

        with open(out_path + "/test_target.txt", "w") as f:
            for trg in test_trg:
                f.write(trg + "\n")


def validate_path(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


if __name__ == "__main__":
    data_prefix = "new"
    p = Preprocess(
        prefix=data_prefix,
        inp_dir="./ghidra/store/new/",
        src_file=f"./ghidra/{data_prefix}_src.txt",
        trg_file=f"./ghidra/{data_prefix}_trg.txt",
        bpe_file=f"./ghidra/{data_prefix}_src_bpe.txt",
    )

    p.split_src_trg()
    p.apply_bpe(10000, data_prefix + ".voc")
    p.split_train_test()
