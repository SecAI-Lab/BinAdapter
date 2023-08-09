from ghidra.app.util.bin.format.elf import *
from ghidra.program.model.block import BasicBlockModel
from ghidra.util.task import ConsoleTaskMonitor
from ghidra.app.util.opinion import ElfLoader
import ghidra.app.util.bin.format.elf.ElfDefaultGotPltMarkup
import ghidra.app.util.bin.format.elf
import ghidra.app.util.bin.MemoryByteProvider
import ghidra.program.model.address.AddressSet
from ghidra.util.task import ConsoleTaskMonitor
from binascii import hexlify
import os


class GhidraAPI:
    def __init__(self, store_bytes=False):
        self.store_bytes = store_bytes
        self.func_corpus = dict()        
        self._init_state()
        self.linker_funcs = [
            "__libc_csu_init",
            "__libc_csu_fini",
            "deno_normister_tm_clones",
            "_start",
            "no_normister_tm_clones",
            "__do_global_dtors_aux",
            "__do_global_ctors_aux",
            "frame_dummy",
        ]

    def _init_state(self):
        state = getState()
        program = state.getCurrentProgram()
        self.bin_name = program.getName()
        self.sections = program.getMemory().getBlocks()
        self.code_block = program.getMemory().getBlock(".text")
        self.listing = program.getListing()
        self.af = program.getAddressFactory()
        self.functions = program.getFunctionManager().getFunctions(True)

    def get_section_names(self):
        names = list()
        for section in self.sections:
            names.append(section.getName())
        return names

    def clean(self, instr):
        instr = (
            instr.replace(" + ", "+")
            .replace(" - ", "-")
            .replace(" ", "_")
            .replace(",_", ", ")
            .replace("!", "")
            .replace(",", "_")
            .replace("#", "")
            .replace(" #", "")
            .replace(" ", "")
            .replace("_-", "-")
            .replace("_+", "+")
            .replace("+-", "-")
            .replace(":", "")
            .lower()
        )
        return instr

    def get_exec_functions(self):
        funcs = self.functions
        textset = self.af.getAddressSet(
            self.code_block.getStart(), self.code_block.getEnd()
        )
        text_funcs = filter(lambda f: textset.contains(f.getEntryPoint()), funcs)
        return text_funcs

    def store_code_units(self):
        exec_funcs = self.get_exec_functions()
        for func in exec_funcs:
            f_name = func.getName()
            if "FUN_" in f_name or f_name in self.linker_funcs or "thunk" in f_name:
                continue
            addrset = func.getBody()
            code_units = self.listing.getCodeUnits(addrset, True)
            instrs = list()
            if self.store_bytes:
                for codeUnit in code_units:
                    instrs.append(hexlify(codeUnit.getBytes())) 
            else:
                for codeUnit in code_units:
                    i = codeUnit.toString()
                    i = self.clean(i)
                    if "nop" in i and i != " ":
                        continue
                    instrs.append(i.lower())
            
            # skip functions containing less than 2 instructions
            if len(instrs) <= 2:
                continue

            code = ", ".join(instrs)
            self.func_corpus[code] = f_name

    def save_bin_info(self, out_dir):
        self.store_code_units()
        out_file = "{}/{}.txt".format(out_dir, self.bin_name)
        with open(out_file, "w") as f:
            for code, fname in self.func_corpus.items():
                f.write(fname + "\t" + code + "\n")


def validate_path(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    out_path = "./ghidra/outputs"
    validate_path(out_path)
    api = GhidraAPI()
    api.save_bin_info(out_dir=out_path)
