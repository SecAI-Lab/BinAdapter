import pickle


class AssemblyVoca:
    def __init__(self, text_path, code_path) -> None:
        self.text_path = text_path
        self.code_path = code_path

    def build(self, text, code, train_data):
        print("Building voca ....")
        code.build_vocab(train_data.code, train_data.code, min_freq=2)
        text.build_vocab(train_data.code, train_data.text)
        return text, code

    def save(self, vocab, path):
        print("Saving vocab ....")
        output = open(path, "wb")
        pickle.dump(vocab, output)
        output.close()

    def read(self, path):
        pkl_file = open(path, "rb")
        vocab = pickle.load(pkl_file)
        pkl_file.close()
        return vocab

    def save_text(self, text_vocab):
        self.save(text_vocab, self.text_path)

    def save_code(self, code_vocab):
        self.save(code_vocab, self.code_path)

    def read_text(self):
        return self.read(self.text_path)

    def read_code(self):
        return self.read(self.code_path)
