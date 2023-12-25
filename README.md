### Source code space for BinAdapter

For base model implementation please refer [here](https://github.com/agwaBom/AsmDepictor)

#### Data preprocessing:

    Install Ghidra open-source disassebler
    update GHIDRA_ANALYZEHEADLESS_PATH and BINARY_PATH in parser/run.py

    $ cd parser/
    $ python run.py

#### Modeling:

    # adding adapter to Asmdepictor
    $ model = AsmdAdapter(pretrained_model)

    # loading adapter and embeddings (if saved)
    $ model.load_state_dict(torch.load(
        '/path/to/adapter/state/'), strict=False)
    
    $ model.module.encoder.src_word_emb = torch.load(
        '/path/to/src_embedding/')
