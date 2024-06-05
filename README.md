## BinAdapter prototype

For base model implementation please refer [here](https://github.com/agwaBom/AsmDepictor)

Steps for training AsmDepictor with incremental data:

### Install prerequisites:
> Working in virtual environment is recommended

    $ pip install -r requirements.txt
    

### Configure

Create following directories for model, voca and dataset in root folder

    $ mkdir checkpoint  
    $ mkdir dataset     

The rest configuration can be updated in `config.py`

### Data preprocessing:

Install Ghidra open-source disassembler
update GHIDRA_ANALYZEHEADLESS_PATH and BINARY_PATH in `preprocess/run.py`

    $ cd preprocess/
    $ python run.py

Next for preprocessing run `preprocess.py`

    $ python preprocess.py


### Modeling BinAdapter:

adding adapter to Asmdepictor

    model = BinAdapter(pretrained_model)


loading adapter and embeddings (if saved)

    model.load_state_dict(torch.load(
        '/path/to/adapter/state/'), strict=False)
    
    model.module.encoder.src_word_emb = torch.load(
        '/path/to/src_embedding/')
