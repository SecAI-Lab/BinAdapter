import torch
from modeling import Adapter

class Adaptered(torch.nn.Module):
    def __init__(self, orig_layer, d_model=512):
        super().__init__()
        self.orig_layer = orig_layer
        self.adapter = Adapter(d_model).to(torch.device("cuda")) 

    def forward(self, x):        
        orig_out = self.orig_layer(x)        
        output = self.adapter(orig_out)
        return output

class AsmdAdapter(torch.nn.Module):
    def __init__(self, model, d_model=512, only_src=False, adapter_size=3) -> None:
        super().__init__()
        self.pretrained = model        
        self.only_src = only_src
        self.adapter_size = adapter_size

        for i in range(adapter_size):
            self.pretrained.module.encoder.layer_stack[i].slf_attn.layer_norm = Adaptered(
                self.pretrained.module.encoder.layer_stack[i].slf_attn.layer_norm, d_model
            )                        

        if not only_src:
            for i in range(adapter_size):
                self.pretrained.module.decoder.layer_stack[i].slf_attn.layer_norm = Adaptered(
                    self.pretrained.module.decoder.layer_stack[i].slf_attn.layer_norm, d_model
                )               
                
            
        for n, p in self.pretrained.named_parameters():
            if 'adapter' not in n:
                p.requires_grad = False  
    
    def get_model(self):
        return self.pretrained

    def save(self, model_name):
       """to do"""
    
    def load(self, adapter_path):
       """to do"""