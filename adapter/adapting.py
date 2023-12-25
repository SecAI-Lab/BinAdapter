import torch
from adapter.modeling import Adapter


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
    def __init__(self, model, d_model=512, adapter_size=3) -> None:
        super().__init__()
        self.pretrained = model
        self.adapter_size = adapter_size

        for i in range(adapter_size):
            self.pretrained.module.encoder.layer_stack[
                i
            ].slf_attn.layer_norm = Adaptered(
                self.pretrained.module.encoder.layer_stack[i].slf_attn.layer_norm,
                d_model,
            )

        for i in range(adapter_size):
            self.pretrained.module.decoder.layer_stack[
                i
            ].slf_attn.layer_norm = Adaptered(
                self.pretrained.module.decoder.layer_stack[i].slf_attn.layer_norm,
                d_model,
            )

        for n, p in self.pretrained.named_parameters():
            if "adapter" not in n:
                p.requires_grad = False


def adapter_state_dict(model: torch.nn.Module):
    my_state_dict = model.state_dict()
    adapter_state = {}
    for k in my_state_dict:
        if "adapter" in k:
            adapter_state[k] = my_state_dict[k]
    return adapter_state
