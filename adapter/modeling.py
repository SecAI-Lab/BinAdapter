import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """
    def __init__(self, hidden_act):

        if hidden_act.lower() == "relu":
            self.f = nn.functional.relu
        elif hidden_act.lower() == "tanh":
            self.f = torch.tanh
        elif hidden_act.lower() == "swish":

            def swish(x):
                return x * torch.sigmoid(x)

            self.f = swish
        elif hidden_act.lower() == "gelu":

            def gelu_new(x):
                return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))) 

            self.f = gelu_new
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu

        super().__init__()

    def forward(self, x):
        return self.f(x)


class Adapter(nn.Module):
    """
    Implementation of a single Adapter block.
    """

    def __init__(
        self,
        input_size,
        down_sample=None,
        non_linearity="gelu",
        init_tf_weights=True,
        add_layer_norm_before=False,
        add_layer_norm_after=True,
        residual_before_ln=False,
    ):
        super().__init__()

        self.input_size = input_size
        self.add_layer_norm_before = add_layer_norm_before
        self.add_layer_norm_after = add_layer_norm_after
        self.residual_before_ln = residual_before_ln
        seq_list = []

        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.input_size // 2

        seq_list.append(nn.Linear(self.input_size, self.down_sample))
        self.non_linearity = Activation_Function_Class(non_linearity.lower())
        seq_list.append(self.non_linearity)

        self.adapter_down = nn.Sequential(*seq_list)
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        if init_tf_weights:
            self.adapter_down.apply(self.init_tf_weights)
            self.adapter_up.apply(self.init_tf_weights)

    def forward(self, x):        
        down = self.adapter_down(x)
        up = self.adapter_up(down)      
        output = up

        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)
        
        if not self.residual_before_ln:
            output = output + x        
        return output
    
