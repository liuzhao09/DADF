import torch

from model.layers import FactorizationMachine, MultiLayerPerceptron

class WideAndDeep(torch.nn.Module):

    def __init__(self, description, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.features = {name: (size, type) for name, size, type in description if (type in ["ctn", 'seq', 'spr'])}
        self.build(embed_dim, mlp_dims, dropout)

    def build(self, embed_dim, mlp_dims, dropout):
        self.emb_layer = torch.nn.ModuleDict()
        self.ctn_emb_layer = torch.nn.ParameterDict()
        self.ctn_linear_layer = torch.nn.ModuleDict()
        embed_output_dim = 0
        for name, (size, type) in self.features.items():
            if type == 'spr':
                self.emb_layer[name] = torch.nn.Embedding(size, embed_dim)
                embed_output_dim += embed_dim
            elif type == 'ctn':
                self.ctn_linear_layer[name] = torch.nn.Linear(1, 1, bias=False)
            elif type == 'seq':
                self.emb_layer[name] = torch.nn.Embedding(size, embed_dim)
                embed_output_dim += embed_dim
            else:
                raise ValueError('unkown feature type: {}'.format(type))
        self.mlp = MultiLayerPerceptron(embed_output_dim, mlp_dims, dropout)
        return

    def init(self):
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def forward_with_hidden(self, x_dict):
        linears = []
        embs = []
        for name, (_, type) in self.features.items():
            x = x_dict[name]
            if type == 'spr':
                embs.append(self.emb_layer[name](x).squeeze(1))
            elif type == 'ctn':
                linears.append(self.ctn_linear_layer[name](x))
            elif type == 'seq':
                seq_emb = self.emb_layer[name](x)
                seq_mask = torch.unsqueeze(x_dict["{}mask".format(name)], dim=2)
                mask_sum = torch.sum(seq_mask, dim=1).clamp(min=1e-9)
                embs.append(torch.sum(seq_emb * seq_mask, dim=1) / mask_sum)
        emb = torch.concat(embs, dim=1)

        hidden = self.mlp.mlp[:-1](emb)
        res    = self.mlp.mlp[-1](hidden)
        if len(linears) > 0:
            linear_part = torch.concat(linears, dim=1).sum(dim=1, keepdims=True)
            res += linear_part
        pred = torch.sigmoid(res.squeeze(1))
        return pred, hidden

    def forward(self, x_dict):
        pred, _ = self.forward_with_hidden(x_dict)
        return pred
