import torch

from model.layers import FactorizationMachine, MultiLayerPerceptron

class Cread(torch.nn.Module):

    def __init__(self, description, embed_dim, share_mlp_dims, output_mlp_dims, head_num, dropout):
        super().__init__()
        self.features = {name: (size, type) for name, size, type in description if (type in ["ctn", 'seq', 'spr'])}
        self.build(embed_dim, share_mlp_dims, output_mlp_dims, head_num, dropout)

    def build(self, embed_dim, share_mlp_dims, output_mlp_dims, head_num, dropout):
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
                embed_output_dim += 1
            elif type == 'seq':
                self.emb_layer[name] = torch.nn.Embedding(size, embed_dim)
                embed_output_dim += embed_dim
            else:
                raise ValueError('unkown feature type: {}'.format(type))
        self.share_mlp = MultiLayerPerceptron(embed_output_dim, share_mlp_dims, dropout, output_layer=False)
        self.output_mlps = torch.nn.ModuleList()
        for idx_head in range(head_num):
            self.output_mlps.append(MultiLayerPerceptron(share_mlp_dims[-1], output_mlp_dims, dropout, output_layer=True))
        return

    def init(self):
        for param in self.parameters():
            torch.nn.init.uniform_(param, -0.01, 0.01)

    def forward(self, x_dict):
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
            else:
                raise ValueError('unkwon feature: {}'.format(name))
        emb = torch.concat(embs + linears, dim=1)
        share_vec = self.share_mlp(emb)
        output_list = []
        for output_mlp in self.output_mlps:
            output_list.append(output_mlp(share_vec))
        return torch.sigmoid(torch.concat(output_list, dim=1))
