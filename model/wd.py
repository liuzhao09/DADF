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
        """
        与 forward 相同逻辑，额外返回 MLP 最后一个 hidden layer 的 32-dim 输出。

        hidden: [B, 32]，即 mlp_dims 最后维（Linear(64,32)+BN+ReLU+Dropout 之后、
                Linear(32,1) 之前）。
        不改 state_dict：通过 self.mlp.mlp[:-1] / [-1] 切片迭代，不修改模块注册结构。
        注意：ctn linear part（wide 分支）不包含在 hidden 里——它是全局校准项，
              不是特征交叉表示，debias_net 已经通过 base_pred 间接感知到 wide 的影响。
        """
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

        # self.mlp.mlp 是 Sequential，[:-1] 取除最后 Linear(32,1) 之外的所有层
        hidden = self.mlp.mlp[:-1](emb)          # [B, 32]
        res    = self.mlp.mlp[-1](hidden)         # [B, 1]
        if len(linears) > 0:
            linear_part = torch.concat(linears, dim=1).sum(dim=1, keepdims=True)
            res += linear_part
        pred = torch.sigmoid(res.squeeze(1))
        return pred, hidden

    def forward(self, x_dict):
        """标准前向，行为与原始 WideAndDeep 完全一致。"""
        pred, _ = self.forward_with_hidden(x_dict)
        return pred
