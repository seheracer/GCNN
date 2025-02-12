import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as pyg_nn


class GINStack(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int, num_conv_layers: int
    ):
        super(GINStack, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.dropout = 0.25
        self.hidden_dim = hidden_dim
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, self.hidden_dim))
        self.lns = nn.ModuleList()
        for l in range(self.num_conv_layers):
            self.convs.append(self.build_conv_model(self.hidden_dim, self.hidden_dim))
            self.lns.append(nn.LayerNorm(self.hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, output_dim),
        )

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        return pyg_nn.GINConv(
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_conv_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            if not i == self.num_conv_layers - 1:
                x = self.lns[i](x)

        x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return x

    def loss(self, pred, value):
        pred_shape = pred.shape
        value_shape = value.shape
        if pred_shape != value_shape:
            value = torch.reshape(value, pred_shape)
        return F.mse_loss(pred, value)

    def loss_rmse(self, pred, value):
        pred_shape = pred.shape
        value_shape = value.shape
        if pred_shape != value_shape:
            value = torch.reshape(value, pred_shape)
        return torch.sqrt(F.mse_loss(pred, value))

    def __str__(self):
        return "GINStack"
