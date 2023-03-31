import torch
import torch.nn as nn
import torch.nn.functional as F

class naverCNN(nn.Module):
    def __init__(self, weights, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(weights)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=self.embedding.embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs) for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)


    def forward(self, input):
        embedded = self.embedding(input)

        embedded = embedded.permute(0, 2, 1)

        conved = [F.relu(conv(embedded)) for conv in self.convs]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        concatenated = self.dropout(torch.cat(pooled, dim=1))
        
        return self.fc(concatenated)