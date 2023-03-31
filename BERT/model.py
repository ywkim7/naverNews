import torch.nn as nn
from transformers import BertModel

class naverModel(nn.Module):
    def __init__(self, num_classes, dr_rate=None):
        super(naverModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-multilingual-cased")
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, ids, masks):
        _, outputs = self.bert(ids, masks, return_dict=False)
        if self.dr_rate:
            outputs = self.dropout(outputs)
        return self.classifier(outputs)
