import torch
import torch.nn as nn
from transformers import BertModel

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x) + torch.randn_like(x) * 0.1

class RomanUrduSentimentModel(nn.Module):
    def __init__(self):
        super(RomanUrduSentimentModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.bayes_fc = BayesianLinear(768, 2)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(output.pooler_output)
        return self.bayes_fc(pooled)
