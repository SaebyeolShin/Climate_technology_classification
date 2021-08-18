from transformers import BertPreTrainedModel, BertModel, XLMRobertaModel
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss


class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """

    def __init__(self, config, num_labels=46):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        # self.apply(self.init_bert_weights)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        _, pooled_output = self.bert(input_ids, attention_mask)  # , output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


class custom_XLMRoberta(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.XLMRoberta = XLMRobertaModel.from_pretrained(args.model_name)
        self.classifier1 = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.Dropout(0.1, inplace=False),
            nn.Linear(1024, args.n_classes)
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(1024, 1024, bias=True),
            nn.Dropout(0.1, inplace=False),
            nn.Linear(1024, args.n_middle_classes)
        )

    def forward(self, ids, mask):
        pooler_output = self.XLMRoberta(ids, mask).pooler_output
        s_output = self.classifier1(pooler_output)
        m_output = self.classifier2(pooler_output)
        return s_output, m_output