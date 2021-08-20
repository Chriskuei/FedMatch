import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.embedding import Embedding
from .modules import Module, ModuleList, ModuleDict
from .modeling import BertConfig, BertModel, BERTLayerNorm


class BERT(Module):
    def __init__(self, args):
        super().__init__()
        self.config = BertConfig.from_json_file(args.bert_config)
        self.bert = BertModel(self.config)
        self.dropout = nn.Dropout(p=args.dropout)
        self.prediction = nn.Sequential(
            nn.Linear(self.config.hidden_size, 2),
        )

        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            elif isinstance(module, BERTLayerNorm):
                module.beta.data.normal_(mean=0.0, std=self.config.initializer_range)
                module.gamma.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    module.bias.data.zero_()
        self.apply(init_weights)

        if args.init_checkpoint:
            self.load_pretrained(args.init_checkpoint)

    def load_pretrained(self, init_checkpoint, patch=False, transfer=True):
        if transfer:
            print('Load all parameters')
            missing_keys, unexpected_keys = self.load_state_dict(torch.load(init_checkpoint, map_location='cpu'),strict=False)
            print("missing keys: {}".format(missing_keys))
            print('unexpected keys: {}'.format(unexpected_keys))

        else:
            print('Load Bert parameters')
            missing_keys, unexpected_keys = self.bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'),strict=False)
            print("missing keys: {}".format(missing_keys))
            print('unexpected keys: {}'.format(unexpected_keys))

    def register_parameters(self, init_checkpoint):
        params = torch.load(init_checkpoint, map_location='cpu')
        for k,v in params.items():
            if 'mean' not in k:
                self.register_buffer(f"{k.replace('.', '_')}__mean", v)

    def forward(self, inputs):
        a = inputs['text1']
        b = inputs['text2']

        input_ids = torch.cat((a, b), dim=-1)
        token_type_ids = torch.cat((
            torch.zeros_like(a),
            torch.ones_like(b)), dim=-1).long()
        attention_mask = (input_ids != 0)
        
        bert_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        last_output = bert_output[1]

        prediction = self.prediction(self.dropout(last_output))

        return prediction
