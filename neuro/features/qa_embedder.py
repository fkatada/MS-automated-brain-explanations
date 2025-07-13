"""""""""
Note: imodelsx has the actual QAEmb class now
"""


from os.path import expanduser, join
from typing import List

import imodelsx.llm
import numpy as np
import pandas as pd
import scipy.special
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import neuro.config as config
import neuro.features.qa_questions as qa_questions

# from vllm import LLM, SamplingParams
# import torch


class MutiTaskClassifier(nn.Module):
    def __init__(self, checkpoint, num_binary_outputs):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            checkpoint, return_dict=True  # , output_hidden_states=True,
        )
        self.classifiers = nn.ModuleList(
            [nn.Linear(self.model.config.hidden_size, 2)
             for _ in range(num_binary_outputs)]
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask)
        logits = torch.stack([classifier(outputs.pooler_output)
                              for classifier in self.classifiers])
        logits = logits.permute(1, 0, 2)
        return logits


class FinetunedQAEmbedder:
    def __init__(self, checkpoint, qa_questions_version='v3_boostexamples'):
        # print(f'{checkpoint=} {num_binary_outputs=}')
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, return_token_type_ids=False)  # , device_map='auto')
        question_counts = {
            'v1': 376,
            'v2': 142,
            'v3_boostexamples': 156,
        }
        self.model = MutiTaskClassifier(
            checkpoint, num_binary_outputs=sum(question_counts.values()),
        ).eval()
        question_idxs = {
            'v1': (0, 376),
            'v2': (376, 376 + 142),
            'v3_boostexamples': (376 + 142, 376 + 142 + 156),
        }
        self.question_idxs = question_idxs[qa_questions_version]

        state_dict = torch.load(
            join(config.ROOT_DIR, 'finetune', f'{checkpoint}.pt'))
        self.model.load_state_dict(state_dict)
        self.model = torch.nn.DataParallel(self.model).to('cuda')

    def get_embs_from_text_list(self, texts: List[str], batch_size=64):
        '''Continuous outputs for each prediction
        '''
        with torch.no_grad():
            inputs = self.tokenizer.batch_encode_plus(
                texts, padding="max_length",
                truncation=True, max_length=512, return_tensors="pt")

            inputs = inputs.to('cuda')

            answer_predictions = []
            for i in tqdm(range(0, len(inputs['input_ids']), batch_size)):
                outputs = self.model(**{k: v[i:i+batch_size]
                                        for k, v in inputs.items()})
                answer_predictions.append(outputs.cpu().detach().numpy())
            answer_predictions = answer_predictions[self.question_idxs[0]
                :self.question_idxs[1]]
            answer_predictions = np.vstack(answer_predictions)
            answer_predictions = scipy.special.softmax(
                answer_predictions, axis=-1)
            return answer_predictions