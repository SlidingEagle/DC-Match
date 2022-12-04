from transformers.data.data_collator import DataCollatorWithPadding
from typing import Any, Dict, List
import torch

from config.global_config import global_config


class DataCollator(DataCollatorWithPadding):

    def __init__(self, args, tokenizer, padding=True):
        super(DataCollator, self).__init__(tokenizer, padding)
        self.args = args
        self.pad_id = 0
        self.improvement = global_config.improvement

    def _pad(self, data, width=-1, dtype=torch.long):
        if width == -1:
            if self.improvement:
                width = max(len(d) for d in data) + 15
            else:
                width = max(len(d) for d in data)
        rtn_data = [d + [self.pad_id] * (width - len(d)) for d in data]
        return torch.tensor(rtn_data, dtype=dtype)

    def handle_special_token(self, token_list):
        return [item.replace("Ġ", "") for item in token_list]

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        """
        features:
            input_ids, token_type_ids, attention_mask, labels,
            keyword_mask, context_mask, special_mask,
            origin_str, keywords
        """
        batch = {}

        # process entity-masked sentence pairs
        features_new = list(map(lambda x: {"input_ids": x['input_ids'],
                                           "token_type_ids": x['token_type_ids'],
                                           "labels": x['labels'] if x.get('labels', 'no') != 'no' else x['label']}, features))

        batch = self.tokenizer.pad(
            features_new,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # process template sentence
        template_keyword = "represent factual information ."
        template_intent = "convey abstract concepts and ideas ."
        encoded_template_keyword_dict = self.tokenizer(template_keyword,
                                                       truncation=True,
                                                       max_length=150,
                                                       return_token_type_ids=True)
        encoded_template_intent_dict = self.tokenizer(template_intent,
                                                       truncation=True,
                                                       max_length=150,
                                                       return_token_type_ids=True)
        origin_template_keyword = self.tokenizer.convert_ids_to_tokens(encoded_template_keyword_dict['input_ids'])
        origin_template_intent = self.tokenizer.convert_ids_to_tokens(encoded_template_intent_dict['input_ids'])
        if "chinese" not in self.args.model:
            origin_template_keyword = self.handle_special_token(origin_template_keyword)
            origin_template_intent = self.handle_special_token(origin_template_intent)

        # get id of <cls>, <sep>, 'and', ','
        id_cls = self.tokenizer(self.tokenizer.cls_token, truncation=True, max_length=150, return_token_type_ids=True)['input_ids'][1]
        id_sep = self.tokenizer(self.tokenizer.sep_token, truncation=True, max_length=150, return_token_type_ids=True)['input_ids'][1]
        id_and = self.tokenizer('and', truncation=True, max_length=150, return_token_type_ids=True)['input_ids'][1]
        id_comma = self.tokenizer(',', truncation=True, max_length=150, return_token_type_ids=True)['input_ids'][1]

        # extract keywords from batch
        keyword_sentence_list = []
        for i in range(len(features)):
            keyword_sentence = []
            keyword_mask = features[i]['keyword_mask']
            input_ids = features[i]['input_ids']
            origin_sentence = self.tokenizer.convert_ids_to_tokens(torch.tensor(input_ids))

            # 'for' loop of one sentence
            j = 0
            while j < len(keyword_mask):
                if keyword_mask[j] == 1:
                    # find a keyword
                    keyword = []
                    for k in range(j, len(keyword_mask)):
                        # break until keyword_mask == 0
                        if keyword_mask[k] == 0:
                            break
                        keyword.append(input_ids[k])
                    keyword_sentence.append(keyword)
                    origin_keyword = self.tokenizer.convert_ids_to_tokens(torch.tensor(keyword))
                    # add the length of keyword
                    j += len(keyword)
                else:
                    j += 1
            keyword_sentence_list.append(keyword_sentence)

        # insert keywords into template_keyword
        keyword_prompt_ids = []
        attention_mask_keyword_prompt = []
        for i in range(len(keyword_sentence_list)):
            keyword_prompt_sentence = [id_cls]
            # insert keywords
            for j in range(len(keyword_sentence_list[i])):
                keyword_prompt_sentence.extend(keyword_sentence_list[i][j])
                if j != len(keyword_sentence_list[i])-1:
                    keyword_prompt_sentence.append(id_comma)
            # insert template
            for j in range(1, len(encoded_template_keyword_dict['input_ids'])):
                keyword_prompt_sentence.append(encoded_template_keyword_dict['input_ids'][j])
            origin_keyword_prompt = self.tokenizer.convert_ids_to_tokens(torch.tensor(keyword_prompt_sentence))
            keyword_prompt_ids.append(keyword_prompt_sentence)
            attention_mask_keyword_prompt.append([1 for i in range(len(keyword_prompt_sentence))])

        # extract intent from batch
        intent_sentence_list = []
        for i in range(len(features)):
            intent_sentence = []
            intent_mask = features[i]['context_mask']
            input_ids = features[i]['input_ids']
            origin_sentence = self.tokenizer.convert_ids_to_tokens(torch.tensor(input_ids))

            # 'for' loop of one sentence
            j = 0
            while j < len(intent_mask):
                if intent_mask[j] == 1:
                    # find a keyword
                    intent = []
                    for k in range(j, len(intent_mask)):
                        # break until keyword_mask == 0
                        if intent_mask[k] == 0:
                            break
                        intent.append(input_ids[k])
                    intent_sentence.append(intent)
                    origin_intent = self.tokenizer.convert_ids_to_tokens(torch.tensor(intent))
                    # add the length of intent
                    j += len(intent)
                else:
                    j += 1
            intent_sentence_list.append(intent_sentence)

        # insert keywords into template_keyword
        intent_prompt_ids = []
        attention_mask_intent_prompt = []
        for i in range(len(intent_sentence_list)):
            intent_prompt_sentence = [id_cls]
            # insert intent
            for j in range(len(intent_sentence_list[i])):
                intent_prompt_sentence.extend(intent_sentence_list[i][j])
                if j != len(intent_sentence_list[i]) - 1:
                    intent_prompt_sentence.append(id_comma)
            # insert template
            for j in range(1, len(encoded_template_intent_dict['input_ids'])):
                intent_prompt_sentence.append(encoded_template_intent_dict['input_ids'][j])
            origin_intent_prompt = self.tokenizer.convert_ids_to_tokens(torch.tensor(intent_prompt_sentence))
            intent_prompt_ids.append(intent_prompt_sentence)
            attention_mask_intent_prompt.append([1 for i in range(len(intent_prompt_sentence))])


        batch['attention_mask'] = self._pad([x['attention_mask'] for x in features])
        if "keyword_mask" in features[0].keys():
            batch['keyword_mask'] = self._pad([x['keyword_mask'] for x in features])
        else:
            batch['keyword_mask'] = []
        if "context_mask" in features[0].keys():
            batch['context_mask'] = self._pad([x['context_mask'] for x in features])
        else:
            batch['context_mask'] = []
        if "special_mask" in features[0].keys():
            batch['special_mask'] = self._pad([x['special_mask'] for x in features])
        else:
            batch['special_mask'] = []

        if self.improvement:
            new_width = batch['attention_mask'].shape[1]

            batch['input_ids'] = self._pad(batch['input_ids'].tolist(), new_width)
            batch['token_type_ids'] = self._pad(batch['token_type_ids'].tolist(), new_width)

            batch['keyword_prompt_ids'] = self._pad(keyword_prompt_ids, new_width)
            batch['attention_mask_keyword_prompt'] = self._pad(attention_mask_keyword_prompt, new_width)
            batch['intent_prompt_ids'] = self._pad(intent_prompt_ids, new_width)
            batch['attention_mask_intent_prompt'] = self._pad(attention_mask_intent_prompt, new_width)

        return batch
