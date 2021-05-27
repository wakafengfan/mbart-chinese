from configuration.config import *


from transformers import MBartTokenizer


class MBartChineseTokenizer():
    def __init__(self):
        self.tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25", src_lang="zh_CN", tgt_lang="zh_CN")
        self.token_mapping = json.load((bart_pt_path / "token_ids_mapping.json").open())  # k: raw_id & v:new_id
        self.unk_token_mapping = self.token_mapping.get(str(self.tokenizer.unk_token_id))
        self.token_mapping_invert = {v:k for k, v in self.token_mapping.items()}

        self.keep_tokens = list(map(int, self.token_mapping.keys()))

    def process_batch_src(self, src_texts, tgt_texts, max_len, max_tgt_len):
        inputs = self.tokenizer.prepare_seq2seq_batch(
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            src_lang="zh_CN",
            tgt_lang="zh_CN",
            max_length=max_len,
            max_target_length=max_tgt_len
        )

        inputs["input_ids"] = [[self.token_mapping.get(str(_), self.unk_token_mapping) for _ in raw_id] for raw_id in inputs["input_ids"]]
        if "labels" in inputs:
            inputs["labels"] = [[self.token_mapping.get(str(_), self.unk_token_mapping) for _ in raw_id] for raw_id in inputs["labels"]]

        return inputs

    def decode(self, token_ids):
        invert_token_ids = [self.token_mapping_invert[pred_id] for pred_id in token_ids]
        texts = self.tokenizer.decode(invert_token_ids)
        return texts
