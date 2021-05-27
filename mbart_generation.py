import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from torch.optim import Adam
from transformers import AutoConfig

from bojone_snippets import DataGenerator, AutoRegressiveDecoder
from configuration.config import *
from mbart_zh_tokenizer import MBartChineseTokenizer
from modeling_mbart import MBartForConditionalGeneration

max_c_len = 256
max_t_len = 32
batch_size = 8
epochs = 40


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            title, content = l.strip().split('\t')
            D.append((title, content))
    return D


# 加载数据集
data_root_path = common_data_path / "open_dataset" / "csl"
train_data = load_data(str(data_root_path/'train.tsv'))
valid_data = load_data(str(data_root_path/'val.tsv'))

mc_tokenizer = MBartChineseTokenizer()

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_c, batch_t = [], []
        for is_end, (title, content) in self.sample(random):
            batch_c.append(content)
            batch_t.append(title)

            if len(batch_c) == self.batch_size or is_end:
                batch_inputs = mc_tokenizer.process_batch_src(src_texts=batch_c,
                                                              tgt_texts=batch_t,
                                                              max_len=max_c_len,
                                                              max_tgt_len=max_t_len)

                batch_c_input_ids = torch.tensor(batch_inputs["input_ids"], dtype=torch.long)
                batch_c_attention_masks = torch.tensor(batch_inputs["attention_mask"], dtype=torch.long)

                batch_t_input_ids = torch.tensor(batch_inputs["labels"], dtype=torch.long)

                yield batch_c_input_ids, batch_c_attention_masks, batch_t_input_ids
                batch_c, batch_t = [], []


config = AutoConfig.from_pretrained(pretrained_model_name_or_path=bart_pt_path/"config.json", keep_tokens=keep_tokens, vocab_size=len(keep_tokens))
tmp_state_dict = torch.load(bart_pt_path/"pytorch_model.bin", map_location="cpu")
tmp_state_dict['model.shared.weight'] = torch.index_select(tmp_state_dict['model.shared.weight'], 0, torch.tensor(keep_tokens, dtype=torch.long))
tmp_state_dict["model.encoder.embed_tokens.weight"] = torch.index_select(tmp_state_dict["model.encoder.embed_tokens.weight"], 0, torch.tensor(keep_tokens, dtype=torch.long))
tmp_state_dict["model.decoder.embed_tokens.weight"] = torch.index_select(tmp_state_dict["model.decoder.embed_tokens.weight"], 0, torch.tensor(keep_tokens, dtype=torch.long))
tmp_state_dict['final_logits_bias'] = torch.index_select(tmp_state_dict['final_logits_bias'], 1, torch.tensor(keep_tokens, dtype=torch.long))

mbart_gen_model = MBartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=bart_pt_path / "pytorch_model.bin", state_dict=tmp_state_dict, config=config)

optimizer = Adam(mbart_gen_model.parameters(), lr=1e-6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mbart_gen_model.to(device)


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):  # output_ids: [1, s']
        c_encoded = inputs[0]
        with torch.no_grad():
            decoder_hidden_state = mbart_gen_model.model.decoder(encoder_hidden_states=torch.tensor(c_encoded, dtype=torch.float, device=device),
                                                                 input_ids=torch.tensor(output_ids, dtype=torch.long, device=device))[0]
            logits = mbart_gen_model.lm_head(decoder_hidden_state) + mbart_gen_model.final_logits_bias  # [1, s', V]
        pred = torch.softmax(logits[:,-1], dim=-1)
        pred = pred.cpu().detach().numpy()
        return pred

    def generate(self, text, topk=1):
        c_inputs = mc_tokenizer.process_batch_src(src_texts=[text],
                                                  tgt_texts=None,
                                                  max_len=max_c_len,
                                                  max_tgt_len=None)
        c_token_ids = c_inputs["input_ids"]

        with torch.no_grad():
            c_encoded = mbart_gen_model.model.encoder(input_ids=torch.tensor(c_token_ids, dtype=torch.long, device=device))[0]  # [1,s,h]
        output_ids = self.beam_search(c_encoded.cpu().detach().numpy(), topk)  # 基于beam search
        return mc_tokenizer.decode([int(i) for i in output_ids])

autotitle = AutoTitle(start_id=mc_tokenizer.tokenizer.bos_token_id, end_id=mc_tokenizer.tokenizer.eos_token_id, maxlen=32)

train_generator = data_generator(train_data, batch_size)

# evaluate
rouge = Rouge()
smooth = SmoothingFunction().method1
best_bleu = 0.
def predict_and_evaluate():
    total = 0
    rouge_1, rouge_2, rouge_l, bleu = 0, 0, 0, 0
    for title, content in valid_data:
        total += 1
        title = ' '.join(title).lower()
        pred_title = ' '.join(autotitle.generate(content)).lower()
        if pred_title.strip():
            scores = rouge.get_scores(hyps=pred_title, refs=title)
            rouge_1 += scores[0]['rouge-1']['f']
            rouge_2 += scores[0]['rouge-2']['f']
            rouge_l += scores[0]['rouge-l']['f']
            bleu += sentence_bleu(
                references=[title.split(' ')],
                hypothesis=pred_title.split(' '),
                smoothing_function=smooth
            )
    rouge_1 /= total
    rouge_2 /= total
    rouge_l /= total
    bleu /= total
    return {
        'rouge-1': rouge_1,
        'rouge-2': rouge_2,
        'rouge-l': rouge_l,
        'bleu': bleu,
    }

# train
mbart_gen_model.zero_grad()
for e in range(epochs):
    mbart_gen_model.train()
    for step, batch in enumerate(train_generator):

        batch = [_.to(device) for _ in batch]
        c_token_ids, c_token_masks, t_token_ids = batch
        output = mbart_gen_model(input_ids=c_token_ids, attention_mask=c_token_masks, labels=t_token_ids)

        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0 and step != 0:
            logger.info(f"epoch: {e} - step: {step} - loss: {output.loss}")


    mbart_gen_model.eval()
    s1 = "为了研究交换超立方体网络容错路由问题,引入了相邻结点集合类的概念,提出了相邻结点集的求解公式。对于满足任意子连通性条件的交换超立方体网络,给出了基于相邻结点集合类的自适应容错路由算法及算法的步长上界。仿真实验结果表明算法是有效的。"
    s1_title = "交换超立方体网络容错路由研究"
    s2 = "研究在已知目标团伙中某节点以及目标团伙特征的前提下,基于通讯痕迹特征寻找社会网络团伙。研究过程中引入了社会圈、节点中心度和事件集合关联矩阵等概念,重点将聚类分析方法与社会团伙发现相结合,以期得到一种基于通讯痕迹的社会网络团伙分析模型。"
    s2_title = "一种基于通讯痕迹的社会网络团伙分析模型"

    for s in [s1, s2]:
        logger.info(f'生成标题: {autotitle.generate(s)}')

    eval_dic = predict_and_evaluate()
    logger.info(eval_dic)







