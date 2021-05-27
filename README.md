# mbart-chinese

基于[mbart-large-cc25](https://huggingface.co/facebook/mbart-large-cc25) 
的中文生成任务

## Input
- source input: `text` + `</s>` + `lang_code`

- target input: `lang_code` + `text` + `</s>`


## Usage

``token_ids_mapping.json``：从全量词表中抽取出的中文字符及高频英文字符，在老新词典中的映射关系表。


## Todo

- mbart在中文标题生成任务的评测结果