import copy
import datetime
import os
import json
import bisect
import numpy as np
import pandas as pd
from tqdm import tqdm
import re
from spacy.tokens import Span
from stanza import Pipeline

"""
===================environmental setting=================
# Basic Setup (One time activity)

# 1. Clone the DYGIE++ repository from: https://github.com/dwadden/dygiepp. This repositiory is managed by Wadden et al., authors of the paper Entity, Relation, and Event Extraction with Contextualized Span Representations (https://www.aclweb.org/anthology/D19-1585.pdf).

# git clone https://github.com/dwadden/dygiepp.git

# 2. Navigate to the root of repo in your system and use the following commands to setup the conda environment:

# conda create --name dygiepp python=3.7
# conda activate dygiepp
# cd dygiepp
# pip install -r requirements.txt
# conda develop .   # Adds DyGIE to your PYTHONPATH

# Running Inference on Radiology Reports

# 3. Activate the conda environment:

# conda activate dygiepp

"""


def remove_duplicate_punctuation(text):
    # 匹配连续出现的标点符号
    pattern = re.compile(r'(\s*[^\w\s]\s*)(\s*[^\w\s]\s*)+')
    # 使用正则表达式替换连续出现的标点符号为第一个标点符号
    result = pattern.sub(lambda match: match.group(1), text)
    return result.strip()

def remove_extra_spaces(text):
    # \s+ 匹配一个或多个空白字符（包括空格、制表符等）
    # 替换为单个空格
    return re.sub(r'\s+', ' ', text).strip()


def split_sentences(text):
    # 使用正则表达式分割句子，句子以'.', '!', '?'结尾，但不分割小数点
    sentences = re.split(r'(?<!\d)[.!?](?!\d)\s*', text)
    # 移除空的句子
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def overlap_ratio(sent1, sent2):
    # 将句子转为单词列表
    words1 = set(sent1.split())
    words2 = set(sent2.split())
    # 计算重叠单词的数量
    overlap = words1 & words2
    # 计算重叠比例
    return len(overlap) / len(words1) if words1 else 0


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            print("MyEncoder-datetime.datetime")
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        if isinstance(obj, int):
            return int(obj)
        elif isinstance(obj, float):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Span):
            return str(obj)
        else:
            return super(MyEncoder, self).default(obj)


class RadGraphNER:
    # change the data architecture
    def __init__(self, corpus=None, ann_path=None, model_path=None, cuda='1', is_get_output=True, data_name='mimic_cxr'):
        """

        Args:
            corpus: dict: {id1: s1, id2: s2, id3: s3, ...}. if corpus is None, temp_dygie_output.json should be at current path
            model_path: the official checkpoint for radgraph
            cuda: the id for gpu
            is_get_input: Whether to convert to the format processed by radgraph
        """
        self.model_path = model_path
        self.cuda = cuda
        # user defined
        self.input_path = "/home/miao/data/Code/MSC-V1212-ablation-study/knowledge_encoder/temp_dygie_input.json"
        self.output_path = '/home/miao/data/Code/MSC-V1212-ablation-study/knowledge_encoder/temp_dygie_output.json'
        if is_get_output:
            if data_name == 'mimic_cxr':
                self.get_mimic_temp_dygie_input(ann_path)
            elif data_name == 'iu_xray':
                self.get_iu_xray_temp_dygie_input(ann_path)
            elif data_name == 'mimic_abn':
                self.get_mimic_abn_temp_dygie_input(ann_path)
            else:
                self.get_corpus_temp_dygie_input(corpus)
            # extract entities and relationships using RadGraph
            self.extract_triplets()

    def get_mimic_temp_dygie_input(self, ann_path):
        # note that only the training corpus can be used.
        ann = json.load(open(ann_path))
        print("initialization the input data")
        del ann['val']
        del ann['test']
        with open(self.input_path, encoding='utf8', mode='w') as f:
            for split, value in ann.items():
                print(f"preprocessing the {split} data...")
                subject_study = []
                for item in tqdm(value):
                    subject, study = str(item['subject_id']), str(item['study_id'])
                    cur_subject_study = subject + '_' + study
                    if cur_subject_study not in subject_study:
                        subject_study.append(cur_subject_study)
                        sen = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
                                     item['report'])
                        input_item = {
                            'doc_key': cur_subject_study,
                            "sentences": [sen.strip().split()]
                        }
                        f.write(json.dumps(input_item, cls=MyEncoder))
                        f.write('\n')

    def get_mimic_abn_temp_dygie_input(self, ann_data):
        # note that only the training corpus can be used.
        with open(self.input_path, encoding='utf8', mode='w') as f:
            for split, value in ann_data.items():
                print(f"preprocessing the {split} data...")
                subject_study = []
                for item in tqdm(value):
                    subject, study = str(item['subject_id']), str(item['study_id'])
                    cur_subject_study = subject + '_' + study
                    if cur_subject_study not in subject_study:
                        subject_study.append(cur_subject_study)
                        sen = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
                                     item['report'])
                        input_item = {
                            'doc_key': cur_subject_study,
                            "sentences": [sen.strip().split()]
                        }
                        f.write(json.dumps(input_item, cls=MyEncoder))
                        f.write('\n')

    def get_corpus_temp_dygie_input(self, corpus):
        # note that only the training corpus can be used.
        with open(self.input_path, encoding='utf8', mode='w') as f:
            for item_id, value in corpus.items():
                input_item = {
                    'doc_key': item_id,
                    "sentences": [value.strip().split()]
                }
                f.write(json.dumps(input_item, cls=MyEncoder))
                f.write('\n')

    def get_iu_xray_temp_dygie_input(self, ann_path):
        # note that only the training corpus can be used.
        ann = json.load(open(ann_path))
        print("initialization the input data")
        with open(self.input_path, encoding='utf8', mode='w') as f:
            for key, value in ann.items():
                print(f"preprocessing the {key} data...")
                findings = value['findings']

                if len(findings) == 0:
                    continue
                findings = re.sub(r'([.,?!:])', r' \1', findings)
                input_item = {
                    'doc_key': key,
                    "sentences": [findings.strip().split()]
                }
                f.write(json.dumps(input_item, cls=MyEncoder))
                f.write('\n')

    def extract_triplets(self):
        print("extract output files using radgraph.")
        os.system(f"allennlp predict {self.model_path} {self.input_path} \
                    --predictor dygie --include-package dygie \
                    --use-dataset-reader \
                    --output-file {self.output_path} \
                    --silent")

    def preprocess_mimic_radgraph_output(self):
        # =====================build the serialized string=============#
        final_dict = {}
        useless_findings = useless_core_findings_new()
        # negative_list = ['no ', 'not ', 'free of', 'negative', 'without', 'clear of']  # delete unremarkable
        print("obtain the triples for each report.")
        with open(self.output_path, 'r') as f:
            for line in tqdm(f):
                data_item = json.loads(line)
                n = data_item['predicted_ner'][0]
                # r = data_item['predicted_relations'][0]
                s = data_item['sentences'][0]
                if len(n) == 0:
                    # print(len(n), " ".join(s))
                    continue

                doc_key = data_item['doc_key']
                # if doc_key in ['10000935_50578979', '10116310_50782200', '17277688_55816958', ]:
                #     print()
                n = preprocessing_entities(n, s, doc_key)
                dict_entity = {'text': ' '.join(s)}

                # ====Remove some useless entities and relationships====
                # initialized the variables
                dot_index = [index for index, token in enumerate(s) if token in ['.', '?', '!']]
                if len(dot_index) != 0:
                    if dot_index[0] != 0:
                        dot_index = [0, *dot_index]
                    if dot_index[-1] != len(s) - 1:
                        dot_index = [*dot_index, len(s)]
                    else:
                        dot_index[-1] += 1
                else:
                    dot_index = [0, len(s)]  # the last index + 1
                core_findings = []
                dot_s_idx, dot_e_idx, pre_sen_idx = -1, -1, -1
                cur_core_findings, previous_node_modified = [], False
                # cur_core_findings: words of each sentence
                # core_findings: finding of each sentence
                for idx, ent_item in enumerate(n):
                    start_idx, end_idx, ent_label = ent_item[0], ent_item[1], ent_item[2].strip()
                    cur_ent = " ".join(s[start_idx:end_idx + 1]).strip('"').strip("'").strip()
                    # delete unrelated entities
                    if cur_ent in list(',:;!()*&-_?'):
                        continue

                    sen_idx = bisect.bisect_left(dot_index, start_idx)
                    if sen_idx != pre_sen_idx:
                        if len(cur_core_findings) != 0:
                            if len(cur_core_findings) == 1 and cur_core_findings[0] in useless_findings:
                                pass
                            else:
                                core_findings.append(' '.join(cur_core_findings))
                        cur_core_findings, previous_node_modified = [], False
                        if start_idx == dot_index[sen_idx]:
                            dot_s_idx = dot_index[sen_idx]
                            dot_e_idx = dot_index[sen_idx] + 1 if sen_idx == len(dot_index) - 1 else dot_index[
                                sen_idx + 1]
                            pre_sen_idx = sen_idx + 1
                        else:
                            dot_s_idx = dot_index[sen_idx - 1]
                            dot_e_idx = dot_index[sen_idx]
                            pre_sen_idx = sen_idx

                    if start_idx <= dot_e_idx < end_idx:
                        print(doc_key, "error!!", cur_ent)
                        cur_ent = cur_ent.split('.')[0].strip()
                    if "DA" in ent_label and not previous_node_modified:
                        cur_core_findings = ['no', *cur_core_findings]
                        previous_node_modified = True
                    elif "U" in ent_label and not previous_node_modified:
                        cur_core_findings = ["maybe", *cur_core_findings]
                        previous_node_modified = True
                    cur_core_findings.append(cur_ent)  # add word

                if len(cur_core_findings) != 0:
                    if len(cur_core_findings) == 1 and cur_core_findings[0] in useless_findings:
                        pass
                    else:
                        core_findings.append(' '.join(cur_core_findings))
                dict_entity['core_findings'] = core_findings
                final_dict.update({
                    doc_key: dict_entity
                })

        # =================== save files ======================#
        # with open(ent_path, 'w') as outfile:
        #     json.dump(final_dict, outfile, indent=2)
        return final_dict

    def preprocess_iu_xray_radgraph_output(self):
        # =====================build the serialized string=============#
        final_dict = {}
        useless_findings = useless_core_findings_new()
        # negative_list = ['no ', 'not ', 'free of', 'negative', 'without', 'clear of']  # delete unremarkable
        print("obtain the triples for each report.")
        with open(self.output_path, 'r') as f:
            for line in tqdm(f):
                data_item = json.loads(line)
                n = data_item['predicted_ner'][0]
                # r = data_item['predicted_relations'][0]
                s = data_item['sentences'][0]
                if len(n) == 0:
                    # print(len(n), " ".join(s))
                    continue

                doc_key = data_item['doc_key']
                # if doc_key in ['10000935_50578979', '10116310_50782200', '17277688_55816958', ]:
                #     print()
                n = preprocessing_entities(n, s, doc_key)
                dict_entity = {'text': ' '.join(s)}

                # ====Remove some useless entities and relationships====
                # initialized the variables
                dot_index = [index for index, token in enumerate(s) if token in ['.', '?', '!']]
                if len(dot_index) != 0:
                    if dot_index[0] != 0:
                        dot_index = [0, *dot_index]
                    if dot_index[-1] != len(s) - 1:
                        dot_index = [*dot_index, len(s)]
                    else:
                        dot_index[-1] += 1
                else:
                    dot_index = [0, len(s)]  # the last index + 1
                core_findings = []
                dot_s_idx, dot_e_idx, pre_sen_idx = -1, -1, -1
                cur_core_findings, previous_node_modified = [], False
                # cur_core_findings: words of each sentence
                # core_findings: finding of each sentence
                for idx, ent_item in enumerate(n):
                    start_idx, end_idx, ent_label = ent_item[0], ent_item[1], ent_item[2].strip()
                    cur_ent = " ".join(s[start_idx:end_idx + 1]).strip('"').strip("'").strip()
                    # delete unrelated entities
                    if cur_ent in list(',:;!()*&-_?'):
                        continue

                    sen_idx = bisect.bisect_left(dot_index, start_idx)
                    if sen_idx != pre_sen_idx:
                        if len(cur_core_findings) != 0:
                            if len(cur_core_findings) == 1 and cur_core_findings[0] in useless_findings:
                                pass
                            else:
                                core_findings.append(' '.join(cur_core_findings))
                        cur_core_findings, previous_node_modified = [], False
                        if start_idx == dot_index[sen_idx]:
                            dot_s_idx = dot_index[sen_idx]
                            dot_e_idx = dot_index[sen_idx] + 1 if sen_idx == len(dot_index) - 1 else dot_index[
                                sen_idx + 1]
                            pre_sen_idx = sen_idx + 1
                        else:
                            dot_s_idx = dot_index[sen_idx - 1]
                            dot_e_idx = dot_index[sen_idx]
                            pre_sen_idx = sen_idx

                    if start_idx <= dot_e_idx < end_idx:
                        print(doc_key, "error!!", cur_ent)
                        cur_ent = cur_ent.split('.')[0].strip()
                    if "DA" in ent_label and not previous_node_modified:
                        cur_core_findings = ['no', *cur_core_findings]
                        previous_node_modified = True
                    elif "U" in ent_label and not previous_node_modified:
                        cur_core_findings = ["maybe", *cur_core_findings]
                        previous_node_modified = True
                    cur_core_findings.append(cur_ent)  # add word

                if len(cur_core_findings) != 0:
                    if len(cur_core_findings) == 1 and cur_core_findings[0] in useless_findings:
                        pass
                    else:
                        core_findings.append(' '.join(cur_core_findings))
                dict_entity['core_findings'] = core_findings
                final_dict.update({
                    doc_key: dict_entity
                })

        # =================== save files ======================#
        # with open(ent_path, 'w') as outfile:
        #     json.dump(final_dict, outfile, indent=2)
        return final_dict

    def preprocess_corpus_radgraph_output(self):
        # =====================build the serialized string=============#
        final_dict = {}
        useless_findings = useless_core_findings_new()
        print("obtain the triples for each report.")
        with open(self.output_path, 'r') as f:
            for line in tqdm(f):
                data_item = json.loads(line)
                n = data_item['predicted_ner'][0]
                # r = data_item['predicted_relations'][0]
                s = data_item['sentences'][0]
                if len(n) == 0:
                    continue

                doc_key = data_item['doc_key']
                # if doc_key in ['10000935_50578979', '10116310_50782200', '17277688_55816958', ]:
                #     print()
                n = preprocessing_entities(n, s, doc_key)
                dict_entity = {'text': ' '.join(s)}

                # ====Remove some useless entities and relationships====
                # initialized the variables
                dot_index = [index for index, token in enumerate(s) if token in ['.', '?', '!']]
                if len(dot_index) != 0:
                    if dot_index[0] != 0:
                        dot_index = [0, *dot_index]
                    if dot_index[-1] != len(s) - 1:
                        dot_index = [*dot_index, len(s)]
                    else:
                        dot_index[-1] += 1
                else:
                    dot_index = [0, len(s)]  # the last index + 1
                core_findings = []
                core_findings_index = []
                cur_ent_index_list = []
                dot_s_idx, dot_e_idx, pre_sen_idx = -1, -1, -1
                cur_core_findings, previous_node_modified = [], False
                # cur_core_findings: words of each sentence
                # core_findings: finding of each sentence
                for idx, ent_item in enumerate(n):
                    start_idx, end_idx, ent_label = ent_item[0], ent_item[1], ent_item[2].strip()
                    cur_ent = " ".join(s[start_idx:end_idx + 1]).strip('"').strip("'").strip()
                    # delete unrelated entities
                    if cur_ent in list(',:;!()*&-_?'):
                        continue
                    cur_ent_index = list(range(start_idx, end_idx + 1))
                    sen_idx = bisect.bisect_left(dot_index, start_idx)
                    if sen_idx != pre_sen_idx:
                        if len(cur_core_findings) != 0:
                            if len(cur_core_findings) == 1 and cur_core_findings[0] in useless_findings:
                                pass
                            else:
                                core_findings.append(' '.join(cur_core_findings))
                                core_findings_index.append(cur_ent_index_list)
                        cur_core_findings, previous_node_modified = [], False
                        cur_ent_index_list = []
                        if start_idx == dot_index[sen_idx]:
                            dot_s_idx = dot_index[sen_idx]
                            dot_e_idx = dot_index[sen_idx] + 1 if sen_idx == len(dot_index) - 1 else dot_index[
                                sen_idx + 1]
                            pre_sen_idx = sen_idx + 1
                        else:
                            dot_s_idx = dot_index[sen_idx - 1]
                            dot_e_idx = dot_index[sen_idx]
                            pre_sen_idx = sen_idx

                    if start_idx <= dot_e_idx < end_idx:
                        print(doc_key, "error!!", cur_ent)
                        temp = cur_ent.split('.')[0].strip()
                        _idx = cur_ent.find(temp)
                        cur_ent_index = cur_ent_index[_idx: (_idx + len(temp))]
                        cur_ent = temp
                    if "DA" in ent_label and not previous_node_modified:
                        cur_core_findings = ['no', *cur_core_findings]
                        previous_node_modified = True
                    elif "U" in ent_label and not previous_node_modified:
                        cur_core_findings = ["maybe", *cur_core_findings]
                        previous_node_modified = True
                    cur_core_findings.append(cur_ent)  # add word
                    cur_ent_index_list.extend(cur_ent_index)

                if len(cur_core_findings) != 0:
                    if cur_core_findings[0] in useless_findings:
                        pass
                    else:
                        core_findings.append(' '.join(cur_core_findings))
                        core_findings_index.append(cur_ent_index_list)
                dict_entity['report'] = s
                dict_entity['core_findings'] = core_findings
                dict_entity['core_findings_index'] = core_findings_index
                final_dict.update({
                    doc_key: dict_entity
                })

        # =================== save files ======================#
        # with open(ent_path, 'w') as outfile:
        #     json.dump(final_dict, outfile, indent=2)
        return final_dict

    def preprocess_indication_corpus_radgraph_output(self):
        # =====================build the serialized string=============#
        final_dict = {}
        useless_findings = useless_core_findings_new()
        print("obtain the triples for each report.")
        with open(self.output_path, 'r') as f:
            for line in tqdm(f):
                data_item = json.loads(line)
                n = data_item['predicted_ner'][0]
                # r = data_item['predicted_relations'][0]
                s = data_item['sentences'][0]
                if len(n) == 0:
                    continue

                doc_key = data_item['doc_key']
                # if doc_key in ['10000935_50578979', '10116310_50782200', '17277688_55816958', ]:
                #     print()
                n = preprocessing_entities(n, s, doc_key)
                dict_entity = {'text': ' '.join(s)}

                # ====Remove some useless entities and relationships====
                # initialized the variables
                dot_index = [index for index, token in enumerate(s) if token in ['.', '?', '!']]
                if len(dot_index) != 0:
                    if dot_index[0] != 0:
                        dot_index = [0, *dot_index]
                    if dot_index[-1] != len(s) - 1:
                        dot_index = [*dot_index, len(s)]
                    else:
                        dot_index[-1] += 1
                else:
                    dot_index = [0, len(s)]  # the last index + 1
                core_findings = []
                core_findings_index = []
                cur_ent_index_list = []
                dot_s_idx, dot_e_idx, pre_sen_idx = -1, -1, -1
                cur_core_findings, previous_node_modified = [], False
                # cur_core_findings: words of each sentence
                # core_findings: finding of each sentence
                for idx, ent_item in enumerate(n):
                    start_idx, end_idx, ent_label = ent_item[0], ent_item[1], ent_item[2].strip()
                    cur_ent = " ".join(s[start_idx:end_idx + 1]).strip('"').strip("'").strip()
                    # delete unrelated entities
                    # if cur_ent in list(',:;!()*&-_?'):
                    #     continue
                    cur_ent = re.sub('[,:;!()*&-_?]', '', cur_ent)
                    cur_ent_index = list(range(start_idx, end_idx + 1))
                    sen_idx = bisect.bisect_left(dot_index, start_idx)
                    if sen_idx != pre_sen_idx:
                        if len(cur_core_findings) != 0:
                            if cur_core_findings[0] in useless_findings:
                                pass
                            else:
                                core_findings.append(' '.join(cur_core_findings))
                                core_findings_index.append(cur_ent_index_list)
                        cur_core_findings, previous_node_modified = [], False
                        cur_ent_index_list = []
                        if start_idx == dot_index[sen_idx]:
                            dot_s_idx = dot_index[sen_idx]
                            dot_e_idx = dot_index[sen_idx] + 1 if sen_idx == len(dot_index) - 1 else dot_index[
                                sen_idx + 1]
                            pre_sen_idx = sen_idx + 1
                        else:
                            dot_s_idx = dot_index[sen_idx - 1]
                            dot_e_idx = dot_index[sen_idx]
                            pre_sen_idx = sen_idx

                    if start_idx <= dot_e_idx < end_idx:
                        print(doc_key, "error!!", cur_ent)
                        temp = cur_ent.split('.')[0].strip()
                        _idx = cur_ent.find(temp)
                        cur_ent_index = cur_ent_index[_idx: (_idx + len(temp))]
                        cur_ent = temp
                    if "DA" in ent_label and not previous_node_modified:
                        # cur_core_findings = ['no', *cur_core_findings]
                        previous_node_modified = True
                    elif "U" in ent_label and not previous_node_modified:
                        # cur_core_findings = ["maybe", *cur_core_findings]
                        previous_node_modified = True
                    cur_core_findings.append(cur_ent)  # add word
                    cur_ent_index_list.extend(cur_ent_index)

                if len(cur_core_findings) != 0:
                    if cur_core_findings[0] in useless_findings:
                        pass
                    else:
                        core_findings.append(' '.join(cur_core_findings))
                        core_findings_index.append(cur_ent_index_list)
                dict_entity['report'] = s
                dict_entity['core_findings'] = core_findings
                dict_entity['core_findings_index'] = core_findings_index
                final_dict.update({
                    doc_key: dict_entity
                })

        # =================== save files ======================#
        # with open(ent_path, 'w') as outfile:
        #     json.dump(final_dict, outfile, indent=2)
        return final_dict


def preprocessing_entities(n, s, doc_key):
    new_n = []
    head_end_idx = -1
    for idx, item in enumerate(n, start=1):
        start_idx, end_idx, ent_label = item[0], item[1], item[2].strip()
        if start_idx > end_idx:
            continue
        elif start_idx <= head_end_idx:
            ori_s_idx, ori_e_idx = new_n[-1][0], new_n[-1][1]
            cur_best_str = ' '.join(s[ori_s_idx: (ori_e_idx + 1)])
            cur_str = ' '.join(s[start_idx: (end_idx + 1)])
            if ' .' in cur_best_str:
                if ' .' not in cur_str:
                    new_n.pop(-1)
                    new_n.append(item)
                    head_end_idx = end_idx
                    print(f"{doc_key} drop entities1: {cur_str} | {cur_best_str}")
                else:
                    print(f"{doc_key} drop entities2: {cur_best_str} | {cur_str}")
            else:
                if ' .' not in cur_str and ori_e_idx - ori_s_idx < (end_idx - start_idx):
                    new_n.pop(-1)
                    new_n.append(item)
                    head_end_idx = end_idx
                    print(f"{doc_key} drop entities3: {cur_str} | {cur_best_str}")
                else:
                    print(f"{doc_key} drop entities4: {cur_best_str} | {cur_str}")
            continue
        else:
            new_n.append(item)
            head_end_idx = end_idx
    return new_n


def useless_core_findings_new():
    result = {'It', 'it', 'otherwise', 'They', 'These', 'This'}
    return result


def get_mimic_cxr_annotations(ann_path, ent_data, file_name):
    ann_data = json.load(open(ann_path))
    new_ann_data = {}
    for split, value in ann_data.items():
        print(f"current preprocessing the {split}....")
        new_ann_data[split] = []
        for item in tqdm(value):
            try:
                doc_key = str(item['subject_id']) + '_' + str(item['study_id'])
                sample_core_finding = ent_data[doc_key]
                core_findings = sample_core_finding['core_findings']
                report = sample_core_finding['text']
            except:
                core_findings = []
                report = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
                                item['report'])
            sample_item = {
                'id': item['id'],
                "study_id": item['study_id'],
                'subject_id': item['subject_id'],
                "report": report,
                'image_path': item['image_path'],
                'core_findings': core_findings,
            }

            new_ann_data[split].append(sample_item)
    with open(file_name, 'w') as f:
        json.dump(new_ann_data, f, indent=2)


def get_mimic_cxr_annotations_temp(ann_path, ent_data, file_name):
    ann_data = json.load(open(ann_path))
    new_ann_data = {}
    for split, value in ann_data.items():
        print(f"current preprocessing the {split}....")
        new_ann_data[split] = []
        for item in tqdm(value):
            try:
                doc_key = str(item['subject_id']) + '_' + str(item['study_id'])
                sample_core_finding = ent_data[doc_key]
                core_findings = sample_core_finding['core_findings']
                report = sample_core_finding['text']
            except:
                core_findings = []
                report = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
                                item['report'])
            sample_item = {
                'id': item['id'],
                "study_id": item['study_id'],
                'subject_id': item['subject_id'],
                "report": report,
                'image_path': item['image_path'],
                'core_findings': core_findings,
            }

            new_ann_data[split].append(sample_item)
    with open(file_name, 'w') as f:
        json.dump(new_ann_data, f, indent=2)


def get_mimic_abn_annotations(ann_path, ent_data, file_name):
    ann_data = json.load(open(ann_path))
    new_ann_data = {}
    for split, value in ann_data.items():
        print(f"current preprocessing the {split}....")
        new_ann_data[split] = []
        for item in tqdm(value):
            try:
                doc_key = item['id']
                sample_core_finding = ent_data[doc_key]
                core_findings = sample_core_finding['core_findings']
                report = sample_core_finding['text']
            except:
                core_findings = []
                report = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
                                item['findings'])
            # new_item = copy.deepcopy(item)
            # new_item['findings_factual_serialization'] = core_findings
            # new_item['findings'] = report
            # construct multi-to-multi paradigm
            for image_path in item['image_path']:

                image_name = image_path.split('/')[-1].split('.')[0]
                subject_id, study_id = item['id'].split('_')
                multiview_image_path = copy.deepcopy(item['image_path'])
                multiview_image_path.remove(image_path)
                new_item = {
                    'id': image_name,
                    'study_id': eval(study_id),
                    'subject_id': eval(subject_id),
                    "report": report,
                    "image_path": [image_path],
                    "core_findings": core_findings,
                    'impression': item['impression'],
                    'comparison': item['comparison'],
                    'indication': item['indication'],
                    'indication_core_findings': item['indication_pure'],
                    'findings': item['findings'],
                    'multiview_image_path': multiview_image_path,
                    'specific_knowledge': []
                }

                new_ann_data[split].append(new_item)
    with open(file_name, 'w') as f:
        json.dump(new_ann_data, f, indent=2)

def get_iu_xray_annotations(ann_path, ent_data, file_name):
    ann_data = json.load(open(ann_path))
    new_ann_data = {}
    for key, value in ann_data.items():
        print(f"current preprocessing the {key}....")
        try:
            sample_core_finding = ent_data[key]
            core_findings = sample_core_finding['core_findings']
            # report = sample_core_finding['text']
        except:
            core_findings = []
            # report = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
            #                 value['findings'])
        new_value = copy.deepcopy(value)
        new_value['core_findings'] = core_findings
        new_ann_data[key] = new_value
    with open(file_name, 'w') as f:
        json.dump(new_ann_data, f, indent=2)


def get_plot_cases_factual_serialization():
    root = '/home/miao/data/Code/results/ablation study/plot_cases/'
    test_pred_path = os.path.join(root, 'test_prediction_temp.csv')
    pred_df = pd.read_csv(test_pred_path)
    image_id_list = pred_df['images_id'].tolist()
    radgraph_path = '/home/miao/data/dataset/checkpoints/radgraph/model.tar.gz'
    corpus = {image_id: gen_text for image_id, gen_text in
              zip(pred_df['images_id'].tolist(), pred_df['pred_report'].tolist())}
    radgraph = RadGraphNER(corpus=corpus, is_get_output=True, is_mimic=False, model_path=radgraph_path, cuda='0')
    pred_fs = radgraph.preprocess_corpus_radgraph_output()
    gen_fs_list = [pred_fs[img_id]['core_findings'] for img_id in image_id_list]
    gen_fs_index_list = [pred_fs[img_id]['core_findings_index'] for img_id in image_id_list]
    pred_df['gen_fs'] = gen_fs_list
    pred_df['gen_fs_index'] = gen_fs_index_list
    pred_df.to_csv(os.path.join(root, 'test_prediction.csv'), index=False)


class SetLogger:
    def __init__(self, filepath, mode, lock=None):
        """
        Implements write routine
        :param filepath: the file where to write
        :param mode: can be 'w' or 'a'
        :param lock: pass a shared lock for multi process write access
        """
        self.filepath = filepath
        if mode not in ['w', 'a']:
            assert False, 'Mode must be one of w, r or a'
        else:
            self.mode = mode
        self.lock = lock

    def info(self, s):
        if self.lock:
            self.lock.acquire()

        try:
            with open(self.filepath, self.mode) as f:
                f.write(s + '\n')
        except Exception as e:
            print(e)

        if self.lock:
            self.lock.release()


def extract_indication_factual_serialization_by_radgraph(radgraph_model_path, logger=None):
    ann_path = '/home/miao/data/dataset/MIMIC-CXR/mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_v0227.json'
    ann_data = json.load(open(ann_path))
    for split, value in ann_data.items():
        for item in tqdm(value):
            if len(item['indication']) == 0:
                item['indication_core_findings'] = []
                continue
            # years old historical
            indication = re.sub('[//?_,!]+', '', item['indication'])
            indication = re.sub('(?<=:)(?=\S)', ' ', indication)
            indication = re.sub('(?<=\s)\s+', '', indication)
            indication = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ', indication)
            indication = indication.lower()
            sex = ''
            if 'f' in indication or 'women' in indication or 'woman' in indication:
                sex = 'woman'
            elif 'm' in indication or 'men' in indication or 'man' in indication:
                sex = 'man'

            indication_list = indication.lower().split()
            # add "."
            if indication_list[-1] != '.':
                indication_list.append('.')

            corpus = {0: ' '.join(indication_list * 10)}
            radgraph = RadGraphNER(corpus=corpus, is_get_output=True, is_mimic=False, model_path=radgraph_model_path,
                                   cuda='0')
            indication_fs = radgraph.preprocess_corpus_radgraph_output()
            try:
                indication_core_findings_list = indication_fs[0]['core_findings']
                indication_core_findings = max(indication_core_findings_list, key=len)
                if len(sex) == 0:
                    item['indication_core_findings'] = indication_core_findings
                else:
                    item['indication_core_findings'] = sex + " " + indication_core_findings
                logger.info(f"indication: {indication}, fs: {item['indication_core_findings']}")
            except:
                item['indication_core_findings'] = []
                print(item['id'], indication, 'not core findings!!')
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                logger.info(f"not fs!!! {' '.join(indication_list)}, {item['id']}")
                print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

    with open(
            '/home/miao/data/dataset/MIMIC-CXR/mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_fs_v0227.json',
            'w') as f:
        json.dump(ann_path, f, indent=2)


def extract_indication_factual_serialization_delete_keywords(logger=None):
    ann_path = '/home/miao/data/dataset/MIMIC-CXR/mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_v0227.json'
    ann_data = json.load(open(ann_path))
    new_ann_data = {}
    for split, value in ann_data.items():
        new_ann_data[split] = []
        for item in tqdm(value):
            new_item = copy.deepcopy(item)
            if len(item['indication']) == 0:
                new_item['indication_core_findings'] = ''
            else:
                if item['id'] in ['a5bb1dd6-32ef2b29-b27f45f5-4980a5b0-34f11cf0',
                                  'ae711ffd-03ebb7b3-cc16c95e-e6f64de7-d2bf7de4']:
                    new_item[
                        'indication'] = 'History: ___F with abd pain and pancreatitis, DKA, WBC elevation to ___, PNA? effusion?'
                # years old historical
                indication = re.sub(r'[//?_!*@]+', '', item['indication'].lower())
                indication = indication.replace('--', '')
                indication = re.sub(r'history:|-year-old|year old', '', indication)
                indication = re.sub(r'\bf\b|\bwomen\b|\bfemale\b', 'woman', indication)
                indication = re.sub(r'\bm\b|\bmen\b|\bmale\b', 'man', indication)
                indication = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ', indication)
                indication = re.sub(r'[.,]$', '', indication.strip())
                indication = re.sub(r'(?<=\s)\s+', '', indication)
                indication = indication.replace('.', ',')
                indication = indication.replace(', ,', ',')
                indication = indication.replace('man ,', 'man')
                indication = indication.replace('woman ,', 'woman')
                indication = indication.replace(': :', ':')
                indication = indication.replace(': ( )', '')
                indication = indication.replace('ped struck cxr : ptxarm : fractures',
                                                'ped struck cxr , ptxarm , fractures')

                print(indication.strip(), item['id'])
                # print(item['id'], indication, 'not core findings!!')
                # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                # logger.info(f"{indication.strip()} {item['id']}")
                # print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                new_item['indication_core_findings'] = indication.strip()
            new_ann_data[split].append(new_item)

    with open('mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_with_fs_v0227.json', 'w') as f:
        json.dump(new_ann_data, f, indent=2)


def plot_attention_for_sei_extract_fs():
    candi_image_path = '/home/miao/data/Code/SEI-Results/mimic-cxr/sample_results_for samples with indication.csv'
    _df = pd.read_csv(candi_image_path)
    _df['image_id'] = _df['image_id'].apply(lambda x: x.split('_')[-1])
    candi_image_list = _df['image_id'].tolist()
    # candi_image_list = ['1d1ad085-bc04d368-4062c6ff-8388f25c-c9acb192', 'befa8b27-2bfd96b0-d50f7eda-deffa4f9-dd7e7314',
    #                     '14ff31ea-afb9a3f3-fca0fe57-1fb4e5d4-9f537945']

    pred_df = pd.read_csv('../results/mimic_cxr/finetune/ft_100_top1/test_prediction.csv')
    pred_df = pred_df.loc[12:, ['images_id', 'ground_truth', 'pred_report_23']]
    candi_ann_data, image_id_list = [], []
    for idx, row in pred_df.iterrows():
        image_id, gt, pred = row.tolist()
        image_id = image_id.split('_')[-1]
        if image_id in candi_image_list:
            candi_ann_data.append({
                'image_id': image_id,
                'ground_truth': gt,
                'gen_text': pred
            })
            image_id_list.append(image_id)
    corpus = {item['image_id']: item['gen_text'] for item in candi_ann_data}
    radgraph_path = '/home/miao/data/dataset/checkpoints/radgraph/model.tar.gz'
    radgraph = RadGraphNER(corpus=corpus, is_get_output=True, is_mimic=False, model_path=radgraph_path, cuda='0')
    pred_fs = radgraph.preprocess_corpus_radgraph_output()
    for i, image_id in enumerate(image_id_list):
        gen_fs_list = pred_fs[image_id]['core_findings']
        gen_fs_index_list = pred_fs[image_id]['core_findings_index']
        candi_ann_data[i].update({'gen_fs': gen_fs_list, 'gen_fs_index': gen_fs_index_list})

    candi_ann_df = pd.DataFrame(candi_ann_data)
    root = '/home/miao/data/Code/SEI-Results/mimic-cxr'
    candi_ann_df.to_csv(os.path.join(root, 'test_prediction_with_factual_serialization.csv'), index=False)


def obtain_multiview_indication_mimic_annotation():
    # deprecate
    # merge samples containing multiple images into one sample
    multiview_ann_path = '/home/miao/data/dataset/MIMIC-CXR/mimic_cxr_annotation_sen_best_reports_keywords_20_multi-view_v0223.json'
    indication_ann_path = '/home/miao/data/Code/Third/knowledge_encoder/mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_with_fs_v0227.json'
    multiview_indication_ann_file_name = '/home/miao/data/dataset/MIMIC-CXR/mimic_cxr_annotation_multiview_indication_similar_cases.json'
    id2indication = {}
    indication_ann_data = json.load(open(indication_ann_path))
    for key, value in indication_ann_data.items():
        for item in value:
            curr_id = "_".join([str(item['subject_id']), str(item['study_id'])])
            if curr_id not in id2indication:
                id2indication[curr_id] = {
                    'impression': item['impression'],
                    'comparison': item['comparison'],
                    'indication': item['indication'],
                    'examination': item['examination'],
                    'indication_core_findings': item['indication_core_findings']
                }
    del indication_ann_data, indication_ann_path
    print("id2indication is finish!")

    # obtain indication, impression,... information.
    multiview_idc_shc_ann_data = {}
    multiview_ann_data = json.load(open(multiview_ann_path))
    for key, value in multiview_ann_data.items():
        multiview_idc_shc_ann_data[key] = []
        for item in tqdm(value):
            new_item = copy.deepcopy(item)
            curr_id = item['id']
            new_item.update(id2indication[curr_id])
            multiview_idc_shc_ann_data[key].append(new_item)

    with open(multiview_indication_ann_file_name, 'w') as outfile:
        json.dump(multiview_idc_shc_ann_data, outfile, indent=2)


def create_multiview_individual_annotation():
    # adding multiview image_paths for each sample
    shc_ann_path = '/home/miao/data/Code/Third/knowledge_encoder/mimic_cxr_annotation_sen_best_reports_keywords_20_all_components_with_fs_v0227.json'
    ori_multiview_ann_path = '/home/miao/data/dataset/MIMIC-CXR/mimic_cxr_annotation_sen_best_reports_keywords_20_multi-view_v0223.json'
    save_ann_path = '/home/miao/data/dataset/MIMIC-CXR/mimic_cxr_annotation_sen_best_reports_keywords_20_multiview_individual_v0331.json'

    # load similar historical cases (SHC) ann_data
    multiview_data = json.load(open(ori_multiview_ann_path))
    id2image_path = {}
    for split, value in multiview_data.items():
        id2image_path[split] = {}
        for item in value:
            cur_id = item['id']  # '{subject_id}_{study_id}'
            id2image_path[split][cur_id] = item['image_path']

    del multiview_data, ori_multiview_ann_path

    # create a new multiview file
    new_ann_data = {}
    shc_ann_data = json.load(open(shc_ann_path))
    for split, value in shc_ann_data.items():
        new_ann_data[split] = []
        for item in tqdm(value):
            new_item = copy.deepcopy(item)
            cur_id = f'{item["subject_id"]}_{item["study_id"]}'  # '{subject_id}_{study_id}'
            cur_image_path = item['image_path'][0]
            multiview_path = copy.deepcopy(id2image_path[split][cur_id])
            multiview_path.remove(cur_image_path)
            new_item['multiview_image_path'] = multiview_path
            new_ann_data[split].append(new_item)
    del shc_ann_data
    with open(save_ann_path, 'w') as f:
        json.dump(new_ann_data, f, indent=2)


def main_extract_factual_serialization_iu_xray():
    radgraph_model_path = '/home/miao/data/dataset/checkpoints/radgraph/model.tar.gz'
    ann_path = '/home/miao/data/dataset/iu_xray/iu_xray_annotation_v0328_partial.json'
    sen_ann_path = '/home/miao/data/dataset/iu_xray/iu_xray_annotation_v0328_partial_fs.json'
    radgraph = RadGraphNER(ann_path=ann_path, is_get_output=True, data_name='iu_xray', model_path=radgraph_model_path)
    factual_serialization = radgraph.preprocess_iu_xray_radgraph_output()
    get_iu_xray_annotations(ann_path, factual_serialization, sen_ann_path)


def main_extract_factual_serialization_mimic_abn():
    # obtain the MIMIC-ABN annotation.json file
    ## 1. remove unrelated information (i.e., only retain abnormal sentences of the findings section)
    ## 2. extract factual serialization for reports
    ## 3. construct annotation.json file for multi-to-multi paradigm

    # ===================1. remove unrelated information ==================================
    ori_ann_path = '/home/miao/data/dataset/MIMIC-CXR/MIMIC-ABN/mimic_abn_report.json'
    mimic_ann_path = '/home/miao/data/dataset/MIMIC-CXR/mimic_cxr_annotation_sen_best_reports_keywords_20_multiview_individual_v0331.json'
    save_ann_path = '/home/miao/data/dataset/MIMIC-CXR/MIMIC-ABN/mimic_abn_report_multi2multi_v0331.json'

    # a. obtain all information for each sample
    mimic_ann_data = json.load(open(mimic_ann_path))
    dicom2item = {}
    for split, value in mimic_ann_data.items():
        for item in tqdm(value):
            new_item = copy.deepcopy(item)
            new_item['specific_knowledge'] = []
            new_item['core_findings'] = []
            new_item['split'] = split
            dicom2item[item['id']] = new_item
    del mimic_ann_data
    # b. preprocessing the mimic-abn report (which mixes the findings, impression (del), and indications (del))
    new_ann_data = {'train': [], 'val': [], 'test': []}
    abn_ann_data = json.load(open(ori_ann_path))
    for dicom_id, value in abn_ann_data.items():
        # note that we solely consider the findings  section
        try:
            item = dicom2item[dicom_id]
        except:
            print('not find!', dicom_id)
            continue
        split = item['split']
        del item['split']
        # preprocessing
        findings = item['report']
        findings = remove_extra_spaces(findings)
        findings = remove_duplicate_punctuation(findings)
        findings = findings.strip().lower()

        if len(findings) == 0:
            print("findings null!", dicom_id)
            continue
        # 分割句子
        sentences_findings = split_sentences(findings)
        sentences_abn = split_sentences(value)

        # 遍历a中的句子并与b中的句子进行比较
        valid_sen = []
        for sent_a in sentences_findings:
            for sent_b in sentences_abn:
                if overlap_ratio(sent_a, sent_b) > 0.5:
                    valid_sen.append(sent_a)
                    break
        new_findings = ' . '.join(valid_sen) + ' .'
        # assert len(item['image_path']) == 2
        item['report'] = new_findings

        new_ann_data[split].append(item)
    del abn_ann_data
    # ===================2. extract factual serialization ==================================
    radgraph_model_path = '/home/miao/data/dataset/checkpoints/radgraph/model.tar.gz'
    # ann_path = '/home/miao/data/dataset/MIMIC-CXR/MIMIC-ABN/mimic_abn_report_multiview_without_factual_serialization.json'
    # save_ann_path = '/home/miao/data/dataset/MIMIC-CXR/MIMIC-ABN/mimic_abn_report_multiview_v0331.json'
    radgraph = RadGraphNER(ann_path=new_ann_data, is_get_output=True, data_name='mimic_abn',
                           model_path=radgraph_model_path)
    factual_serialization = radgraph.preprocess_mimic_radgraph_output()

    # ===================3.construct annotation.json file for multi-to-multi paradigm ==================================

    new_abn_ann_data = {}
    for split, value in new_ann_data.items():
        print(f"current preprocessing the {split}....")
        new_abn_ann_data[split] = []
        for item in tqdm(value):
            try:
                doc_key = str(item['subject_id']) + '_' + str(item['study_id'])
                sample_core_finding = factual_serialization[doc_key]
                core_findings = sample_core_finding['core_findings']
                report = sample_core_finding['text']
            except:
                core_findings = []
                report = re.sub(r'(?<!\d)(?=[/,;,:,.,!?()])|(?<=[/,;,:,.,!?()])(?!\d)|\n', r' ',
                                item['findings'])
            item['core_findings'] = core_findings
            item['report'] = report
            new_abn_ann_data[split].append(item)

    with open(save_ann_path, 'w') as f:
        json.dump(new_abn_ann_data, f, indent=2)

    # get_mimic_abn_annotations(ann_path, factual_serialization, save_ann_path)


if __name__ == '__main__':
    # obtain_multiview_indication_mimic_annotation()
    # create_multiview_individual_annotation()
    # main_extract_factual_serialization_iu_xray()
    main_extract_factual_serialization_mimic_abn()

    # radgraph  from official checkpoint
    # radgraph_model_path = '/home/miao/data/dataset/checkpoints/radgraph/model.tar.gz'

    # root = '/media/miao/data/Dataset/MIMIC-CXR'
    # ann_path = os.path.join(root, 'annotation.json')
    # sen_ann_path = 'mimic_cxr_annotation_sen.json'
    # # extract mimic-cxr factual serialization
    # radgraph = RadGraphNER(ann_path=ann_path, is_get_output=True, is_mimic=True, model_path=radgraph_model_path, cuda=1)
    # factual_serialization = radgraph.preprocess_mimic_radgraph_output()
    # get_mimic_cxr_annotations(ann_path, factual_serialization, sen_ann_path)

    # extract item_report factual serialization
    # hyps = ["patient is status post median sternotomy and cabg . the lungs are clear without focal consolidation . no pleural effusion or pneumothorax is seen . the cardiac and mediastinal silhouettes are unremarkable . no pulmonary edema is seen .",
    #         "___ year old woman with cirrhosis.",
    # ]
    # # note that too short reports cannot be extracted due to the limitation of radgraph
    # hyps = [
    #     "___ year old woman with cirrhosis . ___ year old woman with cirrhosis . ___ year old woman with cirrhosis. ___ year old woman with cirrhosis . ___ year old woman with cirrhosis .",
    #     "___F with new onset ascites // eval for infection . ___F with new onset ascites // eval for infection .",
    #     ]
    # corpus = {i: item for i, item in enumerate(hyps)}
    #
    # # radgraph
    # radgraph = RadGraphNER(corpus=corpus, is_get_output=True, is_mimic=False, model_path=radgraph_model_path, cuda='0')
    # factual_serialization = radgraph.preprocess_corpus_radgraph_output()
    # print(factual_serialization)
    # logger = SetLogger(f'extact_indication_fs.log', 'a')

    # extract_indication_factual_serialization(radgraph_model_path, logger)
    # extract_indication_factual_serialization_delete_keywords(logger)

    # for SEI plot attention for factual serialization
    # extract factual serialization for predications
    # plot_attention_for_sei_extract_fs()
    # get_plot_cases_factual_serialization()
