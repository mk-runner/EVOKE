import os
import copy
import json
import re

import pandas as pd
import numpy as np
from tqdm import tqdm


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


def remove_trailing_punctuation(text):
    # 匹配字符串结尾的一个或多个标点符号
    return re.sub(r'[.,;!?]+$', '', text)


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


def load_mimic_data(mimic_path):
    mimic_data = json.load(open(mimic_path))
    return mimic_data


def load_iu_data(iu_path, iu_knowledge_path):
    iu_data = json.load(open(iu_path))
    iu_knowledge_data = json.load(open(iu_knowledge_path))

    # concat information between iu_data and iu_knowledge_data
    # check the number of sample
    # assert len(iu_data) == len(iu_knowledge_data['train']) + len(iu_knowledge_data['val']) + len(iu_knowledge_data['test'])

    return iu_data, iu_knowledge_data


def build_benchmark(mimic_path, mimic_metadata_path, iu_path, iu_knowledge_path, save_ann_path):
    # initialize the multiview data
    multiview_data = {key: [] for key in ['train', 'val', 'test']}

    # obtain the multiview sample for mimic-cxr
    print("extract multiview samples from mimic-cxr")
    mimic_meta_data = pd.read_csv(mimic_metadata_path)
    mimic_meta_data['id'] = mimic_meta_data[['subject_id', 'study_id', 'dicom_id']].apply(
        lambda x: '_'.join(map(str, x.values.tolist())), axis=1)
    mimic_meta_data['ViewPosition'].fillna('unk', inplace=True)
    mimic_meta_data.index = mimic_meta_data.index.astype(str)
    mimic_meta_data.index = mimic_meta_data['id']
    mimic_data = load_mimic_data(mimic_path)
    for split, value in mimic_data.items():
        for item in tqdm(value):
            # delete sample that has no clinical meaning or not has multiview scans
            if len(item['core_findings']) == 0 or len(item['image_path']) < 2:
                continue
            # choose the anchor scan
            subject2study = item['id']
            # view_position_list, pa_list = [], []
            view_position_list = []
            for image_path in item['image_path']:
                dicom_id = image_path.split('/')[-1].split('.jpg')[0]
                cur_id = f'{subject2study}_{dicom_id}'
                view_position = mimic_meta_data.loc[cur_id, 'ViewPosition']
                # if view_position in ['PA', 'AP', 'PA LLD', 'AP AXIAL', 'AP LLD', 'AP RLD', 'PA RLD']:
                #     pa_list.append(view_position)
                view_position_list.append(view_position)
            new_item = {
                'id': item['id'],
                'findings': item['report'],
                'findings_factual_serialization': item['core_findings'],
                'impression': item['impression'],
                'indication': item['indication'],
                'indication_pure': item['indication_core_findings'],
                'image_path': item['image_path'],
                'view_position': view_position_list,
                'comparison': item['comparison'],
                'similar_historical_cases': item['specific_knowledge']
            }
            multiview_data[split].append(new_item)

    del mimic_meta_data, mimic_data

    print('mimic_cxr', len(multiview_data['train']), len(multiview_data['val']), len(multiview_data['test']))
    # obtain the multiview sample for iu_xray
    print("extract multiview samples from iu_xray")
    iu_data, iu_knowledge = load_iu_data(iu_path, iu_knowledge_path)
    for split, value in iu_knowledge.items():
        for item in tqdm(value):
            # delete sample that has no clinical meaning or not has multiview scans
            if len(item['core_findings']) == 0 or len(item['image_path']) < 2:
                continue
            # choose the anchor scan
            cur_id = item['id'].split('_')[0]
            iu_data_item = iu_data[cur_id]
            del iu_data[cur_id]
            view_position_list = ['unk'] * len(iu_data_item['image_path'])
            images_path = [os.path.join('NLMCXR_png', path.split('.jpg')[0] + '.png') for path in
                           iu_data_item['image_path']]
            indication_core_findings = re.sub(r'\s*,\s*,+', '', item['indication_core_findings'])
            new_item = {
                'id': cur_id,
                'findings': item['report'],
                'findings_factual_serialization': item['core_findings'],
                'impression': item['impression'],
                'indication': item['indication'],
                'indication_pure': indication_core_findings,
                'image_path': images_path,
                'view_position': view_position_list,
                'comparison': iu_data_item['comparison'],
                'similar_historical_cases': item['specific_knowledge']
            }
            multiview_data[split].append(new_item)
    # note that iu_data has multiview samples, but they have not findings section
    print("all_results", len(multiview_data['train']), len(multiview_data['val']), len(multiview_data['test']))
    with open(save_ann_path, 'w') as f:
        json.dump(multiview_data, f, indent=2)


def create_multiview_cxr():
    # multi-to-one paradigm
    # our building Multi-view CXR dataset, aimed at exploring multi-view learning
    mimic_path = '/home/miao/data/dataset/MIMIC-CXR/mimic_cxr_annotation_multiview_indication_similar_cases_v0312.json'
    mimic_metadata_path = '/home/miao/data/dataset/MIMIC-CXR/mimic-cxr-2.0.0-metadata.csv'
    iu_path = '/home/miao/data/dataset/iu_xray/iu_xray_annotation_v0328.json'
    iu_knowledge_path = '/home/miao/data/dataset/iu_xray/iu_xray_annotation_sen_best_reports_keywords_20_all_components_with_fs_v0227.json'
    save_ann_path = '/home/miao/data/dataset/MIMIC-CXR/multiview_CXR.json'
    build_benchmark(mimic_path, mimic_metadata_path, iu_path, iu_knowledge_path, save_ann_path)


def create_multiview_cxr_multi_to_multi():
    multiview_cxr = '/home/miao/data/dataset/MIMIC-CXR/multiview_CXR.json'
    save_ann_path = '/home/miao/data/dataset/MIMIC-CXR/multiview_CXR_multi_to_multi.json'
    new_ann_data = {}
    ann_data = json.load(open(multiview_cxr))
    for split, value in ann_data.items():
        new_ann_data[split] = []
        for item in tqdm(value):
            image_paths = item['image_path']
            for image_path in image_paths:
                # item_id = image_path.split('/')[-1].split('.')[0]
                # current_multiview_paths = copy.deepcopy(image_paths)
                # current_multiview_paths.remove(image_path)
                # cur_item = {
                #     'id': item['id'],
                #     'image_path': [image_path],
                #     'core_findings': item['findings_factual_serialization'],
                #     'specific_knowledge': [],
                #     'findings':item['findings'],
                #     'comparison':item['comparison'],
                #     'indication':item['indication'],
                #     'indication_core_findings':item['indication_pure'],
                #     'multiview_image_path':current_multiview_paths,
                # }
                current_multiview_paths = copy.deepcopy(image_paths)
                current_multiview_paths.remove(image_path)
                cur_item = copy.deepcopy(item)
                del cur_item['view_position']
                del cur_item['similar_historical_cases']
                cur_item['multiview_image_path'] = current_multiview_paths
                cur_item['image_path'] = [image_path]
                new_ann_data[split].append(cur_item)

    with open(save_ann_path, 'w') as f:
        json.dump(new_ann_data, f, indent=2)


def create_multiview_individual_annotation_mimic_cxr():
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


def create_multiview_individual_annotation_iu_xray():
    """
    each study has different views/radiographs, used for multi-to-multi paradigm
    """
    # adding multiview image_paths for each sample
    shc_ann_path = '/home/miao/data/dataset/iu_xray/iu_xray_annotation_sen_best_reports_keywords_20_all_components_with_fs_v0227.json'
    ori_multiview_ann_path = '/home/miao/data/dataset/iu_xray/iu_xray_annotation_v0328.json'
    ori_multiview_partial_ann_path = '/home/miao/data/dataset/iu_xray/iu_xray_annotation_v0328_partial_fs.json'
    save_ann_path = '/home/miao/data/dataset/iu_xray/iu_xray_annotation_sen_best_reports_keywords_20_multiview_individual_v0331.json'

    # load complete iu-xray images
    multiview_data = json.load(open(ori_multiview_ann_path))

    # create a new multiview file
    new_ann_data = {}
    # load similar historical cases (SHC) ann_data
    shc_ann_data = json.load(open(shc_ann_path))
    splits = ['train', 'val', 'test']
    for split, value in shc_ann_data.items():
        new_ann_data[split] = []
        for item in tqdm(value):
            cur_id = item['id'].split('_')[0]
            ori_item = multiview_data[cur_id]
            findings = re.sub(r'([.,?!:])', r' \1', ori_item['findings'])
            impression = re.sub(r'([.,?!:])', r' \1', ori_item['impression'])
            indication_core_findings = remove_duplicate_punctuation(ori_item['indication_core_findings'])
            findings = remove_duplicate_punctuation(findings)
            impression = remove_duplicate_punctuation(impression)

            image_path_list = ori_item['image_path']
            for path in image_path_list:
                ori_path_list = copy.deepcopy(image_path_list)
                ori_path_list.remove(path)
                new_item = {
                    'id': path.split('.')[0],
                    'report': findings,
                    'core_findings': item['core_findings'],
                    'findings': findings,
                    'impression': impression,
                    'indication': ori_item['indication'],
                    'indication_core_findings': indication_core_findings,
                    'comparison': ori_item['comparison'],
                    'mesh_major': ori_item['mesh_major'],
                    'image_path': path,
                    'multiview_image_path': ori_path_list,
                    # 'specific_knowledge': item['specific_knowledge']
                    'specific_knowledge': []
                }
                new_ann_data[split].append(new_item)
            del multiview_data[cur_id]

    del shc_ann_data
    # save other cases
    other_ann_data = json.load(open(ori_multiview_partial_ann_path))
    for key, value in other_ann_data.items():
        if len(value['core_findings']) == 0:
            continue
        idx = np.random.randint(0, 3, 1)[0]
        split = splits[idx]

        findings = re.sub(r'([.,?!:])', r' \1', value['findings'])
        impression = re.sub(r'([.,?!:])', r' \1', value['impression'])
        indication_core_findings = remove_duplicate_punctuation(value['indication_core_findings'])
        findings = remove_duplicate_punctuation(findings)
        impression = remove_duplicate_punctuation(impression)

        image_path_list = value['image_path']
        for path in image_path_list:
            ori_path_list = copy.deepcopy(image_path_list)
            ori_path_list.remove(path)
            new_item = {
                'id': path.split('.')[0],
                'report': findings,
                'core_findings': value['core_findings'],
                'findings': findings,
                'impression': impression,
                'indication': value['indication'],
                'indication_core_findings': indication_core_findings,
                'comparison': value['comparison'],
                'mesh_major': value['mesh_major'],
                'image_path': path,
                'multiview_image_path': ori_path_list,
                'specific_knowledge': []
            }
            new_ann_data[split].append(new_item)

    with open(save_ann_path, 'w') as f:
        json.dump(new_ann_data, f, indent=2)


def create_multiview_same_annotation_iu_xray():
    """
    each study has two views/radiographs, used for multi-to-one paradigm
    """
    # adding multiview image_paths for each sample
    shc_ann_path = '/home/miao/data/dataset/iu_xray/iu_xray_annotation_sen_best_reports_keywords_20_all_components_with_fs_v0227.json'
    # ori_multiview_ann_path = '/home/miao/data/dataset/iu_xray/iu_xray_annotation_v0328.json'
    # ori_multiview_partial_ann_path = '/home/miao/data/dataset/iu_xray/iu_xray_annotation_v0328_partial_fs.json'
    save_ann_path = '/home/miao/data/dataset/iu_xray/iu_xray_annotation_sen_best_reports_keywords_20_twoview_v0331.json'

    # create a new multiview file
    new_ann_data = {}
    # load similar historical cases (SHC) ann_data
    shc_ann_data = json.load(open(shc_ann_path))
    splits = ['train', 'val', 'test']
    for split, value in shc_ann_data.items():
        new_ann_data[split] = []
        for item in tqdm(value):
            new_item = copy.deepcopy(item)
            indication_pure = remove_duplicate_punctuation(new_item['indication_core_findings'])
            findings = remove_duplicate_punctuation(new_item['report'])
            impression = remove_duplicate_punctuation(new_item['impression'])
            assert len(new_item['image_path']) == 2

            new_item = {
                'id': item['id'],
                'findings': findings,
                'findings_factual_serialization': item['core_findings'],
                'impression': impression,
                'indication': item['indication'],
                'indication_pure': indication_pure,
                'image_path': item['image_path'],
                # 'view_position': [],
                # 'comparison': [],
                'mesh_major': item['mesh_major'],
                'mesh_automatic': item['mesh_automatic'],
                'similar_historical_cases': [],
            }

            new_ann_data[split].append(new_item)

    with open(save_ann_path, 'w') as f:
        json.dump(new_ann_data, f, indent=2)


# def create_multiview_individual_annotation_mimic_abn_depreate():
#     """
#     drop out
#     each study has multi-view radiographs, used for multi-to-multi paradigm
#     # only the abnormal sentence of the findings section
#     """
#     # adding multiview image_paths for each sample
#     ori_ann_path = '/home/miao/data/dataset/MIMIC-CXR/MIMIC-ABN/mimic_abn_report.json'
#     multiview_ann_path = '/home/miao/data/dataset/MIMIC-CXR/multiview_CXR.json'
#     save_ann_path = '/home/miao/data/dataset/MIMIC-CXR/MIMIC-ABN/mimic_abn_report_multi2multi_v0331.json'
#
#     # obtain all information for each sample
#     mv_ann_data = json.load(open(multiview_ann_path))
#     dicom2id, id2item = {}, {}
#     for split, value in mv_ann_data.items():
#         for item in tqdm(value):
#             idx = item['id']
#             for image_path in item['image_path']:
#                 dicom_id = image_path.split('/')[-1].split('.jpg')[0]
#                 dicom2id[dicom_id] = idx
#             new_item = copy.deepcopy(item)
#             new_item['split'] = split
#             id2item[idx] = new_item
#     del mv_ann_data
#     # preprocessing the mimic-abn report (which mixes the findings and indications)
#     new_ann_data = {'train': [], 'val': [], 'test': []}
#     abn_ann_data = json.load(open(ori_ann_path))
#     item_id_list = []
#     for dicom_id, value in abn_ann_data.items():
#         # note that we solely consider finding  sections
#         try:
#             study2subject = dicom2id[dicom_id]
#             if study2subject not in item_id_list:
#                 item_id_list.append(study2subject)
#             else:
#                 continue
#         except:
#             continue
#         item = id2item[study2subject]
#         split = item['split']
#         findings = item['findings'].strip().lower()
#         if len(findings) == 0:
#             continue
#         # 分割句子
#         sentences_findings = split_sentences(findings)
#         sentences_abn = split_sentences(value)
#
#         # 遍历a中的句子并与b中的句子进行比较
#         valid_sen = []
#         for sent_a in sentences_findings:
#             for sent_b in sentences_abn:
#                 if overlap_ratio(sent_a, sent_b) > 0.5:
#                     valid_sen.append(sent_a)
#                     break
#         new_findings = ' . '.join(valid_sen) + ' .'
#         # assert len(item['image_path']) == 2
#         item['findings'] = new_findings
#
#         new_ann_data[split].append(item)
#
#     with open(save_ann_path, 'w') as f:
#         json.dump(new_ann_data, f, indent=2)


def obtain_mimic_abn_shc_multi2multi():
    mimic_path = '/home/miao/data/dataset/MIMIC-CXR/mimic_cxr_annotation_sen_best_reports_keywords_20_multiview_individual_v0331.json'
    abn_path = '/home/miao/data/dataset/MIMIC-CXR/MIMIC-ABN/mimic_abn_report_multi2multi_v0331.json'
    save_abn_path = '/home/miao/data/dataset/MIMIC-CXR/MIMIC-ABN/mimic_abn_report_multi2multi_v0331_shc.json'
    mimic_data = json.load(open(mimic_path))
    abn_data = json.load(open(abn_path))
    new_ann_data = {}
    for split, value in mimic_data.items():
        # obtain id-to-similar historical cases
        id2shc = {}
        for item in tqdm(value):
            id2shc[item['id']] = item['specific_knowledge']
        # obtain abn-to-similar historical cases
        abn_value = abn_data[split]
        new_ann_data[split] = []
        for item in tqdm(abn_value):
            try:
                shc = id2shc[item['id']]
            except:
                print('not find!', item['id'])
                continue
            item['specific_knowledge'] = shc
            new_ann_data[split].append(item)

    with open(save_abn_path, 'w') as f:
        json.dump(new_ann_data, f, indent=2)


def create_multiview_cxr_shc_multi_to_multi():
    multiview_cxr = '/home/miao/data/dataset/MIMIC-CXR/multiview_CXR.json'
    save_ann_path = '/home/miao/data/dataset/MIMIC-CXR/multiview_CXR_multi_to_multi_shc.json'
    new_ann_data = {}
    ann_data = json.load(open(multiview_cxr))
    for split, value in ann_data.items():
        new_ann_data[split] = []
        for item in tqdm(value):
            image_paths = item['image_path']
            for image_path in image_paths:
                # item_id = image_path.split('/')[-1].split('.')[0]
                current_multiview_paths = copy.deepcopy(image_paths)
                current_multiview_paths.remove(image_path)
                cur_item = {
                    'id': item['id'],
                    'report': item['findings'],
                    'image_path': [image_path],
                    'core_findings': item['findings_factual_serialization'],
                    'specific_knowledge': item['similar_historical_cases'],
                    'impression': item['impression'],
                    'findings':item['findings'],
                    'comparison':item['comparison'],
                    'indication':item['indication'],
                    'indication_core_findings':item['indication_pure'],
                    'multiview_image_path': current_multiview_paths,
                }
                new_ann_data[split].append(cur_item)

    with open(save_ann_path, 'w') as f:
        json.dump(new_ann_data, f, indent=2)


if __name__ == '__main__':
    # create_multiview_cxr()
    # create_multiview_individual_annotation_iu_xray()
    # create_multiview_same_annotation_iu_xray()
    # create_multiview_individual_annotation_mimic_abn()
    # obtain_mimic_abn_shc_multi2multi()
    print("obtain Multi-view CXR")
    create_multiview_cxr_shc_multi_to_multi()
