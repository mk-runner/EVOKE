import copy
import re
import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


class PretrainDataset(Dataset):  # finetune and inference phase
    def __init__(self, args, tokenizer, split, transform=None):
        self.transform = transform
        self.tokenizer = tokenizer
        # self.image_dir = args['image_dir']
        ann = json.loads(open(args['ann_path'], 'r').read())
        ann = ann[split]
        self.examples = []
        self.max_seq_length = args['max_seq_len']
        if args['align_type'] == 'keywords':
            for i in range(len(ann)):
                if len(ann[i]['findings_factual_serialization']) == 0:
                    continue   # only include the multiview samples
                core_findings = copy.deepcopy(ann[i]['findings_factual_serialization'])
                if args['tokenizer_type'] == 'uncased':
                    core_findings = list(map(lambda x: str(x).lower(), core_findings))
                core_findings = ' [SEP] '.join(core_findings)
                self.examples.append({
                    'image_path': ann[i]['image_path'],  # list
                    'view_position': ann[i]['view_position'],   # list
                    'radgraph': core_findings,    # str
                    'id': ann[i]['id']
                })
        else:   # report
            for i in range(len(ann)):
                if len(ann[i]['findings_factual_serialization']) == 0:
                    continue
                core_findings = ann[i]['findings']
                if args['tokenizer_type'] == 'uncased':
                    core_findings = core_findings.lower()
                # core_findings = sent_tokenize(core_findings)
                item_id = ann[i]['id']
                self.examples.append({
                    'image_path': ann[i]['image_path'],        # list
                    'view_position': ann[i]['view_position'],  # list
                    'radgraph': core_findings,  # str
                    'id': item_id
                })
        # self.examples = self.examples[:100]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        # note that report uses [BOS] and [EOS] tokens
        radgraph_ids = self.tokenizer.encode('[CLS] ' + example['radgraph']).ids[: self.max_seq_length]
        radgraph_len = len(radgraph_ids)
        radgraph_masks = [1] * radgraph_len

        sample = (image_id, example['image_path'], example['view_position'],
                  radgraph_ids, radgraph_masks, radgraph_len)
        return sample


class FinetuneHasIndication(Dataset):  # finetune and inference phase
    def __init__(self, args, tokenizer, split, transform=None):
        self.transform = transform
        self.tokenizer = tokenizer
        # self.image_dir = args['image_dir']
        ann = json.loads(open(args['ann_path'], 'r').read())
        ann = ann[split]
        self.examples = []
        self.max_seq_length = args['max_seq_len']

        for i in range(len(ann)):
            if len(ann[i]['findings_factual_serialization']) == 0 or len(ann[i]['indication_pure']) == 0:
                continue   # only include the multiview samples
            report = copy.deepcopy(ann[i]['findings'])
            if args['tokenizer_type'] == 'uncased':
                report = report.lower()
            self.examples.append({
                "report": report.strip(),
                'image_path': ann[i]['image_path'],  # list
                'view_position': ann[i]['view_position'],   # list
                "indication": ann[i]['indication_pure'].strip().lower(),
                'id': ann[i]['id']
            })
        # self.examples = self.examples[:100]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']

        # report
        report = example['report']
        report_ids = self.tokenizer.encode('[BOS] ' + report + " [EOS]").ids[: self.max_seq_length]
        report_len = len(report_ids)
        report_masks = [1] * report_len

        # indication section
        inc_ids = self.tokenizer.encode('[CLS] ' + example['indication']).ids[: self.max_seq_length]
        inc_len = len(inc_ids)
        inc_masks = [1] * inc_len

        sample = (image_id, example['image_path'], example['view_position'],
                  report_ids, report_masks, report_len, inc_ids, inc_masks, inc_len)
        return sample


class FinetuneNotIndication(Dataset):  # finetune and inference phase
    def __init__(self, args, tokenizer, split, transform=None):
        self.transform = transform
        self.tokenizer = tokenizer
        # self.image_dir = args['image_dir']
        ann = json.loads(open(args['ann_path'], 'r').read())
        ann = ann[split]
        self.examples = []
        self.max_seq_length = args['max_seq_len']

        for i in range(len(ann)):
            if args['is_add_indication']:
                if len(ann[i]['findings_factual_serialization']) == 0 or len(ann[i]['indication_pure']) != 0:
                    continue
            else:
                if len(ann[i]['findings_factual_serialization']) == 0:
                    continue
            report = copy.deepcopy(ann[i]['findings'])
            if args['tokenizer_type'] == 'uncased':
                report = report.lower()
            item_id = ann[i]['id']
            self.examples.append({
                "report": report.strip(),
                'image_path': ann[i]['image_path'],  # list
                'view_position': ann[i]['view_position'],   # list
                'id': item_id
            })
        # self.examples = self.examples[:100]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']

        # report
        report = example['report']
        report_ids = self.tokenizer.encode('[BOS] ' + report + " [EOS]").ids[: self.max_seq_length]
        report_len = len(report_ids)
        report_masks = [1] * report_len

        sample = (image_id, example['image_path'], example['view_position'],
                  report_ids, report_masks, report_len)
        return sample


class PretrainInferenceBaseDataset(Dataset):  # finetune and inference phase
    def __init__(self, args, split, transform=None):
        self.image_dir = args['image_dir']
        self.transform = transform
        ann = json.loads(open(args['ann_path'], 'r').read())
        ann = ann[split]
        self.examples = []
        for i in range(len(ann)):
            if len(ann[i]['core_findings']) == 0:
                continue
            if args['data_name'] == 'mimic_cxr':
                item_id = '_'.join([str(ann[i]['subject_id']), str(ann[i]['study_id']), ann[i]['id']])
            else:
                item_id = ann[i]['id']
            self.examples.append({
                'image_path': ann[i]['image_path'],
                'id': item_id
            })

        # self.examples = self.examples[:100]

    def __len__(self):
        return len(self.examples)


class MimiccxrPretrainInferenceDataset(PretrainInferenceBaseDataset):  # finetune and inference phase
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        sample = (image_id, image)
        return sample


class IuxrayPretrainInferenceDataset(PretrainInferenceBaseDataset):  # finetune and inference phase
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        sample = (image_id, image)
        return sample


class MixSingleImageDataset(Dataset):  # pretrain phase
    def __init__(self, args, tokenizer, split, transform=None):
        # default dataset is mixture
        self.image_dir = {'iu_xray': args['iu_image_dir'],
                          'mimic_cxr': args['mimic_image_dir']}
        ann_path = {'iu_xray': args['iu_ann_path'],
                    'mimic_cxr': args['mimic_ann_path']}
        all_ann = {'iu_xray': json.loads(open(ann_path['iu_xray'], 'r').read()),
                   'mimic_cxr': json.loads(open(ann_path['mimic_cxr'], 'r').read())}
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_seq_length = args['max_seq_len']

        self.examples = []
        if args['pretrain_data_name'] in ['iu_xray', 'mix']:
            ann = all_ann['iu_xray'][split]
            for i in range(len(ann)):
                report = re.sub('Frontal and lateral views of the chest were obtained. ', '', ann[i]['report'])
                if len(report) < 4:
                    # print(f"drop this sample:{ann[i]['id']}, report: {report}")
                    continue
                self.examples.append({
                    "report": report.strip(),
                    'image_path': ann[i]['image_path'],
                    'id': ann[i]['id']
                })
        if args['pretrain_data_name'] in ['mimic_cxr', 'mix']:
            ann = all_ann['mimic_cxr'][split]
            for i in range(len(ann)):
                report = re.sub('Frontal and lateral views of the chest were obtained. ', '', ann[i]['report'])
                if len(report) < 4:
                    # print(f"drop this sample:{ann[i]['id']}, report: {report}")
                    continue
                self.examples.append({
                    "report": report.strip(),
                    'image_path': ann[i]['image_path'],
                    'id': ann[i]['id']
                })
        # self.examples = self.examples[:100]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        report = example['report']
        report_ids = self.tokenizer.encode('[CLS]' + report + "[SEP]").ids
        report_length = len(report_ids)
        report_masks = [1] * report_length

        trunc_ids = report_ids[:self.max_seq_length]
        trunc_length = len(trunc_ids)
        trunc_masks = [1] * trunc_length
        # obtain the image
        if len(image_path) > 1:
            if torch.rand(1) > 0.5:
                image = Image.open(os.path.join(self.image_dir['iu_xray'], image_path[0])).convert('RGB')
            else:
                image = Image.open(os.path.join(self.image_dir['iu_xray'], image_path[1])).convert('RGB')
        else:
            image = Image.open(os.path.join(self.image_dir['mimic_cxr'], image_path[0])).convert('RGB')
        # preprocessing the image
        if self.transform is not None:
            image = self.transform(image)
        sample = (image_id, image, report_ids, report_masks, trunc_ids, trunc_masks, report_length, trunc_length)
        return sample


class MimiccxrPretrainInferenceDatasetOne(Dataset):  # finetune and inference phase
    def __init__(self, args, split, transform=None):
        self.image_dir = args['mimic_cxr_image_dir']
        self.transform = transform
        ann = json.loads(open(args['mimic_cxr_ann_path'], 'r').read())
        ann = ann[split]
        self.examples = []
        for i in range(len(ann)):
            if len(ann[i]['core_findings']) == 0:
                continue

            item_id = '_'.join([str(ann[i]['subject_id']), str(ann[i]['study_id']), ann[i]['id']])
            self.examples.append({
                'image_path': ann[i]['image_path'],
                'id': item_id
            })

        # self.examples = self.examples[:200]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']

        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        sample = (image_id, image)
        return sample
