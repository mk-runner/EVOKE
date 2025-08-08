import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets_m2m import IuxrayFinetuneDatasetNotIndication, MimiccxrFinetuneDatasetNotIndication
from .datasets_m2m import IuxrayFinetuneDatasetHasIndication, MimiccxrFinetuneDatasetHasIndication
from .datasets_m2m import IuxrayPretrainInferenceDataset, MimiccxrPretrainInferenceDataset
from .datasets_m2m import MimiccxrPretrainInferenceDatasetOne
from .datasets_m2m import MimiccxrPretrainDataset, IuxrayPretrainDataset


class PretrainLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, drop_last=None):
        self.batch_size = args['batch_size']
        self.shuffle = shuffle
        self.num_workers = args['num_workers']
        self.image_dir = args['image_dir']
        self.split = split
        self.is_multiview_learning = args['is_multiview_learning']

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        if args['data_name'] == 'iu_xray':
            self.dataset = IuxrayPretrainDataset(args, tokenizer, self.split, transform=None)
        elif args['data_name'] == 'mimic_cxr':
            self.dataset = MimiccxrPretrainDataset(args, tokenizer, self.split, transform=None)
        else:
            raise ValueError

        if drop_last is None:
            drop_last = False
            if len(self.dataset) % args['batch_size'] == 1:
                drop_last = True

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'drop_last': drop_last
        }
        super().__init__(**self.init_kwargs)

    def collate_fn(self, data):
        # note that images is tuple(list, list)
        image_ids, batch_images_list, multiview_images_list, radgraph_ids, radgraph_masks, radgraph_lens = zip(*data)
        # radgraph
        radgraph_max_len = max(radgraph_lens)
        radgraph_input_ids = np.zeros((len(radgraph_ids), radgraph_max_len), dtype=int)
        radgraph_attention_masks = np.zeros((len(radgraph_ids), radgraph_max_len), dtype=int)

        for i in range(len(radgraph_ids)):
            radgraph_input_ids[i, :len(radgraph_ids[i])] = radgraph_ids[i]
            radgraph_attention_masks[i, :len(radgraph_masks[i])] = radgraph_masks[i]

        radgraph_input_ids = torch.LongTensor(radgraph_input_ids)
        radgraph_attention_masks = torch.LongTensor(radgraph_attention_masks)

        # images
        images, patient_ids, patient_info = [], [], []
        # 1. preprocess batch images
        for image_id, image_path in zip(image_ids, batch_images_list):
            # add patient id and information
            cur_patient_id = image_id
            cur_patient_info = image_path
            patient_info.append(cur_patient_info)
            patient_ids.append(cur_patient_id)

            # add image info
            image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        # 2. preprocess multiview images
        if self.is_multiview_learning:
            for image_id, multiview_images in zip(image_ids, multiview_images_list):
                for mv_image in multiview_images:
                    # mv_image_path_split = mv_image.split('/')
                    # # whether add image_list or not, avoid redundancy
                    # assert len(mv_image_path_split) == 4

                    cur_patient_info = mv_image
                    if cur_patient_info not in patient_info:
                        # add patient id and information
                        cur_patient_id = image_id
                        patient_info.append(cur_patient_info)
                        patient_ids.append(cur_patient_id)
                        # add image info
                        image = Image.open(os.path.join(self.image_dir, mv_image)).convert('RGB')
                        if self.transform is not None:
                            image = self.transform(image)
                        images.append(image)

        images = torch.stack(images)
        patient_ids = np.array(patient_ids)
        return image_ids, images, radgraph_input_ids, radgraph_attention_masks, patient_ids


class FinetuneLoaderHasIndication(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, drop_last=None):
        self.batch_size = args['batch_size']
        self.shuffle = shuffle
        self.num_workers = args['num_workers']
        self.image_dir = args['image_dir']
        self.split = split
        self.is_multiview_learning = args['is_multiview_learning']

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        if args['data_name'] == 'iu_xray':
            self.dataset = IuxrayFinetuneDatasetHasIndication(args, tokenizer, self.split, transform=None)
        elif args['data_name'] == 'mimic_cxr':
            self.dataset = MimiccxrFinetuneDatasetHasIndication(args, tokenizer, self.split, transform=None)
        else:
            raise ValueError

        if drop_last is None:
            drop_last = False
            if len(self.dataset) % args['batch_size'] == 1:
                drop_last = True

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'drop_last': drop_last
        }
        super().__init__(**self.init_kwargs)

    def collate_fn(self, data):
        # note that images is tuple(list, list)
        image_ids, batch_images_list, multiview_images_list, report_ids, report_masks, report_lens, inc_ids, inc_masks, inc_lens = zip(*data)

        batch_size = len(report_ids)
        # report
        report_max_len = max(report_lens)
        report_input_ids = np.zeros((batch_size, report_max_len), dtype=int)
        report_attention_masks = np.zeros((batch_size, report_max_len), dtype=int)

        for i in range(batch_size):
            report_input_ids[i, :len(report_ids[i])] = report_ids[i]
            report_attention_masks[i, :len(report_masks[i])] = report_masks[i]

        input_ids = torch.LongTensor(report_input_ids)
        attention_masks = torch.LongTensor(report_attention_masks)

        # indication
        inc_max_len = max(inc_lens)
        inc_input_ids = np.zeros((batch_size, inc_max_len), dtype=int)
        inc_attention_masks = np.zeros((batch_size, inc_max_len), dtype=int)

        for i in range(batch_size):
            inc_input_ids[i, :len(inc_ids[i])] = inc_ids[i]
            inc_attention_masks[i, :len(inc_masks[i])] = inc_masks[i]

        inc_ids = torch.LongTensor(inc_input_ids)
        inc_masks = torch.LongTensor(inc_attention_masks)

        # images
        images, patient_ids, patient_info = [], [], []
        # 1. preprocess batch images
        for image_id, image_path in zip(image_ids, batch_images_list):
            # image_path_split = image_path.split('/')
            # assert len(image_path_split) == 4

            # add patient id and information
            cur_patient_id = image_id
            cur_patient_info = image_path
            patient_info.append(cur_patient_info)
            patient_ids.append(cur_patient_id)

            # add image info
            image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        # 2. preprocess multiview images
        if self.is_multiview_learning:
            for image_id, multiview_images in zip(image_ids, multiview_images_list):
                for mv_image in multiview_images:
                    # mv_image_path_split = mv_image.split('/')
                    # # whether add image_list or not, avoid redundancy
                    # assert len(mv_image_path_split) == 4

                    cur_patient_info = mv_image
                    if cur_patient_info not in patient_info:
                        # add patient id and information
                        cur_patient_id = image_id
                        patient_info.append(cur_patient_info)
                        patient_ids.append(cur_patient_id)
                        # add image info
                        image = Image.open(os.path.join(self.image_dir, mv_image)).convert('RGB')
                        if self.transform is not None:
                            image = self.transform(image)
                        images.append(image)

        images = torch.stack(images)
        patient_ids = np.array(patient_ids)
        sample = [image_ids, images, input_ids, attention_masks, patient_ids, inc_ids, inc_masks, ]
        return sample


class FinetuneLoaderNotIndication(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, drop_last=None):
        self.batch_size = args['batch_size']
        self.shuffle = shuffle
        self.num_workers = args['num_workers']
        self.image_dir = args['image_dir']
        self.split = split
        self.is_multiview_learning = args['is_multiview_learning']

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        if args['data_name'] == 'iu_xray':
            self.dataset = IuxrayFinetuneDatasetNotIndication(args, tokenizer, self.split, transform=None)
        elif args['data_name'] == 'mimic_cxr':
            self.dataset = MimiccxrFinetuneDatasetNotIndication(args, tokenizer, self.split, transform=None)
        else:
            raise ValueError

        if drop_last is None:
            drop_last = False
            if len(self.dataset) % args['batch_size'] == 1:
                drop_last = True

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'drop_last': drop_last
        }
        super().__init__(**self.init_kwargs)

    def collate_fn(self, data):
        # note that images is tuple(list, list)
        image_ids, batch_images_list, multiview_images_list, report_ids, report_masks, report_lens = zip(*data)

        batch_size = len(report_ids)
        # report
        report_max_len = max(report_lens)
        report_input_ids = np.zeros((batch_size, report_max_len), dtype=int)
        report_attention_masks = np.zeros((batch_size, report_max_len), dtype=int)

        for i in range(batch_size):
            report_input_ids[i, :len(report_ids[i])] = report_ids[i]
            report_attention_masks[i, :len(report_masks[i])] = report_masks[i]

        input_ids = torch.LongTensor(report_input_ids)
        attention_masks = torch.LongTensor(report_attention_masks)

        # images
        images, patient_ids, patient_info = [], [], []
        # 1. preprocess batch images
        for image_id, image_path in zip(image_ids, batch_images_list):
            # image_path_split = image_path.split('/')
            # assert len(image_path_split) == 4

            # add patient id and information
            cur_patient_id = image_id
            cur_patient_info = image_path
            patient_info.append(cur_patient_info)
            patient_ids.append(cur_patient_id)

            # add image info
            image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)

        # 2. preprocess multiview images
        if self.is_multiview_learning:
            for image_id, multiview_images in zip(image_ids, multiview_images_list):
                for mv_image in multiview_images:
                    # mv_image_path_split = mv_image.split('/')
                    # # whether add image_list or not, avoid redundancy
                    # assert len(mv_image_path_split) == 4

                    cur_patient_info = mv_image
                    if cur_patient_info not in patient_info:
                        # add patient id and information
                        cur_patient_id = image_id
                        patient_info.append(cur_patient_info)
                        patient_ids.append(cur_patient_id)
                        # add image info
                        image = Image.open(os.path.join(self.image_dir, mv_image)).convert('RGB')
                        if self.transform is not None:
                            image = self.transform(image)
                        images.append(image)

        images = torch.stack(images)
        patient_ids = np.array(patient_ids)
        sample = [image_ids, images, input_ids, attention_masks, patient_ids]
        return sample


class PretrainInferenceLoader(DataLoader):
    def __init__(self, args, split, shuffle=False, drop_last=False):
        self.batch_size = args['batch_size']
        self.shuffle = shuffle
        self.num_workers = args['num_workers']
        self.split = split

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        if args['data_name'] == 'iu_xray':
            self.dataset = IuxrayPretrainInferenceDataset(args, split, transform=self.transform)
        elif args['data_name'] == 'mimic_cxr':
            self.dataset = MimiccxrPretrainInferenceDataset(args, split, transform=self.transform)
        else:
            raise ValueError

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'drop_last': drop_last
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        image_ids, images = zip(*data)
        images = torch.stack(images, 0)

        return image_ids, images


class PretrainInferenceLoaderMIMICOne(DataLoader):
    """
    only used for IU-xray to retrieve similar historical images
    """
    def __init__(self, args, split, shuffle=False, drop_last=False):
        self.batch_size = args['batch_size']
        self.shuffle = shuffle
        self.num_workers = args['num_workers']
        self.split = split

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
        self.dataset = MimiccxrPretrainInferenceDatasetOne(args, split, transform=self.transform)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'drop_last': drop_last
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        image_ids, images = zip(*data)
        images = torch.stack(images, 0)
        return image_ids, images
