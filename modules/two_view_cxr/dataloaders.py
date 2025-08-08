import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import PretrainDataset, FinetuneHasIndication, FinetuneNotIndication


class PretrainLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, drop_last=None):
        self.batch_size = args['batch_size']
        self.shuffle = shuffle
        self.num_workers = args['num_workers']
        self.image_dir = args['image_dir']
        self.split = split

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

        self.dataset = PretrainDataset(args, tokenizer, self.split, transform=None)

        if drop_last is None:
            drop_last = False
            if len(self.dataset) % args['batch_size'] == 1 and split == 'train':
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
        image_ids, batch_images_list, batch_view_position, radgraph_ids, radgraph_masks, radgraph_lens = zip(*data)
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
        # images, patient_ids, patient_info = [], [], []
        anchor_images, other_images, anchor_patient_ids, other_patient_ids = [], [], [], []
        # 1. preprocess batch images
        anchor_view_position = ['AP', 'PA', 'PAO', 'LAO']
        for i, (ids, view_position, images_path) in enumerate(zip(image_ids, batch_view_position, batch_images_list)):
            # select anchor image
            if any(vp in view_position for vp in anchor_view_position):
                anchor_idx = [k for k, vp in enumerate(view_position) if vp in anchor_view_position]
                # random select one image
                rand_idx = np.random.randint(0, len(anchor_idx))
                rand_idx = anchor_idx[rand_idx]
            else:
                # not select lateral as much as possible
                # IU X-ray dataset
                if np.all(np.array(view_position) == 'unk'):
                    rand_idx = 0
                else:
                    # delete LATERAL and LL
                    candi_idx = [k for k, vp in enumerate(view_position) if vp not in ['LATERAL', 'LL']]
                    if len(candi_idx) == 0:
                        rand_idx = np.random.randint(0, len(view_position))
                    else:
                        rand_idx = np.random.randint(0, len(candi_idx))
                        rand_idx = candi_idx[rand_idx]
            for j, image_path in enumerate(images_path):
                image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
                if self.transform is not None:
                    image = self.transform(image)
                # select anchor image or other images
                if j != rand_idx:
                    other_images.append(image)
                    other_patient_ids.append(ids)
                else:
                    anchor_images.append(image)
                    anchor_patient_ids.append(ids)

        anchor_images.extend(other_images)
        anchor_patient_ids.extend(other_patient_ids)
        images = torch.stack(anchor_images)
        patient_ids = np.array(anchor_patient_ids)
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

        self.dataset = FinetuneHasIndication(args, tokenizer, split, transform=None)
        if drop_last is None:
            drop_last = False
            if len(self.dataset) % args['batch_size'] == 1 and split == 'train':
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
        image_ids, batch_images_list, batch_view_position, report_ids, report_masks, report_lens, inc_ids, inc_masks, inc_lens = zip(*data)

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
        anchor_images, other_images, anchor_patient_ids, other_patient_ids = [], [], [], []
        # 1. preprocess batch images
        anchor_view_position = ['AP', 'PA', 'PAO', 'LAO']
        for i, (ids, view_position, images_path) in enumerate(zip(image_ids, batch_view_position, batch_images_list)):
            # select anchor image
            if any(vp in view_position for vp in anchor_view_position):
                anchor_idx = [k for k, vp in enumerate(view_position) if vp in anchor_view_position]
                # random select one image
                rand_idx = np.random.randint(0, len(anchor_idx))
                rand_idx = anchor_idx[rand_idx]
            else:
                # not select lateral as much as possible
                # IU X-ray dataset
                if np.all(np.array(view_position) == 'unk'):
                    rand_idx = 0
                else:
                    # delete LATERAL and LL
                    candi_idx = [k for k, vp in enumerate(view_position) if vp not in ['LATERAL', 'LL']]
                    if len(candi_idx) == 0:
                        rand_idx = np.random.randint(0, len(view_position))
                    else:
                        rand_idx = np.random.randint(0, len(candi_idx))
                        rand_idx = candi_idx[rand_idx]
            for j, image_path in enumerate(images_path):
                image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
                if self.transform is not None:
                    image = self.transform(image)
                # select anchor image or other images
                if j != rand_idx:
                    other_images.append(image)
                    other_patient_ids.append(ids)
                else:
                    anchor_images.append(image)
                    anchor_patient_ids.append(ids)

        anchor_images.extend(other_images)
        anchor_patient_ids.extend(other_patient_ids)
        images = torch.stack(anchor_images)
        patient_ids = np.array(anchor_patient_ids)
        sample = [image_ids, images, input_ids, attention_masks, patient_ids, inc_ids, inc_masks]
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
        self.dataset = FinetuneNotIndication(args, tokenizer, split, transform=None)
        if drop_last is None:
            drop_last = False
            if len(self.dataset) % args['batch_size'] == 1 and split == 'train':
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
        image_ids, batch_images_list, batch_view_position, report_ids, report_masks, report_lens = zip(*data)

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
        anchor_images, other_images, anchor_patient_ids, other_patient_ids = [], [], [], []
        # 1. preprocess batch images
        anchor_view_position = ['AP', 'PA', 'PAO', 'LAO']
        for i, (ids, view_position, images_path) in enumerate(zip(image_ids, batch_view_position, batch_images_list)):
            # select anchor image
            if any(vp in view_position for vp in anchor_view_position):
                anchor_idx = [k for k, vp in enumerate(view_position) if vp in anchor_view_position]
                # random select one image
                rand_idx = np.random.randint(0, len(anchor_idx))
                rand_idx = anchor_idx[rand_idx]
            else:
                # not select lateral as much as possible
                # IU X-ray dataset
                if np.all(np.array(view_position) == 'unk'):
                    rand_idx = 0
                else:
                    # delete LATERAL and LL
                    candi_idx = [k for k, vp in enumerate(view_position) if vp not in ['LATERAL', 'LL']]
                    if len(candi_idx) == 0:
                        rand_idx = np.random.randint(0, len(view_position))
                    else:
                        rand_idx = np.random.randint(0, len(candi_idx))
                        rand_idx = candi_idx[rand_idx]
            for j, image_path in enumerate(images_path):
                image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
                if self.transform is not None:
                    image = self.transform(image)
                # select anchor image or other images
                if j != rand_idx:
                    other_images.append(image)
                    other_patient_ids.append(ids)
                else:
                    anchor_images.append(image)
                    anchor_patient_ids.append(ids)

        anchor_images.extend(other_images)
        anchor_patient_ids.extend(other_patient_ids)
        images = torch.stack(anchor_images)
        patient_ids = np.array(anchor_patient_ids)
        sample = [image_ids, images, input_ids, attention_masks, patient_ids]
        return sample

