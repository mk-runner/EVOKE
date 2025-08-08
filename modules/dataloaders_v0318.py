import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets_v0318 import IuxrayPretrainDatasetHasMultiview, MimiccxrPretrainDatasetHasMultiview
from .datasets_v0318 import IuxrayPretrainDatasetNotMultiview, MimiccxrPretrainDatasetNotMultiview
# from .datasets_two_step import IuxrayFinetuneDatasetHasIndication, MimiccxrFinetuneDatasetHasIndication
# from .datasets_two_step import IuxrayFinetuneDatasetNotIndication, MimiccxrFinetuneDatasetNotIndication
from .datasets_v0318 import IuxrayPretrainInferenceDataset, MimiccxrPretrainInferenceDataset
from .datasets_v0318 import MimiccxrPretrainInferenceDatasetOne


class PretrainLoaderHasMultiview(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, drop_last=None):
        self.batch_size = args['batch_size']
        self.shuffle = shuffle
        self.num_workers = args['num_workers']
        self.split = split
        # self.is_multiview_learning = args['is_multiview_learning']

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
            self.dataset = IuxrayPretrainDatasetHasMultiview(args, tokenizer, self.split, transform=self.transform)
        elif args['data_name'] == 'mimic_cxr':
            self.dataset = MimiccxrPretrainDatasetHasMultiview(args, tokenizer, self.split, transform=self.transform)
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
        image_ids, images, aug_images, radgraph_ids, radgraph_masks, radgraph_lens, sen_idx = zip(*data)
        # radgraph
        radgraph_max_len = max(radgraph_lens)
        radgraph_input_ids = np.zeros((len(radgraph_ids), radgraph_max_len), dtype=int)
        radgraph_attention_masks = np.zeros((len(radgraph_ids), radgraph_max_len), dtype=int)

        for i in range(len(radgraph_ids)):
            radgraph_input_ids[i, :len(radgraph_ids[i])] = radgraph_ids[i]
            radgraph_attention_masks[i, :len(radgraph_masks[i])] = radgraph_masks[i]

        radgraph_input_ids = torch.LongTensor(radgraph_input_ids)
        radgraph_attention_masks = torch.LongTensor(radgraph_attention_masks)

        all_images, all_aug_images, sample_image_idx = [], [], []
        for i, (image, aug_image) in enumerate(zip(images, aug_images)):
            all_images.extend(image)
            all_aug_images.extend(aug_image)
            sample_image_idx.extend([i] * len(image))
        images = torch.stack(all_images)
        aug_images = torch.stack(all_aug_images)
        sample_image_idx = torch.FloatTensor(sample_image_idx)
        return image_ids, images, radgraph_input_ids, radgraph_attention_masks, sen_idx, sample_image_idx, aug_images


class PretrainLoaderNotMultiview(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle, drop_last=None):
        self.batch_size = args['batch_size']
        self.shuffle = shuffle
        self.num_workers = args['num_workers']
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
            self.dataset = IuxrayPretrainDatasetNotMultiview(args, tokenizer, self.split, transform=self.transform)
        elif args['data_name'] == 'mimic_cxr':
            self.dataset = MimiccxrPretrainDatasetNotMultiview(args, tokenizer, self.split, transform=self.transform)
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
        image_ids, images, radgraph_ids, radgraph_masks, radgraph_lens, sen_idx = zip(*data)
        images = torch.stack(images, 0)
        # radgraph
        radgraph_max_len = max(radgraph_lens)
        radgraph_input_ids = np.zeros((len(radgraph_ids), radgraph_max_len), dtype=int)
        radgraph_attention_masks = np.zeros((len(radgraph_ids), radgraph_max_len), dtype=int)

        for i in range(len(radgraph_ids)):
            radgraph_input_ids[i, :len(radgraph_ids[i])] = radgraph_ids[i]
            radgraph_attention_masks[i, :len(radgraph_masks[i])] = radgraph_masks[i]

        radgraph_input_ids = torch.LongTensor(radgraph_input_ids)
        radgraph_attention_masks = torch.LongTensor(radgraph_attention_masks)
        return image_ids, images, radgraph_input_ids, radgraph_attention_masks, sen_idx


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


# class FinetuneLoaderNotIndication(DataLoader):
#     def __init__(self, args, tokenizer, split, shuffle, drop_last=False):
#         self.batch_size = args['batch_size']
#         self.shuffle = shuffle
#         self.num_workers = args['num_workers']
#         self.split = split
#         # self.drop_last = drop_last
#
#         if split == 'train':
#             self.transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.RandomCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406),
#                                      (0.229, 0.224, 0.225))])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406),
#                                      (0.229, 0.224, 0.225))])
#
#         if args['data_name'] == 'iu_xray':
#             self.dataset = IuxrayFinetuneDatasetNotIndication(args, tokenizer, self.split, transform=self.transform)
#         elif args['data_name'] == 'mimic_cxr':
#             self.dataset = MimiccxrFinetuneDatasetNotIndication(args, tokenizer, self.split, transform=self.transform)
#         else:
#             raise ValueError
#
#         drop_last = False
#         if len(self.dataset) % self.batch_size == 1:
#             drop_last = True
#
#         self.init_kwargs = {
#             'dataset': self.dataset,
#             'batch_size': self.batch_size,
#             'shuffle': self.shuffle,
#             'collate_fn': self.collate_fn,
#             'num_workers': self.num_workers,
#             "drop_last": drop_last
#         }
#         super().__init__(**self.init_kwargs)
#
#     @staticmethod
#     def collate_fn(data):
#         image_ids, images, report_ids, report_masks, report_lens, sk_ids, sk_masks, sk_lens = zip(*data)
#         images = torch.stack(images, 0)
#         batch_size = len(images)
#
#         # report
#         report_max_len = max(report_lens)
#         report_input_ids = np.zeros((batch_size, report_max_len), dtype=int)
#         report_attention_masks = np.zeros((batch_size, report_max_len), dtype=int)
#
#         for i in range(batch_size):
#             report_input_ids[i, :len(report_ids[i])] = report_ids[i]
#             report_attention_masks[i, :len(report_masks[i])] = report_masks[i]
#
#         report_input_ids = torch.LongTensor(report_input_ids)
#         report_attention_masks = torch.LongTensor(report_attention_masks)
#
#         # specific knowledge
#         sk_input_ids, sk_attention_masks = [], []
#         if all(len(item) == 0 for item in sk_lens):
#             pass
#         else:
#             specific_knowledge_max_len = np.array(sk_lens).max(0)
#             sk_max_len = np.max(specific_knowledge_max_len)
#             # specific knowledge topk
#             for k, s_len in enumerate(specific_knowledge_max_len):
#                 s_input_ids = np.zeros((batch_size, sk_max_len), dtype=int)
#                 s_attention_masks = np.zeros((batch_size, sk_max_len), dtype=int)
#                 for i in range(batch_size):
#                     cur_ids, cur_masks = sk_ids[i][k], sk_masks[i][k]
#                     s_input_ids[i, :len(cur_ids)] = cur_ids
#                     s_attention_masks[i, :len(cur_masks)] = cur_masks
#                 s_input_ids = torch.LongTensor(s_input_ids)
#                 s_attention_masks = torch.LongTensor(s_attention_masks)
#                 sk_input_ids.append(s_input_ids)
#                 sk_attention_masks.append(s_attention_masks)
#
#         return image_ids, images, report_input_ids, report_attention_masks, sk_input_ids, sk_attention_masks
#
#
# class FinetuneLoaderHaveIndication(DataLoader):
#     def __init__(self, args, tokenizer, split, shuffle, drop_last=False):
#         self.batch_size = args['batch_size']
#         self.shuffle = shuffle
#         self.num_workers = args['num_workers']
#         self.split = split
#         # self.drop_last = drop_last
#
#         if split == 'train':
#             self.transform = transforms.Compose([
#                 transforms.Resize(256),
#                 transforms.RandomCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406),
#                                      (0.229, 0.224, 0.225))])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406),
#                                      (0.229, 0.224, 0.225))])
#
#         if args['data_name'] == 'iu_xray':
#             self.dataset = IuxrayFinetuneDatasetHasIndication(args, tokenizer, self.split, transform=self.transform)
#         elif args['data_name'] == 'mimic_cxr':
#             self.dataset = MimiccxrFinetuneDatasetHasIndication(args, tokenizer, self.split, transform=self.transform)
#         else:
#             raise ValueError
#         drop_last = False
#         if len(self.dataset) % self.batch_size == 1:
#             drop_last = True
#
#         self.init_kwargs = {
#             'dataset': self.dataset,
#             'batch_size': self.batch_size,
#             'shuffle': self.shuffle,
#             'collate_fn': self.collate_fn,
#             'num_workers': self.num_workers,
#             "drop_last": drop_last
#         }
#         super().__init__(**self.init_kwargs)
#
#     @staticmethod
#     def collate_fn(data):
#         image_ids, images, report_ids, report_masks, report_lens, sk_ids, sk_masks, sk_lens, inc_ids, inc_masks, inc_lens = zip(*data)
#         images = torch.stack(images, 0)
#         batch_size = len(images)
#
#         # report
#         report_max_len = max(report_lens)
#         report_input_ids = np.zeros((batch_size, report_max_len), dtype=int)
#         report_attention_masks = np.zeros((batch_size, report_max_len), dtype=int)
#
#         for i in range(batch_size):
#             report_input_ids[i, :len(report_ids[i])] = report_ids[i]
#             report_attention_masks[i, :len(report_masks[i])] = report_masks[i]
#
#         report_input_ids = torch.LongTensor(report_input_ids)
#         report_attention_masks = torch.LongTensor(report_attention_masks)
#
#         # specific knowledge
#         sk_input_ids, sk_attention_masks = [], []
#         if all(len(item) == 0 for item in sk_lens):
#             pass
#         else:
#             specific_knowledge_max_len = np.array(sk_lens).max(0)
#             sk_max_len = np.max(specific_knowledge_max_len)
#             # specific knowledge topk
#             for k, s_len in enumerate(specific_knowledge_max_len):
#                 s_input_ids = np.zeros((batch_size, sk_max_len), dtype=int)
#                 s_attention_masks = np.zeros((batch_size, sk_max_len), dtype=int)
#                 for i in range(batch_size):
#                     cur_ids, cur_masks = sk_ids[i][k], sk_masks[i][k]
#                     s_input_ids[i, :len(cur_ids)] = cur_ids
#                     s_attention_masks[i, :len(cur_masks)] = cur_masks
#                 s_input_ids = torch.LongTensor(s_input_ids)
#                 s_attention_masks = torch.LongTensor(s_attention_masks)
#                 sk_input_ids.append(s_input_ids)
#                 sk_attention_masks.append(s_attention_masks)
#
#         # indication
#         inc_input_ids, inc_attention_masks = [], []
#         if all(item == 0 for item in inc_lens):
#             pass
#         else:
#             inc_max_len = max(inc_lens)
#             inc_input_ids = np.zeros((batch_size, inc_max_len), dtype=int)
#             inc_attention_masks = np.zeros((batch_size, inc_max_len), dtype=int)
#
#             for i in range(batch_size):
#                 inc_input_ids[i, :len(inc_ids[i])] = inc_ids[i]
#                 inc_attention_masks[i, :len(inc_masks[i])] = inc_masks[i]
#
#             inc_input_ids = torch.LongTensor(inc_input_ids)
#             inc_attention_masks = torch.LongTensor(inc_attention_masks)
#
#         return image_ids, images, report_input_ids, report_attention_masks, sk_input_ids, sk_attention_masks, inc_input_ids, inc_attention_masks