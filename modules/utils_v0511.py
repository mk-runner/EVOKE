import copy
import math
import os
import cv2
import json
import os.path
import random
import warnings
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torch import Tensor, device

from .metrics.metrics import compute_ce_scores


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


def setup_seed(seed):
    # seed init
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch seed init
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


def generate_heatmap(image, weights):
    # image = image.transpose(1, 2, 0)
    height, width, _ = image.shape
    weights = weights.reshape(int(weights.shape[0] ** 0.5), int(weights.shape[0] ** 0.5))
    weights = weights - np.min(weights)
    weights = weights / np.max(weights)
    weights = cv2.resize(weights, (width, height))
    weights = np.uint8(255 * weights)
    heatmap = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    result = heatmap * 0.5 + image * 0.5
    return result


class VisualProjectionHeadPretrain(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),  # change the dimension
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)


class TextProjectionHeadPretrain(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),  # change the dimension
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)


class VisualProjectionHeadFinetune(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),  # change the dimension
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)


class TextProjectionHeadFinetune(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),  # change the dimension
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)
        return x.permute(0, 2, 1)


class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        queries, keys, values = queries.unsqueeze(0), keys.unsqueeze(0), values.unsqueeze(0)
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out.squeeze(0)


class GlobalEmbedding(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 2048,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        return self.head(x)


class LocalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim,
                      kernel_size=1, stride=1, padding=0),  # change the dimension
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.head(x)

        return x.permute(0, 2, 1)


class GlobalEmbeddingV1(nn.Module):
    def __init__(self,
                 input_dim: int = 768,
                 output_dim: int = 512) -> None:
        super().__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim, affine=False)

    def forward(self, x):
        if x.shape[0] != 1:
            x = self.bn(self.linear(x))
        else:
            x = self.linear(x)
        return x


class LocalEmbeddingV1(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv1d(input_dim, output_dim,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(output_dim, affine=False)  # output layer
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x = x.unsqueeze(0).permute(0, 2, 1)
        x = self.head(x)
        #
        return x.permute(0, 2, 1)
        # return x.permute(0, 2, 1).squeeze(0)


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class AttentionPool(nn.Module):
    """ Self attention pooling Layer"""

    def __init__(self, in_dim):
        super(AttentionPool, self).__init__()
        self.chanel_in = in_dim
        self.query = nn.Linear(in_dim, in_dim // 8)
        self.key = nn.Linear(in_dim, in_dim // 8)
        self.softmax = nn.Softmax(dim=0)  #

    def forward(self, x):
        """
            inputs :s
                x : input feature in sentence (N, D)
            returns :
                out : self attention value + input feature
                attention: N .
        """
        n, d = x.size()  # N x D
        proj_query = self.query(x)  # N x D  -> N x M
        proj_key = self.key(x)  # N x D  -> N x M
        energy = proj_query @ proj_key.T  # N x M @ M x N = N x N
        attention = self.softmax(energy.sum(1, keepdim=True))  # N x 1
        return attention.T @ x  # 1 * D


class MultiviewFusion(nn.Module):
    def __init__(self, embed_dim, fusion_strategy):
        assert fusion_strategy in ['avg', 'max', 'min', 'attention']
        super().__init__()
        if fusion_strategy == 'avg':
            self.pool = lambda x: torch.mean(x, dim=0, keepdim=True)
        elif fusion_strategy == 'max':
            self.pool = lambda x: torch.max(x, dim=0, keepdim=True)
        elif fusion_strategy == 'min':
            self.pool = lambda x: torch.min(x, dim=0, keepdim=True)
        elif fusion_strategy == 'attention':
            self.attention_pool = AttentionPool(embed_dim)
            self.pool = lambda x: self.attention_pool(x)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.pool(x)


class SentenceGather(nn.Module):
    # Word-level gather
    def __init__(self, pool='avg', embed_dim=768, logger=None):
        super().__init__()
        if pool == 'avg':
            self.pool = lambda x: torch.mean(x, dim=0, keepdim=True)
        elif pool == 'max':
            self.pool = lambda x: torch.max(x, dim=0, keepdim=True)
        elif pool == 'min':
            self.pool = lambda x: torch.min(x, dim=0, keepdim=True)
        elif pool == 'attention':
            self.attention_pool = AttentionPool(embed_dim)
            self.pool = lambda x: self.attention_pool(x)
        else:
            raise NotImplementedError
        if logger is not None:
            logger.info('LocalGather: {}'.format(pool))
        print('LocalGather: {}'.format(pool))

    def forward(self, x, sentence_idx):
        sen_text_embed = []
        # 遍历每个样本
        for b in range(x.shape[0]):
            sample_sen_text_embed = []
            sample_sen_idx = np.array(sentence_idx[b])  # 每个样本的句子划分情况
            sample_text_embed = x[b]
            for sen_idx in np.unique(sample_sen_idx):
                aggregate_word = sample_text_embed[np.where(sample_sen_idx == sen_idx)[0], :]
                if len(aggregate_word) == 0:  # delete the space
                    continue
                sample_sen_text_embed.append(self.pool(aggregate_word))  # 注意添加了self.pool的，将每个句子的表征变为一个向量
            sample_sen_text_embed = torch.cat(sample_sen_text_embed, dim=0)  # 每个句子的表征是一个向量
            sen_text_embed.append(sample_sen_text_embed)
        return sen_text_embed # list(Tensor(sentence_num, dim))


class SentenceAttentionPool(nn.Module):
    def __init__(self, spacial_dim=256, embed_dim=512, num_heads=8, output_dim: int = None, pos_embed=True):
        super().__init__()
        self.pos_embed = pos_embed
        if self.pos_embed:
            self.positional_embedding = nn.Parameter(torch.randn(spacial_dim + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)  # 这个写法感觉怪怪的
        self.num_heads = num_heads

    def forward(self, x):
        # X: [B, N, C], [1, 句子数量，Dim]
        x = x.permute(1, 0, 2)
        # x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (L+1)NC, [句子数量+1, 1, Dim]
        if self.pos_embed:  # 判断是否添加位置编码
            x = x + self.positional_embedding[: x.size(0), None, :].to(x.dtype)  # (L+1)NC

        # 在对x进行，多头注意力机制
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]


class LocalCrossAttention(nn.Module):
    def __init__(self, embed_dim, drop_rate=0):
        super(LocalCrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.query1 = nn.Linear(embed_dim, embed_dim)
        self.key1 = nn.Linear(embed_dim, embed_dim)
        self.value1 = nn.Linear(embed_dim, embed_dim)
        self.dropout1 = nn.Dropout(drop_rate)
        self.query2 = nn.Linear(embed_dim, embed_dim)
        self.key2 = nn.Linear(embed_dim, embed_dim)
        self.value2 = nn.Linear(embed_dim, embed_dim)
        self.dropout2 = nn.Dropout(drop_rate)

    def forward(
            self,
            input_tensor1,
            input_tensor2,
            attention_mask1=None,
            attention_mask2=None,
            return_attention_weights=False,
    ):
        # for vision input [I, D]
        query_layer1 = self.query1(input_tensor1)
        key_layer1 = self.key1(input_tensor1)
        value_layer1 = self.value1(input_tensor1)

        # for text input [T, D]
        query_layer2 = self.query2(input_tensor2)
        key_layer2 = self.key2(input_tensor2)
        value_layer2 = self.value2(input_tensor2)

        # input_tensor2 (T, D) as queries, and input_tensor1 (I, D) as keys and values
        attention_scores1 = query_layer2 @ key_layer1.T  # [T, D] @ [D, I] = [T, I]
        attention_scores1 = attention_scores1 / math.sqrt(self.embed_dim)
        if attention_mask1 is not None:
            attention_scores1 = attention_scores1 + attention_mask1

        # Sigmoid is better in this case
        # TODO: pre-normalize vs. post-normalize
        attention_probs1 = torch.softmax(attention_scores1, dim=-1)
        # attention_probs1 = F.sigmoid(attention_scores1)
        attention_probs1 = self.dropout1(attention_probs1)
        context_layer1 = attention_probs1 @ value_layer1  # [T, I] @ [I, D] = [T, D]

        # input_tensor1 (I, D) as queries, and input_tensor2 (T, D) as keys and values
        attention_scores2 = query_layer1 @ key_layer2.T  # [I, D] @ [D, T] = [I, T]
        attention_scores2 = attention_scores2 / math.sqrt(self.embed_dim)

        if attention_mask2 is not None:
            attention_scores2 = attention_scores2 + attention_mask2

        attention_probs2 = torch.softmax(attention_scores2, dim=-1)
        # attention_probs2 = F.sigmoid(attention_scores2)
        attention_probs2 = self.dropout2(attention_probs2)
        context_layer2 = attention_probs2 @ value_layer2  # [I, T] @ [T, D] = [I, D]
        if return_attention_weights:
            return context_layer2, attention_probs2, context_layer1, attention_probs1
        else:  # text_modal (T, D) and vision_modal (I, D)
            return context_layer1, context_layer2


def temp_compute_scores(ori_report, topk_reports, args):
    ori_report = [ori_report] * len(topk_reports)
    result = compute_ce_scores(ori_report, topk_reports, args)
    return result


def plot_images(images, metrics, images_dir, image_id, specific_knowledge_dir, split):
    fig = plt.figure(figsize=(10, 8))
    for i in range(len(images)):
        image = plt.imread(os.path.join(images_dir, images[i]))
        axes = fig.add_subplot(int(f'22%d' % (i+1)))
        axes.axis('off')
        if i == 0:
            if len(metrics) != 0:
                str_metric = ''
                for k, v in metrics.items():
                    str_metric += f'{k.lower()}:{v:.3f} '
                axes.set_title(str_metric.strip())
            else:
                axes.set_title('original image')
        axes.imshow(image, cmap='viridis')
    plt.savefig(os.path.join(specific_knowledge_dir, f'{split}_{image_id}_specific_knowledge.jpg'))
    # plt.show()
    plt.cla()
    plt.clf()


class PretrainTestAnalysis(object):
    def __init__(self, ann_path, topk, image_dir, sk_analysis_dir=None, data_name='mimic_cxr'):
        assert 1 <= topk <= 30
        # self.topk_ann_path = topk_ann_path
        self.ann_path = ann_path
        self.topk = topk
        self.image_dir = image_dir
        if sk_analysis_dir is not None:
            os.makedirs(sk_analysis_dir, exist_ok=True)
        self.sk_analysis_dir = sk_analysis_dir
        self.data_name = data_name


    def get_report_dict(self):
        id2report = {}
        ann_data = json.load(open(self.ann_path))
        for split, value in ann_data.items():
            for idx, item in tqdm(enumerate(value)):
                if self.data_name == 'mimic_cxr':
                    cur_idx = '_'.join([str(item['subject_id']), str(item['study_id']), item['id']])
                else:
                    cur_idx = item['id']
                id2report[cur_idx] = [item['report'], item['core_findings']]
        return id2report

    def get_specific_knowledge(self, id2image):
        ann_data = json.load(open(self.ann_path))
        # id2image = json.load(open(self.topk_ann_path))
        id2report = self.get_report_dict()
        new_ann_data = {}
        for split, value in ann_data.items():
            new_ann_data[split] = []
            for idx, item in tqdm(enumerate(value)):
                if self.data_name == 'mimic_cxr':
                    cur_idx = '_'.join([str(item['subject_id']), str(item['study_id']), item['id']])
                else:
                    cur_idx = item['id']
                try:
                    topk_images_id = id2image[cur_idx][:self.topk]
                    sk_reports = [id2report[i][0] for i in topk_images_id]
                    sk_keywords = [id2report[i][1] for i in topk_images_id]
                    specific_knowledge = {'reports': sk_reports, 'sk_keywords': sk_keywords}
                except:
                    specific_knowledge = {'reports': [], 'keywords': []}
                new_item = {
                    **item,
                    'specific_knowledge': specific_knowledge
                }
                new_ann_data[split].append(new_item)

        save_file_name = self.ann_path.split('.json')[0] + f'_best_reports_keywords_{self.topk}.json'
        json.dump(new_ann_data, open(save_file_name, 'w'), indent=2)

    def show_topk_images(self, args, id2image):
        ann_data = json.load(open(self.ann_path))
        # id2image = json.load(open(self.topk_ann_path))
        id2report = self.get_report_dict()
        for split, value in ann_data.items():
            idx = torch.randperm(len(value))[:10]
            for i in tqdm(idx):
                item = value[i]
                cur_idx = '_'.join([str(item['subject_id']), str(item['study_id']), item['id']])
                try:
                    topk_images_id = id2image[cur_idx][:self.topk]
                except:
                    continue

                # calculate the similarity between the report and its corresponding topk reports
                cur_reports = item['report']
                topk_reports = [id2report[ids][0] for ids in topk_images_id]
                metrics = temp_compute_scores(cur_reports, topk_reports, args)

                # show_topk_images including images and their reports
                print("--------------------------------------------------------------")
                print(i, metrics)
                print(cur_idx, item['core_findings'])
                topk_images_path = [item['image_path'][0]]
                for k, image_id in enumerate(topk_images_id[:3]):
                    _subject_id, _study_id, _dicom_id = image_id.split('_')
                    _image_path = f"p{_subject_id[:2]}/p{_subject_id}/s{_study_id}/{_dicom_id}.jpg"
                    topk_images_path.append(_image_path)
                    print(image_id, 'topk_image %d' % k, id2report[image_id][1])

                print("--------------------------------------------------------------")
                plot_images(topk_images_path, metrics, self.image_dir, cur_idx, self.sk_analysis_dir, split)


def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):
    if device is not None:
        warnings.warn(
            "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
        )
    else:
        device = attention_mask.device
    batch_size, seq_length = input_shape
    seq_ids = torch.arange(seq_length, device=device)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    # in case past_key_values are used we need to add a prefix ones mask to the causal mask
    # causal and attention masks must have same type with pytorch version < 1.3
    causal_mask = causal_mask.to(attention_mask.dtype)

    if causal_mask.shape[1] < attention_mask.shape[1]:
        prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
        causal_mask = torch.cat(
            [
                torch.ones((batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype),
                causal_mask,
            ],
            axis=-1,
        )

    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    return extended_attention_mask


def get_extended_attention_mask(
        attention_mask: Tensor, input_shape: Tuple[int], device: device = None, dtype: torch.float = None,
        is_decoder: bool = False
) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.
        device
        dtype
        is_decoder
    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if dtype is None:
        dtype = torch.float

    if device is None:
        device = attention_mask.device

    if not (attention_mask.dim() == 2 and is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if is_decoder:
            extended_attention_mask = create_extended_attention_mask_for_decoder(
                input_shape, attention_mask, device
            )
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask


class VisualProjectionHead:
    pass