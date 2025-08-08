import copy
import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from transformers import AutoConfig

from models.language_encoder.language_model import TextDecoderModel, TextEncoderModel
from models.language_encoder.bert_model import BertCrossLayer, BertLayer
from models.vision_encoder.vit import build_vit_extractor
from modules.base_cmn import BaseCMN
from modules.encoder_decoder import EncoderDecoder
from modules.loss import compute_lm_loss
from modules.utils_v0511 import (get_extended_attention_mask, ScaledDotProductAttention,
                                 VisualProjectionHeadPretrain, TextProjectionHeadPretrain,
                                 VisualProjectionHeadFinetune, TextProjectionHeadFinetune)
from modules.visual_extractor import ResNet


class FineTune(nn.Module):
    def __init__(self, args: dict, tokenizer: object, data_name: str):
        super(FineTune, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

        # ==========define model modules===================#
        # define visual encoder
        assert args['visual_encoder'] in ['resnet101', 'ViT-B-32']
        if args["visual_encoder"] == 'resnet101':
            self.visual_extractor = ResNet(args)
            visual_dim = 2048

        elif args['visual_encoder'] == 'ViT-B-32':
            self.visual_extractor = build_vit_extractor(args)
            visual_dim = 768
        else:
            raise ValueError(f'the visual encoder {args["visual_encoder"]} is not support!')

        # self.ln = nn.LayerNorm(visual_dim)

        # define text encoder
        self.text_encoder = TextEncoderModel(args, tokenizer)
        self.layer_norm_1 = nn.LayerNorm(visual_dim)
        self.layer_norm_2 = nn.LayerNorm(visual_dim)

        # define text decoder
        assert args['text_decoder'] in ['r2gen', 'cmn', 'bert']
        if args['text_decoder'] == 'r2gen':
            self.text_decoder = EncoderDecoder(args, tokenizer)
        elif args['text_decoder'] == 'cmn':
            self.text_decoder = BaseCMN(args, tokenizer)
        elif args['text_decoder'] == 'bert':
            self.text_decoder = TextDecoderModel(args, tokenizer)
        else:
            raise ValueError(f'the text decoder {args["text_decoder"]} is not support!')

        # ==========define contrastive learning modules===================#
        # define the local embedding and global embedding for uni-modal
        text_dim = self.text_encoder.encoder.config.hidden_size
        self.visual_head = VisualProjectionHeadFinetune(visual_dim, output_dim=args['output_dim'],
                                                        hidden_dim=args['output_dim'])
        self.text_head = TextProjectionHeadFinetune(text_dim, output_dim=args['output_dim'],
                                                    hidden_dim=args['output_dim'])
        self.multiview_cross_attention = ScaledDotProductAttention(visual_dim, visual_dim, visual_dim, h=8)

        # # fusion module
        fusion_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=args['fusion_checkpoint'],
            vocab_size=args["vocab_size"],
            hidden_size=args["output_dim"],
            num_hidden_layers=args["sk_fusion_num_layers"],
            max_position_embeddings=args["max_seq_len"],
            num_attention_heads=args["fusion_num_heads"],
            eos_token_id=tokenizer.token_to_id('[EOS]'),
            bos_token_id=tokenizer.token_to_id('[BOS]'),
            pad_token_id=tokenizer.token_to_id('[PAD]'),
        )

        # # used to only visual features (i.e., is_add_indication=False, knowledge=False)
        self_att_config = copy.deepcopy(fusion_config)
        self_att_config.is_decoder = False
        self_att_config.add_cross_attention = False
        self.visual_self_atten_layers = nn.ModuleList(
            [BertLayer(self_att_config) for _ in range(args['sk_fusion_num_layers'])])

        # # used to visual and indications features (i.e., is_add_indication=True, knowledge=False)
        # self.indication_layers = nn.ModuleList(
        #     [BertLayer(fusion_config) for _ in range(args['sk_fusion_num_layers'])])
        #
        # # used to visual and knowledge features (i.e., is_add_indication=False, knowledge=True)
        # self.visual_knowledge_layers = nn.ModuleList(
        #     [BertLayer(fusion_config) for _ in range(args['sk_fusion_num_layers'])])
        #
        # # used to visual, indications, and knowledge features (i.e., is_add_indication=True, knowledge=True)
        # self.multimodal_fusion_layers = nn.ModuleList(
        #     [BertLayer(fusion_config) for _ in range(args['sk_fusion_num_layers'])])

        self.multimodal_fusion_layers = nn.ModuleList(
            [BertCrossLayer(fusion_config) for _ in range(args['sk_fusion_num_layers'])])

        # self.modality_type_embeddings = nn.Embedding(3, args['output_dim'])
        # self.modality_type_embeddings.apply(init_weights)

        # define the visual forward
        assert data_name in ['iu_xray', 'mimic_cxr']
        if data_name == 'iu_xray':
            self.visual_forward = self.visual_forward_iu_xray
        else:
            self.visual_forward = self.visual_forward_mimic_cxr

        # define the text_decoder forward
        if args['text_decoder'] in ['r2gen', 'cmn']:
            self.text_decoder_forward = self.text_decoder_forward_r2gen
        else:
            self.text_decoder_forward = self.text_decoder_forward_bert

        self.freeze_encoder_models(args['freeze_image_encoder'], args['freeze_text_encoder'])

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params / 1e6)

    def freeze_encoder_models(self, freeze_image_encoder: bool, freeze_text_encoder: bool):
        if freeze_image_encoder:
            # visual encoder
            for param in self.visual_extractor.parameters():
                param.requires_grad = False
            for param in self.visual_head.parameters():
                param.requires_grad = False
            # for param in self.visual_global.parameters():
            #     param.requires_grad = False
        if freeze_text_encoder:
            # text encoder
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.text_head.parameters():
                param.requires_grad = False
            # for param in self.text_global.parameters():
            #     param.requires_grad = False

    def visual_forward_iu_xray(self, images):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        # fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1], 0), dim=0)
        att_feats = torch.cat((fc_feats_0.unsqueeze(1), att_feats_0, fc_feats_1.unsqueeze(1), att_feats_1), dim=1)
        # att_feats = F.normalize(self.visual_head(att_feats), dim=-1, p=2)
        return fc_feats, att_feats

    def visual_forward_mimic_cxr(self, images):
        att_feats, fc_feats = self.visual_extractor(images)  # attr_feats: patch feature; fc_feats: avg pool feats
        # att_feats = F.normalize(self.visual_local(att_feats), dim=-1, p=2)
        # fc_feats = F.normalize(self.visual_global(fc_feats), dim=-1, p=2)
        # att_feats = self.visual_local(att_feats)  # (b*n_view, 49, 2048)
        # fc_feats = self.visual_global(fc_feats)
        return fc_feats, att_feats

    def text_decoder_forward_r2gen(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, mode='train'):

        if mode == 'train':
            output = self.text_decoder(input_ids, encoder_hidden_states, attention_mask=attention_mask,
                                       encoder_attention_mask=encoder_attention_mask, mode='forward')
            lm_loss = compute_lm_loss(output, input_ids, attention_mask)
            return lm_loss
        else:
            output, _ = self.text_decoder(encoder_hidden_states, encoder_attention_masks=encoder_attention_mask,
                                          mode='sample')
            gen_texts = self.tokenizer.decode_batch(output.cpu().tolist())
            gt_texts = self.tokenizer.decode_batch(input_ids.cpu().tolist())
            if mode == 'sample':
                gen_texts = [text if len(text) > 0 else "there is no evidence of pulmonary." for text in gen_texts]
                return [gen_texts, gt_texts]
            elif mode == 'test':
                return [gen_texts, gt_texts]
            else:  # mode == 'inference'
                gen_texts = [text if len(text) > 0 else "there is no evidence of pulmonary." for text in gen_texts]
                return [gen_texts, output]

    def text_decoder_forward_bert(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, mode='train'):
        if mode == 'train':  # att_feats (16, 49, 2048), fc_feats (16, 2048), target (16, 100)
            lm_loss = self.text_decoder(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        encoder_hidden_states=encoder_hidden_states,
                                        encoder_attention_mask=encoder_attention_mask)
            return lm_loss
        elif mode == 'sample':
            [gen_texts, gt_texts] = self.text_decoder.evaluation(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                tokenizer=self.tokenizer
            )
            return [gen_texts, gt_texts]  # []
        else:
            raise ValueError('it is not implement!')

    def cross_attention_no_parameters(self, q, k_v):
        # q = F.normalize(q, dim=-1, p=2)
        # k_v = F.normalize(k_v, dim=-1, p=2) # all features are already normalized
        att_sim = q @ k_v.t()
        att_sco = F.softmax(att_sim / math.sqrt(q.shape[1]), dim=-1)
        att_output = att_sco @ k_v
        return att_output

    def multiview_fusion(self, global_image_embed, local_image_embed, patient_ids, batch_size):
        # obtain labels indicate corresponding multiview images
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(int)
        labels = torch.from_numpy(labels)
        labels.fill_diagonal_(0)

        # obtain image embed
        image_embed = torch.cat([global_image_embed.unsqueeze(1), local_image_embed], dim=1)
        image_embed = self.layer_norm_1(image_embed)
        new_image_embed = []
        for i in range(batch_size):
            if labels[i].sum() == 0:
                new_image_embed.append(image_embed[i])
                continue
            multiview_image_embed = torch.cat([image_embed[j] for j, tag in enumerate(labels[i]) if tag == 1], dim=0)
            # include multiview images
            # cur_image_embed = self.cross_attention_no_parameters(q=image_embed[i], k_v=multiview_image_embed.detach())
            cur_image_embed = self.multiview_cross_attention(image_embed[i], multiview_image_embed.detach(),
                                                             multiview_image_embed.detach())
            # cur_image_embed = self.multiview_cross_attention(image_embed[i], multiview_image_embed,
            #                                                  multiview_image_embed)
            # add & normalize
            # add & normalize
            cur_image_embed = self.layer_norm_2(cur_image_embed + image_embed[i])
            new_image_embed.append(cur_image_embed)
        new_image_embed = torch.stack(new_image_embed, dim=0)
        new_image_embed = self.visual_head(new_image_embed)
        return new_image_embed[:, 0, :], new_image_embed[:, 1:, :]

    def forward(self, images, report_ids, report_masks, patient_ids, inc_ids=None, inc_masks=None, mode='train'):
        # image embedding
        device = images.device
        # images had been normalized
        v_fc_feats, v_att_feats = self.visual_forward(images)  # fc_feats and att_feats are global and focal features

        if self.args['is_multiview_learning']:
            # multiview fusion based on cross-attention
            v_fc_feats, v_att_feats = self.multiview_fusion(v_fc_feats, v_att_feats, patient_ids, report_ids.shape[0])
        else:
            image_embed = torch.cat([v_fc_feats.unsqueeze(1), v_att_feats], dim=1)
            image_embed = self.layer_norm_1(image_embed)
            image_embed = self.visual_head(image_embed)
            v_fc_feats, v_att_feats = image_embed[:, 0, :], image_embed[:, 1:, :]

        # obtain the encoder_hidden_states and encoder_attention_mask only using image feats and corresponding masks
        encoder_hidden_states = torch.cat([v_fc_feats.unsqueeze(1), v_att_feats], dim=1)
        encoder_attention_mask = torch.ones(encoder_hidden_states.size()[:2], dtype=torch.long).to(device)
        # == Begin: Assign Type Embeddings (uni-modal features + modal_type) ==
        # text modal type is one; vision modal type is zero.
        # encoder_hidden_states = encoder_hidden_states + self.modality_type_embeddings(torch.zeros_like(encoder_attention_mask))

        # obtain the encoder_hidden_states and encoder_attention_mask
        if inc_ids is not None:
            inc_ids, inc_masks = inc_ids.to(device, non_blocking=True), inc_masks.to(device, non_blocking=True)
            inc_feats = self.text_encoder(input_ids=inc_ids, attention_mask=inc_masks)
            inc_feats = self.text_head(inc_feats)
            # if it has knowledge features, please add here
            # knowledge embedding is used for modality_type_embeddings[2]

            # inc_feats = inc_feats + self.modality_type_embeddings(torch.ones(inc_feats.shape[:2]).to(report_masks))

            # concat all features using self-attention
            # cat_attention_mask = torch.cat([encoder_attention_mask, inc_masks], dim=-1)
            # extended_attention_mask = get_extended_attention_mask(cat_attention_mask, cat_attention_mask.size())
            # x = torch.cat([encoder_hidden_states, inc_feats], dim=1)
            # for layer_idx, image_layer in enumerate(self.multimodal_fusion_layers):
            #     # == Begin: Fetch the intermediate outputs (different layers to perform MIM) ==
            #     # == End  : Fetch the intermediate outputs (different layers to perform MIM) ==
            #     # == Begin: Co-Attention ==
            #     x1 = image_layer(x, attention_mask=extended_attention_mask, output_attentions=True)
            #     x = x1[0]
            #     # == End: Co-Attention ==
            #     # == Begin: For visualization: Return the attention weights ==

            extended_image_masks = get_extended_attention_mask(encoder_attention_mask, encoder_attention_mask.size())
            extended_inc_masks = get_extended_attention_mask(inc_masks, inc_masks.size())

            x, y = encoder_hidden_states, inc_feats
            for layer_idx, image_layer in enumerate(self.multimodal_fusion_layers):
                # == Begin: Fetch the intermediate outputs (different layers to perform MIM) ==
                # == End  : Fetch the intermediate outputs (different layers to perform MIM) ==
                # == Begin: Co-Attention ==
                x1 = image_layer(x, y, attention_mask=extended_image_masks,
                                 encoder_attention_mask=extended_inc_masks, output_attentions=True)
                x = x1[0]
                # == End: Co-Attention ==
                # == Begin: For visualization: Return the attention weights ==
            encoder_hidden_states = x

        else:
            x = encoder_hidden_states
            extended_attention_mask = get_extended_attention_mask(encoder_attention_mask, encoder_attention_mask.size())
            for layer_idx, image_layer in enumerate(self.visual_self_atten_layers):
                # == Begin: Fetch the intermediate outputs (different layers to perform MIM) ==
                # == End  : Fetch the intermediate outputs (different layers to perform MIM) ==
                # == Begin: Co-Attention ==
                x1 = image_layer(x, attention_mask=extended_attention_mask, output_attentions=True)
                x = x1[0]
                # == End: Co-Attention ==
                # == Begin: For visualization: Return the attention weights ==
            encoder_hidden_states = x
        ret = self.text_decoder_forward(report_ids, report_masks, encoder_hidden_states, encoder_attention_mask, mode=mode)
        if mode == 'train':  # att_feats (16, 49, 2048), fc_feats (16, 2048), target (16, 100)
            return {
                'lm': ret,
                'all_loss': ret
            }
        elif mode == 'sample':
            gen_texts, gt_texts = ret
            return [gen_texts, gt_texts]
        elif mode == 'inference':
            gen_texts, outputs = ret
            return [gen_texts, outputs]
        else:
            raise ValueError


class Pretrain(nn.Module):
    def __init__(self, args: dict, tokenizer: object, data_name: str):
        super(Pretrain, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

        # ==========define model modules===================#
        # define visual encoder
        assert args['visual_encoder'] in ['resnet101', 'ViT-B-32']
        if args["visual_encoder"] == 'resnet101':
            self.visual_extractor = ResNet(args)
            visual_dim = 2048
        elif args['visual_encoder'] == 'ViT-B-32':
            self.visual_extractor = build_vit_extractor(args)
            visual_dim = 768
        else:
            raise ValueError(f'the visual encoder {args["visual_encoder"]} is not support!')

        # define text encoder
        self.text_encoder = TextEncoderModel(args, tokenizer)
        self.layer_norm_1 = nn.LayerNorm(visual_dim)
        self.layer_norm_2 = nn.LayerNorm(visual_dim)

        # ==========define contrastive learning modules===================#
        # define the local embedding and global embedding for uni-modal
        text_dim = self.text_encoder.encoder.config.hidden_size
        self.visual_head = VisualProjectionHeadPretrain(visual_dim, output_dim=args['output_dim'],
                                                        hidden_dim=args['output_dim'])
        self.text_head = TextProjectionHeadPretrain(text_dim, output_dim=args['output_dim'],
                                                    hidden_dim=args['output_dim'])
        self.multiview_cross_attention = ScaledDotProductAttention(visual_dim, visual_dim, visual_dim, h=8)

        # define the visual forward
        assert data_name in ['iu_xray', 'mimic_cxr', 'mix']
        if data_name == 'iu_xray':
            self.visual_forward = self.visual_forward_iu_xray
        else:
            self.visual_forward = self.visual_forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params / 1e6)

    def visual_forward_iu_xray(self, images):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        # fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1], 0), dim=0)

        # att_feats = F.normalize(self.visual_local(att_feats), dim=-1)
        # fc_feats = F.normalize(self.visual_global(fc_feats), dim=-1)
        return fc_feats, att_feats

    def visual_forward_mimic_cxr(self, images):
        att_feats, fc_feats = self.visual_extractor(images)  # attr_feats: patch feature; fc_feats: avg pool feats
        # att_feats = F.normalize(self.visual_local(att_feats), dim=-1, p=2)
        # fc_feats = F.normalize(self.visual_global(fc_feats), dim=-1, p=2)
        # att_feats = self.visual_local(att_feats)  # (b*n_view, 49, 2048)
        # fc_feats = self.visual_global(fc_feats)
        return fc_feats, att_feats

    def obtain_text_embeds(self, input_ids, attention_mask):
        text_embed = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # fc_feats = self.text_global(text_embed[:, 0, :])
        # att_feats = self.text_local(text_embed[:, 1:, :])

        text_embed = self.text_head(text_embed)
        # fc_feats = F.normalize(self.text_global(text_embed[:, 0, :]), dim=-1)
        # att_feats = F.normalize(self.text_local(text_embed[:, 1:, :]), dim=-1)

        return text_embed[:, 0, :], text_embed[:, 1:, :]

    def simsiam_loss_func(self, x, y):
        p_x = self.predictor(x)
        p_y = self.predictor(y)
        z_x = x.detach()
        z_y = y.detach()
        return - (F.cosine_similarity(p_x, z_y, dim=-1).mean() + F.cosine_similarity(p_y, z_x, dim=-1).mean()) * 0.5

    def text_local_loss_fn(self, embed_A, embed_B, temperature, norm=True):
        if norm:
            embed_A = F.normalize(embed_A, dim=-1, p=2)  # (sen_num, dim)
            embed_B = F.normalize(embed_B, dim=-1, p=2)  # (sen_num, dim)
        lc_labels = torch.arange(embed_B.size(0), device=embed_B.device).long()  # 注意都是用text_token_num来计算最终的损失
        # self.lc_last_local_batch_size = local_batch_size
        logits_per_image = embed_B @ embed_A.t()
        logits_per_text = embed_A @ embed_B.t()
        image_loss = F.cross_entropy(logits_per_image / temperature, lc_labels)  # 直接计算交叉熵，计算文本对应的损失
        text_loss = F.cross_entropy(logits_per_text / temperature, lc_labels)  # 注意这种计算方式，
        loss = (image_loss + text_loss) / 2
        return loss

    def multi_pos_contra_images_v0401(self, global_image_embed, patient_ids):
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(float)
        labels = torch.from_numpy(labels).to(global_image_embed)
        labels.fill_diagonal_(0.0)
        # remove one-view image embed
        idx = torch.argwhere(labels.sum(1) != 0).reshape(-1)
        if len(idx) == 0:  # avoid all samples in a batch are one-view
            return torch.tensor([0.0], requires_grad=True, device=global_image_embed.device)
        global_image_embed, labels = global_image_embed[idx], labels[idx][:, idx]
        labels = labels / labels.sum(1, keepdim=True)

        # normalize
        # global_image_embed = F.normalize(global_image_embed, dim=-1, p=2)  # (b*n_view, dim)

        # calculated multiview loss
        global_image_embed = F.normalize(global_image_embed, dim=-1, p=2)
        logits = global_image_embed @ global_image_embed.T / self.args['region_temp']
        logits.fill_diagonal_(-1e9)

        # stable logits
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits - logits_max.detach()
        loss = F.cross_entropy(logits, labels)
        # if loss.detach().item() >= 10:
        #     pass
        return loss

    def cross_attention_no_parameters(self, q, k_v):
        # q = F.normalize(q, dim=-1, p=2)
        # k_v = F.normalize(k_v, dim=-1, p=2) # all features are already normalized
        att_sim = q @ k_v.t()
        att_sco = F.softmax(att_sim / math.sqrt(q.shape[1]), dim=-1)
        att_output = att_sco @ k_v
        return att_output

    def multiview_fusion(self, global_image_embed, local_image_embed, patient_ids, batch_size):
        # obtain labels indicate corresponding multiview images
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(int)
        labels = torch.from_numpy(labels)
        labels.fill_diagonal_(0)

        # obtain image embed
        image_embed = torch.cat([global_image_embed.unsqueeze(1), local_image_embed], dim=1)
        image_embed = self.layer_norm_1(image_embed)
        new_image_embed = []
        for i in range(batch_size):
            if labels[i].sum() == 0:
                new_image_embed.append(image_embed[i])
                continue
            multiview_image_embed = torch.cat([image_embed[j] for j, tag in enumerate(labels[i]) if tag == 1], dim=0)
            # include multiview images
            # cur_image_embed = self.cross_attention_no_parameters(q=image_embed[i], k_v=multiview_image_embed.detach())
            cur_image_embed = self.multiview_cross_attention(image_embed[i], multiview_image_embed.detach(),
                                                             multiview_image_embed.detach())
            # cur_image_embed = self.multiview_cross_attention(image_embed[i], multiview_image_embed,
            #                                                  multiview_image_embed)
            # add & normalize
            cur_image_embed = self.layer_norm_2(cur_image_embed + image_embed[i])
            new_image_embed.append(cur_image_embed)

        new_image_embed = torch.stack(new_image_embed, dim=0)
        # normalize
        new_image_embed = self.visual_head(new_image_embed)
        return new_image_embed[:, 0, :], new_image_embed[:, 1:, :]

    def global_alignment_loss(self, global_image_embed, global_text_embed, patient_ids):
        # obtain multi-positive target
        patient_ids = patient_ids[:global_image_embed.shape[0]]
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(int)
        labels = torch.from_numpy(labels).float().to(global_image_embed.device)
        labels = labels / labels.sum(1, keepdim=True)
        del patient_ids

        # normalize
        global_image_embed = F.normalize(global_image_embed, dim=-1, p=2)
        global_text_embed = F.normalize(global_text_embed, dim=-1, p=2)

        # calculate the InfoNCE loss
        instance_sim = global_image_embed @ global_text_embed.t()
        instance_sim_1 = global_text_embed @ global_image_embed.t()
        loss_instance_1 = F.cross_entropy(instance_sim / self.args['instance_temp'], labels)
        loss_instance_2 = F.cross_entropy(instance_sim_1 / self.args['instance_temp'], labels)
        global_instance_loss = (loss_instance_1 + loss_instance_2) / 2.0
        return global_instance_loss

    def local_text_token_alignment_loss(self, local_image_embed, local_text_embed):
        # cross-modal alignment between image patches and sentence embed in reports

        t_att_sim = local_text_embed @ local_image_embed.permute(0, 2, 1)
        t_att_sco = F.softmax(t_att_sim / math.sqrt(local_image_embed.shape[2]), dim=-1)
        t_att_output = torch.bmm(t_att_sco, local_image_embed)

        device = local_image_embed.device
        # normalize
        t_att_output = F.normalize(t_att_output, dim=-1, p=2)
        local_text_embed = F.normalize(local_text_embed, dim=-1, p=2)
        # calculate the loss
        word_sim = torch.bmm(local_text_embed, t_att_output.permute(0, 2, 1)) / self.args['region_temp']
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")  # the similarity between each word and each each
        word_targets = torch.arange(word_sim.shape[1]).long().repeat(word_sim.shape[0]).to(device)
        loss_word_1 = F.cross_entropy(word_sim_1, word_targets)

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = F.cross_entropy(word_sim_2, word_targets)
        loss_word = (loss_word_2 + loss_word_1) / 2.0
        return loss_word

    def forward(self, images, radgraph_ids, radgraph_masks, patient_ids):
        # ====obtaining uni-modal embedding====
        # image embedding
        v_fc_feats, v_att_feats = self.visual_forward(images)  # fc_feats and att_feats are global and focal features

        mul_pos_loss = torch.tensor([0.0])
        if self.args['is_multiview_learning']:
            # calculate multiview-enhanced/guided contrastive learning among images
            mul_pos_loss = self.multi_pos_contra_images_v0401(v_fc_feats, patient_ids)
            # multiview fusion based on cross-attention
            v_fc_feats, v_att_feats = self.multiview_fusion(v_fc_feats, v_att_feats, patient_ids, radgraph_ids.shape[0])
        else:
            image_embed = torch.cat([v_fc_feats.unsqueeze(1), v_att_feats], dim=1)
            image_embed = self.layer_norm_1(image_embed)
            image_embed = self.visual_head(image_embed)
            v_fc_feats, v_att_feats = image_embed[:, 0, :], image_embed[:, 1:, :]

        # text embedding
        t_fc_feats, t_att_feats = self.obtain_text_embeds(radgraph_ids, radgraph_masks)

        # ====instance-level contrastive loss====
        instance_loss = self.global_alignment_loss(v_fc_feats, t_fc_feats, patient_ids)

        # ====sentence-level contrastive loss====
        # sen_image_loss, sen_text_loss = self.local_sentence_alignment_loss(v_att_feats, local_sen_embed_stacks)
        sen_text_loss = self.local_text_token_alignment_loss(v_att_feats, t_att_feats)

        sen_image_loss = torch.tensor([0.0])
        # sen_text_loss = torch.tensor([0.0])
        if self.args['is_multiview_learning']:
            return {
                'sen_image_loss': sen_image_loss,
                'sen_text_loss': sen_text_loss,
                'instance_loss': instance_loss,
                'multiview_loss': mul_pos_loss,
                'all_loss': instance_loss + sen_text_loss + mul_pos_loss
            }
        else:
            return {
                'sen_image_loss': sen_image_loss,
                'sen_text_loss': sen_text_loss,
                'instance_loss': instance_loss,
                'multiview_loss': mul_pos_loss,
                'all_loss': instance_loss + sen_text_loss
            }


class PretrainNewMulPos(nn.Module):
    def __init__(self, args: dict, tokenizer: object, data_name: str):
        super(PretrainNewMulPos, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

        # ==========define model modules===================#
        # define visual encoder
        assert args['visual_encoder'] in ['resnet101', 'ViT-B-32']
        if args["visual_encoder"] == 'resnet101':
            self.visual_extractor = ResNet(args)
            visual_dim = 2048
        elif args['visual_encoder'] == 'ViT-B-32':
            self.visual_extractor = build_vit_extractor(args)
            visual_dim = 768
        else:
            raise ValueError(f'the visual encoder {args["visual_encoder"]} is not support!')

        # define text encoder
        self.text_encoder = TextEncoderModel(args, tokenizer)
        self.layer_norm_1 = nn.LayerNorm(visual_dim)
        self.layer_norm_2 = nn.LayerNorm(visual_dim)

        # ==========define contrastive learning modules===================#
        # define the local embedding and global embedding for uni-modal
        text_dim = self.text_encoder.encoder.config.hidden_size

        self.visual_head = VisualProjectionHeadPretrain(visual_dim, output_dim=args['output_dim'],
                                                        hidden_dim=args['output_dim'])
        self.text_head = TextProjectionHeadPretrain(text_dim, output_dim=args['output_dim'],
                                                    hidden_dim=args['output_dim'])
        self.multiview_cross_attention = ScaledDotProductAttention(visual_dim, visual_dim, visual_dim, h=8)
        # self.visual_projection = nn.Linear(visual_dim, visual_dim)

        # define the visual forward
        assert data_name in ['iu_xray', 'mimic_cxr', 'mix']
        if data_name == 'iu_xray':
            self.visual_forward = self.visual_forward_iu_xray
        else:
            self.visual_forward = self.visual_forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params / 1e6)

    def visual_forward_iu_xray(self, images):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0])
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1])
        # fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        fc_feats = torch.mean(torch.stack([fc_feats_0, fc_feats_1], 0), dim=0)

        # att_feats = F.normalize(self.visual_local(att_feats), dim=-1)
        # fc_feats = F.normalize(self.visual_global(fc_feats), dim=-1)
        return fc_feats, att_feats

    def visual_forward_mimic_cxr(self, images):
        att_feats, fc_feats = self.visual_extractor(images)  # attr_feats: patch feature; fc_feats: avg pool feats
        # att_feats = F.normalize(self.visual_local(att_feats), dim=-1, p=2)
        # fc_feats = F.normalize(self.visual_global(fc_feats), dim=-1, p=2)
        # att_feats = self.visual_local(att_feats)  # (b*n_view, 49, 2048)
        # fc_feats = self.visual_global(fc_feats)
        return fc_feats, att_feats

    def obtain_text_embeds(self, input_ids, attention_mask):
        text_embed = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # fc_feats = self.text_global(text_embed[:, 0, :])
        # att_feats = self.text_local(text_embed[:, 1:, :])
        text_embed = self.text_head(text_embed)
        # fc_feats = F.normalize(self.text_global(text_embed[:, 0, :]), dim=-1)
        # att_feats = F.normalize(self.text_local(text_embed[:, 1:, :]), dim=-1)

        return text_embed[:, 0, :], text_embed[:, 1:, :]

    def simsiam_loss_func(self, x, y):
        p_x = self.predictor(x)
        p_y = self.predictor(y)
        z_x = x.detach()
        z_y = y.detach()
        return - (F.cosine_similarity(p_x, z_y, dim=-1).mean() + F.cosine_similarity(p_y, z_x, dim=-1).mean()) * 0.5

    def text_local_loss_fn(self, embed_A, embed_B, temperature, norm=True):
        if norm:
            embed_A = F.normalize(embed_A, dim=-1, p=2)  # (sen_num, dim)
            embed_B = F.normalize(embed_B, dim=-1, p=2)  # (sen_num, dim)
        lc_labels = torch.arange(embed_B.size(0), device=embed_B.device).long()  # 注意都是用text_token_num来计算最终的损失
        # self.lc_last_local_batch_size = local_batch_size
        logits_per_image = embed_B @ embed_A.t()
        logits_per_text = embed_A @ embed_B.t()
        image_loss = F.cross_entropy(logits_per_image / temperature, lc_labels)  # 直接计算交叉熵，计算文本对应的损失
        text_loss = F.cross_entropy(logits_per_text / temperature, lc_labels)  # 注意这种计算方式，
        loss = (image_loss + text_loss) / 2
        return loss

    def multi_pos_contra_images_v0404(self, global_image_embed, patient_ids):
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(float)
        labels = torch.from_numpy(labels).to(global_image_embed)
        labels.fill_diagonal_(0.0)
        # remove one-view image embed
        idx = torch.argwhere(labels.sum(1) != 0).reshape(-1)
        if len(idx) == 0:  # avoid all samples in a batch are one-view
            return torch.tensor([0.0], requires_grad=True, device=global_image_embed.device)
        labels = labels[idx] # multiview_sample to all_sample
        # labels = labels / labels.sum(1, keepdim=True)

        # calculated multiview loss
        global_image_embed = F.normalize(global_image_embed, dim=-1, p=2)
        logits = (global_image_embed @ global_image_embed.T) / self.args['region_temp']
        logits.fill_diagonal_(-1e9)
        logits = logits[idx]  # multiview_sample to all_sample

        # stable logits
        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits = logits - logits_max.detach()
        loss = torch.tensor([0.0], requires_grad=True, device=global_image_embed.device)
        for i in range(logits.shape[0]):
            cur_logits, cur_labels = logits[i], labels[i]

            # obtain the mask of multiple positives
            positive_mask = torch.tensor([False] * len(cur_logits))
            positive_idx = torch.argwhere(cur_labels != 0).reshape(-1)
            positive_mask[positive_idx] = True

            # average the similarities among multiple positives
            positive_logit = cur_logits[positive_mask].sum() / len(positive_idx)
            negative_logit = cur_logits[~positive_mask]

            new_cur_logits = torch.cat([positive_logit.reshape(-1), negative_logit])
            new_cur_label = torch.LongTensor([0]).to(global_image_embed.device)
            loss = loss + F.cross_entropy(new_cur_logits.reshape(1, -1), new_cur_label)

        loss = loss / logits.shape[0]
        return loss

    def cross_attention_no_parameters(self, q, k_v):
        # q = F.normalize(q, dim=-1, p=2)
        # k_v = F.normalize(k_v, dim=-1, p=2) # all features are already normalized
        att_sim = q @ k_v.t()
        att_sco = F.softmax(att_sim / math.sqrt(q.shape[1]), dim=-1)
        att_output = att_sco @ k_v
        return att_output

    def multiview_fusion(self, global_image_embed, local_image_embed, patient_ids, batch_size):
        # obtain labels indicate corresponding multiview images
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(int)
        labels = torch.from_numpy(labels)
        labels.fill_diagonal_(0)

        # obtain image embed
        image_embed = torch.cat([global_image_embed.unsqueeze(1), local_image_embed], dim=1)
        image_embed = self.layer_norm_1(image_embed)
        # layer norm
        # image_embed = self.ln(image_embed)
        new_image_embed = []
        for i in range(batch_size):
            if labels[i].sum() == 0:
                new_image_embed.append(image_embed[i])
                continue
            multiview_image_embed = torch.cat([image_embed[j] for j, tag in enumerate(labels[i]) if tag == 1], dim=0)
            # include multiview images
            # cur_image_embed = self.cross_attention_no_parameters(q=image_embed[i], k_v=multiview_image_embed)
            cur_image_embed = self.multiview_cross_attention(image_embed[i], multiview_image_embed.detach(),
                                                             multiview_image_embed.detach())
            # cur_image_embed = self.multiview_cross_attention(image_embed[i], multiview_image_embed,
            #                                                  multiview_image_embed)
            # # add&norm
            cur_image_embed = self.layer_norm_2(cur_image_embed + image_embed[i])
            new_image_embed.append(cur_image_embed)
        new_image_embed = torch.stack(new_image_embed, dim=0)
        new_image_embed = self.visual_head(new_image_embed)
        return new_image_embed[:, 0, :], new_image_embed[:, 1:, :]

    def global_alignment_loss(self, global_image_embed, global_text_embed, patient_ids):
        # obtain multi-positive target
        patient_ids = patient_ids[:global_image_embed.shape[0]]
        labels = (patient_ids.reshape(-1, 1) == patient_ids.reshape(1, -1)).astype(int)
        labels = torch.from_numpy(labels).float().to(global_image_embed.device)
        del patient_ids

        one_view_idx = torch.argwhere(labels.sum(1) == 1).reshape(-1)
        mul_view_idx = torch.argwhere(labels.sum(1) != 1).reshape(-1)

        # normalize
        global_image_embed = F.normalize(global_image_embed, dim=-1, p=2)
        global_text_embed = F.normalize(global_text_embed, dim=-1, p=2)

        # calculate the similarity
        instance_sim = (global_image_embed @ global_text_embed.t()) / self.args['instance_temp']
        instance_sim_1 = (global_text_embed @ global_image_embed.t()) / self.args['instance_temp']

        global_loss = torch.tensor([0.0], requires_grad=True).to(global_image_embed.device)
        # one-view global alignment loss
        if len(one_view_idx) != 0:
            one_view_sim = instance_sim[one_view_idx]
            one_view_sim_1 = instance_sim_1[one_view_idx]
            one_view_label = labels[one_view_idx]

            one_view_loss = F.cross_entropy(one_view_sim, one_view_label, reduction='sum')
            one_view_loss_1 = F.cross_entropy(one_view_sim_1, one_view_label, reduction='sum')
            global_loss = global_loss + (one_view_loss_1 + one_view_loss) * 0.5
        if len(mul_view_idx) != 0:
            mul_view_sim = instance_sim[mul_view_idx]
            mul_view_sim_1 = instance_sim_1[mul_view_idx]
            mul_view_label = labels[mul_view_idx]

            # obtain the multiple positives contrastive loss
            # mul_view_global_loss = torch.tensor([0.0], requires_grad=True).to(global_image_embed.device)
            for i in range(len(mul_view_idx)):
                cur_mul_view_sim, cur_mul_view_sim_1 = mul_view_sim[i], mul_view_sim_1[i]
                cur_mul_view_label = mul_view_label[i]

                # obtain the mask of multiple positives
                positive_mask = torch.tensor([False] * len(cur_mul_view_sim))
                positive_idx = torch.argwhere(cur_mul_view_label != 0).reshape(-1)
                positive_mask[positive_idx] = True

                # cur_mul_view_sim
                # average the similarities among multiple positives
                positive_logit = cur_mul_view_sim[positive_mask].sum() / len(positive_idx)
                negative_logit = cur_mul_view_sim[~positive_mask]

                new_cur_logits = torch.cat([positive_logit.reshape(-1), negative_logit])
                new_cur_label = torch.LongTensor([0]).to(global_image_embed.device)
                cur_mul_view_loss = F.cross_entropy(new_cur_logits.reshape(1, -1), new_cur_label)

                # cur_mul_view_sim_1
                # average the similarities among multiple positives
                positive_logit = cur_mul_view_sim_1[positive_mask].sum() / len(positive_idx)
                negative_logit = cur_mul_view_sim_1[~positive_mask]

                new_cur_logits = torch.cat([positive_logit.reshape(-1), negative_logit])
                new_cur_label = torch.LongTensor([0]).to(global_image_embed.device)
                cur_mul_view_loss_1 = F.cross_entropy(new_cur_logits.reshape(1, -1), new_cur_label)

                # update the loss
                global_loss = global_loss + (cur_mul_view_loss_1 + cur_mul_view_loss) * 0.5

        global_loss = global_loss / global_image_embed.shape[0]

        return global_loss

    def local_text_token_alignment_loss(self, local_image_embed, local_text_embed):
        # cross-modal alignment between image patches and sentence embed in reports
        t_att_sim = local_text_embed @ local_image_embed.permute(0, 2, 1)
        t_att_sco = F.softmax(t_att_sim / math.sqrt(local_image_embed.shape[2]), dim=-1)
        t_att_output = torch.bmm(t_att_sco, local_image_embed)

        device = local_image_embed.device
        t_att_output = F.normalize(t_att_output, dim=-1, p=2)
        local_text_embed = F.normalize(local_text_embed, dim=-1, p=2)
        word_sim = torch.bmm(local_text_embed, t_att_output.permute(0, 2, 1)) / self.args['region_temp']
        word_sim_1 = rearrange(word_sim, "b n1 n2 -> (b n1) n2")  # the similarity between each word and each each
        word_targets = torch.arange(word_sim.shape[1]).long().repeat(word_sim.shape[0]).to(device)
        loss_word_1 = F.cross_entropy(word_sim_1, word_targets)

        word_sim_2 = rearrange(word_sim, "b n1 n2 -> (b n2) n1")
        loss_word_2 = F.cross_entropy(word_sim_2, word_targets)
        loss_word = (loss_word_2 + loss_word_1) / 2.0
        return loss_word

    def forward(self, images, radgraph_ids, radgraph_masks, patient_ids):
        # ====obtaining uni-modal embedding====
        # image embedding
        v_fc_feats, v_att_feats = self.visual_forward(images)  # fc_feats and att_feats are global and focal features

        mul_pos_loss = torch.tensor([0.0])
        if self.args['is_multiview_learning']:
            # calculate multiview-enhanced/guided contrastive learning among images
            mul_pos_loss = self.multi_pos_contra_images_v0404(v_fc_feats, patient_ids)
            # multiview fusion based on cross-attention
            v_fc_feats, v_att_feats = self.multiview_fusion(v_fc_feats, v_att_feats, patient_ids, radgraph_ids.shape[0])
        else:
            image_embed = torch.cat([v_fc_feats.unsqueeze(1), v_att_feats], dim=1)
            image_embed = self.layer_norm_1(image_embed)
            image_embed = self.visual_head(image_embed)
            v_fc_feats, v_att_feats = image_embed[:, 0, :], image_embed[:, 1:, :]

        # text embedding
        t_fc_feats, t_att_feats = self.obtain_text_embeds(radgraph_ids, radgraph_masks)

        # ====instance-level contrastive loss====
        instance_loss = self.global_alignment_loss(v_fc_feats, t_fc_feats, patient_ids)

        # ====sentence-level contrastive loss====
        # sen_image_loss, sen_text_loss = self.local_sentence_alignment_loss(v_att_feats, local_sen_embed_stacks)
        sen_text_loss = self.local_text_token_alignment_loss(v_att_feats, t_att_feats)

        sen_image_loss = torch.tensor([0.0])
        # sen_text_loss = torch.tensor([0.0])
        if self.args['is_multiview_learning']:
            return {
                'sen_image_loss': sen_image_loss,
                'sen_text_loss': sen_text_loss,
                'instance_loss': instance_loss,
                'multiview_loss': mul_pos_loss,
                'all_loss': instance_loss + sen_text_loss + mul_pos_loss
            }
        else:
            return {
                'sen_image_loss': sen_image_loss,
                'sen_text_loss': sen_text_loss,
                'instance_loss': instance_loss,
                'all_loss': instance_loss + sen_text_loss
            }