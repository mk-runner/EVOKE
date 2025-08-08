import torch
import functools
import torch.nn as nn
import torch.nn.functional as F

import tqdm
import transformers
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import AutoModelForCausalLM, AutoConfig, AutoModel
from transformers import GPT2Model, GPT2Config, GPT2LMHeadModel, EncoderDecoderModel
from transformers.models.bert_generation import BertGenerationConfig, BertGenerationDecoder, BertGenerationEncoder
from .beam_search import prepare_inputs_for_generation, _validate_model_kwargs, beam_search


class TextDecoderModel(nn.Module):
    """
    If proto is mentioned in decoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationDecoder model from decoder dict.
    """

    def __init__(self, config, tokenizer):
        super().__init__()
        if config['text_checkpoint'] is not None:
            dec_config = AutoConfig.from_pretrained(config['text_checkpoint'])
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            dec_config.vocab_size = config['vocab_size']
            dec_config.eos_token_id = tokenizer.token_to_id('[EOS]')
            dec_config.bos_token_id = tokenizer.token_to_id('[BOS]')
            dec_config.pad_token_id = tokenizer.token_to_id('[PAD]')
            dec_config.hidden_size = config['decoder_hidden_size']
            dec_config.num_hidden_layers = config['decoder_num_hidden_layers']
            dec_config.max_length = config['max_seq_len']
            dec_config.num_attention_heads = config['decoder_num_attention_heads']
            self.decoder = AutoModelForCausalLM.from_pretrained(config['text_checkpoint'], config=dec_config,
                                                                ignore_mismatched_sizes=True)
        else:
            dec_config = BertGenerationConfig()
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            dec_config.vocab_size = config['vocab_size']
            dec_config.eos_token_id = tokenizer.token_to_id('[EOS]')
            dec_config.bos_token_id = tokenizer.token_to_id('[BOS]')
            dec_config.pad_token_id = tokenizer.token_to_id('[PAD]')
            dec_config.hidden_size = config['decoder_hidden_size']
            dec_config.num_hidden_layers = config['decoder_num_hidden_layers']
            dec_config.max_length = config['max_seq_len']
            self.decoder = BertGenerationDecoder(dec_config)

        # Evaluation
        self.decoder.prepare_inputs_for_generation = functools.partial(prepare_inputs_for_generation, self.decoder)
        # We override _validate_model_kwargs width empty function because we add custom model kwargs that triggers
        # errors in original _validate_model_kwargs
        self.decoder._validate_model_kwargs = functools.partial(_validate_model_kwargs, self.decoder)

        # Inference
        self.generate = self.decoder.generate
        self.config = self.decoder.config
        self.beam_size = config['beam_size']

    def forward(self, input_ids, attention_mask, encoder_hidden_states=None, encoder_attention_mask=None, **kwargs):
        out = self.decoder(input_ids=input_ids,
                           attention_mask=attention_mask,
                           encoder_hidden_states=encoder_hidden_states,
                           encoder_attention_mask=encoder_attention_mask,
                           labels=input_ids,
                           **kwargs)
        return out.loss

    def evaluation(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, tokenizer):
        # We are in an ensembling scenario, we override huggingface beam-search function
        self.decoder.beam_search = functools.partial(beam_search, self.decoder)

        # Get tokenizer and reference sentences from dataloader
        max_len = self.config.max_length
        bos_token_id, eos_token_id, pad_token_id = tokenizer.token_to_id('[BOS]'), tokenizer.token_to_id(
            '[EOS]'), tokenizer.token_to_id('[PAD]')

        with torch.no_grad():
            batch_size = input_ids.shape[0]
            bos_input_ids = torch.ones((batch_size, 1), dtype=torch.long).to(input_ids) * bos_token_id
            expanded_return_idx = (
                torch.arange(batch_size).view(-1, 1).repeat(1, self.beam_size).view(-1).to(input_ids)
            )
            model_kwargs = {
                'encoders_outputs': [
                    {
                        "encoder_hidden_states": encoder_hidden_states.index_select(0, expanded_return_idx),
                        "encoder_attention_mask": encoder_attention_mask.index_select(0, expanded_return_idx)
                    }
                ],
                "hf_models": [self.decoder]
            }
            output = self.decoder.generate(
                input_ids=bos_input_ids,
                num_return_sequences=1,
                max_length=max_len,
                num_beams=self.beam_size,
                length_penalty=1.0,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                use_cache=True,
                **model_kwargs
            )
            gen_texts, gt_texts = [], []
            for pred, gt in zip(output.cpu().tolist(), input_ids.cpu().tolist()):
                gen_text = tokenizer.decode(pred)
                gt_text = tokenizer.decode(gt)
                gen_texts.append(gen_text)
                gt_texts.append(gt_text)
            gen_texts = [text if len(text) > 0 else "there is no evidence of pulmonary." for text in gen_texts]
            return [gen_texts, gt_texts]

    def __repr__(self):
        s = str(type(self.decoder).__name__) + '(' + str(self.decoder.config) + ')\n'
        return s


class TextEncoderModel(nn.Module):
    """
    If proto is mentioned in decoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationDecoder model from decoder dict.
    """

    def __init__(self, config, tokenizer):
        super().__init__()
        if config['text_checkpoint'] is not None:
            enc_config = AutoConfig.from_pretrained(config['text_checkpoint'])
            enc_config.vocab_size = config['vocab_size']
            enc_config.eos_token_id = tokenizer.token_to_id('[EOS]')
            enc_config.bos_token_id = tokenizer.token_to_id('[BOS]')
            enc_config.pad_token_id = tokenizer.token_to_id('[PAD]')
            enc_config.hidden_size = config['encoder_hidden_size']
            enc_config.num_hidden_layers = config['encoder_num_hidden_layers']
            enc_config.max_length = config['max_seq_len']
            self.encoder = AutoModel.from_pretrained(config['text_checkpoint'], config=enc_config,
                                                     ignore_mismatched_sizes=True)
        else:
            enc_config = BertGenerationConfig()
            enc_config.vocab_size = config['vocab_size']
            enc_config.eos_token_id = tokenizer.token_to_id('[EOS]')
            enc_config.bos_token_id = tokenizer.token_to_id('[BOS]')
            enc_config.pad_token_id = tokenizer.token_to_id('[PAD]')
            enc_config.hidden_size = config['encoder_hidden_size']
            enc_config.num_hidden_layers = config['encoder_num_hidden_layers']
            enc_config.max_length = config['max_seq_len']
            self.encoder = BertGenerationEncoder(enc_config)

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask,
                           **kwargs)[0]
        return out

    def __repr__(self):
        s = str(type(self.encoder).__name__) + '\n'
        return s


class DistilGPT2TextDecoderModel(nn.Module):
    """
    If proto is mentioned in decoder dict, loads pretrained models from proto strings.
    Otherwise, loads a GPT2LMHeadModel model from decoder dict.
    """

    def __init__(self, config, tokenizer):
        super().__init__()
        if config['text_checkpoint'] is not None:
            dec_config = GPT2Config.from_pretrained(config['text_checkpoint'])
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            dec_config.vocab_size = config['vocab_size']
            dec_config.eos_token_id = tokenizer.token_to_id('[EOS]')
            dec_config.bos_token_id = tokenizer.token_to_id('[BOS]')
            dec_config.pad_token_id = tokenizer.token_to_id('[PAD]')
            dec_config.hidden_size = config['decoder_hidden_size']
            dec_config.num_hidden_layers = config['decoder_num_hidden_layers']
            dec_config.max_length = config['max_seq_len']
            dec_config.num_attention_heads = config['decoder_num_attention_heads']
            decoder = GPT2LMHeadModel.from_pretrained(config['text_checkpoint'], config=dec_config,
                                                      ignore_mismatched_sizes=True)
        else:
            dec_config = GPT2Config()
            dec_config.is_decoder = True
            dec_config.add_cross_attention = True
            dec_config.vocab_size = config['vocab_size']
            dec_config.eos_token_id = tokenizer.token_to_id('[EOS]')
            dec_config.bos_token_id = tokenizer.token_to_id('[BOS]')
            dec_config.pad_token_id = tokenizer.token_to_id('[PAD]')
            dec_config.hidden_size = config['decoder_hidden_size']
            dec_config.num_hidden_layers = config['decoder_num_hidden_layers']
            dec_config.max_length = config['max_seq_len']
            decoder = GPT2LMHeadModel(dec_config)

        # We don't actually want to use the encoder of the EncoderDecoderModel, create a dummy encoder:
        class DummyEncoder:
            main_input_name = 'dummy'

            class DummyConfig(PretrainedConfig):
                model_type = 'bert'

            config = DummyConfig()

            def __init__(self, hidden_size):
                self.config.hidden_size = hidden_size

            def get_output_embeddings(cls):
                return None

        # Use Hugging Face Transformers EncoderDecoderModel to generate conditionally:
        dummy_encoder = DummyEncoder(hidden_size=decoder.config.hidden_size)

        # To be compatible with previous the framework (and hence, the available checkpoint):
        class Decoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_decoder = EncoderDecoderModel(encoder=dummy_encoder, decoder=decoder)

        self.decoder = Decoder()

        # # Evaluation
        # self.decoder.prepare_inputs_for_generation = functools.partial(prepare_inputs_for_generation, self.decoder)
        # # We override _validate_model_kwargs width empty function because we add custom model kwargs that triggers
        # # errors in original _validate_model_kwargs
        # self.decoder._validate_model_kwargs = functools.partial(_validate_model_kwargs, self.decoder)

        # Inference
        self.pad_token_id = tokenizer.token_to_id('[PAD]')
        self.eos_token_id = tokenizer.token_to_id('[EOS]')
        self.bos_token_id = tokenizer.token_to_id('[BOS]')
        # self.generate = self.decoder.generate
        # self.config = self.decoder.config
        self.beam_size = config['beam_size']
        self.max_seq_len = config['max_seq_len']

    def forward(self, encoder_hidden_states, encoder_attention_mask, input_ids=None, attention_mask=None, stage='train'):
        assert stage in ['train', 'test']
        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=encoder_hidden_states,
                                                                        attention=encoder_attention_mask)
        if stage == 'train':
            # Teacher forcing: labels are given as input
            # labels should be [-100, 0, ...]
            outputs = self.decoder.encoder_decoder(
                decoder_input_ids=input_ids,
                decoder_attention_mask=attention_mask,
                encoder_outputs=encoder_outputs,
                return_dict=True,
            )

            # Loss:
            loss = F.cross_entropy(
                outputs.logits.permute([0, 2, 1]), input_ids, ignore_index=self.pad_token_id,
            )
            return loss
        else:
            output = self.generate(encoder_outputs)
            return output

    def generate(self, encoder_outputs):
        """
        generate reports in an autoregressive manner.

        Argument/s:
            encoder_outputs: transformers.modeling_outputs.BaseModelOutput of image encoder

        Returns:
            Indices of the tokens for the predicted sequence.
        """

        outputs = self.decoder.encoder_decoder.generate(
            max_length=self.max_seq_len,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            num_beams=self.beam_size,
            return_dict_in_generate=True,
            use_cache=True,
            encoder_outputs=encoder_outputs,
        )

        return outputs['sequences']

    # old ways
    def evaluation(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask, tokenizer):
        # We are in an ensembling scenario, we override huggingface beam-search function
        self.decoder.beam_search = functools.partial(beam_search, self.decoder)

        # Get tokenizer and reference sentences from dataloader
        max_len = self.config.max_length
        bos_token_id, eos_token_id, pad_token_id = tokenizer.token_to_id('[BOS]'), tokenizer.token_to_id(
            '[EOS]'), tokenizer.token_to_id('[PAD]')

        with torch.no_grad():
            batch_size = input_ids.shape[0]
            bos_input_ids = torch.ones((batch_size, 1), dtype=torch.long).to(input_ids) * bos_token_id
            expanded_return_idx = (
                torch.arange(batch_size).view(-1, 1).repeat(1, self.beam_size).view(-1).to(input_ids)
            )
            model_kwargs = {
                'encoders_outputs': [
                    {
                        "encoder_hidden_states": encoder_hidden_states.index_select(0, expanded_return_idx),
                        "encoder_attention_mask": encoder_attention_mask.index_select(0, expanded_return_idx)
                    }
                ],
                "hf_models": [self.decoder]
            }
            output = self.decoder.generate(
                input_ids=bos_input_ids,
                num_return_sequences=1,
                max_length=max_len,
                num_beams=self.beam_size,
                length_penalty=1.0,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                use_cache=True,
                **model_kwargs
            )
            gen_texts, gt_texts = [], []
            for pred, gt in zip(output.cpu().tolist(), input_ids.cpu().tolist()):
                gen_text = tokenizer.decode(pred)
                gt_text = tokenizer.decode(gt)
                gen_texts.append(gen_text)
                gt_texts.append(gt_text)
            gen_texts = [text if len(text) > 0 else "there is no evidence of pulmonary." for text in gen_texts]
            return [gen_texts, gt_texts]

    def __repr__(self):
        s = str(type(self.decoder).__name__) + '(' + str(self.decoder.config) + ')\n'
        return s


class DistilGPT2TextEncoderModel(nn.Module):
    """
    If proto is mentioned in decoder dict, loads pretrained models from proto strings.
    Otherwise, loads a BertGenerationDecoder model from decoder dict.
    """

    def __init__(self, config, tokenizer):
        super().__init__()
        if config['text_checkpoint'] is not None:
            enc_config = GPT2Config.from_pretrained(config['text_checkpoint'])
            enc_config.vocab_size = config['vocab_size']
            enc_config.eos_token_id = tokenizer.token_to_id('[EOS]')
            enc_config.bos_token_id = tokenizer.token_to_id('[BOS]')
            enc_config.pad_token_id = tokenizer.token_to_id('[PAD]')
            enc_config.hidden_size = config['encoder_hidden_size']
            # enc_config.num_hidden_layers = config['encoder_num_hidden_layers']
            enc_config.max_length = config['max_seq_len']
            self.encoder = GPT2Model.from_pretrained(config['text_checkpoint'], config=enc_config,
                                                     ignore_mismatched_sizes=True)
        else:
            enc_config = GPT2Config()
            enc_config.vocab_size = config['vocab_size']
            enc_config.eos_token_id = tokenizer.token_to_id('[EOS]')
            enc_config.bos_token_id = tokenizer.token_to_id('[BOS]')
            enc_config.pad_token_id = tokenizer.token_to_id('[PAD]')
            enc_config.hidden_size = config['encoder_hidden_size']
            # enc_config.num_hidden_layers = config['encoder_num_hidden_layers']
            enc_config.max_length = config['max_seq_len']
            self.encoder = GPT2Model(enc_config)

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask,
                           **kwargs)[0]
        return out

    def __repr__(self):
        s = str(type(self.encoder).__name__) + '\n'
        return s
