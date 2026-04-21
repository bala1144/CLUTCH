import os
from typing import List, Union
import numpy as np
import math
import time
import heapq
import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
import random
from typing import Optional
from .tools.token_emb import NewTokenEmb


class MLM(nn.Module):

    def __init__(
        self,
        model_path: str,
        model_type: str = "t5",
        stage: str = "lm_pretrain",
        new_token_type: str = "insert",
        motion_codebook_size: int = 512,
        framerate: float = 20.0,
        down_t: int = 4,
        predict_ratio: float = 0.2,
        max_length: int = 256,
        lora: bool = False,
        quota_ratio: float = 0.5,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        fine_tune_last_n_layers: int = None,
        motion_ss_rate: int = 3, ## added by the bala to handle the motion subsampling
        supervision_type: str= "default", 
        vqvae_type: str = "standard", # standard, single_hand, single_hand_decomp
        **kwargs,
    ) -> None:

        super().__init__()

        # Parameters
        self.m_codebook_size = motion_codebook_size
        self.max_length = max_length
        self.framerate = framerate
        self.down_t = down_t
        self.motion_ss_rate = motion_ss_rate
        self.predict_ratio = predict_ratio
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.quota_ratio = quota_ratio
        self.stage = stage
        self.supervision_type = supervision_type
        self.vqvae_type = vqvae_type

        if supervision_type == "mixed":
            self.supervision_options = ['text', 'motion', 'supervised', 'supervised', 'supervised']
            # self.supervision_options = ['text', 'motion']
        else:
            self.supervision_options = ['supervised', 'supervised', 'supervised']
        
        # Instantiate language model
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path, token=token,legacy=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,legacy=True)
        if model_type == "t5":
            self.language_model = T5ForConditionalGeneration.from_pretrained(
                model_path)
            self.lm_type = 'encdec'
        elif model_type == "gpt2":
            self.language_model = GPT2LMHeadModel.from_pretrained(model_path)
            self.lm_type = 'dec'
        elif model_type == "qwen":
            from transformers import AutoModelForCausalLM
            self.language_model = AutoModelForCausalLM.from_pretrained(model_path)
            self.lm_type = 'dec'
        else:
            raise ValueError("type must be either seq2seq or conditional")

        self.get_trainable_params()
        if fine_tune_last_n_layers is not None:

            if model_type == "t5":
                self.language_model = self.fine_tune_last_n_layers_t5(self.language_model, fine_tune_last_n_layers)  
            elif model_type == "gpt2":
                self.language_model = self.fine_tune_last_n_layers(fine_tune_last_n_layers)  
            
            trainable_layers = [".".join(name.split(".")[:4]) for name, p in self.named_parameters()  if p.requires_grad]
            print("\n*************************************************************")
            print(f"Number of trainable layers in the LM: {len(set(trainable_layers))}")
            print("\n".join(set(trainable_layers)))
            print("************************************************************\n")
            
        if self.lm_type == 'dec':
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add motion tokens
        self.tokenizer.add_tokens(
            [f'<motion_id_{i}>' for i in range(self.m_codebook_size + 3)])

        if new_token_type == "insert":
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        elif new_token_type == "mlp":
            shared = NewTokenEmb(self.language_model.shared,
                                 self.m_codebook_size + 3)
            # lm_head = NewTokenEmb(self.language_model.lm_head,
            #   self.m_codebook_size + 3)
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            self.language_model.shared = shared
            # self.language_model.lm_head = lm_head

        # Lora
        if lora:
            from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
            from peft.utils.other import fsdp_auto_wrap_policy
            peft_config = LoraConfig(
                bias="none",
                task_type="CAUSAL_LM",
                #  inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05)
            self.language_model = get_peft_model(self.language_model,
                                                 peft_config)
            
    
    def get_trainable_params(self, text="complete trainable model"):
        """
        Returns the trainable parameters of the model.
        Prints the total number of trainable parameters in human-readable format.
        """
        # Filter for trainable parameters only
        trainable_params = [p for name, p in self.named_parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in trainable_params)
        
        # Format the number of parameters into a human-readable format
        if total_params >= 1e6:
            readable_format = f"{total_params / 1e6:.2f}M"
        elif total_params >= 1e3:
            readable_format = f"{total_params / 1e3:.2f}K"
        else:
            readable_format = str(total_params)
        
        print("\n*************************************************************")
        print(f"Number of trainable parameters in the LM ({text}): {readable_format}")
        print("************************************************************\n")
        
        return trainable_params

    
    def fine_tune_last_n_layers(self, n):
        """
        Freezes all layers in the model except for the last n layers.

        Parameters:
        - model: The model to modify (e.g., GPT2Model).
        - n: Number of layers from the end to keep trainable.

        Returns:
        - model: The modified model with only the last n layers trainable.
        """
        # Freeze all parameters by default
        for param in self.language_model.parameters():
            param.requires_grad = False

        # Get all the transformer layers
        if hasattr(self.language_model, "transformer"):
            transformer = self.language_model.transformer
        elif hasattr(self.language_model, "encoder"):
            transformer = self.language_model.encoder
        else:
            raise ValueError("self.language_model architecture not recognized. Transformer layers not found.")

        layers = transformer.h  # GPT-2 uses 'h' for its transformer blocks

        # Unfreeze the last n layers
        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Unfreeze the output layer and final layer normalization (if any)
        if hasattr(transformer, "ln_f"):  # GPT-2 has ln_f
            for param in transformer.ln_f.parameters():
                param.requires_grad = True

        if hasattr(transformer, "lm_head"):  # Unfreeze lm_head if present
            for param in transformer.lm_head.parameters():
                param.requires_grad = True

        self.get_trainable_params(f"Modified model only the last {n} layers trainable")

        return self.language_model

    def fine_tune_last_n_layers_t5(self, model, n):
        """
        Freezes all layers in the T5 model except for the last n layers in the encoder and decoder.

        Parameters:
        - model: The T5 model (e.g., T5ForConditionalGeneration).
        - n: Number of layers from the end of the encoder and decoder to keep trainable.

        Returns:
        - model: The modified model with only the last n layers trainable.
        """
        # Freeze all parameters by default
        for param in model.parameters():
            param.requires_grad = False

        if n > 0:
            # Unfreeze the last n layers of the encoder
            for layer in model.encoder.block[:n]:
                for param in layer.parameters():
                    param.requires_grad = True

            # Unfreeze the last n layers of the decoder
            for layer in model.decoder.block[-n:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Unfreeze the final layer normalization in both encoder and decoder
        for param in model.encoder.final_layer_norm.parameters():
            param.requires_grad = True
        for param in model.decoder.final_layer_norm.parameters():
            param.requires_grad = True

        # Unfreeze the output layer
        if hasattr(model, "lm_head"):
            for param in model.lm_head.parameters():
                param.requires_grad = True
        
        self.get_trainable_params(f"Modified model only the last {n} layers trainable")

        return model


    def forward(self, texts: List[str], motion_tokens: Tensor,
                lengths: List[int], tasks: dict):
        if self.lm_type == 'encdec':
            return self.forward_encdec(texts, motion_tokens, lengths, tasks)
        elif self.lm_type == 'dec':
            return self.forward_dec(texts, motion_tokens, lengths, tasks)
        else:
            raise NotImplementedError("Only conditional_multitask supported")

    def forward_encdec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
    ):

        # Tensor to string
        motion_strings = self.motion_token_to_string(motion_tokens, lengths)
        """
        motion_tokens
        tensor(
        [[10.,  0., 24.,  1.,  1., 20.,  1.,  9.,  9.,  4.],
        [10.,  0., 24.,  1.,  1., 20.,  1.,  9.,  9.,  4.]], device='cuda:0')
        """

        # Supervised or unsupervised
        # condition = random.choice(['text', 'motion'])
        # condition = random.choice(['text', 'motion', 'supervised', 'supervised', 'supervised'])
        condition = random.choice(self.supervision_options)
        # condition = random.choice(['supervised', 'supervised', 'supervised'])
        ### needs to be debug
        if condition == 'text':
            inputs = texts
            outputs = texts
        elif condition == 'motion':
            inputs = motion_strings
            outputs = motion_strings
        else:
            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)
            
            """
            Inputs:

            ['"The person play the flute"', 
            '<motion_id_32> <motion_id_10><motion_id_0><motion_id_24><motion_id_1><motion_id_1><mot...9><motion_id_9><motion_id_4> <motion_id_33>']
            
            tasks:
            
            [
            {'class': 't2m', 'input': ['<Caption_Placeholder>'], 'output': ['<Motion_Placeholder>']},
            {'class': 'm2t', 'input': ['<Motion_Placeholder>'], 'output': ['<Caption_Placeholder>']}
            ]

            Outputs:
            0 =
            '<motion_id_32><motion_id_10><motion_id_0><motion_id_24><motion_id_1><motion_id_1><motion_id_20><motion_id_1><motion_id_9><motion_id_9><motion_id_4><motion_id_33>'
            1 ='"The person play the flute"'

            """

        source_encoding = self.tokenizer(inputs,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")
        
        """
        source_encoding :dict { inputs_ids, attention }
        """

        source_attention_mask = source_encoding.attention_mask.to(motion_tokens.device)
        source_input_ids = source_encoding.input_ids.to(motion_tokens.device)

        """
        source_input_ids : 2, 256 (max length)
        
        source_input_ids [0]
        tensor([   96,   634,   568,   577,     8, 27928,   121,     1,     0,
        
        source_input_ids [1]
        tensor([32132, 32110, 32100, 32124, 32101, 32101, 32120, 32101, 32109, 32109,
        32104, 32133, 
        """

        #### creating target ID's
        if condition in ['text', 'motion']:
            batch_size, expandend_input_length = source_input_ids.shape
            mask_indices = np.asarray([
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ])
            target_mask = ~mask_indices
            input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
            target_sentinel = self.create_sentinel_ids(target_mask.astype(np.int8))

            labels_input_ids = self.filter_input_ids(source_input_ids, target_sentinel).to(motion_tokens.device)
            source_input_ids = self.filter_input_ids(source_input_ids, input_ids_sentinel).to(motion_tokens.device)

            ### if self supervision no need to generate

        else:
            target_inputs = self.tokenizer(outputs,
                                           padding='max_length',
                                           max_length=self.max_length,
                                           truncation=True,
                                           return_attention_mask=True,
                                           add_special_tokens=True,
                                           return_tensors="pt")

            labels_input_ids = target_inputs.input_ids.to(motion_tokens.device)
            lables_attention_mask = target_inputs.attention_mask.to(
                motion_tokens.device)

        labels_input_ids[labels_input_ids == 0] = -100
        outputs = self.language_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask
            if condition == 'supervised' else None,labels=labels_input_ids,
            decoder_attention_mask=lables_attention_mask if condition == 'supervised' else None,
        )

        """
        outputs.keys = dict_keys(['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state'])
        for k, v in outputs.items():
        
            if torch.is_tensor(v):
                print(k, v.shape)
            else:
                print(k, len(v))

        ###
        loss torch.Size([])
        logits torch.Size([1, 256, 32135])
        past_key_values 8
        encoder_last_hidden_state torch.Size([1, 256, 512])
    
        """

        # print()
        # print("************** Forward Encdec **************")
        # print("Input")
        # print(inputs[0])
        # print()
        
        # print("Output")
        # print(outputs[0])

        return outputs

    def forward_dec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
    ):
        self.tokenizer.padding_side = "right"

        # Tensor to string
        motion_strings = self.motion_token_to_string(motion_tokens, lengths)

        # Supervised or unsupervised
        condition = random.choice(
            ['text', 'motion', 'supervised', 'supervised', 'supervised'])

        if condition == 'text':
            labels = texts
        elif condition == 'motion':
            labels = motion_strings
        else:
            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)
            labels = []
            for i in range(len(inputs)):
                labels.append(inputs[i] + ' \n ' + outputs[i] +
                              self.tokenizer.eos_token)

        # Tokenize
        inputs = self.tokenizer(labels,
                                padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors="pt")

        labels_input_ids = inputs.input_ids.to(motion_tokens.device)
        lables_attention_mask = inputs.attention_mask.to(motion_tokens.device)
        outputs = self.language_model(input_ids=labels_input_ids,
                                      attention_mask=lables_attention_mask,
                                      labels=inputs["input_ids"])

        return outputs

    def generate_direct(self,
                        texts: List[str],
                        max_length: int = 256,
                        num_beams: int = 1,
                        do_sample: bool = True,
                        bad_words_ids: List[int] = None):

        # Device
        self.device = self.language_model.device

        # Tokenize
        if self.lm_type == 'dec':
            texts = [text + " \n " for text in texts]

        # texts
        # ['Generate motion: "The person pass the elephant"', 'Generate motion: "The person pass the pyramidlarge"']
        # source_encoding =dict_keys(['input_ids', 'attention_mask'])
        source_encoding = self.tokenizer(texts,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_input_ids = source_encoding.input_ids.to(self.device) # source_input_ids.shape torch.Size([2, 256])
        source_attention_mask = source_encoding.attention_mask.to(self.device) # torch.Size([2, 256])

        if self.lm_type == 'encdec':
            outputs = self.language_model.generate(
                source_input_ids,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                bad_words_ids=bad_words_ids,
            ) # outputs.shape == [2, 21]
        elif self.lm_type == 'dec':
            outputs = self.language_model.generate(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=do_sample,
                max_new_tokens=max_length)
            self.tokenizer.padding_side = 'left'

        outputs_string = self.tokenizer.batch_decode(outputs,
                                                     skip_special_tokens=True) 

        ### [
        # '<motion_id_31>land<motion_id_31>.<motion_id_28><motion_id_33> to the animal<motion_id_29>', 
        # 'A man in a purple suit passes<motion_id_29><motion_id_34><motion_id_34> pyramids<motion_id_30>it through the sky.']

        ### this is the issue string
        outputs_tokens, cleaned_text = self.motion_string_to_token(
            outputs_string)
        
        ### cleaned_text
        # ['<motion_id_31>land<motion_id_31>.<motion_id_28><motion_id_33> to the animal<motion_id_29>', 'A man in a purple suit passes<motion_id_29><motion_id_34><motion_id_34> pyramids<motion_id_30>it through the sky.']
        # print()
        # print("LM : Inputs")
        # print(texts[0])

        # print()
        # print("LM: generated outputs_tokens")
        # print(outputs_tokens[0])
        # print()

        # print("LM: cleaned_text")
        # print(cleaned_text[0])
        # print()

        return outputs_tokens, cleaned_text

    def generate_conditional(self,
                             texts: Optional[List[str]] = None,
                             motion_tokens: Optional[Tensor] = None,
                             lengths: Optional[List[int]] = None,
                             task: str = "t2m",
                             with_len: bool = False,
                             stage: str = 'train',
                             tasks: dict = None):

        self.device = self.language_model.device

        if task == "t2m":
            assert texts is not None
            motion_strings = [''] * len(texts)
            if not with_len:
                if tasks is None:
                    tasks = [{
                        'input':
                        ['Generate motion: <Caption_Placeholder>'],
                        'output': ['']
                    }] * len(texts)

                lengths = [0] * len(texts)
            else:
                tasks = [{
                    'input': [
                        'Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>'
                    ],
                    'output': ['']
                }] * len(texts)

            inputs, outputs = self.template_fulfill(tasks, lengths, motion_strings, texts, stage)

            outputs_tokens, cleaned_text = self.generate_direct(inputs,
                                                                max_length=self.max_length,
                                                                num_beams=1,
                                                                do_sample=True)
            
            # print()
            # print(f"************** Validation/Test Generate Conditional task: {task} **************")
            # print("Input")
            # print(inputs[0])
            # print()
            
            # print("Output")
            # print(outputs[0])

            return outputs_tokens

        elif task == "m2t":
            assert motion_tokens is not None and lengths is not None

            motion_strings = self.motion_token_to_string(
                motion_tokens, lengths)

            if not with_len:
                tasks = [{
                    'input': ['Generate text: <Motion_Placeholder>'],
                    'output': ['']
                }] * len(lengths)
            else:
                tasks = [{
                    'input': [
                        'Generate text with <Frame_Placeholder> frames: <Motion_Placeholder>'
                    ],
                    'output': ['']
                }] * len(lengths)

            texts = [''] * len(lengths)

            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)
            outputs_tokens, cleaned_text = self.generate_direct(
                inputs,
                max_length=40,
                num_beams=1,
                do_sample=False,
                # bad_words_ids=self.bad_words_ids
            )

            # print()
            # print("************** Validation/Test Generate Conditional M2T **************")
            # print(f"Input: {task}")
            # print(inputs[0])
            # print()
            
            # print("Output: ")
            # print(cleaned_text[0])

            return cleaned_text

    def motion_token_to_string(self, motion_token: Tensor, lengths: List[int]):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()[:lengths[i]]
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_token_list_to_string(self, motion_token: Tensor):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_string_to_token(self, motion_string: List[str]):
        motion_tokens = []
        output_string = []
        for i in range(len(motion_string)):
            string = self.get_middle_str(
                motion_string[i], f'<motion_id_{self.m_codebook_size}>',
                f'<motion_id_{self.m_codebook_size + 1}>')
            string_list = string.split('><')
            
            # Modified list comprehension with error handling
            token_list = []
            for s in string_list[1:-1]:
                try:
                    # Extract all digits from the string
                    digit_str = ''.join(filter(str.isdigit, s.split('_')[-1]))
                    if digit_str:
                        token_list.append(int(digit_str))
                except ValueError as e:
                    print(f"Warning: Could not convert '{s}' to int. Skipping.")
            
            if len(token_list) == 0:
                token_list = [0]
            token_list_padded = torch.tensor(token_list, dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(
                string, '<Motion_Placeholder>'))

        return motion_tokens, output_string


    def placeholder_fulfill(self, prompt: str, length: int, motion_string: str,
                            text: str, prompt_type: str = None):

        # if length = 48
        # seconds = math.floor(length / self.framerate)
        # token_length = length / self.down_t ### 12

        token_length = length
        seconds = 10 # hard-coded since, only experiment with the token of similar length
        ### padding is already implemented and I can handle it later

        motion_splited = motion_string.split('>') # length + start token + end_token
        if random.random() < self.quota_ratio:
            text = f'\"{text}\"'
        
        ## prompt = ''
        prompt = prompt.replace('<Caption_Placeholder>', text)
        prompt = prompt.replace('<Motion_Placeholder>',motion_string)
        prompt = prompt.replace('<Frame_Placeholder>', f'{length}')
        prompt = prompt.replace('<Second_Placeholder>', '%.1f' % seconds)
        ### standard model
        if self.vqvae_type != "standard": 
            prompt = self.hand_specific_prompting_fill(prompt, token_length, motion_splited)

        return prompt

    def hand_specific_prompting_fill(self, prompt, token_length, motion_splited ):
        
        """
        Added by Bala for hand masked training to make the model match the hand motions respectively
        """


        if all(x not in prompt for x in ["trg_hand", "src_hand", "Motion_Placeholder_masked_hand"]):
            return prompt

        # print("\n********************************")
        # print("hand_specific_prompting_fill")
        # print("Inprompt", prompt)
        # print("token_length", token_length)


        start_token = f'<motion_id_{self.m_codebook_size}>'
        end_token = f'><motion_id_{self.m_codebook_size+1}>'

        if motion_splited[-1]== "":
            motion_splited = motion_splited[:-1]
        motion_only_tokens = motion_splited[1:-1]
        # print("motion_only_tokens", len(motion_only_tokens))

        if self.vqvae_type == "single_hand":
            # num_time_step = (token_length-2) // 2
            assert len(motion_only_tokens) == token_length
            left_hand_idx = np.arange(0, token_length, 2)
            right_hand_idx = left_hand_idx+1

        elif self.vqvae_type == "single_hand_decomp":

            def interleave_a_b(a,b):
                interleaved = np.empty((a.size + b.size,), dtype=a.dtype)
                interleaved[0::2] = a
                interleaved[1::2] = b
                return interleaved

            assert len(motion_only_tokens) == token_length
            # left
            left_traj_idx = np.arange(0, token_length, 4)
            left_hp_idx = left_traj_idx+1
            left_hand_idx = interleave_a_b(left_traj_idx, left_hp_idx)
            # right
            right_hand_idx = left_hand_idx+2

        else:
            raise("enter valid model type")

        ## predicting left or right hand motion str
        left_hand_motion_str = '>'.join([motion_only_tokens[x] for x in left_hand_idx])
        right_hand_motion_str = '>'.join([motion_only_tokens[x] for x in right_hand_idx]) # motion_splited

        ## choose between the left and right hand
        if random.choice(["left", "right"]) == "left":
            src_hand = "left"
            trg_hand = "right"

            src_hand_motion = left_hand_motion_str
            trg_hand_motion = right_hand_motion_str
            trg_hand_idx = right_hand_idx
            
        else:
            src_hand = "right"
            trg_hand = "left"
            src_hand_motion = right_hand_motion_str
            trg_hand_motion = left_hand_motion_str
            trg_hand_idx = left_hand_idx


        prompt = prompt.replace('<src_hand>', src_hand) # used for pretrain task
        prompt = prompt.replace('<trg_hand>', trg_hand) # used for pretrain task

        ### adding the starting and end token
        src_hand_motion =  start_token + src_hand_motion + end_token
        trg_hand_motion =  start_token + trg_hand_motion + end_token
        prompt = prompt.replace('<Motion_Placeholder_src_hand>', src_hand_motion) # used for pretrain task
        prompt = prompt.replace('<Motion_Placeholder_trg_hand>', trg_hand_motion) # used for pretrain task

        if "<Motion_Placeholder_masked_hand>" in prompt:
            mask_token = f'<motion_id_{self.m_codebook_size+2}'
            ### create masked prompts
            masked_hand = motion_only_tokens
            for idx in trg_hand_idx:
                masked_hand[idx] = mask_token
            masked_hand_motion_str = '>'.join(masked_hand)
            masked_hand_motion_str= start_token + masked_hand_motion_str + end_token
            prompt = prompt.replace('<Motion_Placeholder_masked_hand>', masked_hand_motion_str)
        
        # print("Prompt", prompt)

        return prompt

    def template_fulfill(self,
                         tasks,
                         lengths,
                         motion_strings,
                         texts,
                         stage='test'):
        inputs = []
        outputs = []
        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            inputs.append(
                self.placeholder_fulfill(input_template, length,
                                         motion_strings[i], texts[i]))
            outputs.append(
                self.placeholder_fulfill(output_template, length,
                                         motion_strings[i], texts[i]))

        return inputs, outputs

    def get_middle_str(self, content, startStr, endStr):
        try:
            # startIndex = content.index(startStr)
            startIndex = content.rindex(startStr) 
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except:
            return f'<motion_id_{self.m_codebook_size}><motion_id_0><motion_id_{self.m_codebook_size+1}>'

        return f'<motion_id_{self.m_codebook_size}>' + content[
            startIndex:endIndex] + f'<motion_id_{self.m_codebook_size+1}>'

    def random_spans_noise_mask(self, length):
        # From https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(
            np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens,
                                                  num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens,
                                                     num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length, ), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

    def create_sentinel_ids(self, mask_indices):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        start_indices = mask_indices - np.roll(mask_indices, 1,
                                               axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0,
                                np.cumsum(start_indices, axis=-1),
                                start_indices)
        sentinel_ids = np.where(sentinel_ids != 0,
                                (len(self.tokenizer) - sentinel_ids - (self.m_codebook_size + 3)), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids,
                                  input_ids.to('cpu'))

        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape(
            (batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1),
                        self.tokenizer.eos_token_id,
                        dtype=np.int32),
            ],
            axis=-1,
        )

        # input_ids = torch.tensor(input_ids, device=self.device)
        input_ids = torch.tensor(input_ids)

        return input_ids
