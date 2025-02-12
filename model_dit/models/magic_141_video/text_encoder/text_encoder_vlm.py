from dataclasses import dataclass
from typing import Optional, Tuple
from copy import deepcopy
import numpy as np

import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer, AutoTokenizer, AutoModel, AutoProcessor
from transformers.utils import ModelOutput
from transformers import LlavaForConditionalGeneration

# from ..constants import TEXT_ENCODER_PATH, TOKENIZER_PATH
# from ..constants import PRECISION_TO_TYPE
# from ..utils.token_helper import find_subsequence, multi_slice_to_mask
from PIL import Image

def use_default(value, default):
    return value if value is not None else default


import torch
import torch.nn.functional as F

def find_subsequence(sequence, sub_sequence):

    assert sequence.shape[0]==1
    sequence = sequence[0]
    sub_sequence = sub_sequence[0]

    sub_len = len(sub_sequence)
    indices = []
        
    windows = sequence.unfold(0, sub_len, 1)
    matches = (windows == sub_sequence).all(dim=1)
    indices = matches.nonzero().flatten().tolist()

    return indices, len(indices), sub_len

import ast
import torch

def multi_slice_to_mask(expr, length):
    def process_single_slice(s):
        s = s.replace(':', ',').replace(' ', '')
        while ',,' in s:
            s = s.replace(',,', ',None,')
        if s.startswith(','):
            s = 'None' + s
        if s.endswith(','):
            s = s + 'None'
        return s
    
    try:
        slices = expr.split(',')
        mask = torch.zeros(length, dtype=torch.bool)
        if expr == "":
            return mask
        i = 0
        while i < len(slices):
            if ':' in slices[i]:
                slice_expr = process_single_slice(slices[i])
                slice_args = ast.literal_eval(f"({slice_expr})")
                s = slice(*slice_args)
                mask[s] = True
                i += 1
            else:
                idx = ast.literal_eval(slices[i])
                if idx < 0:
                    idx = length + idx
                if 0 <= idx < length:
                    mask[idx] = True
                i += 1
                
        return mask
    except Exception as e:
        raise ValueError(f"Invalid slice expression: {e}")
PROMPT_TEMPLATE_ENCODE = (
    "<|start_header_id|>user<|end_header_id|>\n\n<image>\n,A Image of {},Describe the image by detailing the color, shape, size, texture, "
    "quantity, text, spatial relationships of the objects and background:<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
) 
PROMPT_TEMPLATE = {
    "dit-llm-encode": {
        "template": PROMPT_TEMPLATE_ENCODE,
    }
}

def load_text_encoder(
    text_encoder_type,
    text_encoder_precision=None,
    text_encoder_path=None,
    logger=None,
    device=None,
    dtype=None,
    quantization_config=None,
):
    # if text_encoder_path is None:
    #     text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]
    if logger is not None:
        logger.info(
            f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}"
        )

    if text_encoder_type == "clipL":
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
    elif text_encoder_type == "llm":
        text_encoder = AutoModel.from_pretrained(
            text_encoder_path, 
            low_cpu_mem_usage=True,
            quantization_config=quantization_config
        )
        text_encoder.final_layer_norm = text_encoder.norm
    elif text_encoder_type == "vlm": #add vlm support for image-prompt video gen 
        text_encoder = LlavaForConditionalGeneration.from_pretrained(
            text_encoder_path, 
            low_cpu_mem_usage=True,
            quantization_config=quantization_config
        )
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
    # from_pretrained will ensure that the model is in eval mode.

    if text_encoder_precision is not None and quantization_config is None and dtype != torch.float8_e4m3fn:
        text_encoder = text_encoder.to(dtype)

    text_encoder.requires_grad_(False)

    if logger is not None:
        logger.info(f"Text encoder to dtype: {dtype}")

    if device is not None and quantization_config is None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path


def load_tokenizer(
    tokenizer_type, tokenizer_path=None, padding_side="right", logger=None
):
    # if tokenizer_path is None:
    #     tokenizer_path = TOKENIZER_PATH[tokenizer_type]
    if logger is not None:
        logger.info(f"Loading tokenizer ({tokenizer_type}) from: {tokenizer_path}")

    if tokenizer_type == "clipL":
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=77)
    elif tokenizer_type == "llm" or tokenizer_type == "vlm":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, padding_side=padding_side
        )
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    return tokenizer, tokenizer_path


@dataclass
class TextEncoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
        hidden_states_list (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        text_outputs (`list`, *optional*, returned when `return_texts=True` is passed):
            List of decoded texts.
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_outputs: Optional[list] = None


class TextEncoderVLM(nn.Module):
    def __init__(
        self,
        text_encoder_type: str,
        max_length: int,
        text_encoder_precision: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        tokenizer_type: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        output_key: Optional[str] = None,
        use_attention_mask: bool = True,
        input_max_length: Optional[int] = None,
        prompt_template: Optional[dict] = PROMPT_TEMPLATE["dit-llm-encode"],
        prompt_template_video: Optional[dict] = None,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        reproduce: bool = False,
        logger=None,
        device=None,
    ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = (
            tokenizer_type if tokenizer_type is not None else text_encoder_type
        )
        self.tokenizer_path = (
            tokenizer_path if tokenizer_path is not None else text_encoder_path
        )
        self.use_attention_mask = use_attention_mask
        
        self.input_max_length = (
            input_max_length if input_max_length is not None else max_length
        )
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.reproduce = reproduce
        self.logger = logger
        self.is_fp8 = False
        self.processor = None
        self.prompt_template = prompt_template
        self.prompt_template_video = prompt_template_video

        self.use_template = self.prompt_template is not None
        if self.use_template:
            assert (
                isinstance(self.prompt_template, dict)
                and "template" in self.prompt_template
            ), f"`prompt_template` must be a dictionary with a key 'template', got {self.prompt_template}"
            assert "{}" in str(self.prompt_template["template"]), (
                "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template['template']}"
            )

        self.use_video_template = self.prompt_template_video is not None
        if self.use_video_template:
            if self.prompt_template_video is not None:
                assert (
                    isinstance(self.prompt_template_video, dict)
                    and "template" in self.prompt_template_video
                ), f"`prompt_template_video` must be a dictionary with a key 'template', got {self.prompt_template_video}"
            assert "{}" in str(self.prompt_template_video["template"]), (
                "`prompt_template_video['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template_video['template']}"
            )


        if "t5" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        elif "clip" in text_encoder_type:
            self.output_key = output_key or "pooler_output"
        elif "llm" in text_encoder_type or "glm" in text_encoder_type or "vlm" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
            if "glm" in text_encoder_type or "vlm" in text_encoder_type:
                self.processor = AutoProcessor.from_pretrained(text_encoder_path, device=device)
        else:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        self.model, self.model_path = load_text_encoder(
            text_encoder_type=self.text_encoder_type,
            text_encoder_precision=self.precision,
            text_encoder_path=self.model_path,
            logger=self.logger,
            device=device,
            # dtype=dtype,
            # quantization_config=quantization_config,
        )
        self.dtype = self.model.dtype
        self.device = self.model.device

        self.tokenizer, self.tokenizer_path = load_tokenizer(
            tokenizer_type=self.tokenizer_type,
            tokenizer_path=self.tokenizer_path,
            padding_side="right",
            logger=self.logger,
        )

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """
        Apply text to template.

        Args:
            text (str): Input text.
            template (str or list): Template string or list of chat conversation.
            prevent_empty_text (bool): If Ture, we will prevent the user text from being empty
                by adding a space. Defaults to True.
        """
        if isinstance(template, str):
            # Will send string to tokenizer. Used for llm
            return template.format(text)
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    # def text2tokens(self, text, image1=None , clip_text_override=None): #not work, have some bugs, need further fix
    #     """
    #     Tokenize the input text.

    #     Args:
    #         text (str or list): Input text.
    #     """
    #     if self.text_encoder_type != "vlm" and image1 is not None:
    #         raise ValueError("Only vision_languague models support image input")
    #     tokenize_input_type = "str"
    #     prompt_template = self.prompt_template
    #     if prompt_template is not None and self.text_encoder_type == "llm" or self.text_encoder_type == "vlm":
    #         if isinstance(text, (list, tuple)):
    #             text = [
    #                 self.apply_text_to_template(one_text, prompt_template["template"])
    #                 for one_text in text
    #             ]
    #             if isinstance(text[0], list):
    #                 tokenize_input_type = "list"
    #         elif isinstance(text, str):
    #             text = self.apply_text_to_template(text, prompt_template["template"])
    #             if isinstance(text, list):
    #                 tokenize_input_type = "list"
    #         else:
    #             raise TypeError(f"Unsupported text type: {type(text)}")
    #     elif clip_text_override is not None and self.text_encoder_type == "clipL":
    #         text = clip_text_override

    #     kwargs = dict(
    #         truncation=True,
    #         max_length=self.max_length,
    #         padding="max_length" if self.text_encoder_type != "vlm" else "do_not_pad",
    #         return_tensors="pt",
    #     )
    #     if tokenize_input_type == "str":
    #         text_tokens = self.tokenizer(
    #             text,
    #             return_length=False,
    #             return_overflowing_tokens=False,
    #             return_attention_mask=True,
    #             **kwargs,
    #         )
    #         if self.text_encoder_type == "vlm":
    #             raw_images = []
    #             if image1 is not None:
    #                 raw_images.append(image1)#.squeeze(0)*255)
    #             # if image2 is not None:
    #             #     raw_images.append(image2.squeeze(0)*255)
    #             text_tokens = self.processor(
    #                 raw_images, 
    #                 text, 
    #                 **kwargs,
    #                 ).to(0, torch.float16)
    #         # import pdb; pdb.set_trace()
    #         return text_tokens #text_tokens
    #     elif tokenize_input_type == "list":
    #         return self.tokenizer.apply_chat_template(
    #             text,
    #             add_generation_prompt=True,
    #             tokenize=True,
    #             return_dict=True,
    #             **kwargs,
    #         )
    #     else:
    #         raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")

    def text2tokens(self, text_prompt, image1=None):
        # Only use text_prompt as text_prompt is same to template
        # text_prompt_formatted = self.prompt_template["template"].format(text_prompt)
        inputs = self.processor(text_prompt, image1, return_tensors='pt').to(0, torch.float16)
        return inputs

    def encode(
        self,
        batch_encoding,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=None,
        hidden_state_skip_layer=None,
        return_texts=False,
        image_token_selection_expr="::4",
        device=None,
    ):
        """
        Args:
            batch_encoding (dict): Batch encoding from tokenizer.
            use_attention_mask (bool): Whether to use attention mask. If None, use self.use_attention_mask.
                Defaults to None.
            output_hidden_states (bool): Whether to output hidden states. If False, return the value of
                self.output_key. If True, return the entire output. If set self.hidden_state_skip_layer,
                output_hidden_states will be set True. Defaults to False.
            do_sample (bool): Whether to sample from the model. Used for Decoder-Only LLMs. Defaults to None.
                When self.produce is False, do_sample is set to True by default.
            hidden_state_skip_layer (int): Number of hidden states to hidden_state_skip_layer. 0 means the last layer.
                If None, self.output_key will be used. Defaults to None.
            return_texts (bool): Whether to return the decoded texts. Defaults to False.
        """
        device = self.model.device if device is None else device
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(
            hidden_state_skip_layer, self.hidden_state_skip_layer
        )
        do_sample = use_default(do_sample, not self.reproduce)
        attention_mask = (
            batch_encoding["attention_mask"].to(device) if use_attention_mask else None
        )
        for k,v in batch_encoding.items():
            batch_encoding[k] = v.to(device) if isinstance(v, torch.Tensor) else v
        outputs = self.model(
            **batch_encoding,
            output_hidden_states=output_hidden_states
            or hidden_state_skip_layer is not None,
            # return_dict=True
        )
        if hidden_state_skip_layer is not None:
            last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
            # Real last hidden state already has layer norm applied. So here we only apply it
            # for intermediate layers.
            if hidden_state_skip_layer > 0 and self.apply_final_norm:
                last_hidden_state = self.model.final_layer_norm(last_hidden_state)
        else:
            last_hidden_state = outputs[self.output_key]
            
        prompt_template = self.prompt_template
        if prompt_template is not None and self.text_encoder_type == "vlm":
            user_start_tokens = self.tokenizer(
                text="<|start_header_id|>user<|end_header_id|>",
                add_special_tokens=False, 
                return_tensors="pt"
                )
            image_token = self.tokenizer(
                text="<image>",
                add_special_tokens=False, 
                return_tensors="pt"
                )
            image_token = image_token["input_ids"].to(device)
            user_start_tokens["input_ids"] = user_start_tokens["input_ids"].to(device)
            tk_idx, tk_n, tk_len = find_subsequence(batch_encoding["input_ids"], user_start_tokens["input_ids"])
            if tk_n != 1:
                raise ValueError("Template seems not in the required format, do you have <|start_header_id|>user<|end_header_id|> in place, and only one round of user input?")
            user_tokens = batch_encoding["input_ids"][:,tk_idx[0]+tk_len:]
            img_idx, img_n, _ = find_subsequence(user_tokens, image_token)
            img_seq_len=outputs["image_hidden_states"].shape[1]
            last_hidden_state = last_hidden_state[:, tk_idx[0]+tk_len:]
            # create image_mask to subset non-image hidden state
            seq_mask = torch.ones_like(last_hidden_state, device=device, dtype=torch.bool)
            img_mask=torch.zeros_like(outputs["image_hidden_states"][0:1], device=device, dtype=torch.bool)
            img_mask[:, multi_slice_to_mask(image_token_selection_expr, img_mask.shape[1])]=True
                
            drift=0  
            for i in img_idx:
                i = i+drift
                seq_mask[:,i:i+img_seq_len,:] = img_mask
                drift+=img_seq_len 
            
            last_hidden_state = last_hidden_state[seq_mask].view(1,-1,outputs["image_hidden_states"].shape[-1])

            attention_mask = torch.ones(last_hidden_state.shape[0], last_hidden_state.shape[1], device=device, dtype=torch.int64)

        elif prompt_template is None and self.text_encoder_type == "vlm":
            raise ValueError("Vlm encoders must use compatiable chat template.")

        if output_hidden_states:
            return TextEncoderModelOutput(
                last_hidden_state, attention_mask, outputs.hidden_states
            )
        return TextEncoderModelOutput(last_hidden_state, attention_mask)

    def forward(
        self,
        text,
        image1,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=False,
        hidden_state_skip_layer=0,
        return_texts=False,
    ):
        batch_encoding = self.text2tokens(text, image1)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            do_sample=do_sample,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts,
        )
LLAVA_LLAMA_3_8B_PROMPT_TEMPLATE_LI_DIT_ENCODE_VIDEO = (
    "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
    "1. The main content and theme of the video."
    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
    "4. Behavior, facial expressions, emotion, speaking style, eye gaze, head orientation, body orientation, body gesture, hand gesture changes of the main character." \
    "5. background environment, light, style and atmosphere."
    "6. camera angles, movements, and transitions used in the video:<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
)
LLAVA_LLAMA_3_8B_HUMAN_IMAGE_PROMPT = LLAVA_LLAMA_3_8B_PROMPT_TEMPLATE_LI_DIT_ENCODE_VIDEO.format('<image>')
# text_encoder = TextEncoderVLM(
#     text_encoder_type="vlm",
#     max_length=512,
#     text_encoder_path="xtuner/llava-llama-3-8b-v1_1-transformers",
#     prompt_template=PROMPT_TEMPLATE["dit-llm-encode"]
# )

# text = ""
# image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
# import requests
# raw_image = Image.open(requests.get(image_file, stream=True).raw)
# model_id = "/mnt/weka/pretrained_weights/llava-llama-3-8b-v1_1-transformers"
# processor = AutoProcessor.from_pretrained(model_id)
# print(processor)
# import pdb; pdb.set_trace()


# raw_image = Image.open(requests.get(image_file, stream=True).raw)

# raw_img_arry = np.array(raw_image)
# raw_img_zero = np.zeros_like(raw_img_arry)
# raw_img = Image.fromarray(raw_img_zero)
# output = text_encoder.forward(text=LLAVA_LLAMA_3_8B_HUMAN_IMAGE_PROMPT, image1=raw_image)
# torch.save(output.hidden_state, "./llava_zeroimg_embed.pt")
# print(output.hidden_state.size())


# output = text_encoder.forward(text=[text], image1=raw_image)  # still have some bugs
# print(output.size())