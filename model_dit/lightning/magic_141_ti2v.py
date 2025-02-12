import os
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import torch
import copy
from omegaconf import OmegaConf
from diffusers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange, repeat
from lightning import LightningModule, seed_everything
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import DictConfig
from torch.nn import functional as F
from tqdm import tqdm

from model_dit.utils.loss import compute_snr, compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from model_dit.utils.util import instantiate
from model_dit.models.magic_141_video.text_encoder import TextEncoder
from model_dit.models.magic_141_video.text_encoder.text_encoder_vlm import TextEncoderVLM, LLAVA_LLAMA_3_8B_HUMAN_IMAGE_PROMPT
from model_dit.models.magic_141_video.constants import PROMPT_TEMPLATE

from model_dit.utils.ds_ema_bk import DSEma


@torch.no_grad()
def noise_inversion(model_infer,
                    noise_tensor,
                    encoder_hidden_states,
                    text_states,
                    text_mask,
                    text_states_2,
                    reference_image, 
                    cur_sigma_t,
                    device,
                    guidance_scale=3.0):
    reference_image = reference_image.repeat(1, 1, noise_tensor.shape[2], 1, 1)
    if guidance_scale != 1.0:
        _reference_image = reference_image[:1]
    else:
        _reference_image = reference_image 
    ori_noisy_latents = noisy_latents = noise_tensor * cur_sigma_t + (1 - cur_sigma_t) * _reference_image
    if guidance_scale != 1.0:
        noisy_latents = noisy_latents.repeat(2, 1, 1, 1, 1)
    else:
        noisy_latents = noisy_latents.repeat(1, 1, 1, 1, 1)
    ts = torch.tensor([cur_sigma_t * 1000] * noisy_latents.shape[0], device=device, dtype=torch.float32)
    print(f"{noisy_latents.shape=}")
    print(f"{ts.shape=}")
    print(f"{encoder_hidden_states.shape=}")
    print(f"{text_states.shape=}")
    print(f"{text_mask.shape=}")
    print(f"{text_states_2.shape=}")
    with torch.autocast(device_type="cuda", dtype=model_infer.dtype, enabled=True):
        print("==============>Text&Audio Cond Inversion=================")
        if guidance_scale != 1.0:
            noise_pred_uncond = model_infer(
                hidden_states=noisy_latents[:1].to(dtype=torch.float32),
                encoder_hidden_states=encoder_hidden_states[:1].to(dtype=torch.float32),
                text_states=text_states[:1].to(device=device, dtype=torch.float32),
                text_mask=text_mask[:1] if text_mask is not None else None,
                text_states_2=text_states_2[:1].to(device=device, dtype=torch.float32) if text_states_2 is not None else None,
                timestep=ts[:1],
                ).sample
            noise_pred_text = model_infer(
                hidden_states=noisy_latents[1:].to(dtype=torch.float32),
                encoder_hidden_states=encoder_hidden_states[1:].to(dtype=torch.float32),
                text_states=text_states[1:].to(device=device, dtype=torch.float32),
                text_mask=text_mask[1:] if text_mask is not None else None,
                text_states_2=text_states_2[1:].to(device=device, dtype=torch.float32) if text_states_2 is not None else None,
                timestep=ts[1:],
                ).sample
            noise_pred = noise_pred_uncond - guidance_scale * (noise_pred_text - noise_pred_uncond)
        else:
            noise_pred_text = model_infer(
                hidden_states=noisy_latents.to(dtype=torch.float32),
                encoder_hidden_states=encoder_hidden_states.to(dtype=torch.float32),
                text_states=text_states.to(device=device, dtype=torch.float32),
                text_mask=text_mask if text_mask is not None else None,
                text_states_2=text_states_2.to(device=device, dtype=torch.float32) if text_states_2 is not None else None,
                timestep=ts,
                ).sample
            noise_pred = noise_pred_text
        H,W = noise_pred_text.shape[-2:]
        initial_noise = ori_noisy_latents + noise_pred * (1 - cur_sigma_t)
    return initial_noise

# This LightningModule support rectify flow denoising
# During Training: reference image condition will be the first frame of current video
class EmoLitModule(LightningModule):
    def __init__(
        self,
        config: DictConfig,
        device=None,
    ) -> None:
        super().__init__()
        self.config = config

        self._get_scheduler()  # load scheduler
        self.to(device)
        self._init_model(device)
        # self.save_hyperparameters(config)  # save hyperparameters for resuming the training

    def on_train_start(self) -> None:
        # Get the rank of the current process after the trainer is attached
        rank = self.global_rank

        # Set a different seed for each process
        base_seed = self.config.seed
        seed = base_seed + rank
        seed_everything(seed)
        print(f"Seed set to {seed} for process {rank}")
    
    def _init_model(self, device):
        config = self.config
        model = instantiate(config.model.denoising_model, instantiate_module=False)
        model_additional_kwargs = config.model.get("model_additional_kwargs", {})
        if not isinstance(model_additional_kwargs, dict):
            model_additional_kwargs = OmegaConf.to_container(model_additional_kwargs)
        model_additional_kwargs["device"] = device if device is not None else "cpu"
        model_additional_kwargs["dtype"] = eval(config.model.dtype)

        self.model = model.from_pretrained(
            config.model.base_model_path,
            from_scratch=config.model.base_model_from_scratch,
            model_additional_kwargs=model_additional_kwargs,
        )
        self.model.requires_grad_(False)
        self.model.to(device="cpu", dtype=eval(config.model.dtype))
        
        # if "strategy" in config.trainer and "deep" in config.trainer.strategy:
        #     self.model.enable_gradient_checkpointing()

        # clip
        if config.model.clip_name is not None:
            pass  # TODO

        # VAE
        if config.model.vae_model_path is not None:
            from ..models.magic_141_video.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
            self.vae = AutoencoderKLCausal3D.from_pretrained(config.model.vae_model_path)
            # freeze vae
            self.vae.requires_grad_(False)
            self.vae.to(device=device, dtype=eval(config.model.vae_dtype))
            
        if config.is_inference:
            self.detector, self.audio_processor = None, None
            prompt_template = PROMPT_TEMPLATE["dit-llm-encode"]
            prompt_template_video = PROMPT_TEMPLATE["dit-llm-encode-video"]
            max_length = 256 + prompt_template_video.get(
                "crop_start", 0
            )
            self.text_encoder = TextEncoder(
                text_encoder_type="llm",
                max_length=max_length,
                text_encoder_path=config.model.text_encoder_path,
                text_encoder_precision=eval(config.model.text_encoder_dtype),
                tokenizer_type="llm",
                prompt_template=prompt_template,
                prompt_template_video=prompt_template_video,
                hidden_state_skip_layer=2,
                device=device,
            )
            self.text_encoder.to("cpu")

            self.text_encoder_2 = TextEncoder(
                text_encoder_type="clipL",
                max_length=77,
                text_encoder_path=config.model.text_encoder2_path,
                text_encoder_precision=eval(config.model.text_encoder2_dtype),
                tokenizer_type="clipL",
                device=device,
            )
            self.text_encoder_2.to("cpu")
            torch.cuda.empty_cache()
            
            self.text_encoder_vlm = TextEncoderVLM(
                text_encoder_type="vlm",
                max_length=512,
                text_encoder_path=config.model.text_encoder_vlm_path,
            ).to(dtype=eval(config.model.text_encoder_vlm_dtype))
            self.text_encoder_vlm.to("cpu")

    def setup(self, stage=None):
        if self.config.enable_ema:
            self.model_ema = DSEma(self.model, rank=self.trainer.strategy.global_rank)

    def inference_setup(self):
        self.model_ema = DSEma(self.model, rank=0)
        self.model_ema.copy_to_non_distributed(self.model)

    def _get_scheduler(self) -> Any:
        if self.config.get("noise_scheduler", "flow") == "flow":
            from ..models.magic_141_video.diffusion.schedulers import FlowMatchDiscreteScheduler
            self.train_noise_scheduler = FlowMatchDiscreteScheduler(
                shift=self.config.scheduler.flow_shift,
                reverse=self.config.scheduler.flow_reverse,
                solver=self.config.scheduler.flow_solver,
            )
        else:
            raise ValueError(f"Invalid denoise type when finetune model")

    def configure_model(self):
        config = self.config
        # Setting trainable params
        trainable_params = config.model.trainable_params
        for name, param in self.model.named_parameters():
            for search_name in trainable_params:
                if search_name in name:
                    param.requires_grad = True

    def configure_optimizers(self) -> Dict[str, Any]:
        params_to_update = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            param_lr = self.config.optimizer.lr
            if "audio" in name and "audio_module_lr" in self.config.optimizer:
                param_lr = self.config.optimizer.audio_module_lr
            params_to_update.append({'params': param, 'lr': param_lr})
        rank_zero_info(f"trainable params len is {len(params_to_update)}")


        # if "strategy" in self.config.trainer \
        #     and "deep" in self.config.trainer.strategy  \
        #     and "offload" in self.config.trainer.strategy:
        #     import deepspeed
        #     optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
        #         params_to_update,
        #         lr=self.config.optimizer.lr,
        #         weight_decay=self.config.optimizer.weight_decay,
        #         betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
        #         eps=self.config.optimizer.adam_epsilon,
        #         adamw_mode=True #you have to use adamw mode
        #     )

        # else:
        optimizer = torch.optim.AdamW(
            params_to_update,
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=(self.config.optimizer.adam_beta1, self.config.optimizer.adam_beta2),
            eps=self.config.optimizer.adam_epsilon,
        )

        def lr_lambda(current_step):
            warmup_steps = self.config.optimizer.warmup_steps

            if current_step < warmup_steps:
                warmup_factor = current_step / warmup_steps
                return warmup_factor
            else:
                return 1

        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [lr_scheduler]

    def training_step(self, batch: Dict, batch_idx: int):
        pass

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        pass

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        pass
    
    def encode_prompt(
        self,
        prompt,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "video",
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`torch.Tensor`, *optional*):
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
            data_type (`str`, *optional*):
        """
        if text_encoder is None:
            text_encoder = self.text_encoder

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

            if clip_skip is None:
                prompt_outputs = text_encoder.encode(
                    text_inputs, data_type=data_type, device=device
                )
                prompt_embeds = prompt_outputs.hidden_state
            else:
                prompt_outputs = text_encoder.encode(
                    text_inputs,
                    output_hidden_states=True,
                    data_type=data_type,
                    device=device,
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                    prompt_embeds
                )

            attention_mask = prompt_outputs.attention_mask
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                bs_embed, seq_len = attention_mask.shape
                attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                attention_mask = attention_mask.view(
                    bs_embed * num_videos_per_prompt, seq_len
                )

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1
            )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # max_length = prompt_embeds.shape[1]
            uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type)

            negative_prompt_outputs = text_encoder.encode(
                uncond_input, data_type=data_type, device=device
            )
            negative_prompt_embeds = negative_prompt_outputs.hidden_state

            negative_attention_mask = negative_prompt_outputs.attention_mask
            if negative_attention_mask is not None:
                negative_attention_mask = negative_attention_mask.to(device)
                _, seq_len = negative_attention_mask.shape
                negative_attention_mask = negative_attention_mask.repeat(
                    1, num_videos_per_prompt
                )
                negative_attention_mask = negative_attention_mask.view(
                    batch_size * num_videos_per_prompt, seq_len
                )

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=prompt_embeds_dtype, device=device
            )

            if negative_prompt_embeds.ndim == 2:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, -1
                )
            else:
                negative_prompt_embeds = negative_prompt_embeds.repeat(
                    1, num_videos_per_prompt, 1
                )
                negative_prompt_embeds = negative_prompt_embeds.view(
                    batch_size * num_videos_per_prompt, seq_len, -1
                )

        return (
            prompt_embeds,
            negative_prompt_embeds,
            attention_mask,
            negative_attention_mask,
        )
    
    @torch.no_grad()
    def predict_step(self, batch: Dict, guidance_scale: float, ema_infer: bool = True) -> torch.Tensor:
        model_infer = self.model
        if hasattr(self, "model_ema"):
            self.model_ema.to("cpu")

        model_infer.to(self.device)
        torch.cuda.empty_cache()
        # 1. get data

        image = batch["image"].to(self.device).unsqueeze(1)
        video_length = batch["video_length"][0].item()
        mask_scale = batch["mask_scale"][0]
        prompt = batch["prompt"][0]
        neg_prompt = "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion"
        ref_image_path = batch["ref_image_path"][0]
        do_classifier_free_guidance = False
        if guidance_scale > 1.0:
            do_classifier_free_guidance = True

        # Get audio and face mask
        img_height, img_width = image.shape[-2:]


        self.text_encoder.to(self.device)
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = self.encode_prompt(
            prompt,
            self.device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=neg_prompt,
            text_encoder=self.text_encoder
        )
        self.text_encoder.to("cpu")
        self.text_encoder_2.to(self.device)
        (
            prompt_embeds_2,
            negative_prompt_embeds_2,
            prompt_mask_2,
            negative_prompt_mask_2,
        ) = self.encode_prompt(
            prompt,
            self.device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=neg_prompt,
            text_encoder=self.text_encoder_2,
        )
        self.text_encoder_2.to("cpu")
        torch.cuda.empty_cache()
        # Get Image text encoder
        self.text_encoder_vlm.to(self.device)
        ref_image = Image.open(ref_image_path)
        image_embeds = self.text_encoder_vlm.forward(text=LLAVA_LLAMA_3_8B_HUMAN_IMAGE_PROMPT, image1=ref_image).hidden_state

        prompt_embeds = torch.cat([image_embeds, prompt_embeds], dim=1)
        image_masks = torch.ones((image_embeds.shape[0], image_embeds.shape[1]), dtype=prompt_mask.dtype)
        prompt_mask = torch.cat([image_masks.to(prompt_mask.device), prompt_mask], dim=1)
        if do_classifier_free_guidance:
            # zero_image = Image.fromarray(np.zeros_like(np.array(ref_image)))
            # zero_image_embeds = self.text_encoder_vlm.forward(text=LLAVA_LLAMA_3_8B_HUMAN_IMAGE_PROMPT, image1=zero_image).hidden_state
            negative_prompt_embeds = torch.cat([image_embeds, negative_prompt_embeds], dim=1)
            negative_prompt_mask = torch.cat([image_masks.to(negative_prompt_mask.device), negative_prompt_mask], dim=1)
        self.text_encoder_vlm.to("cpu")
        torch.cuda.empty_cache()
        
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        # vae
        image = rearrange(image, "b f c h w -> b c f h w")
        image_latents = (
            self.vae.encode(image.to(dtype=self.vae.dtype) * 2.0 - 1.0).latent_dist.sample()
            * self.vae.config.scaling_factor
        )
        B, C, F, H, W = image_latents.shape

        # face_mask b 1 f h w
        # 2. set timesteps
        self.train_noise_scheduler.set_timesteps(
            self.config.inference.num_inference_steps,
            device=self.device,
        )
        timesteps = self.train_noise_scheduler.timesteps
        num_warmup_steps = len(timesteps) - self.config.inference.num_inference_steps * self.train_noise_scheduler.order

        # 3.set inference latent
        generator = torch.manual_seed(torch.randint(0, 100000, (1,)).item())
        latent = (
            randn_tensor((B, C, video_length, H, W), generator=generator, device=self.device, dtype=image_latents.dtype)
        )

        image_latents = (
            torch.cat([image_latents, image_latents])
            if do_classifier_free_guidance
            else image_latents
        )

        if getattr(self.config.inference, "inversion", False):
            latent = noise_inversion(
                model_infer,
                latent,
                encoder_hidden_states=image_latents,
                text_states=prompt_embeds,
                text_mask=prompt_mask,
                text_states_2=prompt_embeds_2,
                reference_image = image_latents,
                cur_sigma_t = 0.9999,
                device = self.device,
                guidance_scale=guidance_scale,
            )
        
        # 4. inference
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            if i < num_warmup_steps:
                continue

            latent_input = latent.repeat(2 if do_classifier_free_guidance else 1, 1, 1, 1, 1)
            ts = torch.tensor([t] * latent_input.shape[0], device=self.device, dtype=torch.float32)

            with torch.autocast(device_type="cuda", dtype=model_infer.dtype, enabled=True):
                if do_classifier_free_guidance:
                    print("==============> Uncond =================")
                    noise_pred_uncond = model_infer(
                        hidden_states=latent_input[:1].to(dtype=torch.float32),
                        encoder_hidden_states=image_latents[:1].to(dtype=torch.float32),
                        text_states=prompt_embeds[:1].to(device=self.device, dtype=torch.float32),  # [2, 256, 4096]
                        text_mask=prompt_mask[:1],  # [2, 256]
                        text_states_2=prompt_embeds_2[:1].to(device=self.device, dtype=torch.float32),  # [2, 768]
                        # auto_guidence_scale=self.config.auto_guidence_scale,
                        timestep=ts[:1],
                    ).sample
                    
                    print("==============>Text Cond =================")
                    noise_pred_text = model_infer(
                        hidden_states=latent_input[1:].to(dtype=torch.float32),
                        encoder_hidden_states=image_latents[1:].to(dtype=torch.float32),
                        text_states=prompt_embeds[1:].to(device=self.device, dtype=torch.float32),  # [2, 256, 4096]
                        text_mask=prompt_mask[1:],  # [2, 256]
                        text_states_2=prompt_embeds_2[1:].to(device=self.device, dtype=torch.float32),  # [2, 768]
                        timestep=ts[1:],
                    ).sample

                    print("CFG:",guidance_scale," Guidance Value Mean:", (guidance_scale  * (noise_pred_text - noise_pred_uncond)).mean()) 
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                else:
                    noise_pred = model_infer(
                        hidden_states=latent_input.to(dtype=torch.float32),
                        encoder_hidden_states=image_latents.to(dtype=torch.float32),
                        text_states=prompt_embeds.to(device=self.device, dtype=torch.float32),  # [2, 256, 4096]
                        text_mask=prompt_mask,  # [2, 256]
                        text_states_2=prompt_embeds_2.to(device=self.device, dtype=torch.float32),  # [2, 768]
                        # auto_guidence_scale=self.config.auto_guidence_scale,
                        timestep=ts,
                    ).sample
                
            latent = self.train_noise_scheduler.step(
                noise_pred,
                t,
                latent,
            ).prev_sample  # outputs are prev_sample and pred_original_sample

        return self.decode_latents(latent)
        # if self.config.inference.do_classifier_free_guidance:

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        latents = 1 / self.vae.config.scaling_factor * latents
        # b c f h w
        image = self.vae.decode(latents.to(dtype=self.vae.dtype)).sample # first frame is the single image.
        return (image / 2.0 + 0.5).clamp(0, 1)

    def loss(
        self,
        model_pred: torch.Tensor,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        sigmas: torch.Tensor,
        noisy_latents: torch.Tensor,
    ):

        if self.config.get("noise_scheduler", "flow") == "flow":
            # print("Using flow matching")
            model_pred = model_pred * (-sigmas) + noisy_latents
            mse_loss_weights = compute_loss_weighting_for_sd3(
                weighting_scheme=self.config.scheduler.weighting_scheme, sigmas=sigmas
            )
            target = latents
            base_loss_pre_weight = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            base_loss_pre_weight = base_loss_pre_weight.mean(dim=list(range(1, len(base_loss_pre_weight.shape))))
            base_loss = (base_loss_pre_weight * mse_loss_weights.squeeze()).mean()
        else:
            # print("Not Using flow matching")
            if self.train_noise_scheduler.config.prediction_type == "epsilon":
                target = noise
                # match the dim 2 to model_pred
                # target = target.repeat(1, 1, model_pred.shape[2] // target.shape[2], 1, 1)

            elif self.train_noise_scheduler.config.prediction_type == "v_prediction":
                target = self.train_noise_scheduler.get_velocity(latents, noise, timesteps)
                # match the dim 2 to model_pred
                # target = target.repeat(1, 1, model_pred.shape[2] // target.shape[2], 1, 1)

            if self.config.loss.snr_gamma == 0:
                base_loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            else:
                snr = compute_snr(self.train_noise_scheduler, timesteps)
                if self.train_noise_scheduler.prediction_type == "v_prediction":
                    # Velocity objective requires that we add one to SNR values before we divide by them.
                    snr = snr + 1
                mse_loss_weights = (
                    torch.stack([snr, self.config.loss.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
                    / snr
                )
                base_loss = (
                    F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    * mse_loss_weights[:, None, None, None, None]
                )

        return base_loss.mean()

    @torch.no_grad()
    def prepare_latents(self, batch):
        # print(f"{self.dtype=}")
        if self.config.stage == 1:
            pixel_values_vid = batch["img"].to(self.device).unsqueeze(1)
            ref_image_list = []
            for ref_img in batch["ref_img"]:
                ref_image_list.append(ref_img)
            pixel_values_ref_img = torch.stack(ref_image_list, dim=0).unsqueeze(1)
            pixel_values_ref_img = pixel_values_ref_img.to(self.device)
        elif "pixel_values_vid" in batch:
            pixel_values_vid = batch["pixel_values_vid"].to(self.device) # b f c h w
            pixel_values_ref_img = batch["pixel_values_ref_img"].to(self.device) # b f c h w
            pixel_values_ref_img = pixel_values_vid[:, :1]
            # ref image latents
            pixel_values_ref_img = rearrange(pixel_values_ref_img, "b f c h w -> b c f h w")
            # Add noise following paper
            if self.config.loss.ref_image_blur:
                image_noise_sigma = torch.normal(mean=-3.0, std=0.5, size=(1,), device=pixel_values_ref_img.device)
                image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=pixel_values_ref_img.dtype)
                pixel_values_ref_img = pixel_values_ref_img + torch.randn_like(pixel_values_ref_img) * image_noise_sigma[:, None, None, None, None]
            ref_image_latents = self.vae.encode(pixel_values_ref_img).latent_dist.sample()
            ref_image_latents = ref_image_latents * self.vae.config.scaling_factor

            # noisy latents
            pixel_values_vid = rearrange(pixel_values_vid, "b f c h w -> b c f h w")
            latents = self.vae.encode(pixel_values_vid).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        else:
            ref_image_latents = batch["ref_image_latent"].to(self.device)
            latents = batch["vid_latent"].to(self.device)
            
        text_states_2 = batch["text_states_2"].to(self.device)
        text_states = batch["text_states"].to(self.device)
        text_mask = batch["text_mask"].to(self.device)    
        pixel_values_pose = batch["pixel_values_pose"].to(self.device) # b c h w    
        audio_feature = batch["target_wav_fea"].to(self.device) # b f n c, n is num of token for audio

            # print(f"{pixel_values_pose.min()=} {pixel_values_pose.max()=} {pixel_values_pose.shape=}")
        # ref_uncond = torch.rand(ref_image_latents.shape[0]) < self.config.loss.uncond_ratio
        
        # face mask conditioning
        # pose_uncond = torch.rand(pixel_values_pose.shape[0]) < self.config.loss.uncond_ratio
        # pixel_values_pose[pose_uncond] = 1.0 # 1.0 is the mask for uncond

        '''
        pose_uncond = [1,1]
        Pixel_value_pose = 1.0 => 全 1
        audio_feature => Cross Attention  => noise_map => loss
        loss  = loss*face_mask(1.0)

        pose_uncond = [1,0]
        Pixel_value_pose = [1,0]
        audio_feature => Cross Attention  => noise_map => loss
        loss  = loss*face_mask

        '''
        #pose uncond => 1.0, pose cond => part area => 1.0

        # audio conditioning
        audio_uncond = torch.rand(audio_feature.shape[0]) < self.config.loss.uncond_ratio
        audio_feature[audio_uncond] = 0
        
        # we do not add face mask dropout during the training.
        # if audio_uncond == True: # if audio is uncond, then the mask condition is 0 as well
        #     pixel_values_pose[audio_uncond] = 0.0

        # face mask b c h w -> b 1 f h w by repeat
        # print(f"{pixel_values_vid.shape=}")
        # print(f"{pixel_values_pose.shape=}")
        # print(f"{audio_feature.shape=}")
        # print(f"{pixel_values_ref_img.shape=}")
        pixel_values_pose = pixel_values_pose[:, :1, :, :].unsqueeze(2).repeat(1, 1, latents.size(2), 1, 1)
        # # # DEBUG latents decode
        # # pixel_values_vid_show = torch.cat([pixel_values_vid, pixel_values_ref_img.repeat(1, 1, pixel_values_vid.size(2), 1, 1)], dim=4)[0]
        # # print(f"{latents.unique()=} {ref_image_latents.unique()=}")
        # decode_latent = self.decode_latents(latents)[0]
        # decode_ref = self.decode_latents(ref_image_latents)
        # # print(f"{decode_ref.shape=} {decode_latent.shape}")
        # decode_ref = decode_ref[0, :, 0].permute(1, 2, 0).cpu().detach().to(dtype=torch.float32).clamp(0, 1).numpy()
        # decode_ref = (decode_ref * 255).astype("uint8")
        # # c f h w [-1, 1]
        # videos_save = []
        # for i in range(decode_latent.size(1)):
        #     img_item = rearrange(decode_latent[:, i, :, :], "c h w -> h w c").cpu().detach()
        #     img_item = img_item.to(dtype=torch.float32).clamp(0, 1).numpy()
        #     img_item = (img_item * 255).astype("uint8")
        #     videos_save.append(np.concatenate([decode_ref, img_item], axis=1))
        # import imageio
        # imageio.mimwrite(f"./DEBUG_2.mp4", videos_save, fps=25)
        if self.config.get("noise_scheduler", "flow") == "flow":
            # print("Using flow matching")
            latents, noisy_latents, timesteps, noise, sigmas = self.get_noisy_latents(latents)
            return (text_states_2, 
                    text_states, 
                    text_mask, 
                    audio_feature, 
                    pixel_values_pose, 
                    latents, 
                    noisy_latents, 
                    ref_image_latents, 
                    timesteps, 
                    noise, 
                    sigmas)
        else:
            # print("Not Using flow matching")
            latents, noisy_latents, timesteps, noise = self.get_noisy_latents(latents)
            return (text_states_2, 
                    text_states, 
                    text_mask, 
                    audio_feature, 
                    pixel_values_pose, 
                    latents, 
                    noisy_latents, 
                    ref_image_latents, 
                    timesteps, 
                    noise)

    @torch.no_grad()
    def get_noisy_latents(
        self, latents: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        get noisy latents
        """

        B = latents.shape[0]
        noise = torch.randn_like(latents)
        if self.config.loss.noise_offset > 0.0:  # offset
            noise += self.config.loss.noise_offset * torch.randn(
                (noise.shape[0], noise.shape[1], 1, 1, 1),
                device=noise.device,
                dtype=noise.dtype,
            )
        if self.config.get("noise_scheduler", "flow") == "flow":
            # print("Using flow matching")
            # u = compute_density_for_timestep_sampling(
            #     weighting_scheme=self.config.scheduler.weighting_scheme,
            #     batch_size=B,
            #     logit_mean=self.config.scheduler.logit_mean,
            #     logit_std=self.config.scheduler.logit_std,
            #     mode_scale=self.config.scheduler.mode_scale,
            # )
            # indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
            # timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=self.device)
            # sigmas = self.get_sigmas(latents, timesteps, n_dim=latents.ndim)
            # noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
            # return latents, noisy_latents, timesteps, noise, sigmas
            sigmas = compute_density_for_timestep_sampling(
                weighting_scheme=self.config.scheduler.weighting_scheme,
                batch_size=B,
                logit_mean=self.config.scheduler.logit_mean,
                logit_std=self.config.scheduler.logit_std,
                mode_scale=self.config.scheduler.mode_scale,
            )

            sigmas = self.train_noise_scheduler.sd3_sigma_shift(sigmas).to(device=latents.device)
            timesteps = self.train_noise_scheduler._sigma_to_t(sigmas).to(device=latents.device)
            while len(sigmas.shape) < len(latents.shape):
                sigmas = sigmas.unsqueeze(-1)
            noisy_latents = sigmas * noise + (1.0 - sigmas) * latents
            return latents, noisy_latents, timesteps, noise, sigmas
        
        else:
            timesteps = torch.randint(
                0,
                self.train_noise_scheduler.config.num_train_timesteps,
                (B,),
                device=latents.device,
            )
            timesteps = timesteps.long()
            noisy_latents = self.train_noise_scheduler.add_noise(latents, noise, timesteps)
            return latents, noisy_latents, timesteps, noise