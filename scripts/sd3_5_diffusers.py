from diffusers.utils import check_min_version
check_min_version("0.31.0")


class SD35Storage:
    ModuleReload = False
    forgeCanvas = False
    usingGradio4 = False
    doneAccessTokenWarning = False
    combined_positive = None
    combined_negative = None
    clipskip = 0
    redoEmbeds = True
    noiseRGBA = [0.0, 0.0, 0.0, 0.0]
    captionToPrompt = False
    lora = None
    lora_scale = 1.0
    LFO = False

    teT5 = None
    teCG = None
    teCL = None
    lastModel = None
    lastControlNet = None
    pipe = None
    loadedLora = False

    locked = False     #   for preventing changes to the following volatile state while generating
    randomSeed = True
    noUnload = False
    useCL = True
    useCG = True
    useT5 = False
    ZN = False
    i2iAllSteps = False
    sharpNoise = False


import gc
import gradio
if int(gradio.__version__[0]) == 4:
    SD35Storage.usingGradio4 = True
import math
import numpy
import os
import torch
import torchvision.transforms.functional as TF
try:
    from importlib import reload
    SD35Storage.ModuleReload = True
except:
    SD35Storage.ModuleReload = False

try:
    from modules_forge.forge_canvas.canvas import ForgeCanvas, canvas_head
    SD35Storage.forgeCanvas = True
except:
    SD35Storage.forgeCanvas = False
    canvas_head = ""

from PIL import Image, ImageFilter

##   from webui
from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.shared import opts
from modules.ui_components import ResizeHandleRow, ToolButton
import modules.infotext_utils as parameters_copypaste

##   diffusers / transformers necessary imports
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast, T5ForConditionalGeneration
from diffusers import SD3Transformer2DModel
from diffusers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler, FlowMatchHeunDiscreteScheduler#, SASolverScheduler

from diffusers.models.controlnet_sd3 import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import logging

##  for Florence-2
from transformers import AutoProcessor, AutoModelForCausalLM 

##   my stuff
import customStylesListSD3_5 as styles
import scripts.SD3_5_pipeline as pipeline

# modules/processing.py
def create_infotext(model, sampler, positive_prompt, negative_prompt, guidance_scale, guidance_rescale, PAG_scale, PAG_adapt, shift, clipskip, steps, seed, width, height, loraSettings, controlNetSettings):
    generation_params = {
        "Steps"         :   steps,
        "Size"          :   f"{width}x{height}",
        "Seed"          :   seed,
        "CFG scale"     :   f"{guidance_scale} ({guidance_rescale})",
        "PAG"           :   f"{PAG_scale} ({PAG_adapt})",
        "Shift"         :   f"{shift}",
        "CLIP skip"     :   f"{clipskip}",
        "LoRA"          :   loraSettings,
        "controlNet"    :   controlNetSettings,
        "CLIP-L"        :   '✓' if SD35Storage.useCL else '✗',
        "CLIP-G"        :   '✓' if SD35Storage.useCG else '✗',
        "T5"            :   '✓' if SD35Storage.useT5 else '✗', #2713, 2717
        "zero negative" :   '✓' if SD35Storage.ZN else '✗',
        "Sampler": f"{sampler}",
    }
#add loras list and scales

    prompt_text = f"{positive_prompt}\n"
    prompt_text += (f"Negative prompt: {negative_prompt}\n")
    generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])
    noise_text = f", Initial noise: {SD35Storage.noiseRGBA}" if SD35Storage.noiseRGBA[3] != 0.0 else ""

    return f"{prompt_text}{generation_params_text}{noise_text}, Model (StableDiffusion3.5): {model}"

def predict(model, sampler, positive_prompt, negative_prompt, width, height, guidance_scale, guidance_rescale, shift, clipskip, 
            num_steps, sampling_seed, num_images, style, i2iSource, i2iDenoise, maskType, maskSource, maskBlur, maskCutOff, 
            controlNet, controlNetImage, controlNetStrength, controlNetStart, controlNetEnd, PAG_scale, PAG_adapt):

    logging.set_verbosity(logging.ERROR)        #   diffusers and transformers both enjoy spamming the console with useless info

    try:
        with open('huggingface_access_token.txt', 'r') as file:
            access_token = file.read().strip()
    except:
        if SD35Storage.doneAccessTokenWarning == False:
            print ("SD3.5: couldn't load 'huggingface_access_token.txt' from the webui directory. Will not be able to download models. Local cache will work.")
            SD35Storage.doneAccessTokenWarning = True
        access_token = 0

    torch.set_grad_enabled(False)
    
    localFilesOnly = SD35Storage.LFO
    
    # do I care about catching this?
#    if SD35Storage.useCL == False and SD35Storage.useCG == False and SD35Storage.useT5 == False:

    if PAG_scale > 0.0:
        guidance_rescale = 0.0

    ####    check img2img
    if i2iSource == None:
        maskType = 0
        i2iDenoise = 1
    
    if maskSource == None:
        maskType = 0

    match maskType:
        case 0:     #   'none'
            maskSource = None
            maskBlur = 0
            maskCutOff = 1.0
        case 1:
            if SD35Storage.forgeCanvas: #  'inpaint mask'
                maskSource = maskSource.getchannel('A').convert('L')#.convert("RGB")#.getchannel('R').convert('L')
            else:                       #   'drawn'
                maskSource = maskSource['layers'][0]  if SD35Storage.usingGradio4 else maskSource['mask']
        case 2:
            if SD35Storage.forgeCanvas: #   sketch
                i2iSource = Image.alpha_composite(i2iSource, maskSource)
                maskSource = None
                maskBlur = 0
                maskCutOff = 1.0
            else:                       #   'image'
                maskSource = maskSource['background'] if SD35Storage.usingGradio4 else maskSource['image']
        case 3:
            if SD35Storage.forgeCanvas: #   inpaint sketch
                i2iSource = Image.alpha_composite(i2iSource, maskSource)
                mask = maskSource.getchannel('A').convert('L')
                short_side = min(mask.size)
                dilation_size = int(0.015 * short_side) * 2 + 1
                mask = mask.filter(ImageFilter.MaxFilter(dilation_size))
                maskSource = mask.point(lambda v: 255 if v > 0 else 0)
                maskCutoff = 0.0
            else:                       #   'composite'
                maskSource = maskSource['composite']  if SD35Storage.usingGradio4 else maskSource['image']
        case _:
            maskSource = None
            maskBlur = 0
            maskCutOff = 1.0

    if i2iSource:
        if SD35Storage.i2iAllSteps == True:
            num_steps = int(num_steps / i2iDenoise)

        if SD35Storage.forgeCanvas:
            i2iSource = i2iSource.convert('RGB')

    if maskBlur > 0:
        dilation_size = maskBlur * 2 + 1
        maskSource = TF.gaussian_blur(maskSource.filter(ImageFilter.MaxFilter(dilation_size)), dilation_size)
    ####    end check img2img
    
    ####    controlnet
    useControlNet = None
    if model == '(medium)': # any 3rd party?
        controlNet = 0

    match controlNet:
        case 1:
            if controlNetImage and controlNetStrength > 0.0:
                useControlNet = 'stabilityai/stable-diffusion-3.5-large-controlnet-blur'
        case 2:
            if controlNetImage and controlNetStrength > 0.0:
                useControlNet = 'stabilityai/stable-diffusion-3.5-large-controlnet-canny'
        case 3:
            if controlNetImage and controlNetStrength > 0.0:
                useControlNet = 'stabilityai/stable-diffusion-3.5-large-controlnet-depth'
        case _:
            controlNetStrength = 0.0
    if useControlNet:
        controlNetImage = controlNetImage.resize((width, height))
    ####    end controlnet

    if model in ['(large)', '(large-turbo)', '(medium)']:
        customModel = None
    else:
        customModel = './/models//diffusers//SD3Custom//' + model + '.safetensors'

    #   triple prompt, automatic support, no longer needs button to enable
    def promptSplit (prompt):
        split_prompt = prompt.split('|')
        c = len(split_prompt)
        prompt_1 = split_prompt[0].strip()
        if c == 1:
            prompt_2 = prompt_1
            prompt_3 = prompt_1
        elif c == 2:
            if SD35Storage.useT5 == True:
                prompt_2 = prompt_1
                prompt_3 = split_prompt[1].strip()
            else:
                prompt_2 = split_prompt[1].strip()
                prompt_3 = ''
        elif c >= 3:
            prompt_2 = split_prompt[1].strip()
            prompt_3 = split_prompt[2].strip()
        return prompt_1, prompt_2, prompt_3

    positive_prompt_1, positive_prompt_2, positive_prompt_3 = promptSplit (positive_prompt)
    negative_prompt_1, negative_prompt_2, negative_prompt_3 = promptSplit (negative_prompt)

    if style:
        for s in style:
            k = 0;
            while styles.styles_list[k][0] != s:
                k += 1
            if "{prompt}" in styles.styles_list[k][1]:
                positive_prompt_1 = styles.styles_list[k][1].replace("{prompt}", positive_prompt_1)
                positive_prompt_2 = styles.styles_list[k][1].replace("{prompt}", positive_prompt_2)
                positive_prompt_3 = styles.styles_list[k][1].replace("{prompt}", positive_prompt_3)
            else:
                positive_prompt_1 += styles.styles_list[k][1]
                positive_prompt_2 += styles.styles_list[k][1]
                positive_prompt_3 += styles.styles_list[k][1]
            
    combined_positive = positive_prompt_1 + " | \n" + positive_prompt_2 + " | \n" + positive_prompt_3
    combined_negative = negative_prompt_1 + " | \n" + negative_prompt_2 + " | \n" + negative_prompt_3

    gc.collect()
    torch.cuda.empty_cache()

    fixed_seed = get_fixed_seed(-1 if SD35Storage.randomSeed else sampling_seed)

    sourceSD3 = "stabilityai/stable-diffusion-3-medium-diffusers"
    sourceLarge = "stabilityai/stable-diffusion-3.5-large"
    sourceTurbo = "stabilityai/stable-diffusion-3.5-large-turbo"
    sourceMedium = "stabilityai/stable-diffusion-3.5-medium"        # used for all text_encoders, to avoid duplicates
    
    match model:
        case "(large)":
            source = sourceLarge
        case "(large-turbo)":
            source = sourceTurbo
        case "(medium)":
            source = sourceMedium
        case _:
            source = sourceMedium

    useCachedEmbeds = (SD35Storage.combined_positive == combined_positive and
                       SD35Storage.combined_negative == combined_negative and
                       SD35Storage.redoEmbeds == False and
                       SD35Storage.clipskip == clipskip)
    #   also shouldn't cache if change model, but how to check if new model has own CLIPs?
    #   maybe just WON'T FIX, to keep it simple

    if useCachedEmbeds:
        print ("SD3.5: Skipping text encoders and tokenizers.")
    else:
        ####    start T5 text encoder
        if SD35Storage.useT5 == True:
            tokenizer = T5TokenizerFast.from_pretrained(
                sourceMedium, local_files_only=localFilesOnly,
                subfolder='tokenizer_3',
                torch_dtype=torch.float16,
                max_length=512,
                token=access_token,
            )

            input_ids = tokenizer(
                [positive_prompt_3, negative_prompt_3],          padding=True, max_length=512, truncation=True,
                add_special_tokens=True,    return_tensors="pt",
            ).input_ids

            # positive_input_ids = input_ids[0:1]
            # negative_input_ids = input_ids[1:]

            del tokenizer

            if SD35Storage.teT5 == None:             #   model not loaded
                if SD35Storage.noUnload == True:     #   will keep model loaded
                    device_map = {
                        'shared': 0,
                        'encoder.embed_tokens': 0,
                        'encoder.block.0': 'cpu',   'encoder.block.1': 'cpu',   'encoder.block.2': 'cpu',   'encoder.block.3': 'cpu', 
                        'encoder.block.4': 'cpu',   'encoder.block.5': 'cpu',   'encoder.block.6': 'cpu',   'encoder.block.7': 'cpu', 
                        'encoder.block.8': 'cpu',   'encoder.block.9': 'cpu',   'encoder.block.10': 'cpu',  'encoder.block.11': 'cpu', 
                        'encoder.block.12': 'cpu',  'encoder.block.13': 'cpu',  'encoder.block.14': 'cpu',  'encoder.block.15': 'cpu', 
                        'encoder.block.16': 'cpu',  'encoder.block.17': 'cpu',  'encoder.block.18': 'cpu',  'encoder.block.19': 'cpu', 
                        'encoder.block.20': 'cpu',  'encoder.block.21': 'cpu',  'encoder.block.22': 'cpu',  'encoder.block.23': 'cpu', 
                        'encoder.final_layer_norm': 0, 
                        'encoder.dropout': 0
                    }
                else:                               #   will delete model after use
                    device_map = 'auto'

                print ("SD3.5: loading T5 ...", end="\r", flush=True)

                if SD35Storage.teT5 == None:             #   model not loaded, try SD3 if already downlaoded
                    try:    #   some potential to error here, if available VRAM changes while loading device_map could be wrong
                        SD35Storage.teT5  = T5EncoderModel.from_pretrained(
                            sourceSD3, local_files_only=True,
                            subfolder='text_encoder_3',
                            torch_dtype=torch.float16,
                            device_map=device_map,
                            token=access_token,
                        )
                    except:
                        try:    #   some potential to error here, if available VRAM changes while loading device_map could be wrong
                            SD35Storage.teT5  = T5EncoderModel.from_pretrained(
                                sourceMedium, local_files_only=localFilesOnly,
                                subfolder='text_encoder_3',
                                torch_dtype=torch.float16,
                                device_map=device_map,
                                token=access_token,
                            )
                        except:
                            print ("SD3.5: loading T5 failed, likely low VRAM at moment of load. Try again, and/or: close other programs, reload/restart webUI, use 'keep models loaded' option.")
                            gradio.Info('Unable to load T5. See console.')
                            SD35Storage.locked = False
                            return sampling_seed, gradio.Button.update(interactive=True), result

            #   if model loaded, then user switches off noUnload, loaded model still used on next run (could alter device_map?: model.hf_device_map)
            #   not a major concern anyway

            print ("SD3.5: encoding prompt (T5) ...", end="\r", flush=True)
            embeds_3 = SD35Storage.teT5(input_ids.to('cuda'))[0]
            positive_embeds_3 = embeds_3[0].unsqueeze(0)
            if SD35Storage.ZN == True:
                negative_embeds_3 = torch.zeros_like(positive_embeds_3)
            else:
                negative_embeds_3 = embeds_3[1].unsqueeze(0)
                
            del input_ids, embeds_3
            
            if SD35Storage.noUnload == False:
                SD35Storage.teT5 = None
            print ("SD3.5: encoding prompt (T5) ... done")
        else:
            #dim 1 (512) is tokenizer max length from config; dim 2 (4096) is transformer joint_attention_dim from its config
            positive_embeds_3 = torch.zeros((1, 1, 4096),    device='cuda', dtype=torch.float16, )
            negative_embeds_3 = torch.zeros((1, 1, 4096),    device='cuda', dtype=torch.float16, )
        ####    end T5

        ####    start CLIP-G
        if SD35Storage.useCG == True:
            tokenizer = CLIPTokenizer.from_pretrained(
                sourceMedium, local_files_only=localFilesOnly,
                subfolder='tokenizer',
                torch_dtype=torch.float16,
                token=access_token,
            )

            input_ids = tokenizer(
                [positive_prompt_1, negative_prompt_1],          padding='max_length', max_length=77, truncation=True,
                return_tensors="pt",
            ).input_ids

            positive_input_ids = input_ids[0:1]
            negative_input_ids = input_ids[1:]

            del tokenizer
            
            #   check if custom model has trained CLIPs
            if model != SD35Storage.lastModel:
                if model in ["(large)", "(large-turbo)", "(medium)"]:
                    SD35Storage.teCG = None
                else:
                    try:    #   maybe custom model has trained CLIPs - not sure if correct way to load
                        SD35Storage.teCG = CLIPTextModelWithProjection.from_single_file(
                            customModel, local_files_only=localFilesOnly,
                            subfolder='text_encoder',
                            torch_dtype=torch.float16,
                            token=access_token,
                        )
                    except:
                        SD35Storage.teCG = None
            if SD35Storage.teCG == None:             #   model not loaded, use base
                try:
                    SD35Storage.teCG = CLIPTextModelWithProjection.from_pretrained(
                        sourceSD3, local_files_only=True,
                        subfolder='text_encoder',
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16,
                        token=access_token,
                    )
                except:
                    SD35Storage.teCG = CLIPTextModelWithProjection.from_pretrained(
                        sourceMedium, local_files_only=localFilesOnly,
                        subfolder='text_encoder',
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16,
                        token=access_token,
                    )
            SD35Storage.teCG.to('cuda')

            positive_embeds = SD35Storage.teCG(positive_input_ids.to('cuda'), output_hidden_states=True)
            pooled_positive_1 = positive_embeds[0]
            positive_embeds_1 = positive_embeds.hidden_states[-(clipskip + 2)]
            
            if SD35Storage.ZN == True:
                negative_embeds_1 = torch.zeros_like(positive_embeds_1)
                pooled_negative_1 = torch.zeros((1, 768),       device='cuda', dtype=torch.float16, )
            else:
                negative_embeds = SD35Storage.teCG(negative_input_ids.to('cuda'), output_hidden_states=True)
                pooled_negative_1 = negative_embeds[0]
                negative_embeds_1 = negative_embeds.hidden_states[-2]

            if SD35Storage.noUnload == False:
                SD35Storage.teCG = None
            else:
                SD35Storage.teCG.to('cpu')
                
        else:
            positive_embeds_1 = torch.zeros((1, 1, 4096),  device='cuda', dtype=torch.float16, )
            negative_embeds_1 = torch.zeros((1, 1, 4096),  device='cuda', dtype=torch.float16, )
            pooled_positive_1 = torch.zeros((1, 768),      device='cuda', dtype=torch.float16, )
            pooled_negative_1 = torch.zeros((1, 768),      device='cuda', dtype=torch.float16, )
        ####    end CLIP-G

        ####    start CLIP-L
        if SD35Storage.useCL == True:
            tokenizer = CLIPTokenizer.from_pretrained(
                sourceMedium, local_files_only=localFilesOnly,
                subfolder='tokenizer_2',
                torch_dtype=torch.float16,
                token=access_token,
            )
            input_ids = tokenizer(
                [positive_prompt_2, negative_prompt_2],          padding='max_length', max_length=77, truncation=True,
                return_tensors="pt",
            ).input_ids

            positive_input_ids = input_ids[0:1]
            negative_input_ids = input_ids[1:]

            del tokenizer

            #   check if custom model has trained CLIPs
            if model != SD35Storage.lastModel:
                if model in ["(large)", "(large-turbo)", "(medium)"]:
                    SD35Storage.teCL = None
                else:
                    try:    #   maybe custom model has trained CLIPs - not sure if correct way to load
                        SD35Storage.teCL = CLIPTextModelWithProjection.from_single_file(
                            customModel, local_files_only=localFilesOnly,
                            subfolder='text_encoder_2',
                            torch_dtype=torch.float16,
                            token=access_token,
                        )
                    except:
                        SD35Storage.teCL = None
            if SD35Storage.teCL == None:             #   model not loaded, use base
                try:
                    SD35Storage.teCL = CLIPTextModelWithProjection.from_pretrained(
                        sourceSD3, local_files_only=True,
                        subfolder='text_encoder_2',
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16,
                        token=access_token,
                    )
                except:
                    SD35Storage.teCL = CLIPTextModelWithProjection.from_pretrained(
                        sourceMedium, local_files_only=localFilesOnly,
                        subfolder='text_encoder_2',
                        low_cpu_mem_usage=True,
                        torch_dtype=torch.float16,
                        token=access_token,
                    )

            SD35Storage.teCL.to('cuda')

            positive_embeds = SD35Storage.teCL(positive_input_ids.to('cuda'), output_hidden_states=True)
            pooled_positive_2 = positive_embeds[0]
            positive_embeds_2 = positive_embeds.hidden_states[-(clipskip + 2)]
            
            if SD35Storage.ZN == True:
                negative_embeds_2 = torch.zeros_like(positive_embeds_2)
                pooled_negative_2 = torch.zeros((1, 1280),      device='cuda', dtype=torch.float16, )
            else:
                negative_embeds = SD35Storage.teCL(negative_input_ids.to('cuda'), output_hidden_states=True)
                pooled_negative_2 = negative_embeds[0]
                negative_embeds_2 = negative_embeds.hidden_states[-2]

            if SD35Storage.noUnload == False:
                SD35Storage.teCL = None
            else:
                SD35Storage.teCL.to('cpu')

        else:
            positive_embeds_2 = torch.zeros((1, 1, 4096),  device='cuda', dtype=torch.float16, )
            negative_embeds_2 = torch.zeros((1, 1, 4096),  device='cuda', dtype=torch.float16, )
            pooled_positive_2 = torch.zeros((1, 1280),     device='cuda', dtype=torch.float16, )
            pooled_negative_2 = torch.zeros((1, 1280),     device='cuda', dtype=torch.float16, )
        ####    end CLIP-L

        #merge
        clip_positive_embeds = torch.cat([positive_embeds_1, positive_embeds_2], dim=-1)
        clip_positive_embeds = torch.nn.functional.pad(clip_positive_embeds, (0, positive_embeds_3.shape[-1] - clip_positive_embeds.shape[-1]) )
        clip_negative_embeds = torch.cat([negative_embeds_1, negative_embeds_2], dim=-1)
        clip_negative_embeds = torch.nn.functional.pad(clip_negative_embeds, (0, negative_embeds_3.shape[-1] - clip_negative_embeds.shape[-1]) )

        positive_embeds = torch.cat([clip_positive_embeds, positive_embeds_3.to('cuda')], dim=-2)
        negative_embeds = torch.cat([clip_negative_embeds, negative_embeds_3.to('cuda')], dim=-2)

        positive_pooled = torch.cat([pooled_positive_1, pooled_positive_2], dim=-1)
        negative_pooled = torch.cat([pooled_negative_1, pooled_negative_2], dim=-1)

        SD35Storage.positive_embeds = positive_embeds.to('cpu')
        SD35Storage.negative_embeds = negative_embeds.to('cpu')
        SD35Storage.positive_pooled = positive_pooled.to('cpu')
        SD35Storage.negative_pooled = negative_pooled.to('cpu')
        SD35Storage.combined_positive = combined_positive
        SD35Storage.combined_negative = combined_negative
        SD35Storage.clipskip = clipskip
        SD35Storage.redoEmbeds = False

        del positive_embeds, negative_embeds, positive_pooled, negative_pooled
        del clip_positive_embeds, clip_negative_embeds
        del pooled_positive_1, pooled_positive_2, pooled_negative_1, pooled_negative_2
        del positive_embeds_1, positive_embeds_2, positive_embeds_3
        del negative_embeds_1, negative_embeds_2, negative_embeds_3

        gc.collect()
        torch.cuda.empty_cache()

    ####    end useCachedEmbeds

    if useControlNet:
        if useControlNet != SD35Storage.lastControlNet:
            controlnet=SD3ControlNetModel.from_pretrained(
                useControlNet, torch_dtype=torch.float16,
            )
    else:
        controlnet = None

    if model != SD35Storage.lastModel:
        SD35Storage.pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    if SD35Storage.pipe == None:
        if model in ["(large)", "(large-turbo)"]:
            SD35Storage.pipe = pipeline.SD35Pipeline_DoE_combined.from_pretrained(
                source,
                local_files_only=localFilesOnly,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,                
                use_safetensors=True,
                token=access_token,
                controlnet=controlnet,
                device_map='balanced'
            )
        elif model in ["(medium)"]:
            SD35Storage.pipe = pipeline.SD35Pipeline_DoE_combined.from_pretrained(
                source,
                local_files_only=localFilesOnly,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,                
                use_safetensors=True,
                token=access_token,
                controlnet=None,
            )
            SD35Storage.pipe.enable_model_cpu_offload()
        else:
            SD35Storage.pipe = pipeline.SD35Pipeline_DoE_combined.from_pretrained(
                source,
                local_files_only=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,                
                use_safetensors=True,
                transformer=SD3Transformer2DModel.from_single_file(customModel, local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.float16),
                token=access_token,
                controlnet=controlnet,
            )
        SD35Storage.lastModel = model
        SD35Storage.lastControlNet = useControlNet

    else:       #   do have pipe
        SD35Storage.pipe.controlnet = controlnet
        SD35Storage.lastControlNet = useControlNet

    del controlnet

    if model == '(medium)':
        SD35Storage.pipe.transformer.to('cuda')
        SD35Storage.pipe.vae.to('cpu')

    shape = (
        num_images,
        SD35Storage.pipe.transformer.config.in_channels,
        int(height) // SD35Storage.pipe.vae_scale_factor,
        int(width) // SD35Storage.pipe.vae_scale_factor,
    )

    #   always generate the noise here
    generator = [torch.Generator(device='cpu').manual_seed(fixed_seed+i) for i in range(num_images)]
    latents = randn_tensor(shape, generator=generator).to('cuda').to(torch.float16)

    if SD35Storage.sharpNoise:
        minDim = 1 + 2*(min(latents.size(2), latents.size(3)) // 4)
        for b in range(len(latents)):
            blurred = TF.gaussian_blur(latents[b], minDim)
            latents[b] = 1.05*latents[b] - 0.05*blurred

    #regen the generator to minimise differences between single/batch - might still be different - batch processing could use different pytorch kernels
    del generator
    generator = torch.Generator(device='cpu').manual_seed(14641)

    #   colour the initial noise
    if SD35Storage.noiseRGBA[3] != 0.0:
        nr = SD35Storage.noiseRGBA[0] ** 0.5
        ng = SD35Storage.noiseRGBA[1] ** 0.5
        nb = SD35Storage.noiseRGBA[2] ** 0.5

        imageR = torch.tensor(numpy.full((8,8), (nr), dtype=numpy.float32))
        imageG = torch.tensor(numpy.full((8,8), (ng), dtype=numpy.float32))
        imageB = torch.tensor(numpy.full((8,8), (nb), dtype=numpy.float32))
        image = torch.stack((imageR, imageG, imageB), dim=0).unsqueeze(0)

        image = SD35Storage.pipe.image_processor.preprocess(image).to('cuda').to(torch.float16)
        image_latents = (SD35Storage.pipe.vae.encode(image).latent_dist.sample(generator) - SD35Storage.pipe.vae.config.shift_factor) * SD35Storage.pipe.vae.config.scaling_factor

        image_latents = image_latents.repeat(num_images, 1, latents.size(2), latents.size(3))

        for b in range(len(latents)):
            for c in range(4):
                latents[b][c] -= latents[b][c].mean()

        torch.lerp (latents, image_latents, SD35Storage.noiseRGBA[3] * 0.25, out=latents)

        del imageR, imageG, imageB, image, image_latents
    #   end: colour the initial noise

    schedulerConfig = dict(SD35Storage.pipe.scheduler.config)
    schedulerConfig['flow_shift'] = shift

    if sampler == "Euler":
        schedulerConfig.pop('algorithm_type', None) 
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(schedulerConfig)
    elif sampler == "Heun":
        schedulerConfig.pop('algorithm_type', None) 
        scheduler = FlowMatchHeunDiscreteScheduler.from_config(schedulerConfig)
    # elif sampler == "SA-solver":
        # schedulerConfig['algorithm_type'] = 'data_prediction'
        # scheduler = SASolverScheduler.from_config(schedulerConfig)
    else:
        schedulerConfig['algorithm_type'] = 'dpmsolver++'
        schedulerConfig['prediction_type'] = 'flow_prediction'
        schedulerConfig['use_flow_sigmas'] = True
        # schedulerConfig['solver_order'] = 2
        # schedulerConfig['solver_type'] = "midpoint"
        # schedulerConfig['beta_end'] = 0.02
        # schedulerConfig['beta_schedule'] = "linear"
        # schedulerConfig['beta_start'] = 0.0001
        scheduler = DPMSolverMultistepScheduler.from_config(schedulerConfig)

    SD35Storage.pipe.scheduler = scheduler


#   load in LoRA, weight passed to pipe
    if SD35Storage.lora and SD35Storage.lora != "(None)" and SD35Storage.lora_scale != 0.0:
        lorafile = ".//models/diffusers//SD35Lora//" + SD35Storage.lora + ".safetensors"
        try:
            SD35Storage.pipe.load_lora_weights(lorafile, local_files_only=True, adapter_name=SD35Storage.lora)
            SD35Storage.loadedLora = True
#            SD35Storage.pipe.set_adapters(SD35Storage.lora, adapter_weights=SD35Storage.lora_scale)    #.set_adapters doesn't exist so no easy multiple LoRAs and weights
        except:
            print ("Failed: LoRA: " + lorafile)
            #   no reason to abort, just carry on without LoRA

#adapter_weight_scales = { "unet": { "down": 1, "mid": 0, "up": 0} }
#pipe.set_adapters("pixel", adapter_weight_scales)
#pipe.set_adapters(["pixel", "toy"], adapter_weights=[0.5, 1.0])

#    print (pipe.scheduler.compatibles)

    # SD35Storage.pipe.transformer.to(memory_format=torch.channels_last)
    # SD35Storage.pipe.vae.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        output = SD35Storage.pipe(
            num_inference_steps             = num_steps,
            guidance_scale                  = guidance_scale,
            guidance_rescale                = guidance_rescale,
            prompt_embeds                   = SD35Storage.positive_embeds.to('cuda'),
            negative_prompt_embeds          = SD35Storage.negative_embeds.to('cuda'),
            pooled_prompt_embeds            = SD35Storage.positive_pooled.to('cuda'),
            negative_pooled_prompt_embeds   = SD35Storage.negative_pooled.to('cuda'),
            num_images_per_prompt           = num_images,
            generator                       = generator,
            latents                         = latents,

            image                           = i2iSource,
            strength                        = i2iDenoise,
            mask_image                      = maskSource,
            mask_cutoff                     = maskCutOff,

            control_image                   = controlNetImage, 
            controlnet_conditioning_scale   = controlNetStrength,  
            control_guidance_start          = controlNetStart,
            control_guidance_end            = controlNetEnd,

            pag_scale                       = PAG_scale,
            pag_adaptive_scale              = PAG_adapt,

#            joint_attention_kwargs          = {"scale": SD35Storage.lora_scale }
        )
        del controlNetImage

    del generator, latents

    if SD35Storage.noUnload:
        if SD35Storage.loadedLora == True:
            SD35Storage.pipe.unload_lora_weights()
            SD35Storage.loadedLora = False
        if model == '(medium)':
            SD35Storage.pipe.transformer.to('cpu')
    else:
        SD35Storage.pipe.transformer = None
        SD35Storage.lastModel = None
        SD35Storage.pipe.controlnet = None
        SD35Storage.lastControlNet = None

    gc.collect()
    torch.cuda.empty_cache()

    if model == '(medium)':
        SD35Storage.pipe.vae.to('cuda')

    if SD35Storage.lora != "(None)" and SD35Storage.lora_scale != 0.0:
        loraSettings = SD35Storage.lora + f" ({SD35Storage.lora_scale})"
    else:
        loraSettings = None

    if useControlNet != None:
        useControlNet += f" strength: {controlNetStrength}; step range: {controlNetStart}-{controlNetEnd}"

    original_samples_filename_pattern = opts.samples_filename_pattern
    opts.samples_filename_pattern = "SD3.5_[datetime]"
    result = []
    total = len(output)
    for i in range (total):
        print (f'SD3.5: VAE: {i+1} of {total}', end='\r', flush=True)
        info=create_infotext(
            model, sampler, combined_positive, combined_negative, 
            guidance_scale, guidance_rescale,
            PAG_scale, PAG_adapt, 
            shift, clipskip, num_steps, 
            fixed_seed + i, 
            width, height,
            loraSettings,
            useControlNet)      #   doing this for every image when only change is fixed_seed

        #   manually handling the VAE prevents hitting shared memory on 8GB
        latent = (output[i:i+1]) / SD35Storage.pipe.vae.config.scaling_factor
        latent = latent + SD35Storage.pipe.vae.config.shift_factor
        image = SD35Storage.pipe.vae.decode(latent, return_dict=False)[0]
        image = SD35Storage.pipe.image_processor.postprocess(image, output_type='pil')[0]

        if maskType > 0 and maskSource is not None:
            image = Image.composite(image, i2iSource, maskSource)

        result.append((image, info))
        
        images.save_image(
            image,
            opts.outdir_samples or opts.outdir_txt2img_samples,
            "",
            fixed_seed + i,
            combined_positive,
            opts.samples_format,
            info
        )
    print ('SD3.5: VAE: done  ')
    opts.samples_filename_pattern = original_samples_filename_pattern

    if not SD35Storage.noUnload:
        SD35Storage.pipe = None
    
    del output
    gc.collect()
    torch.cuda.empty_cache()

    SD35Storage.locked = False
    return fixed_seed, gradio.Button.update(interactive=True), result


def on_ui_tabs():
    if SD35Storage.ModuleReload:
        reload(styles)
        reload(pipeline)

    def buildLoRAList ():
        loras = ["(None)"]
        
        import glob
        customLoRA = glob.glob(".\models\diffusers\SD35Lora\*.safetensors")

        for i in customLoRA:
            filename = i.split('\\')[-1]
            loras.append(filename[0:-12])

        return loras
    def buildModelList ():
        models = ["(large)", "(large-turbo)", "(medium)"]
        
        import glob
        customModel = glob.glob(".\models\diffusers\SD35Custom\*.safetensors")

        for i in customModel:
            filename = i.split('\\')[-1]
            models.append(filename[0:-12])

        return models

    loras = buildLoRAList ()
    models = buildModelList ()

    def refreshLoRAs ():
        loras = buildLoRAList ()
        return gradio.Dropdown.update(choices=loras)
    def refreshModels ():
        models = buildModelList ()
        return gradio.Dropdown.update(choices=models)
   
    def getGalleryIndex (index):
        if index < 0:
            index = 0
        return index

    def getGalleryText (gallery, index, seed):
        if gallery:
            return gallery[index][1], seed+index
        else:
            return "", seed+index
        
    def i2iSetDimensions (image, w, h):
        if image is not None:
            w = 32 * (image.size[0] // 32)
            h = 32 * (image.size[1] // 32)
        return [w, h]

    def i2iImageFromGallery (gallery, index):
        try:
            if SD35Storage.usingGradio4:
                newImage = gallery[index][0]
                return newImage
            else:
                newImage = gallery[index][0]['name'].rsplit('?', 1)[0]
                return newImage
        except:
            return None

    def i2iMakeCaptions (image, originalPrompt):
        if image == None:
            return originalPrompt

        model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', 
                                                     attn_implementation="sdpa", 
                                                     torch_dtype=torch.float16, 
                                                     trust_remote_code=True).to('cuda')
        processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', #-large
                                                  torch_dtype=torch.float32,
                                                  trust_remote_code=True)

        result = ''
        prompts = ['<CAPTION>', '<DETAILED_CAPTION>', '<MORE_DETAILED_CAPTION>']

        for p in prompts:
            inputs = processor(text=p, images=image.convert("RGB"), return_tensors="pt")
            inputs.to('cuda').to(torch.float16)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            del inputs
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            del generated_ids
            parsed_answer = processor.post_process_generation(generated_text, task=p, image_size=(image.width, image.height))
            del generated_text
            print (parsed_answer)
            result += parsed_answer[p]
            del parsed_answer
            if p != prompts[-1]:
                result += ' | \n'

        del model, processor

        if SD35Storage.captionToPrompt:
            return result
        else:
            return originalPrompt
    def toggleC2P ():
        SD35Storage.captionToPrompt ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD35Storage.captionToPrompt])
    def toggleLFO ():
        SD35Storage.LFO ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD35Storage.LFO])

    #   these are volatile state, should not be changed during generation
    def toggleRandom ():
        SD35Storage.randomSeed ^= True
        return gradio.Button.update(variant='primary' if SD35Storage.randomSeed == True else 'secondary')

    def toggleNU ():
        if not SD35Storage.locked:
            SD35Storage.noUnload ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD35Storage.noUnload])
    def unloadM ():
        if not SD35Storage.locked:
            SD35Storage.teT5 = None
            SD35Storage.teCG = None
            SD35Storage.teCL = None
            SD35Storage.pipe = None
            SD35Storage.lastModel = None
            SD35Storage.lastControlNet = None
            gc.collect()
            torch.cuda.empty_cache()
        else:
            gradio.Info('Unable to unload models while using them.')

    def toggleCL ():
        if not SD35Storage.locked:
            SD35Storage.redoEmbeds = True
            SD35Storage.useCL ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD35Storage.useCL])
    def toggleCG ():
        if not SD35Storage.locked:
            SD35Storage.redoEmbeds = True
            SD35Storage.useCG ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD35Storage.useCG])
    def toggleT5 ():
        if not SD35Storage.locked:
            SD35Storage.redoEmbeds = True
            SD35Storage.useT5 ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD35Storage.useT5])
    def toggleZN ():
        if not SD35Storage.locked:
            SD35Storage.redoEmbeds = True
            SD35Storage.ZN ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD35Storage.ZN])
    def toggleAS ():
        if not SD35Storage.locked:
            SD35Storage.i2iAllSteps ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][SD35Storage.i2iAllSteps])
    def toggleSP ():
        if not SD35Storage.locked:
            return gradio.Button.update(variant='primary')
    def superPrompt (prompt, seed):
        tokenizer = getattr (shared, 'SuperPrompt_tokenizer', None)
        superprompt = getattr (shared, 'SuperPrompt_model', None)
        if tokenizer is None:
            tokenizer = T5TokenizerFast.from_pretrained(
                'roborovski/superprompt-v1',
            )
            shared.SuperPrompt_tokenizer = tokenizer
        if superprompt is None:
            superprompt = T5ForConditionalGeneration.from_pretrained(
                'roborovski/superprompt-v1',
                device_map='auto',
                torch_dtype=torch.float16
            )
            shared.SuperPrompt_model = superprompt
            print("SuperPrompt-v1 model loaded successfully.")
            if torch.cuda.is_available():
                superprompt.to('cuda')
 
        torch.manual_seed(get_fixed_seed(seed))
        device = superprompt.device
        systemprompt1 = "Expand the following prompt to add more detail: "
        
        input_ids = tokenizer(systemprompt1 + prompt, return_tensors="pt").input_ids.to(device)
        outputs = superprompt.generate(input_ids, max_new_tokens=256, repetition_penalty=1.2, do_sample=True)
        dirty_text = tokenizer.decode(outputs[0])
        result = dirty_text.replace("<pad>", "").replace("</s>", "").strip()
        
        return gradio.Button.update(variant='secondary'), result

    resolutionList = [
        (1536, 672),    (1344, 768),    (1248, 832),    (1120, 896),
        (1200, 1200),   (1024, 1024),
        (896, 1120),    (832, 1248),    (768, 1344),    (672, 1536)
    ]

    def updateWH (idx, w, h):
        #   returns None to dimensions dropdown so that it doesn't show as being set to particular values
        #   width/height could be manually changed, making that display inaccurate and preventing immediate reselection of that option
        if idx < len(resolutionList):
            return None, resolutionList[idx][0], resolutionList[idx][1]
        return None, w, h

    def randomString ():
        import random
        import string
        alphanumeric_string = ''
        for i in range(8):
            alphanumeric_string += ''.join(random.choices(string.ascii_letters + string.digits, k=8))
            if i < 7:
                alphanumeric_string += ' '
        return alphanumeric_string

    def toggleGenerate (R, G, B, A, lora, scale):
        SD35Storage.noiseRGBA = [R, G, B, A]
        SD35Storage.lora = lora
        SD35Storage.lora_scale = scale# if lora != "(None)" else 1.0
        SD35Storage.locked = True
        return gradio.Button.update(value='...', variant='secondary', interactive=False), gradio.Button.update(interactive=False)


    def parsePrompt (positive, negative, sampler, width, height, seed, steps, CFG, CFGrescale, PAG_scale, PAG_adapt, shift, nr, ng, nb, ns, loraName, loraScale):
        p = positive.split('\n')
        lineCount = len(p)

        negative = ''
        
        if "Prompt" != p[0] and "Prompt: " != p[0][0:8]:               #   civitAI style special case
            positive = p[0]
            l = 1
            while (l < lineCount) and not (p[l][0:17] == "Negative prompt: " or p[l][0:7] == "Steps: " or p[l][0:6] == "Size: "):
                if p[l] != '':
                    positive += '\n' + p[l]
                l += 1
        
        for l in range(lineCount):
            if "Prompt" == p[l][0:6]:
                if ": " == p[l][6:8]:                                   #   mine
                    positive = str(p[l][8:])
                    c = 1
                elif "Prompt" == p[l] and (l+1 < lineCount):            #   webUI
                    positive = p[l+1]
                    c = 2
                else:
                    continue

                while (l+c < lineCount) and not (p[l+c][0:10] == "Negative: " or p[l+c][0:15] == "Negative Prompt" or p[l+c] == "Params" or p[l+c][0:7] == "Steps: " or p[l+c][0:6] == "Size: "):
                    if p[l+c] != '':
                        positive += '\n' + p[l+c]
                    c += 1
                l += 1

            elif "Negative" == p[l][0:8]:
                if ": " == p[l][8:10]:                                  #   mine
                    negative = str(p[l][10:])
                    c = 1
                elif " prompt: " == p[l][8:17]:                         #   civitAI
                    negative = str(p[l][17:])
                    c = 1
                elif " Prompt" == p[l][8:15] and (l+1 < lineCount):     #   webUI
                    negative = p[l+1]
                    c = 2
                else:
                    continue
                
                while (l+c < lineCount) and not (p[l+c] == "Params" or p[l+c][0:7] == "Steps: " or p[l+c][0:6] == "Size: "):
                    if p[l+c] != '':
                        negative += '\n' + p[l+c]
                    c += 1
                l += 1

            elif "Initial noise: " == str(p[l][0:15]):
                noiseRGBA = str(p[l][16:-1]).split(',')
                nr = float(noiseRGBA[0])
                ng = float(noiseRGBA[1])
                nb = float(noiseRGBA[2])
                ns = float(noiseRGBA[3])
            else:
                params = p[l].split(',')
                for k in range(len(params)):
                    pairs = params[k].strip().split(' ')        #split on ':' instead?
                    match pairs[0]:
                        case "Size:":
                            size = pairs[1].split('x')
                            width = 32 * ((int(size[0]) + 16) // 32)
                            height = 32 * ((int(size[1]) + 16) // 32)
                        case "Seed:":
                            seed = int(pairs[1])
                        case "Steps(Prior/Decoder):":
                            steps = str(pairs[1]).split('/')
                            steps = int(steps[0])
                        case "Steps:":
                            steps = int(pairs[1])
                        case "CFG":
                            if "scale:" == pairs[1]:
                                CFG = float(pairs[2])
                        case "CFG:":
                            CFG = float(pairs[1])
                            if len(pairs) >= 3:
                                CFGrescale = float(pairs[2].strip('\(\)'))
                        case "PAG:":
                            if len(pairs) == 3:
                                PAG_scale = float(pairs[1])
                                PAG_adapt = float(pairs[2].strip('\(\)'))
                        case "Shift:":
                            shift = float(pairs[1])
                        case "width:":
                            width = 32 * ((int(pairs[1]) + 16) // 32)
                        case "height:":
                            height = 32 * ((int(pairs[1]) + 16) // 32)
                        case "LoRA:":
                            if len(pairs) == 3:
                                loraName = pairs[1]
                                loraScale = float(pairs[2].strip('\(\)'))
                        case "Sampler:":
                            if len(pairs) == 3:
                                sampler = f"{pairs[1]} {pairs[2]}"
                            else:
                                sampler = pairs[1]
                        #clipskip?
        return positive, negative, sampler, width, height, seed, steps, CFG, CFGrescale, PAG_scale, PAG_adapt, shift, nr, ng, nb, ns, loraName, loraScale

    def style2prompt (prompt, style):
        splitPrompt = prompt.split('|')
        newPrompt = ''
        for p in splitPrompt:
            subprompt = p.strip()
            for s in style:
                #get index from value, working around possible gradio bug
                k = 0;
                while styles.styles_list[k][0] != s:
                    k += 1
                if "{prompt}" in styles.styles_list[k][1]:
                    subprompt = styles.styles_list[k][1].replace("{prompt}", subprompt)
                else:
                    subprompt += styles.styles_list[k][1]
            newPrompt += subprompt
            if p != splitPrompt[-1]:
                newPrompt += ' |\n'
        return newPrompt, []


    def refreshStyles (style):
        if SD35Storage.ModuleReload:
            reload(styles)
        
            newList = [x[0] for x in styles.styles_list]
            newStyle = []
        
            for s in style:
                if s in newList:
                    newStyle.append(s)

            return gradio.Dropdown.update(choices=newList, value=newStyle)
        else:
            return gradio.Dropdown.update(value=style)
            

    def toggleSharp ():
        if not SD35Storage.locked:
            SD35Storage.sharpNoise ^= True
        return gradio.Button.update(value=['s', 'S'][SD35Storage.sharpNoise],
                                variant=['secondary', 'primary'][SD35Storage.sharpNoise])

    def maskFromImage (image):
        if image:
            return image, 'drawn'
        else:
            return None, 'none'

    with gradio.Blocks(analytics_enabled=False, head=canvas_head) as sd35_block:
        with ResizeHandleRow():
            with gradio.Column():
#                    LFO = ToolButton(value='lfo', variant='secondary', tooltip='local files only')

                with gradio.Row():
                    model = gradio.Dropdown(models, label='Model', value='(medium)', type='value')
                    refreshM = ToolButton(value='\U0001f504')
                    nouse0 = ToolButton(value="️|", variant='tertiary', tooltip='', interactive=False)
                    CL = ToolButton(value='CL', variant='primary',   tooltip='use CLIP-L text encoder')
                    CG = ToolButton(value='CG', variant='primary',   tooltip='use CLIP-G text encoder')
                    T5 = ToolButton(value='T5', variant='secondary', tooltip='use T5 text encoder')
                    sampler = gradio.Dropdown(["DPM++ 2M", "Euler", "Heun"], label='Sampler', value="Euler", type='value', scale=0)

                with gradio.Row():
                    positive_prompt = gradio.Textbox(label='Prompt', placeholder='Enter a prompt here ...', lines=1.01)
                    clipskip = gradio.Number(label='CLIP skip', minimum=0, maximum=8, step=1, value=0, precision=0, scale=0)
                with gradio.Row():
                    negative_prompt = gradio.Textbox(label='Negative', placeholder='', lines=1.01)
                    parse = ToolButton(value="↙️", variant='secondary', tooltip="parse")
                    randNeg = ToolButton(value='rng', variant='secondary', tooltip='random negative')
                    ZN = ToolButton(value='ZN', variant='secondary', tooltip='zero out negative embeds')
                    SP = ToolButton(value='ꌗ', variant='secondary', tooltip='zero out negative embeds')

                with gradio.Row():
                    style = gradio.Dropdown([x[0] for x in styles.styles_list], label='Style', value=None, type='value', multiselect=True)
                    strfh = ToolButton(value="🔄", variant='secondary', tooltip='reload styles')
                    st2pr = ToolButton(value="📋", variant='secondary', tooltip='add style to prompt')
                    batch_size = gradio.Number(label='Batch Size', minimum=1, maximum=9, value=1, precision=0, scale=0)

                with gradio.Row():
                    width = gradio.Slider(label='Width', minimum=512, maximum=2048, step=32, value=1024)
                    height = gradio.Slider(label='Height', minimum=512, maximum=2048, step=32, value=1024)
                    swapper = ToolButton(value='\U000021C4')
                    dims = gradio.Dropdown([f'{i} \u00D7 {j}' for i,j in resolutionList],
                                        label='Quickset', type='index', scale=0)

                with gradio.Row():
                    guidance_scale = gradio.Slider(label='CFG', minimum=1, maximum=16, step=0.1, value=5, scale=1)
                    CFGrescale = gradio.Slider(label='rescale CFG', minimum=0.00, maximum=1.0, step=0.01, value=0.0, scale=1)
                    shift = gradio.Slider(label='Shift', minimum=1.0, maximum=8.0, step=0.1, value=3.0, scale=1)
                with gradio.Row():
                    PAG_scale = gradio.Slider(label='Perturbed-Attention Guidance scale', minimum=0, maximum=8, step=0.1, value=0.0, scale=1, visible=True)
                    PAG_adapt = gradio.Slider(label='PAG adaptive scale', minimum=0.00, maximum=0.1, step=0.001, value=0.0, scale=1)
                with gradio.Row(equal_height=True):
                    steps = gradio.Slider(label='Steps', minimum=1, maximum=80, step=1, value=20, scale=2)
                    random = ToolButton(value="\U0001f3b2\ufe0f", variant="primary")
                    sampling_seed = gradio.Number(label='Seed', value=-1, precision=0, scale=0)

                with gradio.Row(equal_height=True, visible=False):
                    lora = gradio.Dropdown([x for x in loras], label='LoRA (place in models/diffusers/SD35Lora)', value="(None)", type='value', multiselect=False, scale=1)
                    refreshL = ToolButton(value='\U0001f504')
                    scale = gradio.Slider(label='LoRA weight', minimum=-1.0, maximum=1.0, value=1.0, step=0.01, scale=1)

                with gradio.Accordion(label='the colour of noise', open=False):
                    with gradio.Row():
                        initialNoiseR = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='red')
                        initialNoiseG = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='green')
                        initialNoiseB = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='blue')
                        initialNoiseA = gradio.Slider(minimum=0, maximum=0.1, value=0.0, step=0.001, label='strength')
                        sharpNoise = ToolButton(value="s", variant='secondary', tooltip='Sharpen initial noise')

                with gradio.Accordion(label='ControlNet', open=False, visible=True):
                    with gradio.Row():
                        CNSource = gradio.Image(label='control image', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                        with gradio.Column():
                            CNMethod = gradio.Dropdown(['(None)', 'blur (large)', 'canny (large)', 'depth (large)'], 
                                                        label='method', value='(None)', type='index', multiselect=False, scale=1)
                            CNStrength = gradio.Slider(label='Strength', minimum=0.00, maximum=1.0, step=0.01, value=0.8)
                            CNStart = gradio.Slider(label='Start step', minimum=0.00, maximum=1.0, step=0.01, value=0.0)
                            CNEnd = gradio.Slider(label='End step', minimum=0.00, maximum=1.0, step=0.01, value=0.8)

                with gradio.Accordion(label='image to image', open=False):
                    if SD35Storage.forgeCanvas:
                        i2iSource = ForgeCanvas(elem_id="SD3.5_img2img_image", height=320, scribble_color=opts.img2img_inpaint_mask_brush_color, scribble_color_fixed=False, scribble_alpha=100, scribble_alpha_fixed=False, scribble_softness_fixed=False)
                        with gradio.Row():
                            i2iFromGallery = gradio.Button(value='Get gallery image')
                            i2iSetWH = gradio.Button(value='Set size from image')
                            i2iCaption = gradio.Button(value='Caption image')
                            toPrompt = ToolButton(value='P', variant='secondary')
                        
                        with gradio.Row():
                            i2iDenoise = gradio.Slider(label='Denoise', minimum=0.00, maximum=1.0, step=0.01, value=0.5)
                            AS = ToolButton(value='AS')
                            maskType = gradio.Dropdown(['i2i', 'inpaint mask', 'sketch', 'inpaint sketch'], value='i2i', label='Type', type='index')
                        with gradio.Row():
                            maskBlur = gradio.Slider(label='Blur mask radius', minimum=0, maximum=64, step=1, value=0)
                            maskCut = gradio.Slider(label='Ignore Mask after step', minimum=0.00, maximum=1.0, step=0.01, value=1.0)
                 
                    else:
                        with gradio.Row():
                            i2iSource = gradio.Image(label='image to image source', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                            if SD35Storage.usingGradio4:
                                maskSource = gradio.ImageEditor(label='mask source', sources=['upload'], type='pil', interactive=True, show_download_button=False, layers=False, brush=gradio.Brush(colors=['#FFFFFF'], color_mode='fixed'))
                            else:
                                maskSource = gradio.Image(label='mask source', sources=['upload'], type='pil', interactive=True, show_download_button=False, tool='sketch', image_mode='RGB', brush_color='#F0F0F0')#opts.img2img_inpaint_mask_brush_color)
                        with gradio.Row():
                            with gradio.Column():
                                with gradio.Row():
                                    i2iDenoise = gradio.Slider(label='Denoise', minimum=0.00, maximum=1.0, step=0.01, value=0.5)
                                    AS = ToolButton(value='AS')
                                with gradio.Row():
                                    i2iFromGallery = gradio.Button(value='Get gallery image')
                                    i2iSetWH = gradio.Button(value='Set size from image')
                                with gradio.Row():
                                    i2iCaption = gradio.Button(value='Caption image (Florence-2)', scale=6)
                                    toPrompt = ToolButton(value='P', variant='secondary')

                            with gradio.Column():
                                maskType = gradio.Dropdown(['none', 'drawn', 'image', 'composite'], value='none', label='Mask', type='index')
                                maskBlur = gradio.Slider(label='Blur mask radius', minimum=0, maximum=25, step=1, value=0)
                                maskCut = gradio.Slider(label='Ignore Mask after step', minimum=0.00, maximum=1.0, step=0.01, value=1.0)
                                maskCopy = gradio.Button(value='use i2i source as template')

                with gradio.Row():
                    noUnload = gradio.Button(value='keep models loaded', variant='primary' if SD35Storage.noUnload else 'secondary', tooltip='noUnload', scale=1)
                    unloadModels = gradio.Button(value='unload models', tooltip='force unload of models', scale=1)

                if SD35Storage.forgeCanvas:
                    ctrls = [model, sampler, positive_prompt, negative_prompt, width, height, guidance_scale, CFGrescale, shift, clipskip, steps, sampling_seed, batch_size, style, i2iSource.background, i2iDenoise, maskType, i2iSource.foreground, maskBlur, maskCut, CNMethod, CNSource, CNStrength, CNStart, CNEnd, PAG_scale, PAG_adapt]
                else:
                    ctrls = [model, sampler, positive_prompt, negative_prompt, width, height, guidance_scale, CFGrescale, shift, clipskip, steps, sampling_seed, batch_size, style, i2iSource, i2iDenoise, maskType, maskSource, maskBlur, maskCut, CNMethod, CNSource, CNStrength, CNStart, CNEnd, PAG_scale, PAG_adapt]
                parseable = [positive_prompt, negative_prompt, sampler, width, height, sampling_seed, steps, guidance_scale, CFGrescale, PAG_scale, PAG_adapt, shift, initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA, lora, scale]

            with gradio.Column():
                generate_button = gradio.Button(value="Generate", variant='primary', visible=True)
                output_gallery = gradio.Gallery(label='Output', height="80vh", type='pil', interactive=False, elem_id="SD3.5_gallery", 
                                            show_label=False, object_fit='contain', visible=True, columns=1, preview=True)

#   caption not displaying linebreaks, alt text does

                gallery_index = gradio.Number(value=0, visible=False)
                infotext = gradio.Textbox(value="", visible=False)
                base_seed = gradio.Number(value=0, visible=False)

                with gradio.Row():
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname,
                        source_text_component=infotext,
                        source_image_component=output_gallery,
                    ))
        noUnload.click(toggleNU, inputs=None, outputs=noUnload)
        unloadModels.click(unloadM, inputs=None, outputs=None, show_progress=True)

        if SD35Storage.forgeCanvas:
            i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource.background, width, height], outputs=[width, height], show_progress=False)
            i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery, gallery_index], outputs=[i2iSource.background])
            i2iCaption.click (fn=i2iMakeCaptions, inputs=[i2iSource.background, positive_prompt], outputs=[positive_prompt])
        else:
            maskCopy.click(fn=maskFromImage, inputs=[i2iSource], outputs=[maskSource, maskType])
            i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource, width, height], outputs=[width, height], show_progress=False)
            i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery, gallery_index], outputs=[i2iSource])
            i2iCaption.click (fn=i2iMakeCaptions, inputs=[i2iSource, positive_prompt], outputs=[positive_prompt])


        SP.click(toggleSP, inputs=None, outputs=SP)
        SP.click(superPrompt, inputs=[positive_prompt, sampling_seed], outputs=[SP, positive_prompt])
        sharpNoise.click(toggleSharp, inputs=None, outputs=sharpNoise)
        strfh.click(refreshStyles, inputs=[style], outputs=[style])
        st2pr.click(style2prompt, inputs=[positive_prompt, style], outputs=[positive_prompt, style])
        parse.click(parsePrompt, inputs=parseable, outputs=parseable, show_progress=False)
        dims.input(updateWH, inputs=[dims, width, height], outputs=[dims, width, height], show_progress=False)
        refreshM.click(refreshModels, inputs=None, outputs=[model])
        refreshL.click(refreshLoRAs, inputs=None, outputs=[lora])
        CL.click(toggleCL, inputs=None, outputs=CL)
        CG.click(toggleCG, inputs=None, outputs=CG)
        T5.click(toggleT5, inputs=None, outputs=T5)
        ZN.click(toggleZN, inputs=None, outputs=ZN)
        AS.click(toggleAS, inputs=None, outputs=AS)
#        LFO.click(toggleLFO, inputs=None, outputs=LFO)
        swapper.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)
        random.click(toggleRandom, inputs=None, outputs=random, show_progress=False)
        randNeg.click(randomString, inputs=None, outputs=[negative_prompt])

        toPrompt.click(toggleC2P, inputs=None, outputs=[toPrompt])

        output_gallery.select(fn=getGalleryIndex, js="selected_gallery_index", inputs=gallery_index, outputs=gallery_index, show_progress=False).then(
            fn=getGalleryText, inputs=[output_gallery, gallery_index, base_seed], outputs=[infotext, sampling_seed], show_progress=False)

        generate_button.click(toggleGenerate, inputs=[initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA, lora, scale], outputs=[generate_button, SP]).then(predict, inputs=ctrls, outputs=[base_seed, SP, output_gallery]).then(fn=lambda: gradio.update(value='Generate', variant='primary', interactive=True), inputs=None, outputs=generate_button).then(fn=getGalleryIndex, js="selected_gallery_index", inputs=gallery_index, outputs=gallery_index).then(fn=getGalleryText, inputs=[output_gallery, gallery_index, base_seed], outputs=[infotext, sampling_seed])

    return [(sd35_block, "StableDiffusion3.5", "sd35_DoE")]

script_callbacks.on_ui_tabs(on_ui_tabs)

