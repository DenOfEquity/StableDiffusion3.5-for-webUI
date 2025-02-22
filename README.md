## StableDiffusion3.5 for Forge webui ##
I don't think there is anything Forge specific here.
### works for me <sup>TM</sup> on 8GB VRAM, 16GB RAM (GTX1070) ###

---
## Install ##
Go to the **Extensions** tab, then **Install from URL**, use the URL for this repository.
### SD3.5 needs *diffusers 0.31.0* ###

Easiest way to ensure necessary requirements are installed is to edit **requirements_versions.txt** in the webUI directory.
```
diffusers>=0.31.0
transformers>=4.40
tokenizers>=0.19
huggingface-hub>=0.23.4
```

Forge2 already has newer versions for all.

>[!IMPORTANT]
> **Also needs a huggingface access token:**
> Sign up / log in, go to your profile, create an access token. **Read** type is all you need, avoid the much more complicated **Fine-grained** option. Copy the token. Make a textfile called `huggingface_access_token.txt` in the main webui folder, e.g. `{forge install directory}\webui`, and paste the token in there. You will also need to accept the terms on the [SD3.5 repository page](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium). Same for large, large-turbo. And for the controlnets (untested).

>[!NOTE]
> Do not download the single file models, this extension cannot use them.

If available, I re-use the text encoders from SD3, to save some GB if you have them downloaded already. 

---
<details>
<summary>possibly necessary /alternate for Automatic1111</summary>

* open a console in the webui directory
* enter ```venv\scripts\activate```
* enter ```pip install -r requirements_versions.txt``` after making the updates listed above
</details>

---
### downloads models on first use - ~15.1GB including T5 text encoder, for medium ###

---
<details>
<summary>Change log</summary>

#### 19/02/2025 ####
* first upload, had this sitting around for months. Derived from my SD3 extension. It should be possible to use SD3.5 with the SD3 extension as the pipeline is the same. But LoRAs will be different, as will controlnets, so I decided to separate.
* Controlnets are untested; I can generate with *large*, but it takes ~10minutes.

