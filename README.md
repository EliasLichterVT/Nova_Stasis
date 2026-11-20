# Nova Stasis

Nova Stasis is a local browser studio for putting two reference characters into a new Stable Diffusion scene. GitHub hosts the source. The Python dashboard and ComfyUI both run on your machine.

## One-step launch

Download and extract the repository, then double-click **`launch.bat`** on 64-bit x86 Windows or **`launch.command`** on Apple Silicon Macs. On Linux, run `./launch.command` from the extracted folder. If your extractor removed the executable bit on macOS or Linux, run `bash launch.command`.

The launcher installs a managed Python 3.12, PyTorch, ComfyUI, IP-Adapter Plus, and a matched SD 1.5 starter stack with the ViT-H image encoder and OpenPose ControlNet. Downloads are verified and cached. The launcher starts both services on loopback, waits for the models to appear, then opens the studio. Close it with Ctrl+C from the launch window. Later launches reuse the installation.

- **First run:** Internet access and roughly 20 GiB of free space are needed, depending on PyTorch and its platform dependencies. The model downloads alone are about 7.6 GB. Linux and Apple Silicon macOS need the system `bash` and `curl`. Git, Python, and a separate ComfyUI install are optional.
- **Acceleration:** The launcher selects CUDA on supported x86-64 Windows/Linux NVIDIA setups with a compatible driver and tries MPS on macOS. It uses CPU on other systems or if the device probe fails. CPU rendering is substantially slower. GPU drivers are installed and maintained by the operating system or GPU vendor.
- **Models:** The starter checkpoint supports text-to-image, image-to-image and masked edits. The app retains manual support for other compatible realism or illustration checkpoints and LoRAs. Pose conditioning uses the bundled ControlNet weights with a rendered OpenPose map you supply. The model sources and licenses are linked in the manual setup section.
- **Storage:** Managed files live in `~/.nova-stasis` by default. ComfyUI keeps launcher-managed references and results in `private/input/` and `private/output/` inside that folder. Set `NOVA_LAUNCHER_HOME` before launch to use another absolute path outside the repository. Logs are in `logs/` in the same folder.
- **Options:** Add `--setup-only` to install and exit, `--no-browser` to start without opening a tab, or `--cpu` to force CPU execution. If the usual ports are occupied, the launcher prints the URL it selected.

## Browser test mode

With the dashboard requirements installed, run `python -m nova.demo` and open `http://127.0.0.1:8765` in the browser on that machine or in the same Codex cloud desktop. Use `--port` to choose another port.

Click **Load sample scene**, then try story translation, LoRA suggestions, the anatomy prompt preview, or **Generate scene**. The test mode uses fixture models and returns labeled preview images. Hardware capacity is simulated, and the Civitai installer is disabled. The session uses a temporary directory that is cleared when it stops.

## Hardware check in the browser

Nova checks the local ComfyUI device when you open the studio and shows the operating system, logical CPU count, active device, and reported memory. It uses the active device, so a CPU-mode launch receives CPU guidance even on a PC with a discrete GPU. Available-memory figures reflect the moment of the scan. Nova uses more conservative settings for CPU and low-memory devices and recognizes the shared memory used by Apple Silicon.

Untouched width, height, steps, and batch fields start with the recommended settings. **Scan again** refreshes the hardware report, and **Use recommended settings** reapplies the defaults. Nova warns when settings exceed the suggested range. These estimates are advisory. A memory-intensive checkpoint, ControlNet, or other apps can lower what the machine can render.

## Manual installation

### Install the dashboard

Use Python 3.11 or 3.12. Install [ComfyUI](https://docs.comfy.org/installation) separately with a PyTorch build for your GPU. ComfyUI and the dashboard can use separate Python environments.

**Linux or macOS**

```sh
git clone https://github.com/EliasLichterVT/Nova_Stasis.git
cd Nova_Stasis
bash scripts/setup.sh
.venv/bin/python -m uvicorn nova.app:app --host 127.0.0.1 --port 8000 --env-file .env
```

**Windows PowerShell**

```powershell
git clone https://github.com/EliasLichterVT/Nova_Stasis.git
cd Nova_Stasis
.\scripts\setup.ps1
.\.venv\Scripts\python.exe -m uvicorn nova.app:app --host 127.0.0.1 --port 8000 --env-file .env
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The dashboard defaults to ComfyUI at `http://127.0.0.1:8188`. To use a different local port or relocate private app storage, edit `.env` using [.env.example](.env.example).

### Set up ComfyUI and models

1. Install [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) through ComfyUI Manager, or clone it into `ComfyUI/custom_nodes` following its installation instructions. Nova uses its `IPAdapterModelLoader` and `IPAdapterAdvanced` nodes.
2. Put a standard SDXL or SD 1.5 checkpoint in `ComfyUI/models/checkpoints`. Start with a text-to-image base checkpoint or a compatible realism- or illustration-tuned checkpoint. The workflow uses ordinary SD latent sampling, including during masked image edits. SDXL refiners, other diffusion families, and dedicated extra-channel inpaint checkpoints are outside this initial workflow.
3. Download the [ViT-H image encoder](https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors) and save it as `ComfyUI/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`. Choose the matching adapter below and place it in `ComfyUI/models/ipadapter`.

| Checkpoint family | Adapter for the ViT-H image encoder |
| --- | --- |
| SDXL | [ip-adapter-plus_sdxl_vit-h.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors) |
| SD 1.5 | [ip-adapter-plus_sd15.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors) |

Optional weights go in `ComfyUI/models/loras` and `ComfyUI/models/controlnet`. For pose conditioning, use an OpenPose ControlNet compatible with your checkpoint family. For example, see [xinsir SDXL OpenPose](https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0) or [ControlNet v1.1 SD 1.5 OpenPose](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose). The launcher installs the [SD 1.5 base checkpoint](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) and the [FP16 SD 1.5 OpenPose checkpoint](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors). Upload a rendered OpenPose skeleton map prepared in an editor or with a preprocessor such as [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux). Nova accepts the rendered map directly.

Restart ComfyUI after adding custom nodes. For a manual installation, run the following from the ComfyUI directory with its Python environment active.

```sh
python main.py --listen 127.0.0.1 --port 8188 --disable-metadata
```

Select **Refresh** in Nova to reload the model lists. The lists reflect installed files. You choose which checkpoint, adapter, encoder, LoRA and ControlNet are compatible. Only download weights and executable custom nodes from sources you trust and follow their licenses.

For a first run, use batch 1. Try 512 × 512 for SD 1.5 or 1024 × 768 for SDXL. A compatible GPU is the practical option. Actual VRAM use depends on the checkpoint, canvas size, batch size, ControlNet, and ComfyUI memory settings. If you run out of VRAM, reduce the batch or canvas size, then retry. CPU mode is possible on supported ComfyUI installations and can be much slower.

## Import models from Civitai

Open **Import models from Civitai** in the running studio. Use **Browse Civitai.com** or **Browse Civitai.red** to open the site in a separate browser tab. Copy the model page or direct download link, then click **Paste link** in Nova to look up its files. You can also paste manually and select **Find files**. Choose the version, weights file, and destination type, then select **Download & import**. Nova shows progress and refreshes its model lists after installation. Follow the model’s license and use weights compatible with your checkpoint family.

The launcher configures the managed ComfyUI model directory automatically. For a manual install, set `NOVA_MODELS_DIR` in `.env` to the absolute path of the running ComfyUI installation’s `models` directory and restart Nova. Public downloads work without a key where Civitai permits them. If Civitai requires account access, enter your Civitai API key in the import panel and click **Save key** to reuse it after restart. Nova encrypts saved keys for the current Windows account using Windows Data Protection. On Linux or macOS it saves the key in an owner-only file in Nova’s private data directory. Click **Forget key** to remove it. You can also set `NOVA_CIVITAI_API_KEY` in `.env` for a manual setup. A typed key overrides a saved key for that request. Nova sends credentials only to the selected Civitai site.

Nova imports `.safetensors` checkpoints, LoRA/LyCORIS, VAEs, ControlNet, IP-Adapter, CLIP Vision, text encoders, embeddings, upscalers, and standalone diffusion models. It verifies the Civitai SHA-256 checksum and tensor container before publishing a file. For a ZIP with a Civitai SHA-256 checksum, Nova verifies the package and extracts supported Safetensors files. Standard ComfyUI folders such as `models/loras/` and `models/controlnet/` determine where members go. Files outside a recognized folder use the default destination you select. The import lists installed files and skips unsupported members such as images, scripts, workflows, and pickle-based weights. Links outside Civitai, other archive formats, and nested archives are excluded. The dashboard selects checkpoints, LoRAs, ControlNet, IP-Adapter and CLIP Vision directly. Supporting weights such as standalone VAEs and upscalers are available in ComfyUI for workflows that use them.

The importer runs one download at a time and keeps it running if you close or reload the tab while Nova remains running. It respects existing files, never overwrites a model, and installs on local filesystems that support hard links, such as NTFS, APFS and ext4. Keep the configured models folder outside Git or in the ignored `models/` directory.

## Generate and reuse scenes

1. Upload one clear reference for each character. Nova accepts still PNG, JPEG and WebP up to 16 MiB and 20 megapixels. Tight crops with distinct features, clothing or silhouettes generally work best.
2. Pick layout and describe both subjects in the prompt, matching their positions and naming distinct visual traits. Each reference is encoded by its own IP-Adapter application with its own weight and attention mask. The default left/right or top/bottom masks blend slightly along the center. In **Mask & LoRA**, upload custom masks for other compositions. White applies the reference, black excludes it, and gray scales it. Use a canvas-aligned mask for each character.
3. Set reference strengths independently. Optional LoRAs are applied to the shared checkpoint and text encoder, so their effects can reach both subjects. The adapter masks are the spatial identity controls. Similar faces, crowded subjects or overlapping regions can still drift or blend.
4. To control position, use **Pose control** and select both a skeleton map and matching OpenPose model. Align each skeleton with that character’s attention mask. For **Image to image**, upload a source and choose a denoise value. In **Inpainting**, also upload a same-size grayscale mask with white over the edit and black elsewhere. Use identical orientation for the source and mask. Nova center-crops and resizes them together to the output dimensions.
5. Leave **Seed** blank for a random seed. Use **Batch size** to request up to four variants in the same sampler run. A history entry records the starting seed and settings for the entire batch. Repeatability depends on the same models, settings and inference environment.
6. Confirm that you have permission to use the references, everyone depicted is an adult, and the scene is non-explicit. Select **Generate scene**. The dashboard polls ComfyUI and shows completed images in **Preview** and **Generation history**. You can download each result or reuse its settings and seed.
7. Save reference images and strengths through **Character profiles**, including an optional LoRA selection. **Saved settings** stores the full prompt and generation settings, including local references. Each generation requires a fresh attestation.

## Turn a story into a prompt

In **Scene**, open **Turn a story into a prompt** and type or paste up to 12,000 characters. Enter the story names for References A and B if you want the prompt to name them. Select **Translate story**, choose a scene, and edit its suggested prompt if needed. Select **Use prompt**, then run **Generate scene** with your usual reference images and settings.

The translator works locally without an API key or additional model. Start a new paragraph or insert a line with three asterisks between scenes. Nova processes up to 12 scene suggestions, extracting short passages and prioritizing visible actions, settings and lighting with English-language rules. It omits quoted dialogue and places both characters according to the selected layout. Recheck the suggested prompt for details the extractor may miss. To generate a different moment, select another scene and apply its prompt.

## LoRA trigger words and story suggestions

Nova records published Civitai trigger words when you import a LoRA. For an existing LoRA, look up its Civitai link and select its version and file. Use **Grab LoRA trigger words** to load the words into the editor when Nova finds one installed file with the same name. Review and save them. In **LoRA keyword library**, you can also enter your own trigger words and descriptive story keywords, one per line.

Story translation suggests installed LoRAs by matching the selected scene prompt against saved keywords, trigger words, and model names. LoRAs you already selected appear first. After editing the suggested prompt, click **Match prompt** to update the suggestions. Choose **Use on A** or **Use on B** to select the LoRA and add missing trigger words to the prompt. Story keywords help find the model and stay out of the generation prompt. Check the displayed Civitai base model against your checkpoint when that metadata is available. LoRA effects still apply to the full image, while the two reference adapters provide the regional conditioning.

## Human anatomy negative prompt

The **Human anatomy** setting in **Scene** defaults to **High priority**. Nova reads the current positive prompt, including any prompt applied from your story, and adds weighted exclusions for common anatomy artifacts. Mentions of hand actions, close-ups, or full-body and foot details add more targeted terms. Your custom negative prompt remains included, and an exact duplicate you wrote takes precedence over the automatic term.

Select **Preview combined negative prompt** to see the text ComfyUI will receive. Choose **Off** to use only your custom negative prompt. The setting is saved with generation settings and history. It guides the sampler during generation. Results still depend on the model, pose, references and seed, and Nova does not validate the anatomy of the finished image.

## Privacy and safeguards

- Both services bind to loopback in the documented setup. Nova has no accounts, telemetry, remote inference or public hosting mode. Its API checks the request host and protects write requests against cross-origin browser access. Use a trusted single-user machine or OS account.
- Normalized uploads and the SQLite database default to `~/.nova-stasis` (the current user’s home directory on Windows too). Keep `NOVA_DATA_DIR` outside any checkout or synced folder. Nova strips upload metadata during re-encoding.
- ComfyUI receives local copies in `input/nova_stasis/` and writes results in `output/nova_stasis/` under its configured input/output roots. Keep those directories outside version control. The `--disable-metadata` flag keeps the prompt graph out of exported ComfyUI PNG metadata. The history still stores prompts and settings locally.
- Removing a selection or deleting a profile/preset does not purge its underlying file or past jobs. To clear the studio, stop both services and remove the Nova private data directory and the two Nova subdirectories under ComfyUI’s input/output roots. For the managed installation, delete the entire launcher home only if you also want to remove the engine, models, dependencies and logs.
- Nova is for authorized, non-explicit adult subjects. Each submission requires an affirmative declaration and is screened server-side for common terms associated with minors, nudity, sexual content, nonconsent and deepfakes. This text filter is heuristic. It cannot establish age or consent, inspect uploads or results, or reliably detect obfuscated or multilingual prompts. Use this initial release privately with trusted users. A hosted rollout would need authentication, per-user storage and stronger reference and output moderation.

## Validation and troubleshooting

```sh
python -m unittest discover -s tests -v
node --check nova/web/studio.js
```

Run the optional Chromium hardware workflow test using a Python environment with the app requirements and Playwright installed:

```sh
python -m pip install playwright
python -m playwright install chromium
python tests/browser_hardware.py
```

This browser test uses simulated ComfyUI hardware reports to check startup defaults, warnings, rescan behavior, and the offline state.

The tests exercise launcher download verification and reuse, graph wiring, independent regional conditioning, local API behavior, saved records, and an inpaint/pose batch with a simulated ComfyUI API. To check actual quality, use two authorized adult references on a GPU with a matched stack, hold the seed fixed, and change one strength at a time. Check whether the expected subject changes while the other remains recognizable.

| Symptom | Check |
| --- | --- |
| ComfyUI offline | Confirm ComfyUI is reachable at the loopback origin in `.env`. |
| ComfyUI connected but models missing | Check the custom-node startup log and the checkpoint, adapter and CLIP Vision directories. |
| Shape or checkpoint errors | Confirm the checkpoint family, adapter and encoder combination, plus any optional ControlNet and LoRAs. |
| Generation failed or stalled | Check the ComfyUI terminal and queue. A ComfyUI restart before Nova sees the result can discard its in-memory history. A job with no result is marked failed after 24 hours when polled. |
| Faces or outfits blend | Try single-subject crops, clearer placement cues, and tighter masks aligned with the pose. |
