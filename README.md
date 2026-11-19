# Nova Stasis

Nova Stasis is a local browser studio for putting two reference characters into a new Stable Diffusion scene. GitHub hosts the source. The Python dashboard and ComfyUI both run on your machine.

## Install the dashboard

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

## Set up ComfyUI and models

1. Install [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) through ComfyUI Manager, or clone it into `ComfyUI/custom_nodes` following its installation instructions. Nova uses its `IPAdapterModelLoader` and `IPAdapterAdvanced` nodes.
2. Put a standard SDXL or SD 1.5 checkpoint in `ComfyUI/models/checkpoints`. Start with a text-to-image base checkpoint or a compatible realism- or illustration-tuned checkpoint. The workflow uses ordinary SD latent sampling, including during masked image edits. SDXL refiners, other diffusion families, and dedicated extra-channel inpaint checkpoints are outside this initial workflow.
3. Download the [ViT-H image encoder](https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors) and save it as `ComfyUI/models/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`. Choose the matching adapter below and place it in `ComfyUI/models/ipadapter`.

| Checkpoint family | Adapter for the ViT-H image encoder |
| --- | --- |
| SDXL | [ip-adapter-plus_sdxl_vit-h.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors) |
| SD 1.5 | [ip-adapter-plus_sd15.safetensors](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors) |

Optional weights go in `ComfyUI/models/loras` and `ComfyUI/models/controlnet`. For pose conditioning, use an OpenPose ControlNet compatible with your checkpoint family. For example, see [xinsir SDXL OpenPose](https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0) or [ControlNet v1.1 SD 1.5 OpenPose](https://huggingface.co/lllyasviel/control_v11p_sd15_openpose). Upload a rendered OpenPose skeleton map prepared in an editor or with a preprocessor such as [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux). Nova accepts the rendered map directly.

Restart ComfyUI after adding custom nodes. For a manual installation, run the following from the ComfyUI directory with its Python environment active.

```sh
python main.py --listen 127.0.0.1 --port 8188 --disable-metadata
```

Select **Refresh** in Nova to reload the model lists. The lists reflect installed files. You choose which checkpoint, adapter, encoder, LoRA and ControlNet are compatible. Only download weights and executable custom nodes from sources you trust and follow their licenses.

For a first run, use batch 1. Try 512 × 512 for SD 1.5 or 1024 × 768 for SDXL. A compatible GPU is the practical option. Actual VRAM use depends on the checkpoint, canvas size, batch size, ControlNet, and ComfyUI memory settings. If you run out of VRAM, reduce the batch or canvas size, then retry. CPU mode is possible on supported ComfyUI installations and can be much slower.

## Generate and reuse scenes

1. Upload one clear reference for each character. Nova accepts still PNG, JPEG and WebP up to 16 MiB and 20 megapixels. Tight crops with distinct features, clothing or silhouettes generally work best.
2. Pick layout and describe both subjects in the prompt, matching their positions and naming distinct visual traits. Each reference is encoded by its own IP-Adapter application with its own weight and attention mask. The default left/right or top/bottom masks blend slightly along the center. In **Mask & LoRA**, upload custom masks for other compositions. White applies the reference, black excludes it, and gray scales it. Use a canvas-aligned mask for each character.
3. Set reference strengths independently. Optional LoRAs are applied to the shared checkpoint and text encoder, so their effects can reach both subjects. The adapter masks are the spatial identity controls. Similar faces, crowded subjects or overlapping regions can still drift or blend.
4. To control position, use **Pose control** and select both a skeleton map and matching OpenPose model. Align each skeleton with that character’s attention mask. For **Image to image**, upload a source and choose a denoise value. In **Inpainting**, also upload a same-size grayscale mask with white over the edit and black elsewhere. Use identical orientation for the source and mask. Nova center-crops and resizes them together to the output dimensions.
5. Leave **Seed** blank for a random seed. Use **Batch size** to request up to four variants in the same sampler run. A history entry records the starting seed and settings for the entire batch. Repeatability depends on the same models, settings and inference environment.
6. Confirm that you have permission to use the references, everyone depicted is an adult, and the scene is non-explicit. Select **Generate scene**. The dashboard polls ComfyUI and shows completed images in **Preview** and **Generation history**. You can download each result or reuse its settings and seed.
7. Save reference images and strengths through **Character profiles**, including an optional LoRA selection. **Saved settings** stores the full prompt and generation settings, including local references. Each generation requires a fresh attestation.

## Privacy and safeguards

- Both services bind to loopback in the documented setup. Nova has no accounts, telemetry, remote inference or public hosting mode. Its API checks the request host and protects write requests against cross-origin browser access. Use a trusted single-user machine or OS account.
- Normalized uploads and the SQLite database default to `~/.nova-stasis` (the current user’s home directory on Windows too). Keep `NOVA_DATA_DIR` outside any checkout or synced folder. Nova strips upload metadata during re-encoding.
- ComfyUI receives local copies in `input/nova_stasis/` and writes results in `output/nova_stasis/` under its configured input/output roots. Keep those directories outside version control. The `--disable-metadata` flag keeps the prompt graph out of exported ComfyUI PNG metadata. The history still stores prompts and settings locally.
- Removing a selection or deleting a profile/preset does not purge its underlying file or past jobs. To clear the studio, stop both services and remove the Nova private data directory and the two Nova subdirectories under ComfyUI’s input/output roots.
- Nova is for authorized, non-explicit adult subjects. Each submission requires an affirmative declaration and is screened server-side for common terms associated with minors, nudity, sexual content, nonconsent and deepfakes. This text filter is heuristic. It cannot establish age or consent, inspect uploads or results, or reliably detect obfuscated or multilingual prompts. Use this initial release privately with trusted users. A hosted rollout would need authentication, per-user storage and stronger reference and output moderation.

## Validation and troubleshooting

```sh
python -m unittest discover -s tests -v
node --check nova/web/studio.js
```

The tests exercise graph wiring, independent regional conditioning, local API behavior, saved records, and an inpaint/pose batch with a simulated ComfyUI API. To check actual quality, use two authorized adult references on a GPU with a matched stack, hold the seed fixed, and change one strength at a time. Check whether the expected subject changes while the other remains recognizable.

| Symptom | Check |
| --- | --- |
| ComfyUI offline | Confirm ComfyUI is reachable at the loopback origin in `.env`. |
| ComfyUI connected but models missing | Check the custom-node startup log and the checkpoint, adapter and CLIP Vision directories. |
| Shape or checkpoint errors | Confirm the checkpoint family, adapter and encoder combination, plus any optional ControlNet and LoRAs. |
| Generation failed or stalled | Check the ComfyUI terminal and queue. A ComfyUI restart before Nova sees the result can discard its in-memory history. A job with no result is marked failed after 24 hours when polled. |
| Faces or outfits blend | Try single-subject crops, clearer placement cues, and tighter masks aligned with the pose. |
