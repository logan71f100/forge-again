import os
import sys
from collections import namedtuple
from pathlib import Path
import re

import torch
import torch.hub

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from modules import devices, paths, shared, modelloader, errors
from backend import memory_management
from backend.patcher.base import ModelPatcher


blip_image_eval_size = 384
clip_model_name = 'ViT-L/14'

Category = namedtuple("Category", ["name", "topn", "items"])

re_topn = re.compile(r"\.top(\d+)$")

def category_types():
    return [f.stem for f in Path(shared.interrogator.content_dir).glob('*.txt')]


def download_default_clip_interrogate_categories(content_dir):
    print("Downloading CLIP categories...")

    tmpdir = f"{content_dir}_tmp"
    category_types = ["artists", "flavors", "mediums", "movements"]

    try:
        os.makedirs(tmpdir, exist_ok=True)
        for category_type in category_types:
            torch.hub.download_url_to_file(f"https://raw.githubusercontent.com/pharmapsychotic/clip-interrogator/main/clip_interrogator/data/{category_type}.txt", os.path.join(tmpdir, f"{category_type}.txt"))
        os.rename(tmpdir, content_dir)

    except Exception as e:
        errors.display(e, "downloading default CLIP interrogate categories")
    finally:
        if os.path.exists(tmpdir):
            os.removedirs(tmpdir)


class InterrogateModels:
    blip_model = None
    clip_model = None
    clip_preprocess = None
    dtype = None
    running_on_cpu = None

    def __init__(self, content_dir):
        self.loaded_categories = None
        self.skip_categories = []
        self.content_dir = content_dir

        self.load_device = memory_management.text_encoder_device()
        self.offload_device = memory_management.text_encoder_offload_device()
        self.dtype = torch.float32

        if memory_management.should_use_fp16(device=self.load_device):
            self.dtype = torch.float16

        self.blip_patcher = None
        self.clip_patcher = None

    def categories(self):
        if not os.path.exists(self.content_dir):
            download_default_clip_interrogate_categories(self.content_dir)

        if self.loaded_categories is not None and self.skip_categories == shared.opts.interrogate_clip_skip_categories:
           return self.loaded_categories

        self.loaded_categories = []

        if os.path.exists(self.content_dir):
            self.skip_categories = shared.opts.interrogate_clip_skip_categories
            category_types = []
            for filename in Path(self.content_dir).glob('*.txt'):
                category_types.append(filename.stem)
                if filename.stem in self.skip_categories:
                    continue
                m = re_topn.search(filename.stem)
                topn = 1 if m is None else int(m.group(1))
                with open(filename, "r", encoding="utf8") as file:
                    lines = [x.strip() for x in file.readlines()]

                self.loaded_categories.append(Category(name=filename.stem, topn=topn, items=lines))

        return self.loaded_categories

    def create_fake_fairscale(self):
        class FakeFairscale:
            def checkpoint_wrapper(self):
                pass

        sys.modules["fairscale.nn.checkpoint.checkpoint_activations"] = FakeFairscale

    def load_blip_model(self):
        self.create_fake_fairscale()

        # The bundled BLIP (repositories/BLIP/models/med.py) does
        #   from transformers.modeling_utils import (apply_chunking_to_forward,
        #       find_pruneable_heads_and_indices, prune_linear_layer, ...)
        # but transformers 5 moved apply_chunking_to_forward and prune_linear_layer
        # to transformers.pytorch_utils and removed find_pruneable_heads_and_indices
        # outright -- so CLIP interrogate crashed with ImportError. Restore the
        # old import path: re-export the two that moved, and provide the removed
        # one from its historical (stable) implementation.
        import transformers.modeling_utils as _mu
        import transformers.pytorch_utils as _pu

        def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        _compat = {
            "apply_chunking_to_forward": getattr(_pu, "apply_chunking_to_forward", None),
            "prune_linear_layer": getattr(_pu, "prune_linear_layer", None),
            "find_pruneable_heads_and_indices": getattr(
                _pu, "find_pruneable_heads_and_indices", _find_pruneable_heads_and_indices),
        }
        for _name, _fn in _compat.items():
            if not hasattr(_mu, _name) and _fn is not None:
                setattr(_mu, _name, _fn)

        import models.blip

        # BLIP ships two captioners. The original code hardwired the base one,
        # whose captions are short and confidently wrong often enough to make
        # the feature hard to trust -- and raising min_length on it only makes
        # it pad a weak guess out into a longer weak guess. The large model
        # (ViT-L/16, 24 layers) is the same pipeline with a much stronger
        # vision tower, so it is the one real quality lever here.
        variant = getattr(shared.opts, 'interrogate_blip_variant', 'large')
        if variant == 'base':
            download_name = 'model_base_caption_capfilt_large.pth'
            vit = 'base'
        else:
            download_name = 'model_large_caption.pth'
            vit = 'large'
        model_dir = os.path.join(paths.models_path, "BLIP")

        files = modelloader.load_models(
            model_path=model_dir,
            model_url=f'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/{download_name}',
            ext_filter=[".pth"],
            download_name=download_name,
        )

        # load_models returns EVERY .pth in the directory, so once both
        # captioners are present, files[0] is whichever sorted first -- taking
        # it would silently load base while the setting says large (and blow up
        # on the 768-vs-1024 vision width). Pick the requested file by name.
        wanted = os.path.join(model_dir, download_name)
        path = next((f for f in files if os.path.normcase(os.path.basename(f)) == os.path.normcase(download_name)), None)
        if path is None:
            path = wanted if os.path.exists(wanted) else files[0]

        print(f'[Interrogate] Loading BLIP caption model: {os.path.basename(path)} (vit={vit})')
        blip_model = models.blip.blip_decoder(pretrained=path, image_size=blip_image_eval_size, vit=vit, med_config=os.path.join(paths.paths["BLIP"], "configs", "med_config.json"))
        blip_model.eval()

        # Stamped here rather than at the call site so that EVERY route which
        # loads a captioner records which one it got -- including SUPIR's boot
        # pre-warm, which assigns interrogator.blip_model directly.
        self.blip_variant_loaded = variant

        return blip_model

    def load_clip_model(self):
        import clip
        import clip.model

        clip.model.LayerNorm = torch.nn.LayerNorm

        model, preprocess = clip.load(clip_model_name, device="cpu", download_root=shared.cmd_opts.clip_models_path)
        model.eval()

        return model, preprocess

    def load(self):
        # The models and their patchers are keyed SEPARATELY: SUPIR's boot
        # pre-warm assigns interrogator.blip_model directly (so captions are
        # fast), and keying the patcher on the model then skipped its creation
        # entirely -- load_models_gpu received None and interrogate died with
        # "'NoneType' object has no attribute 'load_device'". The .to() also
        # has to run for a pre-populated model, which arrives on whatever
        # device and dtype the pre-warmer left it in; .to() is a no-op when
        # nothing needs changing.
        # Switching the BLIP variant has to drop the cached model, or the
        # setting silently does nothing: the first captioner loaded stays for
        # the life of the process and every later "change" returns byte-identical
        # captions (how this was caught -- base and large scoring the same on
        # three test images to the character). load_blip_model() stamps
        # blip_variant_loaded itself, so a model handed to us by SUPIR's
        # pre-warm carries its variant too and is not needlessly reloaded.
        variant = getattr(shared.opts, 'interrogate_blip_variant', 'large')
        if self.blip_model is not None and getattr(self, 'blip_variant_loaded', variant) != variant:
            self.blip_model = None
            self.blip_patcher = None

        if self.blip_model is None:
            self.blip_model = self.load_blip_model()

        # Check the DTYPE, not just whether a patcher exists. SUPIR's boot
        # pre-warm (scripts/supir.py) assigns the model straight from
        # load_blip_model() without converting it, so it can arrive as fp32
        # while the image tensor below is built as self.dtype -- which surfaces
        # as "Input type (struct c10::Half) and bias type (float) should be the
        # same" from the first conv, well away from the actual cause. .to() is a
        # no-op when nothing needs changing, and returns the same module object,
        # so an existing patcher stays valid across the conversion.
        if next(self.blip_model.parameters()).dtype != self.dtype:
            self.blip_model = self.blip_model.to(device=self.offload_device, dtype=self.dtype)

        if self.blip_patcher is None:
            self.blip_model = self.blip_model.to(device=self.offload_device, dtype=self.dtype)
            self.blip_patcher = ModelPatcher(self.blip_model, load_device=self.load_device, offload_device=self.offload_device)

        if self.clip_model is None:
            self.clip_model, self.clip_preprocess = self.load_clip_model()

        if self.clip_patcher is None:
            self.clip_model = self.clip_model.to(device=self.offload_device, dtype=self.dtype)
            self.clip_patcher = ModelPatcher(self.clip_model, load_device=self.load_device, offload_device=self.offload_device)

        memory_management.load_models_gpu([self.blip_patcher, self.clip_patcher])
        return

    def send_clip_to_ram(self):
        pass

    def send_blip_to_ram(self):
        pass

    def unload(self):
        pass

    def rank(self, image_features, text_array, top_count=1):
        import clip

        devices.torch_gc()

        if shared.opts.interrogate_clip_dict_limit != 0:
            text_array = text_array[0:int(shared.opts.interrogate_clip_dict_limit)]

        top_count = min(top_count, len(text_array))
        text_tokens = clip.tokenize(list(text_array), truncate=True).to(self.load_device)
        text_features = self.clip_model.encode_text(text_tokens).type(self.dtype)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = torch.zeros((1, len(text_array))).to(self.load_device)
        for i in range(image_features.shape[0]):
            similarity += (100.0 * image_features[i].unsqueeze(0) @ text_features.T).softmax(dim=-1)
        similarity /= image_features.shape[0]

        top_probs, top_labels = similarity.cpu().topk(top_count, dim=-1)
        return [(text_array[top_labels[0][i].numpy()], (top_probs[0][i].numpy()*100)) for i in range(top_count)]

    def generate_caption(self, pil_image):
        gpu_image = transforms.Compose([
            transforms.Resize((blip_image_eval_size, blip_image_eval_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])(pil_image).unsqueeze(0).type(self.dtype).to(self.load_device)

        # int() on ALL THREE. These are gr.Slider settings, so they come back as
        # floats (96.0), and transformers 5 beam search does
        #   torch.full(size=(..., max_length), ...)
        # which rejects a float outright:
        #   TypeError: full(): argument 'size' failed to unpack the object at
        #   pos 3 with error "type must be tuple of ints, but got float"
        # num_beams and min_length were already cast; max_length was not, so the
        # button died for anyone whose config had ever been written by the
        # settings UI or the options API.
        with torch.no_grad():
            caption = self.blip_model.generate(
                gpu_image, sample=False,
                num_beams=int(shared.opts.interrogate_clip_num_beams),
                min_length=int(shared.opts.interrogate_clip_min_length),
                max_length=int(shared.opts.interrogate_clip_max_length),
            )

        return caption[0]

    def interrogate(self, pil_image):
        res = ""
        shared.state.begin(job="interrogate")
        try:
            self.load()

            caption = self.generate_caption(pil_image)
            self.send_blip_to_ram()
            devices.torch_gc()

            res = caption

            clip_image = self.clip_preprocess(pil_image).unsqueeze(0).type(self.dtype).to(self.load_device)

            with torch.no_grad(), devices.autocast():
                image_features = self.clip_model.encode_image(clip_image).type(self.dtype)

                image_features /= image_features.norm(dim=-1, keepdim=True)

                for cat in self.categories():
                    matches = self.rank(image_features, cat.items, top_count=cat.topn)
                    for match, score in matches:
                        if shared.opts.interrogate_return_ranks:
                            res += f", ({match}:{score/100:.3f})"
                        else:
                            res += f", {match}"

        except Exception:
            errors.report("Error interrogating", exc_info=True)
            res += "<error>"

        self.unload()
        shared.state.end()

        return res
