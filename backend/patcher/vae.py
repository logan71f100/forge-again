import torch
import math
import time
import itertools

from tqdm import trange
from backend import memory_management
from backend.patcher.base import ModelPatcher


@torch.inference_mode()
def tiled_scale_multidim(samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu"):
    dims = len(tile)
    output = torch.empty([samples.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), samples.shape[2:])), device=output_device)

    for b in trange(samples.shape[0]):
        s = samples[b:b + 1]
        out = torch.zeros([s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])), device=output_device)
        out_div = torch.zeros([s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])), device=output_device)

        for it in itertools.product(*map(lambda a: range(0, a[0], a[1] - overlap), zip(s.shape[2:], tile))):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap, it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(pos * upscale_amount))
            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)
            feather = round(overlap * upscale_amount)
            for t in range(feather):
                for d in range(2, dims + 2):
                    m = mask.narrow(d, t, 1)
                    m *= ((1.0 / feather) * (t + 1))
                    m = mask.narrow(d, mask.shape[d] - 1 - t, 1)
                    m *= ((1.0 / feather) * (t + 1))

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o += ps * mask
            o_d += mask

        output[b:b + 1] = out / out_div
    return output


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap=8, upscale_amount=4, out_channels=3, output_device="cpu"):
    return tiled_scale_multidim(samples, function, (tile_y, tile_x), overlap, upscale_amount, out_channels, output_device)


class VAE:
    def __init__(self, model=None, device=None, dtype=None, no_init=False):
        if no_init:
            return

        self.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * memory_management.dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * memory_management.dtype_size(dtype)
        self.downscale_ratio = int(2 ** (len(model.config.down_block_types) - 1))
        self.latent_channels = int(model.config.latent_channels)

        self.first_stage_model = model.eval()

        if device is None:
            device = memory_management.vae_device()

        self.device = device
        offload_device = memory_management.vae_offload_device()

        if dtype is None:
            dtype = memory_management.vae_dtype()

        self.vae_dtype = dtype
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = memory_management.intermediate_device()

        self.patcher = ModelPatcher(
            self.first_stage_model,
            load_device=self.device,
            offload_device=offload_device
        )

    def clone(self):
        n = VAE(no_init=True)
        n.patcher = self.patcher.clone()
        n.memory_used_encode = self.memory_used_encode
        n.memory_used_decode = self.memory_used_decode
        n.downscale_ratio = self.downscale_ratio
        n.latent_channels = self.latent_channels
        n.first_stage_model = self.first_stage_model
        n.device = self.device
        n.vae_dtype = self.vae_dtype
        n.output_device = self.output_device
        return n

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
        steps = samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
        steps += samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)

        decode_fn = lambda a: (self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)) + 1.0).float()
        output = torch.clamp(((tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device) +
                               tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device) +
                               tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device))
                              / 3.0) / 2.0, min=0.0, max=1.0)
        return output

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        steps = pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap)
        steps += pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x * 2, tile_y // 2, overlap)

        encode_fn = lambda a: self.first_stage_model.encode((2. * a - 1.).to(self.vae_dtype).to(self.device)).float()
        samples = tiled_scale(pixel_samples, encode_fn, tile_x, tile_y, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples += tiled_scale(pixel_samples, encode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples += tiled_scale(pixel_samples, encode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples /= 3.0
        return samples

    def decode_inner(self, samples_in):
        if memory_management.VAE_ALWAYS_TILED:
            return self.decode_tiled(samples_in).to(self.output_device)

        try:
            memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
            # The worst-case decode estimate is enormous (~9 GB for a 1 MP fp32
            # decode: 2178*h*w*64*dtype). Passing it wholesale as memory_required
            # made load_models_gpu EVICT THE DIFFUSION MODEL at the end of every
            # run just to decode one image; the next run then paid a full weight
            # reload ("Moving model(s)" every run) or sampled partially
            # CPU-swapped at ~10 s/it. When other models are resident, request
            # only the actual headroom: the VAE weights still load (the
            # minimum-inference floor covers them), the decode batch size below
            # adapts to real free memory, and a genuine shortfall lands in the
            # existing tiled-decode fallback instead of nuking resident weights.
            request = memory_used
            if memory_management.current_loaded_models:
                free_now = memory_management.get_free_memory(self.device)
                request = min(memory_used, max(0, free_now - (1024 * 1024 * 1024)))
            memory_management.load_models_gpu([self.patcher], memory_required=request)
            free_memory = memory_management.get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            # CALIBRATION (temporary): what the estimate claims, what is really
            # free, and -- after the decode -- what it actually peaked at. The
            # estimate is a deliberate worst case (2178*h*w*64*dtype, ~5.7 GB for
            # a 1304x1000 fp16 decode), so "estimate exceeds free" cannot on its
            # own decide to tile until we know how conservative it is.
            # Measured 2026-08-16 on an 11 GB card, Chroma resident: this
            # batch_number is 0 for EVERY decode from 1024x1024 upward, including
            # ones that finish in 1.4 s. memory_used is a whole-VAE worst case
            # (8.7 GB at 1 MP, 10.8 GB at 1304x1000) used to SPLIT BATCHES, and on
            # a consumer card it never approaches free memory -- so it says
            # nothing about whether one image fits, and tiling on `< 1` would
            # tile every single run. What actually matters is the MARGINAL cost
            # of the decode against what is free, which is what this logs.
            _mb = lambda b: b / (1024 * 1024)
            _t0 = time.time()
            _cuda = torch.cuda.is_available() and self.device.type == 'cuda'
            _alloc0 = torch.cuda.memory_allocated(self.device) if _cuda else 0
            if _cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
            batch_number = max(1, batch_number)

            pixel_samples = torch.empty((samples_in.shape[0], 3, round(samples_in.shape[2] * self.downscale_ratio), round(samples_in.shape[3] * self.downscale_ratio)), device=self.output_device)
            for x in range(0, samples_in.shape[0], batch_number):
                samples = samples_in[x:x + batch_number].to(self.vae_dtype).to(self.device)
                pixel_samples[x:x + batch_number] = torch.clamp((self.first_stage_model.decode(samples).to(self.output_device).float() + 1.0) / 2.0, min=0.0, max=1.0)
            # Report only when it MATTERS -- a decode that is slow, or one that
            # asked for more than was free. Measured 2026-08-16 (Chroma, 11 GB
            # card, 5 resolutions from 1304x1000 to 2304x1728): the marginal
            # cost is 0.1563 MB per latent pixel for fp16, to four figures, and
            # exceeding free VRAM degrades GRACEFULLY -- 4.3 GB over cost 7.7 s,
            # because the memory manager evicts the diffusion model to make room.
            # So overshoot alone is NOT the "stuck at the VAE" report; something
            # else has to be holding the VRAM that eviction would otherwise free
            # (the assistant's llama-server is the standing suspect). This line
            # is what will say so, from a real occurrence, in one look.
            _dt = time.time() - _t0
            if _cuda:
                _marginal = torch.cuda.max_memory_allocated(self.device) - _alloc0
                if _dt > 2.0 or _marginal > free_memory:
                    print(f"[VAE decode] latent {tuple(samples_in.shape[-2:])} took {_dt:.2f}s "
                          f"-- needed {_mb(_marginal):.0f} MB on top of "
                          f"{_mb(_alloc0):.0f} MB resident, {_mb(free_memory):.0f} MB was free")
        except memory_management.OOM_EXCEPTION as e:
            print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            # Give the retry somewhere to work. The failed attempt leaves its
            # fragmented cache behind, and the diffusion model is still resident
            # -- a decode deliberately does NOT evict it (see the request cap
            # above) -- so the tiled path, which exists for exactly this case,
            # was retrying inside the same exhausted allocator and OOMing on a
            # 256 MB tile. By the time a decode runs sampling is over and those
            # weights are dead space: the cost of dropping them is one reload on
            # the next run, against a generation that otherwise fails outright
            # AND hands the next run an allocator with ~1 GB free -- which on
            # Windows means silent sysmem fallback and a run that looks hung
            # (measured 2026-09-04: 20 steps at 1304x2048, 8-hour ETA, 0%).
            memory_management.free_memory(0, self.device, free_all=True)
            memory_management.soft_empty_cache(force=True)
            memory_management.load_models_gpu([self.patcher], memory_required=0)
            pixel_samples = self.decode_tiled_(samples_in)

        pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
        return pixel_samples

    def decode(self, samples_in):
        wrapper = self.patcher.model_options.get('model_vae_decode_wrapper', None)
        if wrapper is None:
            return self.decode_inner(samples_in)
        else:
            return wrapper(self.decode_inner, samples_in)

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap=16):
        memory_management.load_model_gpu(self.patcher)
        output = self.decode_tiled_(samples, tile_x, tile_y, overlap)
        return output.movedim(1, -1)

    def encode_inner(self, pixel_samples):
        if memory_management.VAE_ALWAYS_TILED:
            return self.encode_tiled(pixel_samples)

        regulation = self.patcher.model_options.get("model_vae_regulation", None)

        pixel_samples = pixel_samples.movedim(-1, 1)
        try:
            memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
            memory_management.load_models_gpu([self.patcher], memory_required=memory_used)
            free_memory = memory_management.get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)
            samples = torch.empty((pixel_samples.shape[0], self.latent_channels, round(pixel_samples.shape[2] // self.downscale_ratio), round(pixel_samples.shape[3] // self.downscale_ratio)), device=self.output_device)
            for x in range(0, pixel_samples.shape[0], batch_number):
                pixels_in = (2. * pixel_samples[x:x + batch_number] - 1.).to(self.vae_dtype).to(self.device)
                samples[x:x + batch_number] = self.first_stage_model.encode(pixels_in, regulation).to(self.output_device).float()

        except memory_management.OOM_EXCEPTION as e:
            print("Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.")
            # Same as the decode fallback above: free the room the retry needs
            # before asking it to do the same work in smaller pieces.
            memory_management.free_memory(0, self.device, free_all=True)
            memory_management.soft_empty_cache(force=True)
            memory_management.load_models_gpu([self.patcher], memory_required=0)
            samples = self.encode_tiled_(pixel_samples)

        return samples

    def encode(self, pixel_samples):
        wrapper = self.patcher.model_options.get('model_vae_encode_wrapper', None)
        if wrapper is None:
            return self.encode_inner(pixel_samples)
        else:
            return wrapper(self.encode_inner, pixel_samples)

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        memory_management.load_model_gpu(self.patcher)
        pixel_samples = pixel_samples.movedim(-1, 1)
        samples = self.encode_tiled_(pixel_samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap)
        return samples
