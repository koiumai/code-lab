#!/usr/bin/env python3
"""Local-only FLUX.2 Klein 4B quality / Turing-compatible quantization probe.

Suggested isolated venv (do not upgrade the working SDXL venv in place):
  pip install diffusers==0.43.0 transformers==5.22.2 accelerate==1.15.0 \
    bitsandbytes==0.50.2 safetensors pillow numpy sentencepiece
Keep a CUDA-enabled torch build >=2.4 with sm_75 support for an actual 2080.
NF4 FP16 + FP32 VAE path tested on A5000 under allocator limits.
Physical RTX 2080 hardware has NOT been tested.

BF16 reference: --mode bf16 --max-side 1024 (Ampere+ only, no 8GB claim).
NF4 probe: --mode nf4 --max-side 512 --budget-gib 8
NF4 uses FP16 computation, never BF16/FP8; text encoder is quantized, run once,
then deleted BEFORE the denoiser is loaded. Transformer and FP32 tiled VAE stay
on GPU. --nf4-compute fp32 is an overflow fallback, NOT an 8GB guarantee.

Raw output is an unrestricted edit, NOT a mask-preserving deliverable.
--allow-mask enables a separate exact-copy composite; --object-mask excludes the
object from it. Its quality still needs human review. No upload/publish/API call.
"""
import argparse, gc, importlib.metadata, json, math, subprocess, sys, threading, time
try:
    import resource
except ImportError:  # Windows does not provide the Unix resource module.
    resource = None
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

MODEL = 'black-forest-labs/FLUX.2-klein-4B'


def cpu_peak_kib():
    """Best-effort telemetry, not a prerequisite for inference.

    ru_maxrss is KiB on Linux and bytes on macOS. None means unavailable,
    not zero usage. Avoid an optional telemetry error hiding inference errors.
    """
    if resource is None:
        return None
    try:
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(value) / 1024 if sys.platform == 'darwin' else float(value)
    except (AttributeError, OSError, ValueError):
        return None

def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--prompt',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['bf16','nf4'],default='bf16')
    p.add_argument('--max-side',type=int,default=768)
    p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--steps',type=int,default=4)
    p.add_argument('--budget-gib',type=float,default=0)
    p.add_argument('--nf4-compute',choices=['fp16','fp32'],default='fp16')
    p.add_argument('--sequence-length',type=int,default=512)
    p.add_argument('--revision',default=None)
    p.add_argument('--allow-mask')
    p.add_argument('--object-mask')
    return p.parse_args()

def main():
    a=parse_args()
    if a.max_side < 64 or not 1 <= a.sequence_length <= 512 or a.steps < 1:
        raise ValueError('Invalid max-side/sequence-length/steps')
    if not math.isfinite(a.budget_gib) or a.budget_gib < 0:
        raise ValueError('budget-gib must be finite and non-negative')
    import torch
    from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2
    from diffusers import BitsAndBytesConfig as DBnb
    from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast, BitsAndBytesConfig as TBnb
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    if a.mode == 'bf16' and torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError('BF16 reference requires Ampere+; use nf4 mode on RTX2080')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    total=torch.cuda.get_device_properties(0).total_memory
    if a.budget_gib:
        torch.cuda.set_per_process_memory_fraction(min(a.budget_gib*1024**3/total,1))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    src=ImageOps.exif_transpose(Image.open(a.input)).convert('RGB')
    # Match aspect ratio approximately on the model grid; preserve original separately.
    scale=a.max_side/max(src.size)
    w,h=[max(16,int(round(v*scale/16))*16) for v in src.size]
    inp=src.resize((w,h),Image.Resampling.LANCZOS)
    inp.save(out/'model-input.png');src.save(out/'original.png')
    allowed=None
    if a.allow_mask:
        mask=Image.open(a.allow_mask).convert('L')
        if mask.size != src.size: raise ValueError('allow-mask must match original dimensions')
        allowed=np.asarray(mask)>127
        if a.object_mask:
            obj=Image.open(a.object_mask).convert('L')
            if obj.size != src.size: raise ValueError('object-mask must match original dimensions')
            allowed &= np.asarray(obj)==0
    stop=threading.Event();samples=[];started=time.time()
    def monitor():
        while not stop.is_set():
            try:
                val=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits','-i','0'],text=True,timeout=3).strip()
                samples.append({'seconds':time.time()-started,'memory_mib_util':val})
            except Exception: pass
            stop.wait(.5)
    threading.Thread(target=monitor,daemon=True).start()
    versions={p:importlib.metadata.version(p) for p in ['torch','diffusers','transformers','accelerate','bitsandbytes']}
    report={'args':vars(a),'versions':versions,'gpu':torch.cuda.get_device_name(0),'compute_capability':torch.cuda.get_device_capability(), 'total_gpu_bytes':total,'original_size':src.size,'model_size':[w,h],'records':[],'note':'Allocator cap on A5000 is not a physical RTX2080 test; raw images are not mask preserved.'}
    rev={'revision':a.revision} if a.revision else {}
    def save_report():
        report['elapsed_seconds']=time.time()-started
        report['cpu_peak_kib']=cpu_peak_kib()
        report['cpu_peak_available']=report['cpu_peak_kib'] is not None
        report['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        (out/'metrics.json').write_text(json.dumps(report,indent=2))
        (out/'nvidia-smi.json').write_text(json.dumps(samples))
    try:
        torch.cuda.reset_peak_memory_stats()
        emb=None
        if a.mode=='bf16':
            pipe=Flux2KleinPipeline.from_pretrained(MODEL,torch_dtype=torch.bfloat16,**rev)
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()
        else:
            dtype=torch.float16 if a.nf4_compute=='fp16' else torch.float32
            # Two separate 4-bit configs because Transformers and Diffusers use different classes.
            kwargs=dict(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
            tok=Qwen2TokenizerFast.from_pretrained(MODEL,subfolder='tokenizer',**rev)
            enc=Qwen3ForCausalLM.from_pretrained(MODEL,subfolder='text_encoder',quantization_config=TBnb(**kwargs),torch_dtype=dtype,device_map={'':0},attn_implementation='sdpa',**rev)
            enc.eval()
            with torch.inference_mode():
                # Use the pipeline's actual chat template / hidden layers, not guessed token embeddings.
                emb=Flux2KleinPipeline._get_qwen3_prompt_embeds(text_encoder=enc,tokenizer=tok,prompt=a.prompt,dtype=dtype,device=torch.device('cuda'),max_sequence_length=a.sequence_length).detach().cpu()
            if not torch.isfinite(emb).all(): raise RuntimeError('Non-finite text embeddings; retry --nf4-compute fp32')
            torch.save(emb,out/'prompt-embeds.pt')
            report['text_encoding_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            del enc,tok;gc.collect();torch.cuda.empty_cache()
            denoiser=Flux2Transformer2DModel.from_pretrained(MODEL,subfolder='transformer',quantization_config=DBnb(**kwargs),torch_dtype=dtype,device_map={'':0},**rev)
            vae=AutoencoderKLFlux2.from_pretrained(MODEL,subfolder='vae',torch_dtype=torch.float32,**rev)
            vae.enable_tiling();vae.to('cuda')
            # Klein's current decode casts device but not dtype; retain FP32 VAE stability.
            original_decode=vae.decode
            def decode_fp32(z,*args,**kw):
                return original_decode(z.to(dtype=torch.float32),*args,**kw)
            vae.decode=decode_fp32
            pipe=Flux2KleinPipeline.from_pretrained(MODEL,transformer=denoiser,vae=vae,text_encoder=None,tokenizer=None,torch_dtype=dtype,**rev)
            del denoiser,vae
        if not pipe.config.is_distilled:
            raise RuntimeError('Expected distilled Klein4B, not base variant; inspect checkpoint/config')
        def finite_callback(p,i,t,kw):
            if not torch.isfinite(kw['latents']).all():
                raise RuntimeError(f'Non-finite latents at step {i}; retry FP32 compute')
            return kw
        for seed in a.seeds:
            start=time.time()
            inputs={'prompt':a.prompt} if emb is None else {'prompt_embeds':emb.to(device='cuda')}
            with torch.inference_mode():
                arr=pipe(image=inp,height=h,width=w,num_inference_steps=a.steps,guidance_scale=1.0,generator=torch.Generator('cpu').manual_seed(seed),max_sequence_length=a.sequence_length,output_type='np',callback_on_step_end=finite_callback,**inputs).images[0]
            if not np.isfinite(arr).all():raise RuntimeError('Non-finite VAE output')
            raw=Image.fromarray(np.uint8(np.clip(arr*255,0,255)))
            raw.save(out/f'seed{seed}-raw.png')
            rec={'seed':seed,'seconds':time.time()-start,'raw_min':float(arr.min()),'raw_max':float(arr.max()),'raw_std':float(arr.std())}
            if allowed is not None:
                pred=np.asarray(raw.resize(src.size,Image.Resampling.LANCZOS))
                composite=np.asarray(src).copy();composite[allowed]=pred[allowed]
                assert np.array_equal(composite[~allowed],np.asarray(src)[~allowed])
                Image.fromarray(composite).save(out/f'seed{seed}-masked.png')
                rec['outside_allowed_changed']=0
            report['records'].append(rec);save_report();print(json.dumps(rec),flush=True)
    except BaseException as e:
        report['error']=repr(e);raise
    finally:
        stop.set();save_report()

if __name__=='__main__':main()
