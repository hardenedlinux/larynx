#!/usr/bin/env python3
"""Phase-4 end-to-end acceptance: C++ CLI (LLM -> Flow -> HiFT) vs Python.

The C++ CLI runs the *whole* chain natively (Qwen2Tokenizer -> LLM sampling ->
Flow -> HiFT), which is the deliverable of Phase 3. This script cross-checks the
two deterministic decoder stages against the PyTorch reference, holding the
*other* side's inputs fixed so each gap is attributable to a single decoder:

  1. Flow  — feed the CLI's generated speech tokens to the Python flow (same
             prompt bundle, same deterministic CFM noise) and compare its mel to
             the CLI's dumped mel.
  2. HiFT  — feed the CLI's mel to the Python HiFT (same SineGen2 buffers,
             loaded from hift_source.bin) and compare its waveform to the CLI's
             raw PCM.

The LLM sampling itself is stochastic and is excluded here — it is already
covered by verify_generate.py / verify_llm_sampling.py. Everything below is
deterministic, so the expected gap is float32 accumulation (~1e-4..1e-3, see
verify_flow.py), not a bug.

Runs under the CosyVoice python3.10 env (torch 2.3.1+cu121), NOT the larynx
``.venv``. CPU-only (CUDA_VISIBLE_DEVICES unset): the DiT graph at full scale
does not fit the ~2.4 GiB the decoders' GPU load leaves free, and CPU keeps the
comparison deterministic.

Usage:
    ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 \
        tests/verify_e2e.py [--cli build/larynx] [--out-dir wavs] [--text ...]
"""

import argparse
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSYVOICE_DIR = os.path.expanduser("~/Project/CosyVoice")
SITE_PACKAGES = os.path.join(COSYVOICE_DIR, ".local", "lib", "python3.10", "site-packages")
MODEL_DIR = os.path.join(COSYVOICE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")

SAMPLE_RATE = 24000
TOTAL_SCALE = 480            # HiFT samples per mel frame
MEL_DIM = 80
SPK_DIM = 192

# Asset formats (see tests/export_flow_noise.py / export_hift_source.py).
FNSE_MAGIC = 0x45534E46
FNSE_NOISE_MAX = 50 * 300
HSRC_MAGIC = 0x43525348
HSRC_HARMONIC_DIM = 9
HSRC_SINE_MAX = 300 * 24000


def _bootstrap():
    for p in (SITE_PACKAGES, COSYVOICE_DIR, os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS")):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(COSYVOICE_DIR)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU, deterministic


def load_flow_noise(path):
    import numpy as np
    with open(path, "rb") as f:
        magic, version, mel_dim = struct.unpack("<III", f.read(12))
        max_frames = struct.unpack("<Q", f.read(8))[0]
        assert magic == FNSE_MAGIC and version == 1 and mel_dim == MEL_DIM and max_frames == FNSE_NOISE_MAX
        return np.fromfile(f, dtype=np.float32, count=MEL_DIM * FNSE_NOISE_MAX).reshape(1, MEL_DIM, FNSE_NOISE_MAX)


def load_hift_source(path):
    import numpy as np
    with open(path, "rb") as f:
        magic, version, dim = struct.unpack("<III", f.read(12))
        sine_max = struct.unpack("<Q", f.read(8))[0]
        assert magic == HSRC_MAGIC and version == 1 and dim == HSRC_HARMONIC_DIM and sine_max == HSRC_SINE_MAX
        rand_ini = np.fromfile(f, dtype=np.float32, count=dim).reshape(1, dim)
        sine_waves = np.fromfile(f, dtype=np.float32, count=sine_max * dim).reshape(1, sine_max, dim)
    return rand_ini, sine_waves


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", default=os.path.join(ROOT, "build", "larynx"))
    ap.add_argument("--prompt-dir", default=os.path.join(ROOT, "wavs", "flow_inputs"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "wavs"))
    ap.add_argument("--llm-gguf", default=os.path.join(ROOT, "build", "llm.gguf"))
    ap.add_argument("--flow-gguf", default=os.path.join(ROOT, "build", "flow.gguf"))
    ap.add_argument("--hift-gguf", default=os.path.join(ROOT, "build", "hift.gguf"))
    ap.add_argument("--hift-source", default=os.path.join(ROOT, "build", "hift_source.bin"))
    ap.add_argument("--flow-noise", default=os.path.join(ROOT, "build", "flow_noise.bin"))
    ap.add_argument("--tokenizer-dir", default=os.path.join(ROOT, "build", "tokenizer"))
    ap.add_argument("--text", default="今天天气不错，我们一起去公园散步吧。")
    ap.add_argument("--instruct", default="You are a helpful assistant. 请用普通话表达。<|endofprompt|>")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _bootstrap()

    import numpy as np
    import torch
    import torchaudio
    from hyperpyyaml import load_hyperpyyaml

    torch.set_num_threads(1)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, "e2e")
    cli_wav = base + ".wav"
    cli_tokens = base + ".tokens.i32"
    cli_mel = base + ".mel.f32"
    cli_audio = base + ".audio.f32"

    # ---- 1. C++ CLI end-to-end -------------------------------------------------
    cmd = [args.cli,
           "--text", args.text,
           "--instruct", args.instruct,
           "--prompt-dir", args.prompt_dir,
           "--out", cli_wav,
           "--llm", args.llm_gguf,
           "--flow", args.flow_gguf,
           "--hift", args.hift_gguf,
           "--hift-source", args.hift_source,
           "--flow-noise", args.flow_noise,
           "--tokenizer-dir", args.tokenizer_dir,
           "--seed", str(args.seed),
           "--dump-tokens", cli_tokens,
           "--dump-mel", cli_mel,
           "--dump-audio", cli_audio]
    env = dict(os.environ, LARYNX_BACKEND="cpu")
    print(f"[1/4] running C++ CLI (LLM -> Flow -> HiFT):", flush=True)
    subprocess.run(cmd, check=True, env=env)

    tokens = np.fromfile(cli_tokens, dtype=np.int32)
    cli_mel_f = np.fromfile(cli_mel, dtype=np.float32)
    cli_audio_f = np.fromfile(cli_audio, dtype=np.float32)
    print(f"      tokens={tokens.size}  mel={cli_mel_f.size} floats  audio={cli_audio_f.size} samples", flush=True)

    # ---- 2. build Python flow + hift (no LLM) ----------------------------------
    print("[2/4] building Python flow + hift", flush=True)
    with open(os.path.join(MODEL_DIR, "cosyvoice3.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None})
    flow = configs["flow"]
    hift = configs["hift"]

    flow.load_state_dict(torch.load(os.path.join(MODEL_DIR, "flow.pt"),
                                    map_location="cpu", weights_only=True), strict=True)
    flow.to("cpu").eval()
    hift_state_dict = {k.replace("generator.", ""): v for k, v in
                       torch.load(os.path.join(MODEL_DIR, "hift.pt"),
                                  map_location="cpu", weights_only=True).items()}
    hift.load_state_dict(hift_state_dict, strict=True)
    hift.to("cpu").eval()

    # Freeze the two fixed RNG buffers to the same bytes the C++ side loads.
    noise = load_flow_noise(args.flow_noise)                       # (1, 80, 15000)
    fresh = flow.decoder.rand_noise.detach().cpu().numpy()          # seed-0, should match
    print(f"      flow_noise.bin vs fresh seed-0 rand_noise max|diff| = "
          f"{np.abs(fresh - noise).max():.3e}", flush=True)
    flow.decoder.rand_noise = torch.from_numpy(noise.copy())

    rand_ini, sine_waves = load_hift_source(args.hift_source)
    hift.m_source.l_sin_gen.rand_ini = torch.from_numpy(rand_ini.copy())
    hift.m_source.l_sin_gen.sine_waves = torch.from_numpy(sine_waves.copy())

    # ---- 3. prompt bundle (read the same files the CLI read) --------------------
    prompt_tokens = np.fromfile(os.path.join(args.prompt_dir, "prompt_tokens.i32"), dtype=np.int32)
    prompt_feat = np.fromfile(os.path.join(args.prompt_dir, "prompt_feat.f32"), dtype=np.float32)
    spk_embedding = np.fromfile(os.path.join(args.prompt_dir, "spk_embedding.f32"), dtype=np.float32)
    P = prompt_tokens.size
    mel_len1 = prompt_feat.size // MEL_DIM
    assert prompt_feat.size == mel_len1 * MEL_DIM
    assert spk_embedding.size == SPK_DIM

    tt = torch.from_numpy(tokens).reshape(1, -1).to(torch.int32)
    pt = torch.from_numpy(prompt_tokens).reshape(1, -1).to(torch.int32)
    pf = torch.from_numpy(prompt_feat).reshape(1, mel_len1, MEL_DIM).to(torch.float32)
    emb = torch.from_numpy(spk_embedding).reshape(1, SPK_DIM).to(torch.float32)

    # ---- 4. Flow cross-check -----------------------------------------------------
    print("[3/4] Python flow on the CLI's tokens", flush=True)
    with torch.no_grad():
        ref_mel, _ = flow.inference(token=tt, token_len=torch.tensor([tokens.size], dtype=torch.int32),
                                    prompt_token=pt, prompt_token_len=torch.tensor([P], dtype=torch.int32),
                                    prompt_feat=pf, prompt_feat_len=torch.tensor([mel_len1], dtype=torch.int32),
                                    embedding=emb, streaming=False, finalize=True)
    ref_mel = ref_mel[0].numpy().astype(np.float32)              # (80, OUT_FRAMES)
    out_frames = ref_mel.shape[1]
    assert cli_mel_f.size == MEL_DIM * out_frames, \
        f"CLI mel {cli_mel_f.size} floats != 80*{out_frames}"
    cli_mel_2d = cli_mel_f.reshape(MEL_DIM, out_frames)
    mel_err = report("flow mel", cli_mel_2d, ref_mel)

    # ---- 5. HiFT cross-check ------------------------------------------------------
    print("[4/4] Python hift on the CLI's mel", flush=True)
    with torch.no_grad():
        ref_speech, _ = hift.inference(speech_feat=torch.from_numpy(cli_mel_2d).reshape(1, MEL_DIM, out_frames),
                                       finalize=True)
    ref_speech = ref_speech[0].numpy().astype(np.float32)        # (L_s,)
    L_s = ref_speech.shape[0]
    assert L_s == out_frames * TOTAL_SCALE, f"hift L_s={L_s} != {out_frames}*{TOTAL_SCALE}"
    assert cli_audio_f.size == L_s, f"CLI audio {cli_audio_f.size} != {L_s}"
    wav_err = report("hift pcm", cli_audio_f, ref_speech)

    # ---- deliverable files (for listening) ----------------------------------------
    ref_wav = base + "_reference.wav"
    torchaudio.save(ref_wav, torch.from_numpy(ref_speech.copy()).reshape(1, -1), SAMPLE_RATE)

    print("\n" + "-" * 72)
    print("deliverables:")
    print(f"  C++ CLI wav      : {cli_wav}")
    print(f"  Python ref wav   : {ref_wav}")
    print(f"  C++ CLI tokens   : {cli_tokens}  ({tokens.size} tokens)")
    print(f"  C++ CLI mel      : {cli_mel}  (80 x {out_frames})")
    print(f"  C++ CLI pcm      : {cli_audio}  ({L_s} samples)")
    print(f"  flow mel max abs err = {mel_err[0]:.3e}  (rel {mel_err[2]:.3e})")
    print(f"  hift pcm max abs err = {wav_err[0]:.3e}  (rel {wav_err[2]:.3e})")

    # ---- verdict -------------------------------------------------------------------
    fails = []
    if mel_err[2] > 1e-2:
        fails.append(f"flow mel (rel {mel_err[2]:.3e})")
    if wav_err[2] > 1e-2:
        fails.append(f"hift pcm (rel {wav_err[2]:.3e})")
    if fails:
        print("FAIL: " + ", ".join(fails) + " diverges beyond float32 accumulation.")
        sys.exit(1)
    print("PASS: C++ LLM -> Flow -> HiFT chain matches the PyTorch reference within float32 accumulation.")


def report(name, cpp, ref):
    import numpy as np
    cpp = np.asarray(cpp, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    diff = np.abs(cpp - ref)
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    scale = float(np.abs(ref).max())
    rel = max_err / scale if scale > 0 else max_err
    print(f"  {name:<10} max abs err = {max_err:.3e}  mean = {mean_err:.3e}  rel = {rel:.3e}", flush=True)
    return max_err, mean_err, rel


if __name__ == "__main__":
    main()
