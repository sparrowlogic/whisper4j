#!/usr/bin/env python3
"""
Generate stage-by-stage reference values as JSON for cross-validation with Java.
Usage: python3 tools/generate_reference_values.py <model_path> <audio_path>
Outputs JSON to stdout with intermediate values at each pipeline stage.
"""
import json, struct, sys, os
import numpy as np

def load_ggml(path):
    tensors = {}
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        hp = struct.unpack("<11i", f.read(44))
        n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer = hp[:5]
        n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype = hp[5:]

        n_mel = struct.unpack("<i", f.read(4))[0]
        n_fft = struct.unpack("<i", f.read(4))[0]
        mel_filters = np.frombuffer(f.read(n_mel * n_fft * 4), dtype=np.float32).reshape(n_mel, n_fft)

        vocab_count = struct.unpack("<i", f.read(4))[0]
        vocab = {}
        for i in range(vocab_count):
            tlen = struct.unpack("<i", f.read(4))[0]
            if tlen > 0:
                vocab[i] = f.read(tlen).decode("utf-8", errors="replace")
            else:
                vocab[i] = ""

        while True:
            hdr = f.read(4)
            if len(hdr) < 4: break
            nDims = struct.unpack("<i", hdr)[0]
            if nDims < 1 or nDims > 4: break
            nameLen = struct.unpack("<i", f.read(4))[0]
            ttype = struct.unpack("<i", f.read(4))[0]
            ne = [struct.unpack("<i", f.read(4))[0] for _ in range(nDims)]
            name = f.read(nameLen).decode().strip()
            nEl = 1
            for d in ne: nEl *= d
            if ttype == 0:
                raw = np.frombuffer(f.read(nEl * 4), dtype=np.float32).copy()
            elif ttype == 1:
                raw = np.frombuffer(f.read(nEl * 2), dtype=np.float16).astype(np.float32)
            else:
                sizes = {2: 18, 3: 20, 6: 18, 7: 20, 8: 34}
                bs = sizes.get(ttype)
                if bs is None: break
                f.read(nEl // 32 * bs)
                continue
            tensors[name] = raw.reshape(list(reversed(ne)))

    dims = {
        'n_audio_state': n_audio_state, 'n_audio_head': n_audio_head,
        'n_audio_layer': n_audio_layer, 'n_text_state': n_text_state,
        'n_text_head': n_text_head, 'n_text_layer': n_text_layer,
        'n_mels': n_mels, 'n_audio_ctx': n_audio_ctx, 'n_text_ctx': n_text_ctx,
        'n_vocab': n_vocab,
    }
    return tensors, dims, mel_filters

def layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return w * (x - mean) / np.sqrt(var + eps) + b

def linear(x, w, b=None):
    out = x @ w.T
    return out + b if b is not None else out

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def mha(x, xa, W, prefix, n_head, mask=None):
    q = linear(x, W[prefix+'query.weight'], W[prefix+'query.bias'])
    src = xa if xa is not None else x
    k = linear(src, W[prefix+'key.weight'], W.get(prefix+'key.bias'))
    v = linear(src, W[prefix+'value.weight'], W[prefix+'value.bias'])
    batch, seq, state = q.shape
    hd = state // n_head
    ks = k.shape[1]
    def heads(t, s): return t.reshape(batch, s, n_head, hd).transpose(0,2,1,3).reshape(batch*n_head, s, hd)
    attn = (heads(q, seq) @ heads(k, ks).transpose(0,2,1)) / np.sqrt(hd)
    if mask is not None and seq > 1: attn += mask[:seq, :ks]
    attn = softmax(attn)
    out = attn @ heads(v, ks)
    merged = out.reshape(batch, n_head, seq, hd).transpose(0,2,1,3).reshape(batch, seq, state)
    return linear(merged, W[prefix+'out.weight'], W[prefix+'out.bias'])

def conv1d(x, w, b, stride=1, padding=1):
    batch, in_ch, length = x.shape
    out_ch, _, k = w.shape
    if padding > 0: x = np.pad(x, ((0,0),(0,0),(padding,padding)))
    out_len = (x.shape[2] - k) // stride + 1
    out = np.zeros((batch, out_ch, out_len), dtype=np.float32)
    for t in range(out_len):
        s = t * stride
        out[:,:,t] = np.einsum('bik,oik->bo', x[:,:,s:s+k], w)
    if b is not None: out += b.reshape(1,-1,1)
    return out

def first_n(arr, n=8):
    return [float(x) for x in arr.flat[:n]]

def main():
    model_path = sys.argv[1]
    audio_path = sys.argv[2]

    import wave
    with wave.open(audio_path, 'rb') as wf:
        nch, sr, nf = wf.getnchannels(), wf.getframerate(), wf.getnframes()
        raw = np.frombuffer(wf.readframes(nf), dtype=np.int16).astype(np.float32) / 32768.0
        if nch == 2: raw = raw.reshape(-1, 2).mean(axis=1)

    W, hp, mel_f = load_ggml(model_path)
    result = {
        "metadata": {
            "model": os.path.basename(model_path),
            "audio": os.path.basename(audio_path),
            "whisper_ref_commit": "cba3768",
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "dims": hp,
        },
        "stages": {}
    }

    # Stage 1: Mel spectrogram
    audio = np.zeros(30 * 16000, dtype=np.float32)
    audio[:min(len(raw), len(audio))] = raw[:min(len(raw), len(audio))]
    n_fft, hop = 400, 160
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    padded = np.pad(audio, (n_fft // 2, n_fft // 2))
    n_frames = 1 + (len(padded) - n_fft) // hop
    mags = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame = padded[i*hop:i*hop+n_fft] * window
        spec = np.fft.rfft(frame)
        mags[:, i] = np.abs(spec) ** 2
    mel_spec = mel_f @ mags[:mel_f.shape[1], :]
    log_mel = np.log10(np.maximum(mel_spec, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    mel = log_mel[:, :3000]
    result["stages"]["mel"] = {"shape": list(mel.shape), "first8": first_n(mel)}

    # Stage 2: Conv1 + GELU
    mel_3d = mel.reshape(1, hp['n_mels'], 3000)
    c1 = gelu(conv1d(mel_3d, W['encoder.conv1.weight'], W['encoder.conv1.bias'].flatten(), stride=1, padding=1))
    result["stages"]["conv1_gelu"] = {"shape": list(c1.shape), "first8": first_n(c1)}

    # Stage 3: Conv2 + GELU
    c2 = gelu(conv1d(c1, W['encoder.conv2.weight'], W['encoder.conv2.bias'].flatten(), stride=2, padding=1))
    result["stages"]["conv2_gelu"] = {"shape": list(c2.shape), "first8": first_n(c2)}

    # Stage 4: Transpose + positional embedding
    x = c2.transpose(0, 2, 1) + W['encoder.positional_embedding']
    result["stages"]["pos_embed"] = {"shape": list(x.shape), "first8": first_n(x)}

    # Stage 5: Encoder block 0
    bp = 'encoder.blocks.0.'
    ln = layer_norm(x, W[bp+'attn_ln.weight'], W[bp+'attn_ln.bias'])
    sa = mha(ln, None, W, bp+'attn.', hp['n_audio_head'])
    x = x + sa
    ml = layer_norm(x, W[bp+'mlp_ln.weight'], W[bp+'mlp_ln.bias'])
    x = x + linear(gelu(linear(ml, W[bp+'mlp.0.weight'], W[bp+'mlp.0.bias'])), W[bp+'mlp.2.weight'], W[bp+'mlp.2.bias'])
    result["stages"]["encoder_block0"] = {"shape": list(x.shape), "first8": first_n(x)}

    # Stage 6: Full encoder output
    for i in range(1, hp['n_audio_layer']):
        bp = f'encoder.blocks.{i}.'
        if bp+'attn_ln.weight' not in W: continue
        ln = layer_norm(x, W[bp+'attn_ln.weight'], W[bp+'attn_ln.bias'])
        sa = mha(ln, None, W, bp+'attn.', hp['n_audio_head'])
        x = x + sa
        ml = layer_norm(x, W[bp+'mlp_ln.weight'], W[bp+'mlp_ln.bias'])
        x = x + linear(gelu(linear(ml, W[bp+'mlp.0.weight'], W[bp+'mlp.0.bias'])), W[bp+'mlp.2.weight'], W[bp+'mlp.2.bias'])
    enc_out = layer_norm(x, W['encoder.ln_post.weight'], W['encoder.ln_post.bias'])
    result["stages"]["encoder_output"] = {"shape": list(enc_out.shape), "first8": first_n(enc_out)}

    # Stage 7: Decoder first step logits
    n_vocab = hp['n_vocab']
    num_langs = n_vocab - 51765
    dt = num_langs - 98
    is_multilingual = n_vocab >= 51865
    sot = 50257 if is_multilingual else 50256
    en_tok = 50258 if is_multilingual else -1
    transcribe = 50358 + dt if is_multilingual else 50257
    notimestamps = 50362 + dt if is_multilingual else 50361
    prompt = [sot]
    if is_multilingual:
        prompt += [en_tok, transcribe]
    prompt.append(notimestamps)

    te = W['decoder.token_embedding.weight']
    dpe = W['decoder.positional_embedding']
    ns = hp['n_text_state']
    x_dec = np.zeros((1, len(prompt), ns), dtype=np.float32)
    for t, tok in enumerate(prompt):
        x_dec[0, t] = te[tok] + dpe[t]
    mask = np.full((hp['n_text_ctx'], hp['n_text_ctx']), -np.inf, dtype=np.float32)
    mask = np.triu(mask, k=1)
    for i in range(hp['n_text_layer']):
        bp = f'decoder.blocks.{i}.'
        if bp+'attn_ln.weight' not in W: continue
        ln = layer_norm(x_dec, W[bp+'attn_ln.weight'], W[bp+'attn_ln.bias'])
        sa = mha(ln, None, W, bp+'attn.', hp['n_text_head'], mask)
        x_dec = x_dec + sa
        ca_ln = layer_norm(x_dec, W[bp+'cross_attn_ln.weight'], W[bp+'cross_attn_ln.bias'])
        ca = mha(ca_ln, enc_out, W, bp+'cross_attn.', hp['n_text_head'])
        x_dec = x_dec + ca
        ml = layer_norm(x_dec, W[bp+'mlp_ln.weight'], W[bp+'mlp_ln.bias'])
        x_dec = x_dec + linear(gelu(linear(ml, W[bp+'mlp.0.weight'], W[bp+'mlp.0.bias'])), W[bp+'mlp.2.weight'], W[bp+'mlp.2.bias'])
    x_dec = layer_norm(x_dec, W['decoder.ln.weight'], W['decoder.ln.bias'])
    logits = x_dec @ te.T
    last = logits[0, -1]
    top5_idx = np.argsort(last)[-5:][::-1].tolist()
    top5_scores = [float(last[i]) for i in top5_idx]
    result["stages"]["decoder_logits"] = {
        "shape": list(logits.shape),
        "first8": first_n(logits[0, -1]),
        "top5_tokens": top5_idx,
        "top5_scores": top5_scores,
        "prompt": prompt,
    }

    # Stage 8: Decoder step 1 — run prompt+[token0] without cache to get ground truth
    token0 = top5_idx[0]
    tokens_step1 = prompt + [token0]
    x_dec2 = np.zeros((1, len(tokens_step1), ns), dtype=np.float32)
    for t, tok in enumerate(tokens_step1):
        x_dec2[0, t] = te[tok] + dpe[t]
    for i in range(hp['n_text_layer']):
        bp = f'decoder.blocks.{i}.'
        if bp+'attn_ln.weight' not in W: continue
        ln = layer_norm(x_dec2, W[bp+'attn_ln.weight'], W[bp+'attn_ln.bias'])
        sa = mha(ln, None, W, bp+'attn.', hp['n_text_head'], mask)
        x_dec2 = x_dec2 + sa
        ca_ln = layer_norm(x_dec2, W[bp+'cross_attn_ln.weight'], W[bp+'cross_attn_ln.bias'])
        ca = mha(ca_ln, enc_out, W, bp+'cross_attn.', hp['n_text_head'])
        x_dec2 = x_dec2 + ca
        ml = layer_norm(x_dec2, W[bp+'mlp_ln.weight'], W[bp+'mlp_ln.bias'])
        x_dec2 = x_dec2 + linear(gelu(linear(ml, W[bp+'mlp.0.weight'], W[bp+'mlp.0.bias'])), W[bp+'mlp.2.weight'], W[bp+'mlp.2.bias'])
    x_dec2 = layer_norm(x_dec2, W['decoder.ln.weight'], W['decoder.ln.bias'])
    logits2 = x_dec2 @ te.T
    last2 = logits2[0, -1]
    top5_step1 = np.argsort(last2)[-5:][::-1].tolist()
    result["stages"]["decoder_step1"] = {
        "first8": first_n(logits2[0, -1]),
        "top5_tokens": top5_step1,
        "input_tokens": tokens_step1,
    }

    json.dump(result, sys.stdout, indent=2)
    print()

if __name__ == "__main__":
    main()
