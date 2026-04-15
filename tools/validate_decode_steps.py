#!/usr/bin/env python3
"""Run N greedy decode steps with KV cache, output per-step logits for parity checking."""
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
        for i in range(vocab_count):
            tlen = struct.unpack("<i", f.read(4))[0]
            f.read(tlen) if tlen > 0 else None
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
    return tensors, {'n_audio_state': n_audio_state, 'n_audio_head': n_audio_head,
        'n_audio_layer': n_audio_layer, 'n_text_state': n_text_state,
        'n_text_head': n_text_head, 'n_text_layer': n_text_layer,
        'n_mels': n_mels, 'n_audio_ctx': n_audio_ctx, 'n_text_ctx': n_text_ctx,
        'n_vocab': n_vocab}, mel_filters

def layer_norm(x, w, b, eps=1e-5):
    m = x.mean(axis=-1, keepdims=True)
    v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
    return w * (x - m) / np.sqrt(v + eps) + b

def linear(x, w, b=None):
    return x @ w.T + b if b is not None else x @ w.T

def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def mha_cached(x, xa, W, prefix, n_head, kv_cache, mask=None):
    q = linear(x, W[prefix+'query.weight'], W[prefix+'query.bias'])
    batch, seq, state = q.shape
    hd = state // n_head
    kr_key, vr_key = prefix+'kr', prefix+'vr'
    def heads(t, s): return t.reshape(batch, s, n_head, hd).transpose(0,2,1,3).reshape(batch*n_head, s, hd)

    if xa is not None and kr_key in kv_cache:
        kr, vr = kv_cache[kr_key], kv_cache[vr_key]
    elif xa is None and kr_key in kv_cache:
        k_new = heads(linear(x, W[prefix+'key.weight'], W.get(prefix+'key.bias')), seq)
        v_new = heads(linear(x, W[prefix+'value.weight'], W[prefix+'value.bias']), seq)
        kr = np.concatenate([kv_cache[kr_key], k_new], axis=1)
        vr = np.concatenate([kv_cache[vr_key], v_new], axis=1)
        kv_cache[kr_key], kv_cache[vr_key] = kr, vr
    else:
        src = xa if xa is not None else x
        kr = heads(linear(src, W[prefix+'key.weight'], W.get(prefix+'key.bias')), src.shape[1])
        vr = heads(linear(src, W[prefix+'value.weight'], W[prefix+'value.bias']), src.shape[1])
        kv_cache[kr_key], kv_cache[vr_key] = kr, vr

    qr = heads(q, seq)
    attn = (qr @ kr.transpose(0,2,1)) / np.sqrt(hd)
    if mask is not None and seq > 1: attn += mask[:seq, :kr.shape[1]]
    attn = softmax(attn)
    out = attn @ vr
    merged = out.reshape(batch, n_head, seq, hd).transpose(0,2,1,3).reshape(batch, seq, state)
    return linear(merged, W[prefix+'out.weight'], W[prefix+'out.bias'])

def decoder_step(tokens, enc_out, W, hp, kv_cache, offset):
    te = W['decoder.token_embedding.weight']
    dpe = W['decoder.positional_embedding']
    ns = hp['n_text_state']
    seq = len(tokens)
    x = np.zeros((1, seq, ns), dtype=np.float32)
    for t, tok in enumerate(tokens):
        x[0, t] = te[tok] + dpe[offset + t]
    mask = np.full((hp['n_text_ctx'], hp['n_text_ctx']), -np.inf, dtype=np.float32)
    mask = np.triu(mask, k=1)
    for i in range(hp['n_text_layer']):
        bp = f'decoder.blocks.{i}.'
        ln = layer_norm(x, W[bp+'attn_ln.weight'], W[bp+'attn_ln.bias'])
        sa = mha_cached(ln, None, W, bp+'attn.', hp['n_text_head'], kv_cache, mask)
        x = x + sa
        ca_ln = layer_norm(x, W[bp+'cross_attn_ln.weight'], W[bp+'cross_attn_ln.bias'])
        ca = mha_cached(ca_ln, enc_out, W, bp+'cross_attn.', hp['n_text_head'], kv_cache)
        x = x + ca
        ml = layer_norm(x, W[bp+'mlp_ln.weight'], W[bp+'mlp_ln.bias'])
        x = x + linear(gelu(linear(ml, W[bp+'mlp.0.weight'], W[bp+'mlp.0.bias'])), W[bp+'mlp.2.weight'], W[bp+'mlp.2.bias'])
    x = layer_norm(x, W['decoder.ln.weight'], W['decoder.ln.bias'])
    return x @ te.T

def main():
    model_path = sys.argv[1]
    audio_path = sys.argv[2]
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 15

    W, hp, mel_f = load_ggml(model_path)

    # Mel + encoder (reuse from generate_reference_values logic)
    import wave
    with wave.open(audio_path, 'rb') as wf:
        nch, sr, nf = wf.getnchannels(), wf.getframerate(), wf.getnframes()
        raw = np.frombuffer(wf.readframes(nf), dtype=np.int16).astype(np.float32) / 32768.0
        if nch == 2: raw = raw.reshape(-1, 2).mean(axis=1)
    audio = np.zeros(30*16000, dtype=np.float32)
    audio[:min(len(raw),len(audio))] = raw[:min(len(raw),len(audio))]
    n_fft, hop = 400, 160
    window = np.hanning(n_fft+1)[:-1].astype(np.float32)
    padded = np.pad(audio, (n_fft//2, n_fft//2))
    nfr = 1 + (len(padded)-n_fft)//hop
    mags = np.zeros((n_fft//2+1, nfr), dtype=np.float32)
    for i in range(nfr):
        frame = padded[i*hop:i*hop+n_fft]*window
        mags[:, i] = np.abs(np.fft.rfft(frame))**2
    mel_spec = mel_f @ mags[:mel_f.shape[1],:]
    log_mel = np.log10(np.maximum(mel_spec, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max()-8.0)
    mel = ((log_mel+4.0)/4.0)[:,:3000]

    # Encoder
    def conv1d(x, w, b, stride=1, padding=1):
        batch, in_ch, length = x.shape; out_ch, _, k = w.shape
        if padding > 0: x = np.pad(x, ((0,0),(0,0),(padding,padding)))
        out_len = (x.shape[2]-k)//stride+1
        out = np.zeros((batch, out_ch, out_len), dtype=np.float32)
        for t in range(out_len):
            s = t*stride; out[:,:,t] = np.einsum('bik,oik->bo', x[:,:,s:s+k], w)
        return out + b.reshape(1,-1,1) if b is not None else out

    mel_3d = mel.reshape(1, hp['n_mels'], 3000)
    x = gelu(conv1d(mel_3d, W['encoder.conv1.weight'], W['encoder.conv1.bias'].flatten(), 1, 1))
    x = gelu(conv1d(x, W['encoder.conv2.weight'], W['encoder.conv2.bias'].flatten(), 2, 1))
    x = x.transpose(0,2,1) + W['encoder.positional_embedding']
    for i in range(hp['n_audio_layer']):
        bp = f'encoder.blocks.{i}.'
        if bp+'attn_ln.weight' not in W: continue
        ln = layer_norm(x, W[bp+'attn_ln.weight'], W[bp+'attn_ln.bias'])
        q = linear(ln, W[bp+'attn.query.weight'], W[bp+'attn.query.bias'])
        k = linear(ln, W[bp+'attn.key.weight'], W.get(bp+'attn.key.bias'))
        v = linear(ln, W[bp+'attn.value.weight'], W[bp+'attn.value.bias'])
        b,s,st = q.shape; hd = st//hp['n_audio_head']; nh = hp['n_audio_head']
        def h(t,l): return t.reshape(b,l,nh,hd).transpose(0,2,1,3).reshape(b*nh,l,hd)
        a = softmax((h(q,s) @ h(k,s).transpose(0,2,1))/np.sqrt(hd))
        o = a @ h(v,s)
        sa = linear(o.reshape(b,nh,s,hd).transpose(0,2,1,3).reshape(b,s,st), W[bp+'attn.out.weight'], W[bp+'attn.out.bias'])
        x = x + sa
        ml = layer_norm(x, W[bp+'mlp_ln.weight'], W[bp+'mlp_ln.bias'])
        x = x + linear(gelu(linear(ml, W[bp+'mlp.0.weight'], W[bp+'mlp.0.bias'])), W[bp+'mlp.2.weight'], W[bp+'mlp.2.bias'])
    enc_out = layer_norm(x, W['encoder.ln_post.weight'], W['encoder.ln_post.bias'])

    # Build prompt
    nv = hp['n_vocab']; is_multi = nv >= 51865
    dt = (nv - 51765 - (1 if is_multi else 0)) - 98
    sot = 50257 if is_multi else 50256
    prompt = [sot]
    if is_multi: prompt += [50258, 50358+dt]
    prompt.append(50362+dt)

    # Greedy decode with KV cache
    kv_cache = {}
    cur = prompt
    offset = 0
    steps = []
    for step in range(n_steps):
        logits = decoder_step(cur, enc_out, W, hp, kv_cache, offset)
        last = logits[0, -1]
        tok = int(np.argmax(last))
        top3_idx = np.argsort(last)[-3:][::-1].tolist()
        top3_val = [float(last[i]) for i in top3_idx]
        steps.append({
            "step": step, "token": tok,
            "top3": top3_idx, "top3_scores": top3_val,
            "first4": [float(x) for x in last[:4]],
        })
        offset += len(cur)
        cur = [tok]
        if tok == 50256: break

    json.dump({"model": os.path.basename(model_path), "steps": steps}, sys.stdout, indent=2)
    print()

if __name__ == "__main__":
    main()
