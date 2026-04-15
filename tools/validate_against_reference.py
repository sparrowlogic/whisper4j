#!/usr/bin/env python3
"""
Validate whisper4j components against reference Python implementation.
Loads GGML weights directly and runs each stage, printing outputs for comparison with Java.
"""
import struct, numpy as np, sys

MODEL = "models/ggml-base.en.bin"
AUDIO = "src/test/resources/data/stereo_diarization.wav"

# ---- GGML loader ----
def load_ggml(path):
    tensors = {}
    with open(path, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        hp = struct.unpack("<11i", f.read(44))
        n_vocab, n_audio_ctx, n_audio_state, n_audio_head, n_audio_layer = hp[:5]
        n_text_ctx, n_text_state, n_text_head, n_text_layer, n_mels, ftype = hp[5:]
        print(f"hparams: state={n_audio_state} heads={n_audio_head} layers={n_audio_layer} mels={n_mels} ftype={ftype}")

        n_mel = struct.unpack("<i", f.read(4))[0]
        n_fft = struct.unpack("<i", f.read(4))[0]
        mel_filters = np.frombuffer(f.read(n_mel * n_fft * 4), dtype=np.float32).reshape(n_mel, n_fft)

        vocab_count = struct.unpack("<i", f.read(4))[0]
        for i in range(vocab_count):
            tlen = struct.unpack("<i", f.read(4))[0]
            f.read(tlen)

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
                # skip quantized
                sizes = {2: 18, 3: 20, 6: 18, 7: 20, 8: 34}
                bs = sizes.get(ttype)
                if bs is None: break
                f.read(nEl // 32 * bs)
                continue

            # Column-major ne -> row-major reversed shape
            tensors[name] = raw.reshape(list(reversed(ne)))

    return tensors, {
        'n_audio_state': n_audio_state, 'n_audio_head': n_audio_head,
        'n_audio_layer': n_audio_layer, 'n_text_state': n_text_state,
        'n_text_head': n_text_head, 'n_text_layer': n_text_layer,
        'n_mels': n_mels, 'n_audio_ctx': n_audio_ctx, 'n_text_ctx': n_text_ctx,
        'n_vocab': n_vocab,
    }, mel_filters

# ---- Neural network ops (matching PyTorch/whisper exactly) ----
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

def multi_head_attention(x, xa, qw, qb, kw, kb, vw, vb, ow, ob, n_head, mask=None):
    q = linear(x, qw, qb)
    source = xa if xa is not None else x
    k = linear(source, kw, kb)
    v = linear(source, vw, vb)

    batch, seq, state = q.shape
    head_dim = state // n_head
    kv_seq = k.shape[1]

    def to_heads(t, s):
        return t.reshape(batch, s, n_head, head_dim).transpose(0, 2, 1, 3).reshape(batch * n_head, s, head_dim)

    qr = to_heads(q, seq)
    kr = to_heads(k, kv_seq)
    vr = to_heads(v, kv_seq)

    scale = 1.0 / np.sqrt(head_dim)
    attn = (qr @ kr.transpose(0, 2, 1)) * scale

    if mask is not None and seq > 1:
        attn += mask[:seq, :kv_seq]

    attn = softmax(attn)
    out = attn @ vr

    merged = out.reshape(batch, n_head, seq, head_dim).transpose(0, 2, 1, 3).reshape(batch, seq, state)
    return linear(merged, ow, ob)

def conv1d(x, w, b, stride=1, padding=1):
    """x: (batch, in_ch, length), w: (out_ch, in_ch, k), b: (out_ch,)"""
    batch, in_ch, length = x.shape
    out_ch, _, k = w.shape

    # Pad
    if padding > 0:
        x = np.pad(x, ((0,0), (0,0), (padding, padding)))

    out_len = (x.shape[2] - k) // stride + 1
    out = np.zeros((batch, out_ch, out_len), dtype=np.float32)

    for t in range(out_len):
        start = t * stride
        patch = x[:, :, start:start+k]  # (batch, in_ch, k)
        # (batch, out_ch) = sum over (in_ch, k)
        out[:, :, t] = np.einsum('bik,oik->bo', patch, w)

    if b is not None:
        out += b.reshape(1, -1, 1)
    return out

# ---- Load model ----
print("Loading GGML model...")
W, hp, mel_f = load_ggml(MODEL)
print(f"Loaded {len(W)} tensors\n")

# ---- Stage 1: Mel spectrogram ----
import wave
with wave.open(AUDIO, 'rb') as wf:
    nch, sr, nf = wf.getnchannels(), wf.getframerate(), wf.getnframes()
    raw = np.frombuffer(wf.readframes(nf), dtype=np.int16).astype(np.float32) / 32768.0
    if nch == 2: raw = raw.reshape(-1, 2).mean(axis=1)

# Pad to 30s
audio = np.zeros(30 * 16000, dtype=np.float32)
audio[:len(raw)] = raw[:min(len(raw), len(audio))]

# STFT
n_fft, hop = 400, 160
window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
padded = np.pad(audio, (n_fft // 2, n_fft // 2))
n_frames = 1 + (len(padded) - n_fft) // hop
mags = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
for i in range(n_frames):
    frame = padded[i * hop: i * hop + n_fft] * window
    spec = np.fft.rfft(frame)
    mags[:, i] = np.abs(spec) ** 2

mel_spec = mel_f @ mags
log_mel = np.log10(np.maximum(mel_spec, 1e-10))
log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
log_mel = (log_mel + 4.0) / 4.0

# Trim to 3000 frames
mel = log_mel[:, :3000]
print(f"[Stage 1] Mel: shape={mel.shape} mean={mel.mean():.6f} first4={mel.flat[:4]}")

# ---- Stage 2: Conv1 + GELU ----
c1w = W['encoder.conv1.weight']  # (512, 80, 3)
c1b = W['encoder.conv1.bias'].flatten()
mel_3d = mel.reshape(1, hp['n_mels'], 3000)
conv1_out = gelu(conv1d(mel_3d, c1w, c1b, stride=1, padding=1))
print(f"[Stage 2] Conv1+GELU: shape={conv1_out.shape} first4={conv1_out.flat[:4]}")

# ---- Stage 3: Conv2 + GELU ----
c2w = W['encoder.conv2.weight']  # (512, 512, 3)
c2b = W['encoder.conv2.bias'].flatten()
conv2_out = gelu(conv1d(conv1_out, c2w, c2b, stride=2, padding=1))
print(f"[Stage 3] Conv2+GELU: shape={conv2_out.shape} first4={conv2_out.flat[:4]}")

# ---- Stage 4: Transpose + positional embedding ----
x = conv2_out.transpose(0, 2, 1)  # (1, 1500, 512)
pe = W['encoder.positional_embedding']  # (1500, 512)
x = x + pe
print(f"[Stage 4] After pos_emb: shape={x.shape} first4={x.flat[:4]}")

# ---- Stage 5: Encoder block 0 ----
bp = 'encoder.blocks.0.'
ln_out = layer_norm(x, W[bp+'attn_ln.weight'], W[bp+'attn_ln.bias'])
attn_out = multi_head_attention(
    ln_out, None,
    W[bp+'attn.query.weight'], W[bp+'attn.query.bias'],
    W[bp+'attn.key.weight'], W.get(bp+'attn.key.bias'),
    W[bp+'attn.value.weight'], W[bp+'attn.value.bias'],
    W[bp+'attn.out.weight'], W[bp+'attn.out.bias'],
    hp['n_audio_head']
)
x = x + attn_out
mlp_ln = layer_norm(x, W[bp+'mlp_ln.weight'], W[bp+'mlp_ln.bias'])
mlp_out = linear(gelu(linear(mlp_ln, W[bp+'mlp.0.weight'], W[bp+'mlp.0.bias'])),
                 W[bp+'mlp.2.weight'], W[bp+'mlp.2.bias'])
x = x + mlp_out
print(f"[Stage 5] Encoder block 0: shape={x.shape} first4={x.flat[:4]}")

# ---- Stage 6: All encoder blocks + ln_post ----
for i in range(1, hp['n_audio_layer']):
    bp = f'encoder.blocks.{i}.'
    if bp+'attn_ln.weight' not in W:
        print(f"  Skipping block {i} (weights not loaded)")
        continue
    ln_out = layer_norm(x, W[bp+'attn_ln.weight'], W[bp+'attn_ln.bias'])
    attn_out = multi_head_attention(
        ln_out, None,
        W[bp+'attn.query.weight'], W[bp+'attn.query.bias'],
        W[bp+'attn.key.weight'], W.get(bp+'attn.key.bias'),
        W[bp+'attn.value.weight'], W[bp+'attn.value.bias'],
        W[bp+'attn.out.weight'], W[bp+'attn.out.bias'],
        hp['n_audio_head']
    )
    x = x + attn_out
    mlp_ln = layer_norm(x, W[bp+'mlp_ln.weight'], W[bp+'mlp_ln.bias'])
    mlp_out = linear(gelu(linear(mlp_ln, W[bp+'mlp.0.weight'], W[bp+'mlp.0.bias'])),
                     W[bp+'mlp.2.weight'], W[bp+'mlp.2.bias'])
    x = x + mlp_out

encoder_out = layer_norm(x, W['encoder.ln_post.weight'], W['encoder.ln_post.bias'])
print(f"[Stage 6] Encoder output: shape={encoder_out.shape} first8={encoder_out.flat[:8]}")

# ---- Stage 7: Decoder (first token step) ----
# SOT sequence for base.en: [50257, 50258, 50359, 50363]
n_vocab = hp['n_vocab']
num_langs = n_vocab - 51765
dt = num_langs - 98
sot = 50257
en_tok = 50258
transcribe = 50358 + dt
notimestamps = 50362 + dt
prompt = [sot, en_tok, transcribe, notimestamps]
print(f"\n[Stage 7] Decoder prompt: {prompt}")

te = W['decoder.token_embedding.weight']  # (n_vocab, n_state)
dpe = W['decoder.positional_embedding']    # (n_text_ctx, n_state)
n_state = hp['n_text_state']

# Token + positional embedding
x_dec = np.zeros((1, len(prompt), n_state), dtype=np.float32)
for t, tok in enumerate(prompt):
    x_dec[0, t] = te[tok] + dpe[t]
print(f"[Stage 7] Decoder embed: first4={x_dec.flat[:4]}")

# Causal mask
mask = np.full((hp['n_text_ctx'], hp['n_text_ctx']), -np.inf, dtype=np.float32)
mask = np.triu(mask, k=1)

# Decoder blocks
for i in range(hp['n_text_layer']):
    bp = f'decoder.blocks.{i}.'
    if bp+'attn_ln.weight' not in W:
        print(f"  Skipping decoder block {i}")
        continue
    # Self-attention
    ln_out = layer_norm(x_dec, W[bp+'attn_ln.weight'], W[bp+'attn_ln.bias'])
    sa_out = multi_head_attention(
        ln_out, None,
        W[bp+'attn.query.weight'], W[bp+'attn.query.bias'],
        W[bp+'attn.key.weight'], W.get(bp+'attn.key.bias'),
        W[bp+'attn.value.weight'], W[bp+'attn.value.bias'],
        W[bp+'attn.out.weight'], W[bp+'attn.out.bias'],
        hp['n_text_head'], mask
    )
    x_dec = x_dec + sa_out
    # Cross-attention
    ca_ln = layer_norm(x_dec, W[bp+'cross_attn_ln.weight'], W[bp+'cross_attn_ln.bias'])
    ca_out = multi_head_attention(
        ca_ln, encoder_out,
        W[bp+'cross_attn.query.weight'], W[bp+'cross_attn.query.bias'],
        W[bp+'cross_attn.key.weight'], W.get(bp+'cross_attn.key.bias'),
        W[bp+'cross_attn.value.weight'], W[bp+'cross_attn.value.bias'],
        W[bp+'cross_attn.out.weight'], W[bp+'cross_attn.out.bias'],
        hp['n_text_head']
    )
    x_dec = x_dec + ca_out
    # MLP
    mlp_ln = layer_norm(x_dec, W[bp+'mlp_ln.weight'], W[bp+'mlp_ln.bias'])
    mlp_out = linear(gelu(linear(mlp_ln, W[bp+'mlp.0.weight'], W[bp+'mlp.0.bias'])),
                     W[bp+'mlp.2.weight'], W[bp+'mlp.2.bias'])
    x_dec = x_dec + mlp_out

x_dec = layer_norm(x_dec, W['decoder.ln.weight'], W['decoder.ln.bias'])
logits = x_dec @ te.T  # (1, seq, n_vocab)
last_logits = logits[0, -1]  # logits for last position
top5 = np.argsort(last_logits)[-5:][::-1]
print(f"[Stage 7] Logits shape: {logits.shape}")
top5_str = ', '.join([f"({int(t)}, {float(last_logits[t]):.3f})" for t in top5])
print(f"[Stage 7] Top 5 tokens: [{top5_str}]")
print(f"[Stage 7] EOT score: {last_logits[50256]:.3f}")
