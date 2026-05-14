"""
PixelNarrator – Image Captioning Demo
======================================
Local Gradio demo for the CNN Encoder + LSTM Decoder (Bahdanau Attention)
model trained on MS COCO 2017.

Usage
-----
  # Install deps (once)
  pip install gradio torch torchvision pillow nltk numpy matplotlib

  # Run
  python app.py

Image input methods
-------------------
  1. Upload a file via the file-chooser button
  2. Drag-and-drop an image onto the drop zone
  3. Paste directly from clipboard (Ctrl+V / Cmd+V inside the image box)

Checkpoint
----------
  Place any of the four checkpoint files produced by the notebook next to
  this script, or anywhere – then paste the path into the "Checkpoint path"
  textbox inside the app.

  Expected filenames (notebook defaults):
    checkpoints_caption/1_baseline.pth
    checkpoints_caption/2_pretrained.pth
    checkpoints_caption/3_attention.pth
    checkpoints_caption/4_full_model.pth

  If no checkpoint is found the app will still run with an *untrained*
  Full Model so you can verify the pipeline end-to-end.
"""

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import os
import sys
import warnings
import io
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

import nltk
for _res in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{_res}")
    except LookupError:
        nltk.download(_res, quiet=True)
from nltk.tokenize import word_tokenize

import gradio as gr

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PixelNarrator] Device: {DEVICE}")


# ─────────────────────────────────────────────────────────────
# Model definition  (identical to notebook)
# ─────────────────────────────────────────────────────────────
class Vocabulary:
    """Vocab with special tokens <PAD>=0 <SOS>=1 <EOS>=2 <UNK>=3."""

    def __init__(self, captions=None, max_vocab_size=10_000, min_freq=5,
                 word2idx=None):
        if word2idx is not None:
            self.word2idx = word2idx
        else:
            self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
            counts = Counter()
            for cap in (captions or []):
                counts.update(word_tokenize(cap.lower()))
            for word, cnt in counts.most_common():
                if cnt < min_freq:
                    break
                if word not in self.word2idx and len(self.word2idx) < max_vocab_size:
                    self.word2idx[word] = len(self.word2idx)

        self.idx2word   = {i: w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

    def encode(self, caption, max_length=52):
        tokens = word_tokenize(caption.lower())[: max_length - 2]
        ids    = [self.word2idx.get(t, 3) for t in tokens]
        return [1] + ids + [2]

    def decode(self, ids):
        return " ".join(
            self.idx2word[i]
            for i in ids
            if i > 3 and i in self.idx2word
        )


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim=2048, decoder_dim=512, attention_dim=512):
        super().__init__()
        self.enc_proj = nn.Linear(encoder_dim, attention_dim)
        self.dec_proj = nn.Linear(decoder_dim, attention_dim)
        self.score    = nn.Linear(attention_dim, 1)

    def forward(self, encoder_out, decoder_hidden):
        proj_enc = self.enc_proj(encoder_out)
        proj_dec = self.dec_proj(decoder_hidden).unsqueeze(1)
        energy   = self.score(torch.tanh(proj_enc + proj_dec)).squeeze(-1)
        alpha    = F.softmax(energy, dim=1)
        context  = (alpha.unsqueeze(-1) * encoder_out).sum(dim=1)
        return context, alpha


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet          = models.resnet50(weights="DEFAULT" if pretrained else None)
        self.backbone   = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, images):
        features = self.backbone(images)
        features = self.adaptive_pool(features)
        return features.flatten(2).transpose(1, 2)   # (B, 49, 2048)


class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, encoder_dim=2048,
                 decoder_dim=512, dropout=0.3, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.decoder_dim   = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)

        if use_attention:
            self.attention = BahdanauAttention(encoder_dim, decoder_dim)

        self.lstm   = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.fc     = nn.Linear(decoder_dim, vocab_size)

    def _init_hidden(self, encoder_out):
        mean_enc = encoder_out.mean(dim=1)
        return torch.tanh(self.init_h(mean_enc)), torch.tanh(self.init_c(mean_enc))


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, pretrained=True, use_attention=True,
                 embed_dim=256, decoder_dim=512, dropout=0.3):
        super().__init__()
        self.encoder = ImageEncoder(pretrained=pretrained)
        self.decoder = CaptionDecoder(
            vocab_size    = vocab_size,
            embed_dim     = embed_dim,
            decoder_dim   = decoder_dim,
            dropout       = dropout,
            use_attention = use_attention,
        )

    @torch.no_grad()
    def generate(self, image_tensor, vocab, device, max_len=40, temperature=1.0):
        """
        Greedy / temperature-scaled generation for a single pre-processed image.
        Returns (caption_str, attention_weights_tensor_or_None).
        """
        self.eval()
        img     = image_tensor.unsqueeze(0).to(device)
        enc_out = self.encoder(img)                        # (1, 49, 2048)

        h, c    = self.decoder._init_hidden(enc_out)
        context = enc_out.mean(dim=1)
        token   = torch.tensor([[1]], device=device)       # SOS

        generated  = []
        all_alphas = []

        for _ in range(max_len):
            embed = self.decoder.embedding(token.squeeze(1))

            if self.decoder.use_attention:
                context, alpha = self.decoder.attention(enc_out, h)
                all_alphas.append(alpha.squeeze(0).cpu())

            lstm_in = torch.cat([embed, context], dim=1)
            h, c    = self.decoder.lstm(lstm_in, (h, c))
            logit   = self.decoder.fc(h)

            probs      = F.softmax(logit / max(temperature, 1e-3), dim=-1)
            next_token = probs.argmax(-1).item()

            if next_token == 2:      # EOS
                break
            generated.append(next_token)
            token = torch.tensor([[next_token]], device=device)

        caption      = vocab.decode(generated)
        attn_weights = torch.stack(all_alphas) if all_alphas else None
        return caption, attn_weights


# ─────────────────────────────────────────────────────────────
# Image pre-processing
# ─────────────────────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────
# Checkpoint loader
# ─────────────────────────────────────────────────────────────
_model_cache: dict = {}   # path → (model, vocab)

CONFIG_LABELS = {
    "1. Baseline"  : (False, False),
    "2. Pretrained": (True,  False),
    "3. Attention" : (False, True),
    "4. Full Model": (True,  True),
}

DEFAULT_CKPT_PATHS = {
    "1. Baseline"  : "checkpoints_caption/1_baseline.pth",
    "2. Pretrained": "checkpoints_caption/2_pretrained.pth",
    "3. Attention" : "checkpoints_caption/3_attention.pth",
    "4. Full Model": "checkpoints_caption/4_full_model.pth",
}


def load_model(ckpt_path: str, config_name: str):
    """
    Load (and cache) a model + vocab from a checkpoint file.
    Falls back to an untrained model when the checkpoint does not exist.
    Returns (model, vocab, status_message).
    """
    cache_key = (ckpt_path, config_name)
    if cache_key in _model_cache:
        model, vocab = _model_cache[cache_key]
        return model, vocab, f"✅ Loaded from cache – {config_name}"

    pretrained, use_attn = CONFIG_LABELS[config_name]

    # ── Try to load checkpoint ────────────────────────────────
    if ckpt_path and os.path.isfile(ckpt_path):
        try:
            ckpt  = torch.load(ckpt_path, map_location=DEVICE)
            w2i   = ckpt["vocab_word2idx"]
            vocab = Vocabulary(word2idx=w2i)
            cfg   = ckpt.get("config", {})
            model = ImageCaptioningModel(
                vocab_size    = vocab.vocab_size,
                pretrained    = cfg.get("pretrained",    pretrained),
                use_attention = cfg.get("use_attention", use_attn),
            ).to(DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            _model_cache[cache_key] = (model, vocab)
            return model, vocab, f"✅ Checkpoint loaded – {config_name}"
        except Exception as exc:
            status = f"⚠️ Could not load checkpoint: {exc}\nFalling back to untrained model."
    else:
        status = (
            f"ℹ️ No checkpoint found at '{ckpt_path}'.\n"
            "Running with an *untrained* model – captions will be random.\n"
            "Train the notebook first and supply the .pth path."
        )

    # ── Untrained fallback ────────────────────────────────────
    # Build a tiny dummy vocab so the model can at least run
    dummy_words = (
        "a an the is are on in with and of to for that this it"
        " man woman person people dog cat car tree house"
    ).split()
    dummy_vocab       = Vocabulary.__new__(Vocabulary)
    dummy_vocab.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    for w in dummy_words:
        dummy_vocab.word2idx.setdefault(w, len(dummy_vocab.word2idx))
    dummy_vocab.idx2word   = {i: w for w, i in dummy_vocab.word2idx.items()}
    dummy_vocab.vocab_size = len(dummy_vocab.word2idx)

    model = ImageCaptioningModel(
        vocab_size    = dummy_vocab.vocab_size,
        pretrained    = pretrained,
        use_attention = use_attn,
    ).to(DEVICE)
    model.eval()
    _model_cache[cache_key] = (model, dummy_vocab)
    return model, dummy_vocab, status


# ─────────────────────────────────────────────────────────────
# Attention visualisation helper
# ─────────────────────────────────────────────────────────────
def make_attention_figure(pil_image: Image.Image,
                          caption: str,
                          attn_weights) -> Image.Image:
    """
    Render per-word Bahdanau attention heat-maps as a clean grid.
    Each tile = original image + viridis overlay + bold word label.
    Returns a PIL Image of the combined figure.
    """
    if attn_weights is None:
        return None

    # Strip punctuation tokens for cleaner display
    words = [w for w in caption.replace(".", "").replace(",", "").split() if w]
    T     = min(len(words), attn_weights.shape[0], 16)   # max 16 tiles
    if T == 0:
        return None

    COLS      = 4
    rows      = (T + COLS - 1) // COLS
    TILE_SIZE = 3.2          # inches per tile
    BG        = "#0f0f1a"    # deep navy background
    LABEL_BG  = "#6366f1"    # indigo badge behind word
    LABEL_FG  = "#ffffff"

    fig = plt.figure(
        figsize=(COLS * TILE_SIZE, rows * TILE_SIZE + 0.7),
        facecolor=BG,
        dpi=140,
    )

    # Caption banner at the top
    fig.text(
        0.5, 0.995,
        f"📝  {caption}",
        ha="center", va="top",
        fontsize=10, color="#e2e8f0",
        fontweight="bold",
        wrap=True,
        transform=fig.transFigure,
    )

    img_resized = np.array(pil_image.resize((224, 224)))

    for t in range(T):
        ax = fig.add_subplot(rows, COLS, t + 1)
        ax.set_facecolor(BG)

        # ── Attention map ──────────────────────────────────────
        alpha = attn_weights[t].reshape(7, 7).numpy().astype(np.float32)
        alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-8)
        # Upsample to 224×224 with bilinear-like smoothing via PIL
        alpha_up = np.array(
            Image.fromarray((alpha * 255).astype(np.uint8))
                 .resize((224, 224), resample=Image.BILINEAR)
        ).astype(np.float32) / 255.0

        # Composite: image + semi-transparent viridis heat overlay
        ax.imshow(img_resized, interpolation="bilinear")
        ax.imshow(alpha_up, cmap="inferno", alpha=0.52,
                  vmin=0, vmax=1, interpolation="bilinear")

        # ── Word badge label ───────────────────────────────────
        word = words[t]
        ax.text(
            0.5, -0.04, word,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=9, fontweight="bold",
            color=LABEL_FG,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=LABEL_BG,
                      edgecolor="none", alpha=0.9),
        )

        # Thin indigo border around each tile
        for spine in ax.spines.values():
            spine.set_edgecolor("#4f46e5")
            spine.set_linewidth(1.2)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for t in range(T, rows * COLS):
        ax = fig.add_subplot(rows, COLS, t + 1)
        ax.set_visible(False)

    plt.subplots_adjust(
        left=0.02, right=0.98,
        top=0.93,  bottom=0.06,
        wspace=0.06, hspace=0.22,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor(), dpi=140)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


# ─────────────────────────────────────────────────────────────
# Core inference function (called by Gradio)
# ─────────────────────────────────────────────────────────────
def run_captioning(image,           # PIL Image from Gradio
                   config_name,
                   ckpt_path,
                   temperature,
                   max_len,
                   show_attention):
    """
    Main inference handler.
    Returns (caption_str, attention_pil_or_None, status_str).
    """
    # ── Validate image ────────────────────────────────────────
    if image is None:
        return "", None, "⚠️  Please upload or paste an image first."

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    # ── Load model ────────────────────────────────────────────
    resolved_path = ckpt_path.strip() if ckpt_path and ckpt_path.strip() \
                    else DEFAULT_CKPT_PATHS.get(config_name, "")
    model, vocab, load_status = load_model(resolved_path, config_name)

    # ── Pre-process ───────────────────────────────────────────
    img_tensor = TRANSFORM(image)

    # ── Generate caption ──────────────────────────────────────
    caption, attn = model.generate(
        img_tensor, vocab, DEVICE,
        max_len     = int(max_len),
        temperature = float(temperature),
    )

    # Capitalise and clean up
    caption = caption.strip()
    if caption:
        caption = caption[0].upper() + caption[1:]
        if not caption.endswith("."):
            caption += "."

    # ── Attention figure ──────────────────────────────────────
    attn_img = None
    if show_attention and attn is not None:
        attn_img = make_attention_figure(image, caption, attn)

    status = load_status + (
        "\n\n🔍 Attention maps generated." if attn_img else
        ("\n\n(No attention – model has no attention module.)"
         if show_attention else "")
    )

    return caption, attn_img, status


# ─────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────
DESCRIPTION = textwrap.dedent("""
## 🖼️ PixelNarrator — Image Captioning with MS COCO

**CNN Encoder (ResNet-50) + LSTM Decoder + Bahdanau Attention**

### How to provide an image
| Method | How |
|--------|-----|
| 📁 File chooser | Click the image box → *Upload image* |
| 🖱️ Drag & drop | Drag any image file onto the box |
| 📋 Paste from clipboard | Copy an image anywhere, then press **Ctrl+V / Cmd+V** inside the box |

*Load a trained checkpoint (`.pth`) from the notebook to get meaningful captions.*
""").strip()

EXAMPLE_IMAGES = []   # populated below if sample images are found
for _p in Path(".").glob("**/*.jpg"):
    EXAMPLE_IMAGES.append(str(_p))
    if len(EXAMPLE_IMAGES) >= 4:
        break

with gr.Blocks(
    title="PixelNarrator – Image Captioning",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
    ),
    css="""
    /* ── Caption output ── */
    #caption-output textarea {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        color: #f0f4ff !important;
        background: #1e1b4b !important;
        border: 2px solid #6366f1 !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
        line-height: 1.6 !important;
        letter-spacing: 0.01em !important;
    }
    /* ── Status box ── */
    .status-box textarea {
        font-size: 0.82rem !important;
        color: #cbd5e1 !important;
    }
    """,
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Row(equal_height=False):
        # ── Left column: inputs ───────────────────────────────
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Input Image  (upload · drag-and-drop · Ctrl+V paste)",
                type="pil",
                sources=["upload", "clipboard"],
                height=320,
            )

            with gr.Accordion("⚙️ Model & Generation Settings", open=True):
                config_dd = gr.Dropdown(
                    choices=list(CONFIG_LABELS.keys()),
                    value="4. Full Model",
                    label="Model configuration",
                )
                ckpt_box = gr.Textbox(
                    label="Checkpoint path  (leave blank for notebook default)",
                    placeholder="e.g.  checkpoints_caption/4_full_model.pth",
                    value="",
                )
                temp_slider = gr.Slider(
                    minimum=0.1, maximum=2.0, value=1.0, step=0.05,
                    label="Temperature  (lower = more confident, higher = more diverse)",
                )
                maxlen_slider = gr.Slider(
                    minimum=5, maximum=60, value=30, step=1,
                    label="Max caption length (tokens)",
                )
                attn_check = gr.Checkbox(
                    value=True,
                    label="Show attention heat-maps  (only for attention models)",
                )

            run_btn = gr.Button("✨  Generate Caption", variant="primary", size="lg")

            if EXAMPLE_IMAGES:
                gr.Examples(
                    examples=[[p] for p in EXAMPLE_IMAGES],
                    inputs=[image_input],
                    label="📷 Example images (found on disk)",
                )

        # ── Right column: outputs ─────────────────────────────
        with gr.Column(scale=1):
            caption_out = gr.Textbox(
                label="Generated Caption",
                lines=3,
                interactive=False,
                elem_id="caption-output",
            )
            attn_out = gr.Image(
                label="Attention Heat-Maps  (one tile per generated word)",
                type="pil",
                height=480,
            )
            status_out = gr.Textbox(
                label="Status / Model info",
                lines=4,
                interactive=False,
                elem_classes=["status-box"],
            )

    # ── Wire up events ────────────────────────────────────────
    run_btn.click(
        fn=run_captioning,
        inputs=[image_input, config_dd, ckpt_box,
                temp_slider, maxlen_slider, attn_check],
        outputs=[caption_out, attn_out, status_out],
    )

    # Also trigger on Enter inside the checkpoint box
    ckpt_box.submit(
        fn=run_captioning,
        inputs=[image_input, config_dd, ckpt_box,
                temp_slider, maxlen_slider, attn_check],
        outputs=[caption_out, attn_out, status_out],
    )

    gr.Markdown(
        "---\n"
        '**Tip:** Train the notebook first, then point the **Checkpoint path** field at one of the saved `.pth` files '
        "in `checkpoints_caption/` to get real captions.\n\n"
        "Made with ❤️ using PyTorch + Gradio · ResNet-50 + LSTM + Bahdanau Attention"
    )


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # accessible on LAN; use "127.0.0.1" for localhost-only
        server_port=7860,
        share=False,             # set True to get a temporary public URL via Gradio tunnel
        inbrowser=True,          # auto-opens browser tab
    )