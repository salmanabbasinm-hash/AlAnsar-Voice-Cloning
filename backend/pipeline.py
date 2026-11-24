"""Al-Ansar Voice Cloning - core pipeline (speech -> text -> translation -> cloned voice)."""
import os, gc, re, difflib, subprocess
import numpy as np
import soundfile as sf
import librosa
import torch

SR = 24000
MAX_SECONDS = int(os.environ.get("ALANSAR_MAX_SECONDS", "600"))
WORK = os.environ.get("ALANSAR_WORK") or ("/content/alansar" if os.path.isdir("/content") else os.path.abspath("./work"))
os.makedirs(WORK, exist_ok=True)

NLLB = {"ur": "urd_Arab", "ar": "arb_Arab", "en": "eng_Latn", "hi": "hin_Deva", "fa": "pes_Arab",
        "tr": "tur_Latn", "fr": "fra_Latn", "es": "spa_Latn", "de": "deu_Latn", "ru": "rus_Cyrl",
        "zh": "zho_Hans", "ja": "jpn_Jpan", "ko": "kor_Hang", "id": "ind_Latn", "it": "ita_Latn",
        "pt": "por_Latn", "bn": "ben_Beng"}


class UserError(Exception):
    """Error whose message is safe to show to the user."""


M = {}


def load_models():
    from faster_whisper import WhisperModel
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from omnivoice import OmniVoice
    if not torch.cuda.is_available():
        raise RuntimeError("GPU نہیں ملا (Colab میں Runtime type میں T4 GPU چنیں)")
    M["asr"] = WhisperModel("large-v3", device="cuda", compute_type="float16")
    name = "facebook/nllb-200-distilled-1.3B"
    M["tok"] = AutoTokenizer.from_pretrained(name)
    M["mt"] = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float16).to("cuda")
    M["tts"] = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)


def sh(cmd):
    subprocess.run(cmd, shell=True, check=True)


def translate(text, src, tgt):
    tok, mt = M["tok"], M["mt"]
    tok.src_lang = NLLB[src]
    x = tok(text, return_tensors="pt", truncation=True, max_length=256).to("cuda")
    y = mt.generate(**x, forced_bos_token_id=tok.convert_tokens_to_ids(NLLB[tgt]),
                    num_beams=5, max_new_tokens=300)
    return tok.batch_decode(y, skip_special_tokens=True)[0]


def build_chunks(segs):
    chunks, cur = [], None
    for s in segs:
        t = s.text.strip()
        if not t:
            continue
        if cur is None:
            cur = {"start": s.start, "end": s.end, "text": t}
            continue
        unfinished = len(cur["text"]) < 70 or cur["text"][-1] not in ".؟?!۔"
        if unfinished and (s.end - cur["start"]) <= 10 and (s.start - cur["end"]) < 1.0:
            cur["text"] += " " + t
            cur["end"] = s.end
        else:
            chunks.append(cur)
            cur = {"start": s.start, "end": s.end, "text": t}
    if cur:
        chunks.append(cur)
    return chunks


def gen_once(text, ref_audio, ref_text, tgt):
    tts = M["tts"]
    kw = dict(text=text, ref_audio=ref_audio, ref_text=ref_text)
    for extra in ({"language": tgt, "num_step": 32}, {"language": tgt}, {}):
        try:
            a = tts.generate(**kw, **extra)
            x = a[0]
            if hasattr(x, "detach"):
                x = x.detach().float().cpu().numpy()
            return np.asarray(x).squeeze().astype(np.float32)
        except TypeError:
            continue
    raise RuntimeError("OmniVoice generate کال ناکام")


def _norm(s):
    return re.sub(r"[^\w\s]", "", s).lower().strip()


def clarity(y, text, tgt):
    y16 = librosa.resample(y.astype(np.float32), orig_sr=SR, target_sr=16000)
    segs, _ = M["asr"].transcribe(y16, language=tgt, beam_size=1)
    hyp = " ".join(s.text for s in segs)
    return difflib.SequenceMatcher(None, _norm(hyp), _norm(text)).ratio()


def speak(text, ref_audio, ref_text, tgt, takes):
    best, best_score = None, -1.0
    for _ in range(takes):
        y = gen_once(text, ref_audio, ref_text, tgt)
        if takes == 1:
            return y
        sc = clarity(y, text, tgt)
        if sc > best_score:
            best, best_score = y, sc
    return best


def stretch(y, f, outdir):
    if abs(f - 1.0) < 0.03:
        return y
    a, b = os.path.join(outdir, "st_in.wav"), os.path.join(outdir, "st_out.wav")
    sf.write(a, y, SR)
    sh(f'ffmpeg -y -loglevel error -i "{a}" -filter:a "atempo={f:.4f}" "{b}"')
    out, _ = sf.read(b, dtype="float32")
    return out if out.ndim == 1 else out.mean(axis=1)


def process(inp, src, tgt, mode, sample, quality, timing, progress, outdir):
    """mode: 'emotion' | 'sample'; quality: 'fast' | 'best'. progress(frac, text)."""
    if src not in NLLB or tgt not in NLLB:
        raise UserError("یہ زبان ابھی سپورٹ نہیں ہے")
    if src == tgt:
        raise UserError("تقریر اور ہدف زبان الگ الگ چنیں")
    if mode == "sample" and not sample:
        raise UserError("اس طریقے میں اپنی آواز کا نمونہ دینا ضروری ہے")
    os.makedirs(outdir, exist_ok=True)
    asr = M["asr"]
    is_video = str(inp).lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".3gp"))
    a16, a24 = os.path.join(outdir, "s16.wav"), os.path.join(outdir, "s24.wav")

    progress(0.05, "آڈیو تیار ہو رہی ہے")
    try:
        sh(f'ffmpeg -y -loglevel error -i "{inp}" -vn -ac 1 -ar 16000 "{a16}"')
        sh(f'ffmpeg -y -loglevel error -i "{inp}" -vn -ac 1 -ar 24000 -af "highpass=f=70,afftdn=nf=-25" "{a24}"')
    except subprocess.CalledProcessError:
        raise UserError("فائل پڑھی نہیں جا سکی، دوسری فائل آزمائیں")
    full, _ = sf.read(a24, dtype="float32")
    if len(full) / SR > MAX_SECONDS:
        raise UserError("تقریر %d منٹ سے لمبی ہے، اسے چھوٹے حصوں میں دیں" % (MAX_SECONDS // 60))

    progress(0.15, "تقریر سنی جا رہی ہے")
    segs, _ = asr.transcribe(a16, language=src, vad_filter=True, beam_size=5)
    chunks = build_chunks(list(segs))
    if not chunks:
        raise UserError("فائل میں تقریر نہیں ملی")

    def cut(c, name):
        s0 = max(0.0, c["start"] - 0.05)
        s1 = min(len(full) / SR, c["end"] + 0.05)
        p = os.path.join(outdir, f"{name}.wav")
        sf.write(p, full[int(s0 * SR):int(s1 * SR)], SR)
        return p

    g_ref, g_text = None, None
    if sample:
        g_ref = os.path.join(outdir, "sample.wav")
        try:
            sh(f'ffmpeg -y -loglevel error -i "{sample}" -t 10 -ac 1 -ar 24000 -af "highpass=f=80,afftdn=nf=-25,loudnorm" "{g_ref}"')
        except subprocess.CalledProcessError:
            raise UserError("آواز کا نمونہ پڑھا نہیں جا سکا")
        rs, _ = asr.transcribe(g_ref, beam_size=5)
        g_text = " ".join(s.text.strip() for s in rs)
    else:
        cand = [c for c in chunks if 3.0 <= c["end"] - c["start"] <= 10.0]
        if cand:
            c = max(cand, key=lambda c: c["end"] - c["start"])
            g_ref, g_text = cut(c, "global"), c["text"]
    use_emotion = (mode != "sample")

    progress(0.3, "ترجمہ ہو رہا ہے")
    for c in chunks:
        c["tr"] = translate(c["text"], src, tgt)

    takes = 1 if quality == "fast" else 3
    audios = []
    for i, c in enumerate(chunks):
        progress(0.35 + 0.55 * i / len(chunks), f"آواز بن رہی ہے {i + 1}/{len(chunks)}")
        d = c["end"] - c["start"]
        if use_emotion and 2.0 <= d <= 10.0:
            ref, rtext = cut(c, f"c{i}"), c["text"]
        elif g_ref is not None:
            ref, rtext = g_ref, g_text
        else:
            raise UserError("اس تقریر میں صاف نمونہ نہیں ملا۔ اپنی آواز کا نمونہ ریکارڈ کر کے دیں")
        audios.append(speak(c["tr"], ref, rtext, tgt, takes))
        torch.cuda.empty_cache()

    progress(0.92, "جوڑا جا رہا ہے")
    pieces, cursor = [], 0.0
    for c, y in zip(chunks, audios):
        if timing:
            d_o = max(c["end"] - c["start"], 0.3)
            f = float(np.clip((len(y) / SR) / d_o, 0.85, 1.3))
            y = stretch(y, f, outdir)
            gap = min(max(c["start"] - cursor, 0.12), 6.0)
        else:
            gap = 0.25
        pieces.append(np.zeros(int(gap * SR), dtype=np.float32))
        pieces.append(y.astype(np.float32))
        cursor += gap + len(y) / SR
    out = np.concatenate(pieces)
    out = out / (float(np.max(np.abs(out))) or 1.0) * 0.95

    out_wav = os.path.join(outdir, "AlAnsar.wav")
    sf.write(out_wav, out, SR)
    out_vid = None
    if is_video:
        out_vid = os.path.join(outdir, "AlAnsar.mp4")
        sh(f'ffmpeg -y -loglevel error -i "{inp}" -i "{out_wav}" -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac "{out_vid}"')

    gc.collect()
    torch.cuda.empty_cache()
    return {"audio": out_wav, "video": out_vid,
            "source_text": "\n".join(c["text"] for c in chunks),
            "target_text": "\n".join(c["tr"] for c in chunks)}
