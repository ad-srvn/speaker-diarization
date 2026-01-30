import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import soundfile as sf
import os
import subprocess

_SILERO_MODEL = None
_ECAPA = None
_ECAPA_DEVICE = None

def get_silero():
    global _SILERO_MODEL
    if _SILERO_MODEL is None:
        from silero_vad import load_silero_vad
        _SILERO_MODEL = load_silero_vad()
    return _SILERO_MODEL

def get_ecapa(device=None):
    global _ECAPA, _ECAPA_DEVICE
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if _ECAPA is None or _ECAPA_DEVICE != device:
        _ECAPA = EncoderClassifier.from_hparams(
            "speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": device}
        )
        _ECAPA_DEVICE = device
    return _ECAPA

def audioload(path, torchmode=False, target_sr=16000):
    import soundfile as sf
    import librosa
    import numpy as np
    import torch

    audio, sr = sf.read(path, dtype="float32") 
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    
    audio = np.clip(audio, -1.0, 1.0)

    if torchmode:
        return torch.from_numpy(audio)
    return audio
def vad(path):
    from silero_vad import get_speech_timestamps
    wav = audioload(path, torchmode=True)
    model = get_silero()
    return get_speech_timestamps(wav, model, return_seconds=True)

def chunker(speech_timestamps, chunk_length=1.5, hop_length=0.5):
    chunk = []
    for seg in speech_timestamps:
        start = float(seg["start"])
        end = float(seg["end"])
        dur = end - start
        if dur <= 0:
            continue
        
        if dur <= chunk_length:
            chunk.append({"start": start, "end": end})
            continue

        t = start
        while t < end:
            e = min(t + chunk_length, end)
            if e > t:
                chunk.append({"start": t, "end": e})
            if e >= end:
                break
            t += hop_length

    return chunk
def extract_ecapa(audio_16k_mono: torch.Tensor, chunks, device=None):
    import numpy as np
    import torch

    model = get_ecapa(device=device)
    device = model.device  # ensure consistent
    wav = audio_16k_mono.to(device)

    embs = []
    with torch.no_grad():
        for c in chunks:
            s, e = int(c["start"] * 16000), int(c["end"] * 16000)
            if e > s:
                v = model.encode_batch(wav[s:e].unsqueeze(0)).squeeze()
                v = v / (v.norm(p=2) + 1e-12)
                embs.append(v.cpu().numpy())

    if len(embs) == 0:
        return np.zeros((0, 192), dtype=np.float32)
    return np.vstack(embs)

def ahc_cluster_labels(embs: np.ndarray, distance_threshold=1.0, linkage="average"):
    from sklearn.cluster import AgglomerativeClustering
    embs = np.asarray(embs)
    if embs.ndim != 2 or embs.shape[0] == 0:
        return np.zeros((0,), dtype=int), 0
    if embs.shape[0] == 1:
        return np.array([0], dtype=int), 1

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage=linkage,
        distance_threshold=distance_threshold,
    )
    labels = model.fit_predict(embs).astype(int)
    return labels, labels.max() + 1
def play_with_moving_cursor_stream(
    audio_path: str,
    chunks: list,
    labels: np.ndarray,
    sr: int = 16000,
    dt: float = 0.05,
    title: str = "Diarization playback",
    show_nospeech: bool = True,
):
    import soundfile as sf
    import librosa
    import sounddevice as sd

    
    x, fs = sf.read(audio_path, dtype="float32", always_2d=True)  
    x = x.mean(axis=1)  
    if fs != sr:
        x = librosa.resample(x, orig_sr=fs, target_sr=sr).astype(np.float32)
        fs = sr

    
    x = np.asarray(x, dtype=np.float32)
    m = float(np.max(np.abs(x))) + 1e-12
    if m > 0.99:
        x = 0.99 * (x / m)
    
    x = np.ascontiguousarray(x)

    duration = len(x) / float(fs)
    t_audio = np.arange(len(x), dtype=np.float32) / float(fs)

    chunks = list(chunks) if chunks is not None else []
    labels = np.asarray(labels) if labels is not None else np.array([], dtype=int)
    if len(chunks) != len(labels):
        raise ValueError(f"chunks and labels must match: {len(chunks)} vs {len(labels)}")

    n_bins = int(np.ceil(duration / dt))
    votes = [{} for _ in range(n_bins)]

    for c, lab in zip(chunks, labels):
        if "start" not in c or "end" not in c:
            continue
        s = float(c["start"]); e = float(c["end"])
        s = max(0.0, min(duration, s))
        e = max(0.0, min(duration, e))
        if e <= s:
            continue
        b0 = max(0, int(np.floor(s / dt)))
        b1 = min(n_bins, int(np.ceil(e / dt)))
        lab = int(lab)
        for b in range(b0, b1):
            votes[b][lab] = votes[b].get(lab, 0) + 1

    bin_label = [None] * n_bins
    for b, v in enumerate(votes):
        if v:
            bin_label[b] = max(v.items(), key=lambda kv: kv[1])[0]

    segments = []
    cur = bin_label[0] if n_bins > 0 else None
    start_bin = 0
    for b in range(1, n_bins + 1):
        nxt = bin_label[b] if b < n_bins else None
        if nxt != cur:
            s = start_bin * dt
            e = b * dt
            if e > s:
                segments.append((cur, s, e))
            cur = nxt
            start_bin = b

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t_audio, x, color="black", linewidth=0.6, zorder=3)

    uniq = sorted(set(int(l) for l in labels.tolist())) if len(labels) else []
    cmap = plt.get_cmap("tab20")
    color_for = {lab: cmap(i % cmap.N) for i, lab in enumerate(uniq)}
    nospeech_color = (0.85, 0.85, 0.85, 0.6)

    for lab, s, e in segments:
        if lab is None:
            if show_nospeech:
                ax.axvspan(s, e, color=nospeech_color, linewidth=0, zorder=0)
        else:
            ax.axvspan(s, e, color=color_for[lab], alpha=0.25, linewidth=0, zorder=0)

    handles = [plt.Line2D([0], [0], color=color_for[lab], linewidth=6, alpha=0.6) for lab in uniq]
    texts = [f"SPEAKER_{lab:02d}" for lab in uniq]
    if show_nospeech:
        handles.append(plt.Line2D([0], [0], color=nospeech_color, linewidth=6))
        texts.append("NO_SPEECH")
    if handles:
        ax.legend(handles, texts, loc="upper right", ncol=2, frameon=True)

    ax.set_xlim(0, duration)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)

    cursor = ax.axvline(0.0, color="red", linewidth=2.0, zorder=5)
    fig.tight_layout()

    sd.stop()  

    state = {
        "idx": 0,
        "playing": True,
    }

    blocksize = 1024  

    def callback(outdata, frames, time_info, status):
        if status:
            pass

        i = state["idx"]
        j = i + frames

        if i >= len(x):
            outdata[:] = 0
            state["playing"] = False
            raise sd.CallbackStop()

        chunk = x[i:min(j, len(x))]
        out = np.zeros((frames, 1), dtype=np.float32)
        out[:len(chunk), 0] = chunk
        outdata[:] = out

        state["idx"] = j

    stream = sd.OutputStream(
        samplerate=fs,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    )
    stream.start()

    def update(_frame):
        if not state["playing"]:
            cursor.set_xdata([duration, duration])
            return (cursor,)
        t = state["idx"] / float(fs)
        t = min(duration, max(0.0, t))
        cursor.set_xdata([t, t])
        return (cursor,)

    anim = FuncAnimation(fig, update, interval=33, blit=True)

    def _on_close(_evt):
        try:
            stream.stop()
            stream.close()
        except Exception:
            pass
        try:
            sd.stop()
        except Exception:
            pass

    fig.canvas.mpl_connect("close_event", _on_close)
    plt.show()

    _on_close(None)
def diarization_json_from_chunks_labels(
    audio_url: str,
    audio_path: str,
    chunks: list,
    labels,
    merge_gap: float = 0.15,
    min_dur: float = 0.20,
    decimals: int = 3,
) -> dict:
    """
    Build the JSON payload for the frontend, automatically determining
    the true audio duration from the audio file.

    Output format:
    {
      "audioUrl": "...",
      "duration": <float>,
      "speakers": [
        {"start":..., "end":..., "speaker":...},
        ...
      ]
    }

    Parameters
    ----------
    audio_url : str
        URL served to the browser (e.g. "/files/abcd.wav").
    audio_path : str
        Local filesystem path to the audio file.
    chunks : list[dict]
        [{"start": float, "end": float}, ...] in seconds.
    labels : array-like
        Speaker label per chunk (same length as chunks).
    merge_gap : float
        Merge same-speaker segments if gap <= merge_gap seconds.
    min_dur : float
        Drop segments shorter than this duration.
    decimals : int
        Decimal precision for times in output.

    Returns
    -------
    dict
        JSON-ready diarization payload.
    """

    # ---- determine real duration from file ----
    info = sf.info(audio_path)
    duration = info.frames / float(info.samplerate)

    # ---- validation ----
    chunks = list(chunks) if chunks is not None else []
    labels = np.asarray(labels) if labels is not None else np.array([], dtype=int)

    if len(chunks) != len(labels):
        raise ValueError(
            f"chunks and labels must have same length: {len(chunks)} vs {len(labels)}"
        )

    # ---- build raw labeled segments ----
    items = []
    for c, lab in zip(chunks, labels):
        if not isinstance(c, dict) or "start" not in c or "end" not in c:
            continue
        s = float(c["start"])
        e = float(c["end"])
        if e <= s:
            continue
        try:
            spk = int(lab)
        except Exception:
            continue
        items.append((s, e, spk))

    # sort by time
    items.sort(key=lambda x: (x[0], x[1]))

    # clamp to valid audio range
    def clamp(t: float) -> float:
        return max(0.0, min(duration, float(t)))

    # ---- merge adjacent same-speaker segments ----
    merged = []
    for s, e, spk in items:
        s = clamp(s)
        e = clamp(e)
        if e <= s:
            continue

        if not merged:
            merged.append([s, e, spk])
            continue

        ps, pe, pspk = merged[-1]
        if spk == pspk and s <= pe + merge_gap:
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e, spk])

    # ---- finalize output ----
    speakers = []
    for s, e, spk in merged:
        if (e - s) >= min_dur:
            speakers.append(
                {
                    "start": round(s, decimals),
                    "end": round(e, decimals),
                    "speaker": int(spk),
                }
            )

    return {
        "audioUrl": audio_url,
        "duration": round(duration, decimals),
        "speakers": speakers,
    }
def to_wav_16k_mono_ffmpeg(in_path: str, out_path: str, target_sr: int = 16000) -> float:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", in_path,
        "-ac", "1",
        "-ar", str(target_sr),
        "-vn",
        out_path,
    ]
    subprocess.run(cmd, check=True)

    info = sf.info(out_path)
    return info.frames / float(info.samplerate)
def make_diarization_payload(
    audio_url: str,
    audio_path: str,  
    chunks: list,
    labels,
    merge_gap: float = 0.15,
    min_dur: float = 0.20,
    decimals: int = 3,
) -> dict:
    info = sf.info(audio_path)
    duration = info.frames / float(info.samplerate)

    chunks = list(chunks) if chunks is not None else []
    labels = np.asarray(labels) if labels is not None else np.array([], dtype=int)

    if len(chunks) != len(labels):
        raise ValueError(f"chunks and labels length mismatch: {len(chunks)} vs {len(labels)}")

    items = []
    for c, lab in zip(chunks, labels):
        if not isinstance(c, dict) or "start" not in c or "end" not in c:
            continue
        s = float(c["start"])
        e = float(c["end"])
        if e <= s:
            continue
        items.append((s, e, int(lab)))

    items.sort(key=lambda x: (x[0], x[1]))

    def clamp(t: float) -> float:
        return max(0.0, min(duration, float(t)))

    merged = []
    for s, e, spk in items:
        s = clamp(s); e = clamp(e)
        if e <= s:
            continue
        if not merged:
            merged.append([s, e, spk]); continue
        ps, pe, pspk = merged[-1]
        if spk == pspk and s <= pe + merge_gap:
            merged[-1][1] = max(pe, e)
        else:
            merged.append([s, e, spk])

    speakers = []
    for s, e, spk in merged:
        if (e - s) >= min_dur:
            speakers.append({
                "start": round(s, decimals),
                "end": round(e, decimals),
                "speaker": int(spk),
            })

    return {
        "audioUrl": audio_url,
        "duration": round(duration, decimals),
        "speakers": speakers,
    }
def diarize_file(audio_path: str):
    voice = vad(audio_path)
    chunks = chunker(voice, chunk_length=2.0, hop_length=1.0)

    if len(chunks) == 0:
        return chunks, np.array([], dtype=int)

    Taudio = audioload(audio_path, torchmode=True)
    embs = extract_ecapa(Taudio, chunks)

    if embs.shape[0] == 0:
        return chunks, np.array([], dtype=int)

    labels, K = ahc_cluster_labels(embs, distance_threshold=0.71)

    remap = {}
    next_id = 0
    new_labels = []
    for lab in labels:
        lab = int(lab)
        if lab not in remap:
            remap[lab] = next_id
            next_id += 1
        new_labels.append(remap[lab])
    labels = np.array(new_labels, dtype=int)

    return chunks, labels

def diarizer_with_plot(path):
    voice=vad(path)
    chunks=chunker(voice)
    if len(chunks) == 0:
        print("No chunks found (maybe no speech detected).")
        quit()
    Taudio=audioload(path,torchmode=True)
    embs=extract_ecapa(Taudio,chunks)
    if embs.shape[0] == 0:
        print("No embeddings extracted (chunks too short or no speech).")
        quit()
    labels, K = ahc_cluster_labels(embs, distance_threshold=0.75, linkage="average")
    print("Speakers:", K)
    play_with_moving_cursor_stream(audio_path=path,chunks=chunks,labels=labels,title="Waveform colored by speakers")
