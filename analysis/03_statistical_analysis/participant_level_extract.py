#!/usr/bin/env python
"""
Participant-level event-related analysis, step 1 of 3: raw recordings -> per-participant epochs.

Reads the raw BIOPAC recordings and the conversation logs directly, so it does not depend on
the outputs of signal_cleaner.py or event_related_signal_analysis.py; it needs only the
per-participant experimental-info files written by step 4 (condition and recording start).

Signal processing
  EDA   absolute microsiemens, 4th-order zero-phase Butterworth low-pass at 5 Hz applied to
        the whole recording. No offset removal and no sample-level artefact correction, so
        skin conductance level is preserved.
  fEMG  one 4th-order zero-phase Butterworth band-pass (20-500 Hz) applied to the whole
        recording, then a 50 ms (101-sample) moving root-mean-square envelope. No software
        notch filter: the recordings already carry AcqKnowledge's real-time line-frequency
        comb filter (50 Hz and harmonics). No sample-level artefact correction; outlier
        epochs are handled in step 2.

Time-locking: t = 0 is the conversation-log `timestamp_start` of each chatbot message,
written by the chat interface when the participant's message has been submitted and the
model request is issued. The reply-stream duration (`timestamp_end` - `timestamp_start`)
is stored for sensitivity analyses. Epochs span -2 to +18 s (EDA) and -2 to +13 s (fEMG);
messages whose window falls outside the recording are skipped.

Output: one .npz per participant in $STUDY_DATA_ROOT/event_related_participant_level/participants/.
Waveforms are stored at 100 Hz (block means of 20 samples); epoch peaks are computed at 2000 Hz.
"""
import json
import os
from datetime import datetime
from multiprocessing import Pool

import bioread
from dotenv import load_dotenv
import numpy as np
from scipy import signal as sps

FS = 2000.0
DS = 20                                    # 2000 Hz -> 100 Hz block means

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
RAW = os.path.join(DATA_ROOT, "psychophysiology_signal")
LOGS = os.path.join(DATA_ROOT, "conversation_logs")
META = os.path.join(DATA_ROOT, "cleaned_signals", "cleaned_signals")      # experimental-info files from step 4
OUT = os.path.join(DATA_ROOT, "event_related_participant_level", "participants")
EDA = "EDA2111000305"
EMG = {"corrugator": "fEMG2212000226", "zygomaticus": "fEMG2212000228"}
EDA_PRE, EDA_POST = 2.0, 18.0
EMG_PRE, EMG_POST = 2.0, 13.0


def ts(s):
    # Conversation logs carry local wall-clock time; a trailing UTC label is ignored.
    return datetime.fromisoformat(s.replace("Z", "+00:00").split("+")[0])


def eda_filter(x):
    b, a = sps.butter(4, 5.0 / (FS / 2), btype="low")
    return sps.filtfilt(b, a, x)


def emg_envelope(x):
    """Single 4th-order 20-500 Hz band-pass, then 50 ms (101-sample) moving RMS."""
    b, a = sps.butter(4, [20.0 / (FS / 2), 500.0 / (FS / 2)], btype="band")
    y = sps.filtfilt(b, a, x - np.mean(x))
    w = int(0.05 * FS)
    w += (w % 2 == 0)
    h = w // 2
    c = np.cumsum(np.pad(y, (h, h), mode="reflect") ** 2)
    c = np.concatenate([[0.0], c])
    return np.sqrt((c[w:] - c[:-w]) / w)[: len(y)]


def epochs(sig, onsets, pre, post):
    ps, qs = int(pre * FS), int(post * FS)
    kept, E = [], []
    for i, s in enumerate(onsets):
        a, b = s - ps, s + qs
        if a >= 0 and b <= len(sig):
            E.append(sig[a:b])
            kept.append(i)
    return np.array(E), np.array(kept, int)


def block(E):
    n = E.shape[1] // DS
    return E[:, : n * DS].reshape(E.shape[0], n, DS).mean(2)


def tonic_phasic(x):
    """Conventional split of the absolute EDA signal: tonic < 0.05 Hz, phasic 0.05-1 Hz."""
    q = int(FS // 20)
    xd = sps.decimate(x, q, ftype="fir", zero_phase=True)
    fsd = FS / q
    bt, at = sps.butter(2, 0.05 / (fsd / 2), btype="low")
    bp, ap = sps.butter(2, [0.05 / (fsd / 2), 1.0 / (fsd / 2)], btype="band")
    tonic, phasic = sps.filtfilt(bt, at, xd), sps.filtfilt(bp, ap, xd)
    return dict(eda_tonic_mean=float(tonic.mean()), eda_tonic_sd=float(tonic.std()),
                eda_phasic_rms=float(np.sqrt(np.mean(phasic ** 2))),
                eda_phasic_var_share=float(np.var(phasic) / np.var(xd)))


def channels(path, names):
    d = bioread.read(path)
    out = {}
    for c in d.channels:
        if c.name in names and c.name not in out:      # one file stores EDA three times
            if abs(c.samples_per_second - FS) > 1e-6:
                raise ValueError(f"{path}:{c.name} sampled at {c.samples_per_second} Hz")
            out[c.name] = np.asarray(c.data, float)
    return out


def assistant_messages():
    """Assistant messages per participant, from the conversation logs (one file each)."""
    out = {}
    for f in sorted(os.listdir(LOGS)):
        if f.endswith(".json"):
            j = json.load(open(f"{LOGS}/{f}"))
            uid = j.get("user_id") or j.get("metadata", {}).get("user_id")
            if uid in out:
                raise ValueError(f"two conversation logs for {uid}")
            out[uid] = [m for m in j["messages"] if m.get("role") == "assistant"]
    return out


MESSAGES = assistant_messages()


def one(pid):
    info = json.load(open(f"{META}/{pid}/{pid}_experimental_info.json"))
    cond = "SL" if info["experimental_group"] == "group_1" else "DT"
    t0 = ts(info["biopac_recording_timestamps"]["main"]["recording_start_time"])
    resp = MESSAGES[pid]
    on_s = np.array([(ts(r["timestamp_start"]) - t0).total_seconds() for r in resp])
    stream = np.array([(ts(r["timestamp_end"]) - ts(r["timestamp_start"])).total_seconds() for r in resp])
    onsets = np.round(on_s * FS).astype(int)
    nxt = np.append(np.diff(on_s), np.inf)                 # time to next assistant onset
    prv = np.insert(np.diff(on_s), 0, np.inf)              # time since previous onset

    raw = channels(f"{RAW}/{pid}_main.acq", [EDA] + list(EMG.values()))
    rest = channels(f"{RAW}/{pid}_baseline.acq", list(EMG.values()))
    out = dict(pid=pid, cond=cond)

    # ---- EDA (absolute microsiemens) ----------------------------------------
    y = eda_filter(raw[EDA])
    E, k = epochs(y, onsets, EDA_PRE, EDA_POST)
    pre = E[:, : int(EDA_PRE * FS)].mean(1)
    Ec = E - pre[:, None]
    post = Ec[:, int(EDA_PRE * FS):]
    out.update(eda_epochs=block(Ec).astype(np.float32), eda_pre_scl=pre,
               eda_epoch_peak=post.max(1), eda_kept=k, eda_onset_s=on_s[k],
               eda_stream_s=stream[k], eda_next_s=nxt[k], eda_prev_s=prv[k],
               eda_session_scl=float(y.mean()))
    out.update(tonic_phasic(raw[EDA]))

    # ---- facial EMG -----------------------------------------------------------
    for lab, ch in EMG.items():
        env = emg_envelope(raw[ch])
        E, k = epochs(env, onsets, EMG_PRE, EMG_POST)
        renv = emg_envelope(rest[ch])
        out[f"{lab}_epochs_abs"] = block(E).astype(np.float32)       # absolute envelope, mV
        out[f"{lab}_kept"] = k
        out[f"{lab}_rest"] = float(renv[int(5 * FS): -int(5 * FS)].mean())
        out[f"{lab}_session"] = float(env.mean())
    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(f"{OUT}/{pid}.tmp.npz", **out)
    os.replace(f"{OUT}/{pid}.tmp.npz", f"{OUT}/{pid}.npz")
    return pid, cond, len(out["eda_kept"]), len(out["corrugator_kept"]), len(resp)


def main():
    pids = sorted(p for p in os.listdir(META) if os.path.isfile(f"{META}/{p}/{p}_experimental_info.json"))
    with Pool(4, maxtasksperchild=1) as pool:
        res = list(pool.imap_unordered(one, pids))
    res.sort()
    for cond in ("DT", "SL"):
        r = [x for x in res if x[1] == cond]
        print(f"{cond}: {len(r)} participants, chatbot messages {sum(x[4] for x in r)}, "
              f"EDA epochs {sum(x[2] for x in r)}, fEMG epochs {sum(x[3] for x in r)}")


if __name__ == "__main__":
    main()
