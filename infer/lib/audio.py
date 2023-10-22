import os
import traceback

import librosa
import numpy as np
import av
from io import BytesIO

from gradio import File


def wav2(i, o, format):
    inp = av.open(i, "rb")
    if format == "m4a":
        format = "mp4"
    out = av.open(o, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "mp4":
        format = "aac"

    ostream = out.add_stream(format)

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    for p in ostream.encode(None):
        out.mux(p)

    out.close()
    inp.close()


def audio2(i, o, format, sr):
    inp = av.open(i, "rb")
    out = av.open(o, "wb", format=format)
    if format == "ogg":
        format = "libvorbis"
    if format == "f32le":
        format = "pcm_f32le"

    ostream = out.add_stream(format, channels=1)
    ostream.sample_rate = sr

    for frame in inp.decode(audio=0):
        for p in ostream.encode(frame):
            out.mux(p)

    out.close()
    inp.close()


def load_audio(file_or_path, sr):
    try:
        if isinstance(file_or_path, File):
            # as Gradio File object
            with BytesIO(file_or_path.read()) as f:
                with BytesIO() as out:
                    audio2(f, out, "f32le", sr)
                    return np.frombuffer(out.getvalue(), np.float32).flatten()

        elif isinstance(file_or_path, str):
            # as File path
            file_or_path = (
                file_or_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )

            if not os.path.exists(file_or_path):
                raise RuntimeError(
                    "You input a wrong audio path that does not exist, please fix it!"
                )

            with open(file_or_path, "rb") as f:
                with BytesIO() as out:
                    audio2(f, out, "f32le", sr)
                    return np.frombuffer(out.getvalue(), np.float32).flatten()

        else:
            raise RuntimeError(
                "Invalid input type. Please provide either a Gradio File object or a file path."
            )

    except AttributeError:
        audio = file_or_path[1] / 32768.0
        if len(audio.shape) == 2:
            audio = np.mean(audio, -1)
        return librosa.resample(audio, orig_sr=file_or_path[0], target_sr=sr)

    except:
        raise RuntimeError(traceback.format_exc())
