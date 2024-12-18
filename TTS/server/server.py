#!flask/bin/python
import time
import boto3
from numba import jit
import numpy as np
import hashlib
import argparse
import io
import json
import os
import sys
from pathlib import Path
from threading import Lock
from typing import Union
from urllib.parse import parse_qs

from flask import Flask, render_template, render_template_string, request, \
        send_file, stream_with_context, Response

from TTS.config import load_config
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


def create_argparser():
    def convert_boolean(x):
        return x.lower() in ["true", "1", "yes"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list_models",
        type=convert_boolean,
        nargs="?",
        const=True,
        default=False,
        help="list available pre-trained tts and vocoder models.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument("--vocoder_name", type=str, default=None, help="name of one of the released vocoder models.")

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file.",
    )
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--port", type=int, default=5002, help="port to listen on.")
    parser.add_argument("--use_cuda", type=convert_boolean, default=False, help="true to use CUDA.")
    parser.add_argument("--debug", type=convert_boolean, default=False, help="true to enable Flask debug mode.")
    parser.add_argument("--show_details", type=convert_boolean, default=False, help="Generate model detail page.")
    return parser


# parse the args
args = create_argparser().parse_args()

path = Path(__file__).parent / "../.models.json"
manager = ModelManager(path)

if args.list_models:
    manager.list_models()
    sys.exit()

# update in-use models to the specified released models.
model_path = None
config_path = None
speakers_file_path = None
vocoder_path = None
vocoder_config_path = None

# CASE1: list pre-trained TTS models
if args.list_models:
    manager.list_models()
    sys.exit()

# CASE2: load pre-trained model paths
if args.model_name is not None and not args.model_path:
    model_path, config_path, model_item = manager.download_model(args.model_name)
    if not config_path:
        config_path = model_path + '/config.json'

    args.vocoder_name = model_item["default_vocoder"] if args.vocoder_name is None else args.vocoder_name

if args.vocoder_name is not None and not args.vocoder_path:
    vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

# CASE3: set custom model paths
if args.model_path is not None:
    model_path = args.model_path
    config_path = args.config_path
    speakers_file_path = args.speakers_file_path

if args.vocoder_path is not None:
    vocoder_path = args.vocoder_path
    vocoder_config_path = args.vocoder_config_path

print('model_path:', model_path)
print('config_path:', config_path)
print('speakers_file_path:', speakers_file_path)
# load models
synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    tts_speakers_file=speakers_file_path,
    tts_languages_file=None,
    vocoder_checkpoint=vocoder_path,
    vocoder_config=vocoder_config_path,
    encoder_checkpoint="",
    encoder_config="",
    use_cuda=args.use_cuda,
)

use_multi_speaker = hasattr(synthesizer.tts_model, "num_speakers") and (
    synthesizer.tts_model.num_speakers > 1 or synthesizer.tts_speakers_file is not None
)
speaker_manager = getattr(synthesizer.tts_model, "speaker_manager", None)

use_multi_language = hasattr(synthesizer.tts_model, "num_languages") and (
    synthesizer.tts_model.num_languages > 1 or synthesizer.tts_languages_file is not None
)
language_manager = getattr(synthesizer.tts_model, "language_manager", None)

# TODO: set this from SpeakerManager
use_gst = synthesizer.tts_config.get("use_gst", False)
app = Flask(__name__)


def style_wav_uri_to_dict(style_wav: str) -> Union[str, dict]:
    """Transform an uri style_wav, in either a string (path to wav file to be use for style transfer)
    or a dict (gst tokens/values to be use for styling)

    Args:
        style_wav (str): uri

    Returns:
        Union[str, dict]: path to file (str) or gst style (dict)
    """
    if style_wav:
        if os.path.isfile(style_wav) and style_wav.endswith(".wav"):
            return style_wav  # style_wav is a .wav file located on the server

        style_wav = json.loads(style_wav)
        return style_wav  # style_wav is a gst dictionary with {token1_id : token1_weigth, ...}
    return None


@app.route("/")
def index():
    return render_template(
        "index.html",
        show_details=args.show_details,
        use_multi_speaker=use_multi_speaker,
        use_multi_language=use_multi_language,
        speaker_ids=speaker_manager.name_to_id if speaker_manager is not None else None,
        language_ids=language_manager.name_to_id if language_manager is not None else None,
        use_gst=use_gst,
    )


@app.route("/details")
def details():
    if args.config_path is not None and os.path.isfile(args.config_path):
        model_config = load_config(args.config_path)
    else:
        if args.model_name is not None:
            model_config = load_config(config_path)

    if args.vocoder_config_path is not None and os.path.isfile(args.vocoder_config_path):
        vocoder_config = load_config(args.vocoder_config_path)
    else:
        if args.vocoder_name is not None:
            vocoder_config = load_config(vocoder_config_path)
        else:
            vocoder_config = None

    return render_template(
        "details.html",
        show_details=args.show_details,
        model_config=model_config,
        vocoder_config=vocoder_config,
        args=args.__dict__,
    )


lock = Lock()


@app.route("/api/tts", methods=["GET", "POST"])
def tts():
    with lock:
        text = request.headers.get("text") or request.values.get("text", "")
        speaker_idx = request.headers.get("speaker-id") or request.values.get("speaker_id", "")
        language_idx = request.headers.get("language-id") or request.values.get("language_id", "")
        style_wav = request.headers.get("style-wav") or request.values.get("style_wav", "")
        style_wav = style_wav_uri_to_dict(style_wav)

        print(f" > Model input: {text}")
        print(f" > Speaker Idx: {speaker_idx}")
        print(f" > Language Idx: {language_idx}")
        wavs = synthesizer.tts(text, speaker_name=speaker_idx, language_name=language_idx, style_wav=style_wav)
        out = io.BytesIO()
        synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")


# Amazon SageMaker compatibility layer
references = {}
frame_bytes_10ms = 24000 * 1 * 2 // 100  # sample_rate * channel_num * bytes_per_sample // 100


def get_bucket_and_key(s3uri):
    """
    get_bucket_and_key is helper function
    """
    pos = s3uri.find('/', 5)
    bucket = s3uri[5: pos]
    key = s3uri[pos + 1:]
    return bucket, key


def download_s3_wav(source_s3_url, local_file_path):
    s3 = boto3.client('s3')
    bucket_name, s3_file_path = get_bucket_and_key(source_s3_url)
    # 下载文件
    try:
        if os.path.exists(local_file_path):
            return

        s3.download_file(bucket_name, s3_file_path, local_file_path)
        print(f's3 wav file {s3_file_path} saved to {local_file_path}')
    except Exception as e:
        print(f's3 wav file download failed: {e}')


@jit(nopython=True)
def process_audio(wav):
    wav = np.clip(wav, -1, 1)
    return (wav * 32767).astype(np.int16)


def get_latent_and_embedding(speaker_wav=None, speaker_name=None):
    if speaker_wav:  # mannual set speakers
        for idx, wav_file in enumerate(speaker_wav):
            if wav_file.startswith("s3://"):
                file_name_hash = hashlib.sha1(wav_file.encode('utf-8')).digest().hex()
                local_file_name = f'/tmp/{file_name_hash}_{wav_file.split("/")[-1]}'
                download_s3_wav(wav_file, local_file_name)
                speaker_wav[idx] = local_file_name

        speaker_wav_str = '__'.join(speaker_wav)
        cached_speaker_wav = references.get(speaker_wav_str, {})
        # should limit cached count
        if not cached_speaker_wav:
            gpt_cond_latent, speaker_embedding = synthesizer.tts_model.get_conditioning_latents(audio_path=speaker_wav)
            references[speaker_wav_str] = {
                "gpt_cond_latent": gpt_cond_latent,
                "speaker_embedding": speaker_embedding
            }
        else:
            gpt_cond_latent = cached_speaker_wav["gpt_cond_latent"]
            speaker_embedding = cached_speaker_wav["speaker_embedding"]
    else:  # get speaker by speaker name
        if synthesizer.tts_config.model == "xtts":
            speaker_id = synthesizer.tts_model.speaker_manager.name_to_id[speaker_name]
        else:
            # defaults to the first speaker
            speaker_id = list(synthesizer.tts_model.speaker_manager.name_to_id.values())[0]
        gpt_cond_latent, speaker_embedding = synthesizer.speaker_manager.speakers[speaker_id].values()

    return gpt_cond_latent, speaker_embedding


@app.route('/ping', methods=["GET", "POST"])
def ping():
    return "ok", 200


@app.route('/invocations', methods=["POST"])
def invocations():
    with lock:
        data = request.json
        print(data)

        text = data.get("text", "")
        speaker_name = data.get("speaker_name", "")
        speaker_wav = data.get("speaker_wav", [])  # reference wav
        language_idx = data.get("language_id", "")  # en, zh, etc..
        temperature = data.get("temperature", 0.75)
        top_k = data.get("top_k", 50)
        top_p = data.get("top_p", 0.85)
        speed = data.get("speed", 1.0)
        style_wav = data.get("style_wav", "")
        style_wav = style_wav_uri_to_dict(style_wav)

        if not (speaker_wav or speaker_name):
            return "should specify 'speaker_wav' or 'speaker_name'", 400

        if not (language_idx):
            return "should specify 'language_id'", 400

        if isinstance(speaker_wav, str):
            speaker_wav = speaker_wav.split(',')

        try:
            gpt_cond_latent, speaker_embedding = get_latent_and_embedding(speaker_wav, speaker_name)
        except Exception as e:
            return f"An unexpected error occurred: {e}", 400

        print(f" > Model input: {text}")
        print(f" > Speaker wav: {speaker_wav}")
        print(f" > Speaker Name: {speaker_name}")
        print(f" > Language Idx: {language_idx}")

        t0 = time.time()

        @stream_with_context
        def generate():
            chunks = synthesizer.tts_model.inference_stream(
                text,
                language_idx,
                gpt_cond_latent,
                speaker_embedding,
                stream_chunk_size=20,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                speed=speed,
            )

            for i, chunk in enumerate(chunks):
                if i == 0:
                    print(f"Time to first chunck: {time.time() - t0}")
                # print(f"Received chunk {i} of audio length {chunk.shape[-1]}")
                wav = chunk.clone().detach().cpu().numpy()
                wav = wav[None, : int(wav.shape[0])]
                wav = process_audio(wav)
                wav = wav.tobytes()

                for i in range(0, len(wav), frame_bytes_10ms):
                    chunk = wav[i:i + frame_bytes_10ms]
                    yield chunk

                # wav_chuncks.append(chunk)

        return Response(generate(), mimetype="audio/wav")


# Basic MaryTTS compatibility layer
@app.route("/locales", methods=["GET"])
def mary_tts_api_locales():
    """MaryTTS-compatible /locales endpoint"""
    # NOTE: We currently assume there is only one model active at the same time
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    return render_template_string("{{ locale }}\n", locale=model_details[1])


@app.route("/voices", methods=["GET"])
def mary_tts_api_voices():
    """MaryTTS-compatible /voices endpoint"""
    # NOTE: We currently assume there is only one model active at the same time
    if args.model_name is not None:
        model_details = args.model_name.split("/")
    else:
        model_details = ["", "en", "", "default"]
    return render_template_string(
        "{{ name }} {{ locale }} {{ gender }}\n", name=model_details[3], locale=model_details[1], gender="u"
    )


@app.route("/process", methods=["GET", "POST"])
def mary_tts_api_process():
    """MaryTTS-compatible /process endpoint"""
    with lock:
        if request.method == "POST":
            data = parse_qs(request.get_data(as_text=True))
            # NOTE: we ignore param. LOCALE and VOICE for now since we have only one active model
            text = data.get("INPUT_TEXT", [""])[0]
        else:
            text = request.args.get("INPUT_TEXT", "")
        print(f" > Model input: {text}")
        wavs = synthesizer.tts(text)
        out = io.BytesIO()
        synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype="audio/wav")


def main():
    app.run(debug=args.debug, host="::", port=args.port)


if __name__ == "__main__":
    main()