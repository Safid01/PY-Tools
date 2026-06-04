# transcribe_bn.py
import os, sys, tempfile
import speech_recognition as sr
from pydub import AudioSegment

LANG = "bn-BD"  # Bangla (Bangladesh)

def to_mono16k_wav(in_path):
    audio = AudioSegment.from_file(in_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_path = os.path.splitext(in_path)[0] + ".wav"
    audio.export(wav_path, format="wav")
    return wav_path

def iter_chunks(audio, chunk_ms=30000, overlap_ms=500):
    start = 0
    length = len(audio)
    while start < length:
        end = min(start + chunk_ms, length)
        yield (start, end, audio[start:end])
        # small overlap helps avoid cutting off words
        start = end - overlap_ms

def transcribe_file(in_path, language=LANG):
    recognizer = sr.Recognizer()

    # Convert to mono 16k wav (or reuse if already wav)
    if in_path.lower().endswith(".wav"):
        wav_path = in_path
    else:
        wav_path = to_mono16k_wav(in_path)

    # Load normalized wav
    audio = AudioSegment.from_wav(wav_path)
    text_parts = []

    for (s, e, piece) in iter_chunks(audio, chunk_ms=30000, overlap_ms=500):
        # export chunk to a temp wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            piece.export(tmp.name, format="wav")
            tmp_path = tmp.name

        # recognize
        try:
            with sr.AudioFile(tmp_path) as source:
                audio_data = recognizer.record(source)
            # You can tune for noisy audio:
            # recognizer.energy_threshold = 300
            # recognizer.dynamic_energy_threshold = True
            txt = recognizer.recognize_google(audio_data, language=language)
            text_parts.append(txt)
            print(f"Chunk {s}-{e} ms → OK")
        except sr.UnknownValueError:
            print(f"Chunk {s}-{e} ms → (unintelligible, skipped)")
        except Exception as ex:
            print(f"Chunk {s}-{e} ms → ERROR: {ex}")
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    return " ".join(text_parts).strip()

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_bn.py <audio-file.mp3|wav>")
        sys.exit(1)

    in_path = sys.argv[1]
    if not os.path.exists(in_path):
        print(f"File not found: {in_path}")
        sys.exit(1)

    out_path = os.path.splitext(in_path)[0] + "_bn.txt"
    text = transcribe_file(in_path, language=LANG)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    print(f"\nSaved transcript → {out_path}")

if __name__ == "__main__":
    main()