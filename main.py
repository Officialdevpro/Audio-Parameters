from flask import Flask, request, jsonify
import numpy as np
import librosa
import tempfile
import os

app = Flask(__name__)


def extract_audio_features(audio_path):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Ensure mono audio
    if len(y.shape) > 1:
        y = librosa.to_mono(y)

    # Pitch detection using librosa.pyin
    pitch, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch = np.nanmean(pitch)  # Average pitch, ignoring NaNs
    pitch = round(float(pitch), 2) if not np.isnan(pitch) else None

    # Amplitude (RMS)
    amplitude = round(float(np.sqrt(np.mean(y ** 2))), 4)

    # Frequency (Dominant frequency using FFT)
    fft_spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / sr)
    dominant_freq = freqs[np.argmax(fft_spectrum)]
    dominant_freq = round(float(dominant_freq), 2)

    # Tempo detection using librosa.beat.tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = round(float(tempo), 2)

    return {
        "pitch": pitch,
        "amplitude": amplitude,
        "dominant_frequency": dominant_freq,
        "tempo": tempo
    }


@app.route('/analyze', methods=['POST'])
def analyze_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.endswith('.wav'):
        return jsonify({"error": "Only .wav files are supported"}), 400

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        file.save(temp_audio.name)
        temp_audio_path = temp_audio.name

    try:
        # Extract features
        features = extract_audio_features(temp_audio_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_audio_path)  # Clean up temp file

    return jsonify(features)


if __name__ == '__main__':
    app.run(debug=True)