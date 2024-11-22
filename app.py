from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import whisper
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['TRANSCRIPT_FOLDER'] = './transcripts'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TRANSCRIPT_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    audio = request.files['audio']
    model_name = request.form.get('model', 'turbo')  # Default to "turbo" model
    
    
    # Define the path for the uploaded audio and the transcript
    audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
    transcript_file_path = os.path.join(app.config['TRANSCRIPT_FOLDER'], f"{audio.filename}.json")
    
    # Check if the audio and transcript already exist
    if os.path.exists(audio_file_path) and os.path.exists(transcript_file_path):
        # If they exist, return the existing data
        with open(transcript_file_path, 'r') as transcript_file:
            transcript_data = json.load(transcript_file)
        return jsonify({
            'segments': transcript_data,
            'audio_path': audio_file_path,
        })

    # Save the audio file
    audio.save(audio_file_path)

    # Load the Whisper model
    model = whisper.load_model(model=model_name, download_root="models")
    result = model.transcribe(audio_file_path)

    # Save the transcription
    with open(transcript_file_path, 'w') as transcript_file:
        json.dump(result['segments'], transcript_file)

    # Return the result with audio path
    return jsonify({
        'segments': result['segments'],
        'audio_path': audio_file_path,
    })

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/src/<path:filename>')
def serve_static(filename):
    return send_from_directory('src', filename)

if __name__ == '__main__':
    app.run(debug=True)
