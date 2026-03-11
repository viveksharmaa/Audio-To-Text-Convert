from flask import Flask, render_template, request
import whisper
import os
import ffmpeg
from textblob import TextBlob

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Whisper Model
model = whisper.load_model("base")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "audio" not in request.files:
        return "No audio file"

    file = request.files["audio"]

    input_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], "converted.wav")

    file.save(input_path)

    # Convert audio to wav using FFmpeg
    ffmpeg.input(input_path).output(
        output_path,
        ac=1,
        ar="16000"
    ).run(overwrite_output=True)

    # Speech to Text
    result = model.transcribe(output_path)
    text = result["text"]

    # Sentiment Analysis
    score = TextBlob(text).sentiment.polarity

    if score > 0:
        sentiment = "Positive 😊"
    elif score < 0:
        sentiment = "Negative 😡"
    else:
        sentiment = "Neutral 😐"

    return render_template(
        "index.html",
        transcription=text,
        sentiment=sentiment
    )


if __name__ == "__main__":
    app.run(debug=True)