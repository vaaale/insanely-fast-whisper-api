import json
from base64 import b64encode

import requests

from app.schema import TranscribeRequest


def main():
    url = "http://localhost:9000"
    audio_file = "/home/alex/video-ts/meeting.mp3"
    print("Encoding audio.....")
    audio_base64 = b64encode(open(audio_file, 'rb').read())

    request = TranscribeRequest(
        audio=audio_base64,
        task="transcribe",
        language="en",
        diarize_audio=True
    )
    print("Transcribing audio.....")
    response = requests.post(url, json=request.model_dump())
    response.raise_for_status()
    print("Results:")
    text = response.json()
    print(text)

    with open("output.json", "w") as out:
        json.dump(text, out)


if __name__ == '__main__':
    main()