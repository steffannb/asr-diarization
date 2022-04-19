def main():

    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection")
    output = pipeline("audio-reduced/afjiv.wav")

    for speech in output.get_timeline().support():
        print(f'Start: {0}, End: {1}', speech.start, speech.end)

def seg():
    from pyannote.audio import Pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    output = pipeline("audio-reduced/afjiv.wav")

    for turn, _, speaker in output.itertracks(yield_label=True):
        print(f'Start: {turn.start}, DURATION: {turn.end- turn.start}. SPEAKER {speaker}')


if __name__ == '__main__':
    seg()