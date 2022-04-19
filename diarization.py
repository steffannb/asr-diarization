from pyannote.audio.pipelines import SpeakerDiarization

def main():
    pipeline = SpeakerDiarization()
    HYPER_PARAMETERS_VAD = {
        # onset/offset activation thresholds
        "onset": 0.767, "offset": 0.713,
        # remove speech regions shorter than that many seconds.
        "min_duration_on": 0.182,
        # fill non-speech regions shorter than that many seconds.
        "min_duration_off": 0.501
    }
    pipeline.instantiate(HYPER_PARAMETERS_VAD)
    diarization = pipeline("audio-reduced/afjiv.wav")
    print(diarization)

if __name__ == '__main__':
    main()