from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines import Resegmentation
import noisereduce as nr
import soundfile as sf

# data, rate = sf.read('audio/afjiv.wav')
# reduced_noise = nr.reduce_noise(y=data.T, sr=rate)
# sf.write('audio-reduced/afjiv.wav', reduced_noise, rate)

pipeline_vad = VoiceActivityDetection(segmentation="pyannote/segmentation")


HYPER_PARAMETERS_VAD = {
  # onset/offset activation thresholds
  "onset": 0.767, "offset": 0.713,
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.182,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.501
}
pipeline_vad.instantiate(HYPER_PARAMETERS_VAD)

vad_result = pipeline_vad('audio-reduced/afjiv.wav')
# segmentation = pipeline("audio/afjiv.wav")

#print(vad_result)

pipeline_res = Resegmentation(segmentation="pyannote/segmentation", diarization="baseline")

HYPER_PARAMETERS_RES = {
  # onset/offset activation thresholds
  "onset": 0.537, "offset": 0.724,
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.410,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.563
}
pipeline_res.instantiate(HYPER_PARAMETERS_RES)
resegmented_baseline = pipeline_res({"audio": "audio-reduced/afjiv.wav", "baseline": vad_result})

print(resegmented_baseline)


