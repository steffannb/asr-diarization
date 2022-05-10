from pyannote.audio import Pipeline

#


"""
SEGMENTATION
"""

"""
pipeline = Pipeline.from_pretrained("pyannote/speaker-segmentation")
output = pipeline("audio/ampme.wav")

print(output)
"""


"""
EMBEDDINGS
"""

"""
DIARIZATION
"""

from pytorch_lightning.core.mixins import DeviceDtypeModuleMixin
from pyannote.audio.pipelines import SpeakerDiarization
import torch

a = SpeakerDiarization()
a.from_pretrained("pyannote/speaker-diarization")
a.segmentation.device = torch.device("cuda")
output = a("audio/ampme.wav")

# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
# output = pipeline("audio/ampme.wav")

print(output)

exit(0)

for turn, _, speaker in output.itertracks(yield_label=True):
    # speaker speaks between turn.start and turn.end
    ...


