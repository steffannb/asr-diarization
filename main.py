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
from pyannote.audio.pipeline import SpeakerDiarization
import torch
import pyannote.audio
import torch
model = torch.hub.load('pyannote/pyannote-audio', 'emb')
from pyannote.audio.features.pretrained import Pretrained
print(model.get_dimension())
exit()



#
# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
# output = pipeline("audio/abjxc.wav")
# print(output)
# exit(0)
# pipeline = SpeakerDiarization(embedding=model)
# output = pipeline("audio/ampme.wav")

# print(output)

# for turn, _, speaker in output.itertracks(yield_label=True):
#     speaker speaks between turn.start and turn.end
#     ...


