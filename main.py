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
from pyannote.audio.pipelines.utils import (
    PipelineModel,
    SpeakerDiarizationMixin,
    get_devices,
    get_model,
)
from ecapa import ECAPAModel

model = get_model({"checkpoint": "ecapa/exps/pretrain.model", "map_location": torch.device("cuda")})
# model = get_model(ECAPAModel.ECAPAModel)

print(model.specifications)
exit(0)
pipeline = SpeakerDiarization(embedding=model)
output = pipeline("audio/ampme.wav")

print(output)

exit(0)

for turn, _, speaker in output.itertracks(yield_label=True):
    # speaker speaks between turn.start and turn.end
    ...


