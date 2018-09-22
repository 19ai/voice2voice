import torch
from torch import nn

from voice2voice_pytorch import Voice2VoiceModel

def voice2voice():
    from voice2voice_pytorch.voice2voice import Encoder, Decoder
    encoder = Encoder()
    decoder = Decoder()
    model = Voice2VoiceModel(encoder, decoder)
    return model
