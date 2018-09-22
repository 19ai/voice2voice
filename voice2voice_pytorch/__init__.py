# coding: utf-8

from .version import __version__

import torch
from torch import nn

class Voice2VoiceModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(Voice2VoiceModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, content_voice, style_speaker=None, style=None):
        if style_speaker is not None:
            style = self.encoder(style_speaker)
        if style is None:
            return None
        y = self.decoder(content_voice, style)
        return y
