import os
import math
import random
import numpy as np
import h5py
import torch
import audio

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
_frontend = None  # to be set later

def get_text(vocab=['a','b','c']):
    text = ''
    word_num = np.random.randint(5,20)
    for i in range(word_num):
        idx = np.random.randint(0,len(vocab)-1)
        text+=vocab[idx]
    return text

def _resample(x, time_size):
    x = x.reshape((x.size(0),1,x.size(1),x.size(2)))
    x = torch.nn.functional.interpolate(x, size=(time_size[1],x.size(3)), mode='bilinear', align_corners=True)
    x = x.reshape((x.size(0),time_size[1],x.size(3)))
    return x

def _reverse(x):
    idx = [i for i in range(x.size(0)-1, -1, -1)]
    idx = torch.LongTensor(idx).to(device)
    inverted_tensor = x.index_select(0, idx)
    return inverted_tensor.contiguous()

def _pitch_shifter(spec, shift):
        # Suggest shift = [-25, 25]
        N = spec.shape[2]
        shifted_freq = torch.zeros_like(spec)

        S = np.round(shift if shift > 0 else N + shift, 0)
        s = N - S
        
        shifted_freq[:,:,:S] = spec[:,:,s:]
        shifted_freq[:,:,S:] = spec[:,:,:s]

        return shifted_freq.contiguous()

def deepvoice3_generator(model, vocab=['a','b','c'], bsz=32, p=0, speaker_num=None, fast=False):
    """Convert text to speech spectrogram batch given a deepvoice3 model for voice2voice training.

    Args:
        vocab (Array) : Input vocab array to be synthesized
        p (float) : Replace word to pronounciation if p > 0. Default is 0.
    """
    model = model.to(device)
    model.eval()
    if fast:
        model.make_generation_fast_()
    def get_sequence_and_position():
        text = get_text(vocab)
        sequence = np.array(_frontend.text_to_sequence(text, p=p))
        sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
        text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)
        return (sequence, text_positions)
    t1 = get_sequence_and_position()
    t2 = get_sequence_and_position()
    speaker_ids = None if speaker_num is None else torch.LongTensor([np.random.randint(0,speaker_num-1,size=bsz)]).to(device)
    
    # Greedy decoding
    with torch.no_grad():
        spec1 = None
        spec2 = None
        _spec = [(model(
            t1[0], text_positions=t1[1], speaker_ids=speaker_ids[:,i])[1],
            model(
            t2[0], text_positions=t2[1], speaker_ids=speaker_ids[:,i])[1]) for i in range(bsz//2 if bsz//2 else 1)]
        _s = None
        for i in _spec:
            if _s is None:
                _s= True
                spec1 = i[0]
                size = spec1.size()
            else:
                i = _resample(i[0],size)
                spec1 = torch.cat([spec1, i])
        _s = None
        for i in _spec:
            if _s is None:
                _s= True
                spec2 = i[1]
                size = spec2.size()
            else:
                i = _resample(i[1],size)
                spec2 = torch.cat([spec2, i])
        shift = np.random.randint(-25,25)
        spec2_shift = _pitch_shifter(spec2, shift)
        spec2 = _reverse(spec2)
        y = _reverse(spec1)
        y2 = _pitch_shifter(spec1, shift)
        x = torch.cat([spec1,spec1])
        style = torch.cat([spec2, spec2_shift])
        y = torch.cat([y,y2])
        
    return x.to(device), style.to(device), y.to(device)

class H5List:
    def __init__(self, path, hold_time=50):
        self.f = h5py.File(path,"r")
        self.len = self.f['len'].value
        self.hold_time = hold_time
        self.left_hold_time = 0
    def sample(self, num):
        if self.left_hold_time:
            self.left_hold_time -= 1
        else:
            self.left_hold_time = self.hold_time
            length = len(self)
            idxs = np.random.randint(0, length-1, num)
            self.hold_samples = [torch.from_numpy(self.f[str(i)].value).float() for i in idxs]
        return self.hold_samples
    def __len__(self):
        return self.len

script_dir = os.path.split(os.path.realpath(__file__))[0]
real_data = H5List(os.path.join(script_dir, 'real_data', 'voice2voice_real_data.h5'))
def real_data_generator(bsz=32):
    import hparams
    import json

    preset = "./presets/deepvoice3_vctk.json"

    # Newly added params. Need to inject dummy values
    for dummy, v in [("fmin", 0), ("fmax", 0), ("rescaling", False),
                    ("rescaling_max", 0.999), 
                    ("allow_clipping_in_normalization", False)]:
        if hparams.hparams.get(dummy) is None:
            hparams.hparams.add_hparam(dummy, v)
        
    # Load parameters from preset
    with open(preset) as f:
        hparams.hparams.parse_json(f.read())

    # Tell we are using multi-speaker DeepVoice3
    hparams.hparams.builder = "deepvoice3_multispeaker"


    _specs = real_data.sample(bsz)
    _s0 = _specs[0].to(device)
    if _s0.size(1) > 2000:
        frames = 2000
        start = int(np.floor(np.random.random()*(_s0.size(1)-frames)))
        _s0 = _s0[:,start:start+frames,:].contiguous()
    specs = [_s0]
    frames = specs[0].size(1)
    for _i in range(bsz-1):
        i = _i+1
        spec = _specs[i].to(device)
        spec = spec.repeat(1,math.ceil(frames/spec.size(1)),1)
        start = int(np.floor(np.random.random()*(spec.size(1)-frames)))
        spec = spec[:,start:start+frames,:].contiguous()
        specs.append(spec)
    _x = torch.cat(specs)
    shift1 = np.random.randint(-25,25)
    shift2 = np.random.randint(-25,25)
    x = _pitch_shifter(_x, shift1)
    style = _pitch_shifter(_x, shift2)
    y = style.clone()

    return x, style, y
