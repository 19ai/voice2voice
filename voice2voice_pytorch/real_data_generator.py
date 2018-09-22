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


import torch
import audio
import os
import h5py
def run():
    f=h5py.File("voice2voice_pytorch/real_data/voice2voice_real_data.h5","w")

    rootdir = 'voice2voice_real_data'
    _list = os.listdir(rootdir) #列出文件夹下所有的目录与文件
    j=0
    for i in range(0,len(_list)):
        path = os.path.join(rootdir,_list[i])
        if os.path.isfile(path):
            try:
                waveform = audio.load_wav(path)
                spec = audio.spectrogram(waveform)
                spec = spec.T
                spec = spec.reshape((1,spec.shape[0],spec.shape[1]))
                print(j, spec.shape)
                f.create_dataset(str(j), data=spec)
                j+=1
            except Exception as e:
                continue
    f.create_dataset('len', data=j)
    f.close()