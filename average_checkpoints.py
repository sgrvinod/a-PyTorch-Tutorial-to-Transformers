import torch
import os
from collections import OrderedDict

source_folder = "./"  # folder containing checkpoints that need to be averaged
starts_with = "step"  # checkpoints' names begin with this string
ends_with = ".pth.tar"  # checkpoints' names end with this string

# Get list of checkpoint names
checkpoint_names = [f for f in os.listdir(source_folder) if f.startswith(starts_with) and f.endswith(ends_with)]
assert len(checkpoint_names) > 0, "Did not find any checkpoints!"

# Average parameters from checkpoints
averaged_params = OrderedDict()
for c in checkpoint_names:
    checkpoint = torch.load(c)['model']
    checkpoint_params = checkpoint.state_dict()
    checkpoint_param_names = checkpoint_params.keys()
    for param_name in checkpoint_param_names:
        if param_name not in averaged_params:
            averaged_params[param_name] = checkpoint_params[param_name].clone() * 1 / len(checkpoint_names)
        else:
            averaged_params[param_name] += checkpoint_params[param_name] * 1 / len(checkpoint_names)

# Use one of the checkpoints as a surrogate to load the averaged parameters into
averaged_checkpoint = torch.load(checkpoint_names[0])['model']
for param_name in averaged_checkpoint.state_dict().keys():
    assert param_name in averaged_params
averaged_checkpoint.load_state_dict(averaged_params)

# Save averaged checkpoint
torch.save({'model': averaged_checkpoint}, "averaged_transformer_checkpoint.pth.tar")
