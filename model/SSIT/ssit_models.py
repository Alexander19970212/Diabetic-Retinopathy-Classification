import torch
import torch.nn as nn
from functools import partial
from vits import archs, resize_pos_embed

def load_checkpoint(model, checkpoint_path, checkpoint_key, linear_key):
    checkpoint = torch.load(checkpoint_path)

    # replace all torch-10 GELU's by torch-12 GELU
    
    # def torchmodify(name) :
    #     a=name.split('.')
    #     for i,s in enumerate(a) :
    #         if s.isnumeric() :
    #             a[i]="_modules['"+s+"']"
    #     return '.'.join(a)
    # import torch.nn as nn
    # for name, module in checkpoint.named_modules() :
    #     if isinstance(module,nn.GELU) :
    #         exec('checkpoint.'+torchmodify(name)+'=nn.GELU()')

    state_dict = checkpoint.state_dict()
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith(checkpoint_key) and not k.startswith('%s.%s' % (checkpoint_key, linear_key)):
            # remove prefix
            state_dict[k[len("%s." % checkpoint_key):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    # position embedding
    pos_embed_w = state_dict['pos_embed']
    pos_embed_w = resize_pos_embed(pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    state_dict['pos_embed'] = pos_embed_w

    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"%s.weight" % linear_key, "%s.bias" % linear_key}
    print('Load weights form {}'.format(checkpoint_path))

class SSitEncoder(nn.Module):
    def __init__(self, arch, checkpoint=None, input_size=224):
        super(SSitEncoder, self).__init__()
        self.model = archs[arch](
            num_classes=5,
            pretrained=True,
            img_size=input_size,
            feat_concat=True
        )

        linear_key = 'head'
        checkpoint_key = 'base_encoder'
        
        if checkpoint:
            load_checkpoint(self.model, checkpoint, checkpoint_key, linear_key)
        else:
            print('No checkpoint provided. Training from scratch.')

    def forward(self, pixel_values):
        # define function in transformers library maner
        one_image = False
        if pixel_values.dim() == 3:
            pixel_values = pixel_values[None, :, :, :]
            one_image = True
        _, f = self.model(pixel_values)

        if one_image:
            return f[0]
        else:
            return f

