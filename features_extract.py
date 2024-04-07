import torch
import numpy as np
from segment_anything import sam_model_registry

def deep_features(image_tensor, extractor, layer = 11, facet = 'key', bin: bool = False, include_cls: bool = False, device = 'cuda'):
    """
    Extract descriptors from pretrained DINO model; Create an adj matrix from descriptors
    @param image_tensor: Tensor of size (batch, height, width)
    @param extractor: Initialized model to extract descriptors from
    @param layer: Layer to extract the descriptors from
    @param facet: Facet to extract the descriptors from (key, value, query)
    @param bin: apply log binning to the descriptor. default is False.
    @param include_cls: To include CLS token in extracted descriptor
    @param device: Training device
    @return: W: adjacency matrix, F: feature matrix, D: row wise diagonal of W
    """
    # images to deep_features.
    # input is a tensor of size batch X height X width,
    deep_features = extractor.extract_descriptors(image_tensor.to(device), layer, facet, bin, include_cls).cpu().numpy()
    deep_features = np.squeeze(deep_features, axis=1)
    deep_features = deep_features.reshape((deep_features.shape[0] * deep_features.shape[1], deep_features.shape[2]))
    return deep_features

## MedSAM features

class FeatEx:
  def __init__(self, enc):
    self.enc = enc 
    self.feat = []
  
  def func(self, module, inp, output):
    inp = inp[0]
    B, H, W, _ = inp.shape
    qkv = (
        module.qkv(inp).reshape(B, H * W, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
      )
    self.feat.append(qkv[1])

  def extract(self, image):
    self.enc.blocks[11].attn.register_forward_hook(self.func)
    self.enc(image)
    return self.feat[0]
  

def medsam_features(image_tensor, medsam_vit_path = './medsam_vit_b.pth', device = 'cuda'):
    """
    Extract descriptors from pretrained DINO model; Create an adj matrix from descriptors
    @param image_tensor: Tensor of size (batch, height, width)
    @param medsam_vit_path: Path to the MedSAM checkpoint ('medsam_vit_b.pth')
    @return: W: adjacency matrix, F: feature matrix, D: row wise diagonal of W
    """
    medsam_model = sam_model_registry['vit_b'](checkpoint=medsam_vit_path).to(torch.device('cpu')) 
    medsam_model.eval()
    image_encoder = medsam_model.image_encoder
    image_encoder.eval()

    ex = FeatEx(image_encoder.to(device))
    x = ex.extract(image_tensor.to(device))

    x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)
    x = x.detach().cpu().numpy()
    deep_features = np.squeeze(x, axis=1)
    deep_features = deep_features.reshape((deep_features.shape[0] * deep_features.shape[1], deep_features.shape[2]))
    
    return deep_features
