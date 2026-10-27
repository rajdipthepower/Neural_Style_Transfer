from pathlib import Path
import torch,torchvision
import torch.nn as nn
from PIL import Image,ImageOps
import time 
import sys

torch.set_num_threads(2)

Root = Path(__file__).resolve().parent.parent # Path of Root directory 'deployment'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vgg_weights = torch.load(
    Root/"models"/"vgg_normalised.pth",
    map_location=device
)

decoder_weights = torch.load(
    Root/"models"/"decoder.pth",
    map_location=device
)

def load_and_preprocess(img,category,style_size = 512):
  max_size = 640
  short_size = 512
  if category == 'content':
    with Image.open(img) as img:
      img = ImageOps.exif_transpose(img)
      img = img.convert("RGB")
      w,h = img.size
      if min(w,h) == w:
        new_w = short_size
        new_h = h*(short_size/w)
        if new_h > max_size:
          new_w = new_w*(max_size/new_h)
          new_h = max_size
      else:
        new_h = short_size
        new_w = w * (short_size/h)
        if new_w > max_size:
          new_h = new_h * (max_size/new_w)
          new_w = max_size
      img = img.resize((max(8,int(new_w)//8*8), max(8,int(new_h)//8*8)))
      data = torchvision.transforms.ToTensor()(img)
      #data = data[[2,1,0],:,:] # RGB to BGR conversion done in standard preprocessing in official paper
      #data = data - torch.tensor([103.939, 116.779, 123.68]).view(3,1,1)
      data = data.unsqueeze(0).to(device) # Adding batch dimension at axis 0
      return data
  else:
    with Image.open(img) as img:
      img = ImageOps.exif_transpose(img)
      img = img.convert("RGB")
      w,h = img.size
      if min(w,h) == w:
        new_w = style_size
        new_h = h*(style_size/w)
      else:
        new_h = style_size
        new_w = w*(style_size/h)
      img = img.resize((int(new_w),int(new_h)))
      style_data = torchvision.transforms.ToTensor()(img)
# Resize only the smaller edge not both edge (preserving aspect ratio)
    #style_data = style_data[[2,1,0],:,:]
    #style_data = style_data - torch.tensor([103.939, 116.779, 123.68]).view(3,1,1) # standard preprocessing using means from Imagenet Dataset
      style_data = style_data.unsqueeze(0).to(device)
      return style_data

# For post processing
def tensor_to_pil(tensor):
    # Clamp to [0, 1], move to cpu, drop batch dim, permute from (C, H, W) to (H, W, C), and scale to 255
    image = tensor.squeeze(0).clamp(0.0, 1.0).cpu()
    image = torchvision.transforms.ToPILImage()(image)
    return image

# Setting Up and Loading the Encoder 

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.Conv2d(3,3,(1,1),stride = 1,padding = 0), # this subtracts [103.939, 116.779, 123.68] from each channel (BGR)
        nn.ReflectionPad2d(1),
        nn.Conv2d(3,64,(3,3),stride=1,padding=0),
        nn.ReLU(), # 1_1
        nn.ReflectionPad2d(1),
        nn.Conv2d(64,64,(3,3),stride=1,padding=0),
        nn.ReLU(), # 1_2
        nn.MaxPool2d(2,stride=2,ceil_mode = True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64,128,kernel_size = (3,3),stride = 1,padding = 0),
        nn.ReLU(), # 2_1
        nn.ReflectionPad2d(1),
        nn.Conv2d(128,128,kernel_size = (3,3),stride = 1,padding = 0),
        nn.ReLU(), # 2_2
        nn.MaxPool2d(2,stride = 2,ceil_mode = True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128,256,(3,3),stride = 1,padding = 0),
        nn.ReLU(), # 3_1
        nn.ReflectionPad2d(1),
        nn.Conv2d(256,256,(3,3),stride = 1,padding = 0),
        nn.ReLU(), # 3_2
        nn.ReflectionPad2d(1),
        nn.Conv2d(256,256,(3,3),stride = 1,padding = 0),
        nn.ReLU(), # 3_3
        nn.ReflectionPad2d(1),
        nn.Conv2d(256,256,(3,3),stride = 1,padding = 0),
        nn.ReLU(), # 3_4
        nn.MaxPool2d(2,stride = 2,ceil_mode = True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256,512,(3,3),stride = 1,padding = 0),
        nn.ReLU() # 4_1
    )
  def forward(self,x):
    return self.model(x)

encoder = Encoder()

#print("Output Shape ",encoder(torch.rand((3,512,512))).shape)

encoder.load_state_dict({
    "model.0.weight":  vgg_weights["0.weight"],
    "model.0.bias":    vgg_weights["0.bias"],

    "model.2.weight":  vgg_weights["2.weight"],
    "model.2.bias":    vgg_weights["2.bias"],

    "model.5.weight":  vgg_weights["5.weight"],
    "model.5.bias":    vgg_weights["5.bias"],

    "model.9.weight":  vgg_weights["9.weight"],
    "model.9.bias":    vgg_weights["9.bias"],

    "model.12.weight":  vgg_weights["12.weight"],
    "model.12.bias":    vgg_weights["12.bias"],

    "model.16.weight": vgg_weights["16.weight"],
    "model.16.bias":   vgg_weights["16.bias"],

    "model.19.weight": vgg_weights["19.weight"],
    "model.19.bias":   vgg_weights["19.bias"],

    "model.22.weight": vgg_weights["22.weight"],
    "model.22.bias":   vgg_weights["22.bias"],

    "model.25.weight": vgg_weights["25.weight"],
    "model.25.bias":   vgg_weights["25.bias"],

    "model.29.weight": vgg_weights["29.weight"],
    "model.29.bias":   vgg_weights["29.bias"],
})  # Downloaded vgg19 weights are for full model but we use only upto relu4_1 hence need explicit mapping

encoder.to(device).eval()

# Setting Up and Loading the Deocoder

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(512,256,(3,3),stride = 1,padding = 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2,mode="nearest"),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256,256,(3,3),stride = 1,padding = 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256,256,(3,3),stride = 1,padding = 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256,256,(3,3),stride = 1,padding = 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(256,128,(3,3),stride = 1,padding = 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2,mode="nearest"),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128,128,(3,3),stride = 1,padding = 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(128,64,(3,3),stride = 1,padding = 0),
        nn.ReLU(),
        nn.Upsample(scale_factor=2,mode="nearest"),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64,64,(3,3),stride = 1,padding = 0),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        nn.Conv2d(64,3,(3,3),stride = 1,padding = 0),
    )
  def forward(self,x):
    return self.model(x)

decoder = Decoder()
# print(decoder(torch.rand(1,512,64,64)).shape)
decoder.model.load_state_dict(decoder_weights)
decoder.to(device).eval()

#Defining the adain function 

def adain(content,alpha=1.0,style=None,content_mean = None,content_std = None , style_mean = None,style_std = None):
  
  eps = 1e-5
  content1 = content.view(content.size(0),content.size(1),-1) # Shape: (1,512,4096) Flattening height and width

  if style_mean is None and content_mean is None:
    style1 = style.view(style.size(0),style.size(1),-1)

    mu_content = content1.mean(axis = 2).unsqueeze(2) # Shape : (1,512,1)
    mu_style = style1.mean(axis = 2).unsqueeze(2)
    # To subtract a tensor (1,512) from (1,512,1024) , we need to reshape : (1,512) -> (1,512,1)
    std_content = torch.sqrt(((content1 - mu_content)**2).mean(axis = 2) + eps).unsqueeze(2) # Shape : (1,512,1)
    std_style = torch.sqrt(((style1 - mu_style)**2).mean(axis = 2) + eps).unsqueeze(2) # Shape : (1,512,1)

    transformed_content = ((content1 - mu_content)/std_content)*std_style + mu_style # Shape:(1,512,4096)

    #print('Content ',content.shape)
    #print('Mean of content ',mu_content.shape)
    #print('Std of content ',std_content.shape)
    #print('Transformed Content ',transformed_content.shape)
    #print('final Content ',final.shape)
    
  elif content_mean is None:
    content1 = content.view(content.size(0),content.size(1),-1)
    mu_content = content1.mean(axis = 2).unsqueeze(2)
    std_content = torch.sqrt(((content1 - mu_content)**2).mean(axis = 2) + eps).unsqueeze(2)
    transformed_content = ((content1 - mu_content)/std_content)*style_std + style_mean

  else:
    transformed_content = ((content1 - content_mean)/content_std)*style_std + style_mean

  final =  alpha * transformed_content.view(content.size()) + (1-alpha) * content # Shape:(1,512,64,64)  
  return final

def stylize(content_path,style_stats,alpha = 1.0):
  content = load_and_preprocess(content_path,'content')
  style_mean,style_std = style_stats

  with torch.inference_mode():
    content_feat = encoder(content)
    stylized_embed = adain(content_feat,alpha,style_mean=style_mean,style_std=style_std)
    output = decoder(stylized_embed)

  return tensor_to_pil(output)

def get_style_stats(style_path, size=512):
  style = load_and_preprocess(style_path, 'style', size)
  with torch.inference_mode():
    feat = encoder(style).view(1, 512, -1)
    mean = feat.mean(dim=2, keepdim=True)
    std = torch.sqrt(((feat - mean)**2).mean(dim=2, keepdim=True) + 1e-5)
  return mean, std

if __name__ == "__main__":
    # Expects arguments: python inference.py <content_path> <style_path> [alpha]
    if len(sys.argv) < 3:
        print("Usage: python inference.py <content_path> <style_path> [alpha]")
        sys.exit(1)
# argument 0 is the file name
    content_file = sys.argv[1]
    style_file = sys.argv[2]
    alpha_val = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

    style_stats = get_style_stats(style_file)
    print("Stylizing...")
    # Time *only* the stylize execution process as requested
    stylize(content_file, style_stats, alpha_val)  # warm-up, not timed
    start_time = time.time()
    result_image = stylize(content_file, style_stats, alpha=alpha_val)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Stylization completed in {elapsed_time:.4f} seconds.")

    # Save output
    output_path = Root / "output.jpg"
    result_image.save(output_path, quality=90)
    print(f"Saved stylized output to {output_path}")