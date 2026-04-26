import os
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from PIL import Image
import torchvision.transforms as T
from transformers.models.swinv2.modeling_swinv2 import Swinv2Model, Swinv2PreTrainedModel, Swinv2Stage
from transformers import Swinv2Config

class Swinv2PatchExpanding(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        """
        Args:
            input_resolution (tuple[int]): Resolution of input feature.
            dim (int): Number of input channels.
            dim_scale (int): Scale for expanding dimension.
            norm_layer (nn.Module): Normalization layer.
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = self.expand(x)
        
        x = x.view(B, H, W, 2 * C)
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)
        x = x.view(B, H * 2 * W * 2, C // 2)
        x = self.norm(x)

        return x

class Swinv2UNetDecoderLayer(nn.Module):
    def __init__(self, config, input_resolution, dim, depth, num_heads, drop_path=0.0):
        """
        Args:
            config: Swinv2Config
            input_resolution (tuple[int]): Resolution of input feature.
            dim (int): Number of input channels.
            depth (int): Depth of Swin stage.
            num_heads (int): Number of attention heads.
            drop_path (float): Stochastic depth rate.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        
        # Upsample blocks (PatchExpanding)
        self.upsample = Swinv2PatchExpanding(
            input_resolution=input_resolution,
            dim=dim,
            dim_scale=2,
            norm_layer=nn.LayerNorm
        )
        
        # After upsample, resolution doubles and channels halve: dim // 2
        # We concatenate with skip connection which has dim // 2 channels
        # Total channels after concat: dim
        self.linear = nn.Linear(dim, dim // 2, bias=False)
        
        # Output resolution after upsample
        out_resolution = (input_resolution[0] * 2, input_resolution[1] * 2)
        out_dim = dim // 2
        
        # Create a modified config for the stage
        stage_config = Swinv2Config(**config.to_dict())
        stage_config.embed_dim = out_dim
        stage_config.depths = [depth]
        stage_config.num_heads = [num_heads]
        
        # Use Swinv2Stage to process the features
        self.stage = Swinv2Stage(
            config=stage_config,
            dim=out_dim,
            input_resolution=out_resolution,
            depth=depth,
            num_heads=num_heads,
            drop_path=[drop_path for _ in range(depth)],
            downsample=None
        )

    def forward(self, x, skip_connection):
        """
        Args:
            x: features from previous decoder stage (B, L, C)
            skip_connection: features from corresponding encoder stage (B, 4L, C/2)
        """
        x = self.upsample(x)
        
        # Concatenate upsampled features with skip connection
        x = torch.cat([x, skip_connection], dim=-1)
        x = self.linear(x)
        
        # Pass through Swinv2Stage
        x = self.stage(x)
        return x

class Swinv2UNetForMaskedImageModeling(Swinv2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        self.swinv2 = Swinv2Model(config, add_pooling_layer=False, use_mask_token=True)
        
        # U-Net Decoder
        # The encoder stages usually have depths like [2, 2, 6, 2]
        # And heads like [3, 6, 12, 24]
        # Let's create a symmetric decoder
        encoder_depths = config.depths
        encoder_heads = config.num_heads
        
        num_layers = len(encoder_depths)
        self.decoder_layers = nn.ModuleList()
        
        # Patch resolutions for each stage
        # E.g., for 224x224 patch 4, window 7: 56 -> 28 -> 14 -> 7
        embed_dim = config.embed_dim
        resolutions = []
        dims = []
        h = config.image_size // config.patch_size
        w = config.image_size // config.patch_size
        
        for i in range(num_layers):
            resolutions.append((h, w))
            dims.append(embed_dim)
            h //= 2
            w //= 2
            embed_dim *= 2
            
        # resolutions: [(56, 56), (28, 28), (14, 14), (7, 7)]
        # dims: [96, 192, 384, 768]
        
        # Build decoder from bottleneck upwards
        for i in range(num_layers - 1, 0, -1):
            dim = dims[i]
            input_res = resolutions[i]
            # Use symmetric depth and heads from the layer we are expanding into
            depth = encoder_depths[i-1]
            num_heads = encoder_heads[i-1]
            
            decoder_layer = Swinv2UNetDecoderLayer(
                config=config,
                input_resolution=input_res,
                dim=dim,
                depth=depth,
                num_heads=num_heads
            )
            self.decoder_layers.append(decoder_layer)
            
        # Final projection to mask image modeling output
        self.decoder_head = nn.Linear(config.embed_dim, config.num_channels * config.patch_size**2)

        # Initialize weights
        self.post_init()

    def forward(self, pixel_values, bool_masked_pos=None):
        """
        Forward pass for Masked Image Modeling.
        bool_masked_pos: (B, H*W) boolean mask.
        """
        # Run encoder
        outputs = self.swinv2(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            output_hidden_states=True,
            return_dict=True
        )
        
        # hidden_states:
        # idx 0: patch embedding output (B, 56x56, 96)
        # idx 1: stage 0 output (B, 56x56, 96)
        # idx 2: stage 1 output (B, 28x28, 192)
        # idx 3: stage 2 output (B, 14x14, 384)
        # idx 4: stage 3 output (B, 7x7, 768)
        
        hidden_states = outputs.hidden_states
        
        # Skip connections
        # Since standard Swinv2 outputs states after each stage:
        # Skip connection 0 (for 56x56): hidden_states[1]
        # Skip connection 1 (for 28x28): hidden_states[2]
        # Skip connection 2 (for 14x14): hidden_states[3]
        
        num_layers = len(self.config.depths)
        x = hidden_states[-1] # bottleneck, e.g., (B, 7x7, 768)
        
        # Reverse iterate through skip connections
        skip_connections = hidden_states[1:-1] # e.g., [state0, state1, state2]
        
        for i, decoder_layer in enumerate(self.decoder_layers):
            skip_idx = len(skip_connections) - 1 - i
            skip_conn = skip_connections[skip_idx]
            # When x has shape (B, 7x7, 768)
            # decoder layer 0 upsamples x to (B, 14x14, 384)
            # concatenates with skip_conn which is state2 (B, 14x14, 384)
            x = decoder_layer(x, skip_conn)
            
        # Final projection
        # x is now (B, 56x56, 96)
        sequence_output = self.decoder_head(x)
        
        return sequence_output

def generate_random_mask(mask_size=(64, 64), max_shapes=5):
    """
    Generates a boolean mask of shape (mask_size[0], mask_size[1]) with random shapes.
    """
    mask = np.zeros(mask_size, dtype=np.uint8)
    num_shapes = random.randint(1, max_shapes)
    
    for _ in range(num_shapes):
        shape_type = random.choice(['rectangle', 'circle', 'ellipse'])
        
        if shape_type == 'rectangle':
            x1 = random.randint(0, mask_size[1] // 2)
            y1 = random.randint(0, mask_size[0] // 2)
            x2 = random.randint(mask_size[1] // 2, mask_size[1])
            y2 = random.randint(mask_size[0] // 2, mask_size[0])
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
            
        elif shape_type == 'circle':
            center_x = random.randint(0, mask_size[1])
            center_y = random.randint(0, mask_size[0])
            radius = random.randint(mask_size[0] // 8, mask_size[0] // 2)
            cv2.circle(mask, (center_x, center_y), radius, 1, -1)
            
        elif shape_type == 'ellipse':
            center_x = random.randint(0, mask_size[1])
            center_y = random.randint(0, mask_size[0])
            axes = (random.randint(mask_size[0] // 8, mask_size[0] // 2),
                    random.randint(mask_size[1] // 8, mask_size[1] // 2))
            angle = random.randint(0, 180)
            cv2.ellipse(mask, (center_x, center_y), axes, angle, 0, 360, 1, -1)
            
    return torch.from_numpy(mask).bool().view(-1)

class Pix2PixMaskedDataset(Dataset):
    def __init__(self, root_dir, image_size=256, patch_size=4):
        self.image_paths = glob.glob(os.path.join(root_dir, "*.*"))
        self.image_size = image_size
        self.patch_size = patch_size
        self.mask_size = (image_size // patch_size, image_size // patch_size)
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Pix2Pix images are usually concatenated A|B. Split in half and take B.
        w, h = image.size
        if w > h:
            image = image.crop((w // 2, 0, w, h))
            
        pixel_values = self.transform(image)
        bool_masked_pos = generate_random_mask(self.mask_size)
        
        return pixel_values, bool_masked_pos

def train_model(model_config, dataset_dir, epochs=10, batch_size=4, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Swinv2UNetForMaskedImageModeling(model_config).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    
    dataset = Pix2PixMaskedDataset(dataset_dir, image_size=model_config.image_size, patch_size=model_config.patch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    print(f"Starting training on {device}...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (pixel_values, bool_masked_pos) in enumerate(dataloader):
            pixel_values = pixel_values.to(device)
            bool_masked_pos = bool_masked_pos.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
            
            # Extract ground truth patches
            patch_size = model_config.patch_size
            unfolded = F.unfold(pixel_values, kernel_size=patch_size, stride=patch_size)
            target_patches = unfolded.transpose(1, 2) # (B, L, C * patch_size**2)
            
            # Compute loss only on masked positions
            loss = F.l1_loss(outputs[bool_masked_pos], target_patches[bool_masked_pos])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
                
        if len(dataloader) > 0:
            print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {epoch_loss / len(dataloader):.4f}")
        else:
            print(f"Epoch [{epoch+1}/{epochs}] No batches found. Check dataset.")
        
    return model

if __name__ == "__main__":
    # Small test
    config = Swinv2Config(
        image_size=256,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
    )
    
    model = Swinv2UNetForMaskedImageModeling(config)
    print("Model created successfully.")
    
    # Dummy input (batch_size, num_channels, height, width)
    pixel_values = torch.randn(2, 3, 256, 256)
    
    # Dummy mask
    # Sequence length for patch size 4 and image 256 is (256/4) * (256/4) = 64 * 64 = 4096
    bool_masked_pos = torch.zeros(2, 4096).bool()
    bool_masked_pos[:, :10] = True # mask first 10 patches
    
    output = model(pixel_values, bool_masked_pos=bool_masked_pos)
    print("Output shape:", output.shape)
    
    # Check backprop
    loss = output.sum()
    loss.backward()
    print("Backward pass successful.")
    
    # Test random mask generation
    dummy_mask = generate_random_mask((64, 64))
    print("Dummy mask generated with shape:", dummy_mask.shape, "and sum:", dummy_mask.sum().item())
