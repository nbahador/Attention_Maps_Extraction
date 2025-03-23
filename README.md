## Installation

To install the [`Attention-Maps-Extraction`](https://pypi.org/project/Attention-Maps-Extraction/) package, run the following command:

```bash
pip install Attention-Maps-Extraction
```

---

### Usage Example

Here is an example of how to use this package:

```python
from Attention_Maps_Extraction import ViTForRegression, SpectrogramDataset, visualize_attention_maps, load_data
import torch
import os
from torchvision import transforms  # Add this import

# User inputs
data_dir = "path/to/spectrogram_images"
csv_path = "path/to/labels.csv"
output_dir = "path/to/output"
model_path = "path/to/best_vit_regression.pth"

# Load data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataloader = load_data(data_dir, csv_path, transform=transform)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForRegression().to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Visualize attention maps
patch_contributions_dir = os.path.join(output_dir, "attention_maps")
os.makedirs(patch_contributions_dir, exist_ok=True)
visualize_attention_maps(model, dataloader, save_dir=patch_contributions_dir, device=device)
```
