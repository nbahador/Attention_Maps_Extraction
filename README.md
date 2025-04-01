### Installation

To install the [`Attention-Maps-Extraction`](https://pypi.org/project/Attention-Maps-Extraction/) package, run the following command:

```bash
pip install Attention-Maps-Extraction
```
---

### Citation
Bahador N. Mechanistic interpretability of fine-tuned vision transformers on distorted images: Decoding attention head behavior for transparent and trustworthy AI. arXiv [csLG]. Published online 24 March 2025. http://arxiv.org/abs/2503.18762. 

**[[PDF]](https://arxiv.org/pdf/2503.18762)**

---

### Sample Generated Map

<img src="https://github.com/nbahador/Attention_Maps_Extraction/raw/main/Example/Sample%20Generated%20Map.jpg" alt="Sample Generated Map" width="200" height="200" />

---

### Usage Example

Here is an example of how to use this package:

```python
from Attention_Maps_Extraction import ViTForRegression, SpectrogramDataset, visualize_attention_maps, load_data
import torch
import os
from torchvision import transforms

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
