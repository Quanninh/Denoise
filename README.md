# Denoise

The ERB stage is implemented in [`erb_stage.py`](erb_stage.py). It contains
only the ERB encoder, recurrent embedding, and ERB mask decoder; the complex
spectral and multi-frame filtering branches are deliberately excluded.

```python
import torch
from erb_stage import ERBStage

model = ERBStage(erb_bins=32)
erb_features = torch.randn(1, 1, 100, 32)
erb_mask = model(erb_features)  # [1, 1, 100, 32], values in [0, 1]
```
