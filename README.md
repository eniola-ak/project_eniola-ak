# DNNLS Final Assessment  
**Author:** Eniola Akinlua

---

## Introduction and Problem Statement
This project extends a baseline sequence predictor model for visual storytelling by incorporating advanced deep learning techniques. Given 4 frames with text descriptions, the model must predict the 5th frame (both image and text description).

The impact of each architectural change and the combined architectural changes on the reasoning quality and narrative coherence are evaluated.

**Dataset**: https://huggingface.co/datasets/daniel3303/StoryReasoning

### Problem Definition

The baseline model is enhanced with two key architectural improvements:
1. **Cross-Modal Attention Mechanism** - Aligns visual and textual features for better multimodal understanding
2. **Variational Autoencoder (VAE) Decoder** - Enables probabilistic image generation with improved generalization


### Evaluation Metrics
- **Image Quality**: Mean Absolute Error (MAE), Mean Squared Error (MSE)
- **Text Quality**: Perplexity (lower is better)
- **Training**: Loss curves

---

## Methodology
**Baseline Limitations**:
- Small latent space (16-dim) limits representation capacity
- No explicit cross-modal interaction
- Deterministic image generation
- Simple attention mechanism

### Improvements

#### 1. Cross-Modal Attention Mechanism (4 heads)
**Motivation**: The baseline handles the text and visual features separately without establishing the relationships but the CMA mechanism aligns both modes to generate a more coherent output.

**Code Snippet**:

```python
class CrossModalAttention(nn.Module):
    def __init__(self, visual_dim, text_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(text_dim, visual_dim)
        self.key_proj = nn.Linear(visual_dim, visual_dim)
        self.value_proj = nn.Linear(visual_dim, visual_dim)
        self.out_proj = nn.Linear(visual_dim, visual_dim)
    
    def forward(self, text_features, visual_features):
        #returns text and visual features that match
```

#### 2. Variational Autoencoder (VAE) Decoder

**Motivation**: The VA decoder allows the model generalize better  by not learning the images directly. Instead, it learns a distribution and uses the distribution samples to generate images to prevent overfitting.

**Implementation**:
```python
class VariationalDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.fc_mean = nn.Linear(latent_dim, latent_dim)
        self.fc_logvar = nn.Linear(latent_dim, latent_dim)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, z):
        mean = self.fc_mean(z)
        logvar = self.fc_logvar(z)
        z_sample = self.reparameterize(mean, logvar)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return image, image_context, kl_loss
```

#### 3. Enhanced Model Capacity
- Latent dimension: 16 -> 256 (16× increase)
- Embedding dimension: 16 -> 768
- GRU hidden dimension: 16 -> 256
- LSTM layers: 1 -> 2 
- Dropout: 0.1 -> 0.2

#### 4. Training Optimizations

**OneCycleLR Scheduler**:
```python
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.003,
    epochs=50,
    steps_per_epoch=len(train_dataloader),
    pct_start=0.1,           
    anneal_strategy='cos',   # Cosine annealing
    div_factor=3.0,
    final_div_factor=100
)
```

---

## Experiments

### Experiment 1: Baseline

**Configuration**: `configs/config_baseline.yaml`
**Results**: `results/baseline/metrics.csv`

**Observations**:
- Quick convergence due to small model and short training
- Limited capacity for complex patterns


### Experiment 2: Cross-Modal Attention

**Configuration**: `configs/config_crossmodal.yaml`
**Results**: `results/cross_modal_attention/metrics.csv`

**Observations**:
- Higher capacity model
- Longer training time
- Similar image generation capacity
- Best text perplexity (lowest)


### Experiment 3: VAE Decoder

**Configuration**: `configs/config_vae.yaml`

**Results**: `results/vae_decoder/metrics.csv`

**Observations**:
- Similar image generation capacity 
- Second best Perplexity values


### Experiment 4: Cross-Modal Attention + VAE

**Configuration**: `configs/config_both.yaml`

**Results**: `results/cross_modal_plus_vae/metrics.csv`\

**Observations**:
- The combination does not actualy improve the text perplexity
- Similar image generation capacity

---

## Results

### Quantitative Evaluation
- MAE, MSE, textperplxity comparison: `results/Results.txt`
- Loss curves: `results/baseline/baseline_loss_curve.png`, `results/cross_modal_attention/cross_modal_attention_loss_curve.png`, `results/cross_modal_plus_vae/VAE_decoder+CMA_loss_curve.png`, `results/vae_decoder/VAE_decoder_loss_curve.png`

**loss** = (image_weight * loss_image + text_weight * loss_text + context_weight * loss_context + kl_weight * kl_divergence)

### Qualitative Analysis

Example visualizations for each experiment are shown in:
- Baseline experiment:`results/baseline/visualizations/epoch_004.png`
- Cross modal Attention experiment: `results/cross_modal_attention/visualizations/epoch_049.png`
- VAE decoder experiment: `results/vae_decoder/visualizations/epoch_049.png`
- Cross Modal + VAE decoder experiment: `results/cross_modal_plus_vae/visualizations/epoch_049.png`

### Key Findings

#### 1. Text Generation Improved
**42% improvement** in text perplexity (66.5 -> 38.5). OneCycleLR enabled faster convergence.

#### 2. Little to no improvement in Image Quality
MAE remained **~0.24**  and MSE remained **~0.08** across all models.

#### 3. Little returns for huge change in model complexity
100× parameter increase (2M -> 198M), Longer training time: 10 mins -> 2 hours.

#### 4. Weight and number of epoch training Optimization 
**Experinents with different loss weights**:
1. Initial (baseline): image=1.0, text=1.0, 5 epochs : Equal priority : worst text, worst images
2. Attempt 1: image=1.0, text=5.0, 10-50 epochs : No significant improvement
3. Attempt 2: image=10.0, text=1.0, 10-50 epochs : No significant improvement
3. Attempt 3: image=0.2, text=10.0, 80 epochs : Bad text, Bad images (overfitting)
3. Attempt 4 (**Optimal**): image=0.2, text=10.0, 50 epochs → Best text and image performance.

Reducing the image weight acknowledged the architectural limitations of the model and allowed the model focus more resources on the text generation aspects.

---

## How to reproduce
Google Colab (recommended) or local GPU
Disk space: ~15GB (dataset + checkpoints + results)

### Setup
1. Open the experiments.ipynb file 
2. set config path by removing comment

# CONFIG_NAME = 'configs/config_crossmodal.yaml'   # + Cross-modal
# CONFIG_NAME = 'configs/config_vae.yaml'          # + VAE
# CONFIG_NAME = 'configs/config_both.yaml'         # Both 

3. To save time, set the code to load final saved checkpoint and evaluate else train for all epochs by setting RESUME_FROM_CHECKPOINT to False. 
```python
RESUME_FROM_CHECKPOINT = True
CHECKPOINT_TO_LOAD = "cross_modal_plus_vae_epoch_049.pth"

if RESUME_FROM_CHECKPOINT:
    checkpoint_path = CHECKPOINT_DIR / CHECKPOINT_TO_LOAD
```
4. run all cells
5. view results

### Summary of Achievements

This project successfully enhanced the baseline visual storytelling model through cross-Modal Attention, a VAE Decoder, Optimized Training (Longer epochs, More robust model with more parameters, OneCycleLR scheduler, AdamW optimizer, Loss-weight-epoch tuning).

**Challenges**:
- Image generation quality limited by shallow decoder architecture
- 100× parameter increase with modest performance gain
- Training time increased significantly (10 min → 2 hours)
- Finding optimal loss weights required multiple iterations with very limited colab resources.

**Conclusion**: Achieved significant text generation improvement at the cost of substantially increased model complexity. Future work should focus on more efficient architectures.

---

## Future Work
1. Deeper Visual Decoder
2. Data Augmentation such as Image transformations (rotation, crop, color jitter) and Text paraphrasing to improve generalization.