# Refusal Direction is Universal Across Safety-Aligned Languages

This repository contains the code and dataset for the paper "Refusal Direction is Universal Across Safety-Aligned Languages".

## PolyRefuse Dataset

The **PolyRefuse** dataset is a multilingual safety evaluation dataset covering 14 languages: ar, de, en, es, fr, it, ja, ko, nl, pl, ru, th, zh, yo.

You can find the dataset in the [`PolyRefuse/`](PolyRefuse) directory, which contains:
- Harmful prompts (train/val/test splits) translated to all languages
- Harmless prompts (train/val/test splits) translated to all languages
- Back-translated versions for analysis

## Setup

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup script
bash setup.sh
```

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See `requirements.txt` for full list

## Usage

### Running Experiments

```bash
# Configure experiment settings in configs/cfg.yaml
# Run the main pipeline
python scripts/experiment.py --config configs/cfg.yaml
```

### Dataset Loading

```python
from dataset.load_dataset import load_multilingual_data

# Load dataset for a specific language
data = load_multilingual_data(language='en', split='test', dataset_type='harmful')
```

## Repository Structure

```
.
├── PolyRefuse/              # Multilingual safety dataset
├── configs/                 # Configuration files
├── dataset/                 # Dataset loading and processing
├── evaluators/              # Safety evaluators
├── pipeline/                # Main experimental pipeline
│   ├── model_utils/        # Model implementations
│   ├── submodules/         # Pipeline components
│   └── run_pipeline.py     # Main pipeline runner
├── scripts/                 # Utility scripts and experiments
├── utils/                   # Helper utilities
└── requirements.txt        # Python dependencies
```

## Citation

If you use this code or dataset, please cite our paper:

```bibtex
@inproceedings{polyrefuse2025,
  title={Refusal Direction is Universal Across Safety-Aligned Languages},
  author={[Authors]},
  booktitle={[Conference]},
  year={2025}
}
```

## License

See [LICENSE](LICENSE) for details.

## Baseline vs English Refusal Vector Ablation

![Baseline vs English Refusal Vector Ablation](images/baseline_vs_harm_ablation-1.png)
