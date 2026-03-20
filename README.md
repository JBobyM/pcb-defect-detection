# PCB Defect Detection

Trained a YOLOv8m model to detect defects on printed circuit boards. It covers 9 defect types and hits **84.7% mAP50** on the validation set after 100 epochs — good enough to be useful in a real production line.

The dataset is [DsPCBSD+](https://github.com/aub-mind/DsPCBSD), a public PCB defect dataset. Training ran on two RTX 3090s in parallel, taking just over 2 hours.

---

## What it detects

| Code | Defect | mAP50 |
|------|--------|-------|
| HB | Hole Break | 98.5% |
| OP | Open Circuit | 89.9% |
| SH | Short Circuit | 89.5% |
| BMFO | Base Material Foreign Object | 87.2% |
| SP | Spur | 85.2% |
| MB | Mousebite | 84.5% |
| SC | Spurious Copper | 83.2% |
| CS | Conductor Scratch | 74.3% |
| CFO | Copper Foreign Object | 70.4% |

Hole breaks are almost perfectly detected. Conductor scratches and copper foreign objects are the trickiest — they're visually inconsistent and the model reflects that.

---

## Setup

Clone the repo, then create a virtual environment and install dependencies:

```bash
python3 -m venv pcb_env
source pcb_env/bin/activate
pip install ultralytics torch torchvision pyyaml
```

You'll also need the DsPCBSD+ dataset. Download it and place it under `data/DsPCBSD+/`. The expected structure is:

```
data/DsPCBSD+/
├── Data_YOLO/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
└── Data_COCO/
    └── annotations/
```

---

## Training

```bash
python train.py
```

This trains from scratch using the config in `dataset.yaml`. It'll use both GPUs automatically if available (`device='0,1'`). Checkpoints are saved every 5 epochs under `runs/pcb_defect_detection/weights/`.

Key settings:
- Model: YOLOv8m (pretrained on COCO)
- Optimizer: AdamW, lr=0.001
- Image size: 640px
- Batch: 16 per GPU (32 total)
- Augmentation: mosaic, mixup, copy-paste, horizontal flip

The augmentation choices are intentional — no vertical flips or perspective warping, since PCB images are always taken from directly above.

---

## Inference

```python
from ultralytics import YOLO

model = YOLO("runs/pcb_defect_detection/weights/best.pt")
results = model("path/to/pcb_image.jpg", conf=0.25)
results[0].show()
```

An ONNX export is also available at `runs/pcb_defect_detection/weights/best.onnx` if you need to run it outside of Python.

---

## Results

Overall on the validation set:

| Metric | Value |
|--------|-------|
| mAP@0.5 | 84.7% |
| mAP@0.5:0.95 | 49.9% |
| Precision | 81.6% |
| Recall | 79.4% |

Training curves and confusion matrices are in `runs/pcb_defect_detection/`.

---

## Notes

- `yolo11n.pt` is sitting in the repo root from an earlier experiment. It wasn't used for the final training run.
- The COCO-format annotations (`Data_COCO/`) are there if you want to fine-tune with a different framework.
- Short circuit (SH) performs surprisingly well despite having the fewest training examples. Hole break (HB) is almost a solved problem at this point.
