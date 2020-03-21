# fix-match

In progress pytorch implementation of [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/pdf/2001.07685.pdf)

## Run

    python train.py --batch-size=64 --mu=7 --dataset-name=CIFAR10C --data-dir=path/to/your/data
    

## Results
CIFAR10 classification accuracy for now. Approx 85% after 90 epochs

| Epochs| 100 | 500 
| ------ |-----| -----|
| Paper | xx | xx
| This repo | 85.0 | 93.7500

