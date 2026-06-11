# Deep Learning Assignment 2

The current version of the repo has an auto encoder implemented, with variable encoder and decoder models. Currently, only a Convolution Neural Network is implemented as Encoder/decoder. The current `main.py` file runs the training of the model on the data.

## Setup
### Data
Download the data from `https://surfdrive.surf.nl/s/ks6qE3xLm37wmrS` (4.5 GB) and extract the **content** of the zip in the `data` folder.

You should have the following structure:
```
.
└── data/
    ├── Cross/
    │   └── ...
    └── Intra/
        └── ...
```

### Run the files

We use `uv` to run everything. To donwload all the required packages, run `uv sync`. Then you can run a file with `uv run file.py`


