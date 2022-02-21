# CoordX

PyTorch implementation of the paper CoordX: Accelerating Implicit Neural Representation with a Split MLP Architecture.

### [Project Page](https://chrissun06.github.io/CoordX/) | [Paper](https://arxiv.org/abs/2201.12425)

## Setup
Python dependencies:
- ConfigArgParse
- imageio
- scikit-image
- scipy
- tensorboard
- PyTorch 1.9.1
- Torchvision 0.10.1
- trimesh

Run
`pip install -r requirements.txt`

## Running Experiments
### Image experiments
Run
`python experiment_scripts/train_img.py --split_mlp --split_train --experiment_name=image_test`
to train a CoordX network to fit the default photographer image for 10,000 epochs, results can be found under `./logs/image_test`

Some arguments and their definitions:

`--model_type`  
: Options currently are "sine" (all sine activations), "relu" (all relu activations), "nerf" (relu activations and positional encoding as in NeRF), "rbf" (input rbf layer, rest relu), and in the future: "mixed" (first layer sine, other layers tanh)'.

`--split_mlp`  
: Using the split MLP architecture.

`--split_train`  
: Using split training acceleration.

`--fusion_operator`  
: Options currently are "sum" (element-wise summation for features from each channel), "prod" (element-wise multiplication) and "concat" (concatenation).

### Video experiments
Run
`python experiment_scripts/train_video.py --split_mlp --split_train --st_split --experiment_name=video_test`
to train a CoordX network to fit the default `cat_video.mp4` under `./data`


`--st_split`  
: Split channels into [pixel location, frame]

### 3D shape experiments
Run `python experiment_scripts/train_3d_occupancy.py --split_mlp --mesh_path=<./path/to/mesh> --experiment_name=shape_test` to train a CoordX network to fit a 3d shape stored in `./path/to/mesh`

`--mesh_path`  
: Path to the mesh file

## Contact
If you have any questions, please feel free to email the authors.