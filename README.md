## DeepParametricShapes | [Webpage](https://people.csail.mit.edu/smirnov/deep-parametric-shapes/) | [Paper](https://arxiv.org/abs/1904.08921) | [Video](https://youtu.be/v_0UrjbTtHg)

<img src="https://people.csail.mit.edu/smirnov/deep-parametric-shapes/im.png" width="75%" alt="Deep Parametric Shapes" />

**Deep Parametric Shape Predictions using Distance Fields**<br>
Dmitriy Smirnov, Matthew Fisher, Vladimir G. Kim, Richard Zhang, Justin Solomon<br>
[Conference on Computer Vision and Pattern Recognition (CVPR) 2020](http://cvpr2020.thecvf.com)

### Set-up
To install the code, run:
```
sudo apt install libcairo2-dev pkg-config python3-dev
conda create -n dps python=3.6 -y
conda activate dps
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch -y
pip install -r requirements.txt
```

Also, be sure to execute `export PYTHONPATH=:$PYTHONPATH` prior to running any of the scripts.

### 2D: Font Vectorization

#### Demo
First, download a pretrained font vectorization model:
```
mkdir -p models/dps_2d
wget -O models/dps_2d/ckpt.pth https://www.dropbox.com/s/46tp19h6npqhuuh/dps_2d.pth\?dl\=0
```

Then, run the following to generate a PDF file with the vectorization for a given input glyph PNG image:
```
python scripts/run_2d.py demo/P1.png P out.pdf
```
Make sure to specify the letter of the input glyph (in this case `P`). The `demo` directory contains PNGs of the GAN-generated glyphs used for Figure 13 of the paper.

#### Training
To prepare the training dataset, first download and extract the font TTF files:
```
wget -O fonts.tar.gz https://www.dropbox.com/s/7oreepk5gm0efj4/fonts.tar.gz?dl=0
tar -xvf fonts.tar.gz
```
Then, process the TTFs to generate the input images and target distance fields:
```
python scripts/generate_fonts.py
```

To train a model from scratch, run:
```
python scripts/train_2d.py --output models/model_name --data data/fonts
```

### BibTeX
```
@inproceedings{smirnov2020dps,
  title={Deep Parametric Shape Predictions using Distance Fields},
  author={Smirnov, Dmitriy and Fisher, Matthew and Kim, Vladimir G. and Zhang, Richard and Solomon, Justin},
  year={2020},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)}
}
```
