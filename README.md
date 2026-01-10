
<h1><strong>Paired Uniform Cubic B-Splines are Strong Approximation to Represent Skeleton Activity</strong></h1>
Muhammad Amirul Raziq Rosman, Owais Ahmed Malik, Daphne Teck Ching Lai, Wee Hong Ong

This is the PyTorch implementation of our MPSCP.

<h2> Abstract </h2>
<p>
In a real-life scenario where an agent needs to learn the patterns of a human’s daily activities, labels are not available, making it difficult to model a representation using supervised training strategies. This has fueled research in unsupervised and self-supervised strategies. However, most of these strategies require a large amount of computing power and time to learn the patterns of these activities. In this paper, we present an approach to reduce the number of embeddings per activity, thus reducing the need for high computational power and, in turn, reducing the time needed to learn the patterns of these activities using motion and velocity splines as training targets using a masked autoencoder we dubbed the Masked Paired Spline Coefficient Predictor (MPSCP). We have performed extensive experiments on NTU RGB+D 60 and 120, as well as PKU-MMD, to verify that MPSCP has indeed learnt good latent representations of the activities while increasing the efficiency of the pre-training and linear evaluation protocol task and mitigating performance loss using spline coefficients.
</p>

<h2> Installation </h2>

```bash
conda env create -f environment.yml
```

<strong>OR</strong>

```bash
pip install -r requirements.txt
```

<strong>OR</strong>

Download the packages manually needed in the <strong>Requirements</strong> section.

<h2> Requirements </h2>

```bash
python==3.8.19
pyyaml==6.0.3
numpy==1.24.4
einops==0.8.1
scikit-learn==1.3.2
matplotlib==3.7.5
seaborn==0.13.2
torch==2.4.1
torchvision==0.19.1
tensorboard==2.14.0
tqdm==4.67.1
```

<h2> Data Preparation </h2>

This section follows <a href="https://github.com/maoyunyao/MAMP/tree/main">MAMP's</a> Data Preparation.

<h3> Datasets </h3>
<h4> NTU RGB+D & NTU 120 </h4>
<ol>
    <li>Request for the dataset <a href="https://rose1.ntu.edu.sg/dataset/actionRecognition/">here</a></li>
    <li>Download the skeletons: </li>
    <ul>
        <li>nturgbd_skeletons_s001_to_s017.zip (NTU 60)</li>
        <li>nturgbd_skeletons_s018_to_s032.zip (NTU 120)</li>
    </ul>
    <li>Extract the zip files to ./data/nturgd_raw following the directory structure in the next section.</li>
</ol>

<h4> PKU MMD Phase I & II </h4>
<ol>
    <li>Download the dataset from <a href="https://struct002.github.io/PKUMMD/">here</a></li>
    <li>Specifically download the skeleton data, label data and the split files:</li>
    <ul>
        <li>Data/Skeleton.7z + Label/Label_PKUMMD.7z + Split/cross_subject.txt + Split/cross_view.txt (Phase I)</li>
        <li>Data/Skeleton_v2.7z + Label/Label_PKUMMD_v2.7z + Split/cross_subject_v2.txt + Split/cross_view_v2.txt (Phase II)</li>
    </ul>
    <li>Extract the files to ./data/pku_raw following the directory structure in the next section.</li>
</ol>

<h3> Data Generation </h3>

<h4> Directory Structure </h4>

```
- data/
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
  - pku_v1/
  - pku_v2/
  - pku_raw/
    - v1/
      - label/
      - skeleton/
      - cross_subject.txt
      - cross_view.txt
    - v2/
      - label/
      - skeleton/
      - cross_subject_v2.txt
      - cross_view_v2.txt
```

<h4> Generating the Data </h4>

<ul>
<li>Generate the NTU 60 or the NTU 120 dataset:</li>

```
cd ./data/ntu # or cd ./data/ntu120
# Get skeleton of each subject
python get_raw_skes_data.py
# Remove the noisy skeletons
python get_raw_denoised_data.py
# Transform the skeleton to the center of the first frame
python seq_transformation.py
```

<li>Generate the PKU-MMD Phase I or Phase II dataset:</li>
    
```
cd ./data/pku_v1 # or cd ./data/pku_v2
python pku_gendata.py
```

</ul>

<h2> Pre-training and Linear Evaluation Protocol (LEP) </h2>

The general command for training can be seen  below:

```bash
python main.py --config ./config/base/{config}.yaml --seed {seed} --work-dir {work_dir} --weights-transformer-path {work_dir + final_weights/weights.pt} --train {train} --train-lep {train-lep}
```

where:
<ul>
<li>--config requires the location of the yaml file</li>
<li>--seed requires any integer number for reproducibility</li>
<li>--work-dir is an optional argument for the output directory of the model</li>
<li>--weights-transformer-path is an optional argument for the weights used for the transformer during LEP</li>
<li>--train is an optional argument where setting to True performs pre-training</li>
<li>--train-lep is an optional argument where setting to True performs LEP</li>
</ul>

For our paper, we have used seeds 2021, 2022, 2023, 2024 and 2025.

<h2> Acknowledgements </h2>
Parts of our code have been based on <a href="https://github.com/maoyunyao/MAMP/tree/main">MAMP</a> and <a href="https://github.com/facebookresearch/mae">MAE</a>