# System-level time computation and representation in the suprachiasmatic nucleus revealed by large-scale calcium imaging and machine learning

> [**System-level time computation and representation in the suprachiasmatic nucleus revealed by large-scale calcium imaging and machine learning**]
>  <br>
> Zichen Wang†, Jing Yu†, Muyue Zhai, Zehua Wang, Kaiwen Sheng, Yu Zhu, Tianyu Wang, Mianzhi Liu, Lu Wang, Miao Yan, Jue Zhang, Ying Xu, Xianhua Wang, Lei Ma*, Wei Hu*, Heping Cheng* <br>
> > <https://www.nature.com/articles/s41422-024-00956-x> <br>

This repository contains the implementation for the paper [System-level time computation and representation in the suprachiasmatic nucleus revealed by large-scale calcium imaging and machine learning]. 

## The Suprachiasmatic Nucleus

> The suprachiasmatic nucleus (SCN) is the mammalian central circadian pacemaker with heterogeneous neurons acting in concert while each harboring self-sustained molecular clockwork. It comprises a pair of oval structures each containing ~10,000 heterogeneous neurons. The central clock integrates the external light and time cues to generate multichannel signals to command peripheral clocks across diverse tissues, thereby regulating physiological functions and daily behaviors of the animal.
> >
> ![The SCN Universe](./The SCN Universe.jpg)

## How to run the code
### 1. Dependencies
Before running our codes, please make sure you have installed conda/miniconda and use the following commands to install the requirements.
```shell script
conda env create -f torch.yml
``` 

### 2. Datasets
We have SCN Ca2+ signal data from six different mice, and experimental analyses, including the graph-based state classifier, the time prediction workflow, and the time-series analyzer TraceContrast, rely on these data to be completed. We provided the full data in the following link 
[SCNdata_link](https://pkueducn-my.sharepoint.com/:f:/g/personal/wangzichen_pkueducn_onmicrosoft_com/ElZ-3W0GFl9Hrs0Kh_i2_70B7F5ReKX9hZxlUl837WON8A?e=9kM5cR)

Download the SCN data and put them in ```./SCNData ``` folder.

### 3. TimePredictor
#### 3.1 Definition:
(1) **general time predictor $f_{n}$**: this model is trained on the full training dataset. For example, one model is trained on the training dataset of ```Dataset1_SCNProject.mat```, we name it as general time predictor $f_{1}$.

(2) **submodule time predictor $g_{n}$**: this model is trained on a splitted dataset. For example, one model is trained on the training dataset of the **spatial class 1** in ```Dataset1_SCNProject.mat```, we name it as submodule time predictor $g_{1}$. As shown in the FigS8, we have trained separately on three training dataset of the **spatial class 1, 2, 3** in ```Dataset1_SCNProject.mat```.

#### 3.2 code introduction.
(1) For general time predictor, we provide its full training and testing pipeline in folder ```./TimePredictor```. And we also provided the code for general time predictor testing on sub-spatial modules in the same folder. 
(2) For submodule time predictor, we provide its full training and testing pipeline in folder ```./SubModule_TimePredictor```..

#### 3.2 training and testing.
##### 3.2.1 general time predictor.
```
cd TimePredictor
python train.py

python test_on_FullTestSet.py             # test on the full test set.

python test_on_SubModuleInFullTestSet.py  # test on the submodule in the full test set.
```

##### 3.2.2 submodule time predictor.
$g_{n}$ is trained on one spatial submodule, then test on its own testset and other two submodule.
```
cd SubModule_TimePredictor
python train_1for3.py

python training_base_1for3.py  
```

### 4. Attribution Analysis.

```
cd Attribution_analysis
python train.py

python test_AttributionAnalysis.py
```


### 5. TraceContrast.

#### 5.1 Preprocessing data.

In ```./TraceContrast``` folder, the original data should be preprocessed by ```scn_data_process.m``` (MATLAB) first for different tasks, which includes "standard", "pc-sample" (sampling neurons), "time-sample" (sampling timestamps), "1_3-sample" (sampling 1-8 hours of 24 hours), "2_3-sample" (sampling 9-16 hours of 24 hours), "3_3-sample" (sampling 17-24 hours of 24 hours).

By specifying ```scn_data_path``` (the file path of the original data) and ```dataset_order``` in ```scn_data_process.m``` and running ```scn_data_process.m```, the code will generate ```*.mat``` files automatically for all the above tasks, which can be used for the latter training procedure.

- For "1_3-sample", "2_3-sample" and "3_3-sample" tasks, the ```*_standard.mat``` is used.


#### 5.2 Training.

```
cd TraceContrast
python main $your_input_file_path$ $task$
```

The $task$ represents the name of the task for the input data, which has been mentioned above. More parameters are optional, and the instruction of them can be found in ```main.py```.


### 6. StateClassifier.

##### 6.1 Prepare Data

In ```./StateClassifier``` folder, the original data should be preprocessed by ```scn_phase_space_process.m``` (MATLAB) first to convert raw data into graph dataset.

##### 6.2 Install dependencies

You may install PyTorch Geometric as follows:

```bash
# Install PyTorch Geometric
pip install torch-cluster==1.6.3 -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html
pip install torch-sparse==0.6.18 -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html
pip install torch-scatter==2.1.2 -f https://pytorch-geometric.com/whl/torch-2.1.0+cu118.html
pip install torch-geometric
```

##### 6.3 Training and testing

Execute the main script:

```
cd StateClassifier
python main.py
```

This will train the GCN model, perform validation, and finally test the model on the provided dataset. The best model is saved in the `./result/best_scn.pt` file.

## Citation
If you use our data or code in a research project leading to a publication, please cite the paper.

```
Wang, Z., Yu, J., Zhai, M. et al. System-level time computation and representation in the suprachiasmatic nucleus revealed by large-scale calcium imaging and machine learning. Cell Res (2024). https://doi.org/10.1038/s41422-024-00956-x
```