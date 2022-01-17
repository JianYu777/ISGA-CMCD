

# Our approach

Pytorch Code of method for "An Intrinsic Structured Graph Alignment Module
with Modality-Invariant Representations for NIR-VIS Face Recognition" 

### Results on the  CASIA NIR-VIS 2.0 Dataset an the  LMAP-HQ Dataset 
| Method | Datasets           | Rank@1  | VR@FPR=1% | VR@FPR=0.1% |
| ------ | ------------------ | ------- | --------- | ----------- |
| LLM    | #CASIA NIR-VIS 2.0 | ~ 99.9% | ~ 99.9%   | ~ 99.9%     |
| LLM    | #LMAP-HQ           | ~ 99.3% | ~ 99.5%   | ~ 98.7%     |



*The code has been tested in Python 3.7, PyTorch=1.0. 

### 1. Prepare the datasets.

- CASIA NIR-VIS 2.0 involves nearly 18,000 images from 725 subjects, with 1-22 VIS and 5-
   50 NIR images per subject.

-  The LAMP-HQ dataset is a newly proposed large-scale dataset for NIR-VIS FR, consisting of approximately 74,000 faces from 573 subjects

- run 

   ```
   data_load.py, data_manager.py
   ```

    in to pepare the dataset

### 2. Test.
  Test a model by
  ```bash
python test.py
  ```

  - `--dataset`: which dataset "sysu" or "regdb".

  - `--lr`: initial learning rate.
  
  - `--gpu`:  which gpu to run.



### 3. References

```
[1] Stan Li, Dong Yi, Zhen Lei, and Shengcai Liao. The casia nir-vis 2.0
face database. In Computer Vision and Pattern Recognition Workshops,
pages 348–353, 2013
```

```
[2] Aijing Yu, Haoxue Wu, Huaibo Huang, Zhen Lei, and Ran He. Lamp-
hq: A large-scale multi-pose high-quality database and benchmark for
nir-vis face recognition. International Journal of Computer Vision,
129(5):1467–1483, 2021
```

```
[3] M. Ye, Z. Wang, X. Lan, and P. C. Yuen. Visible thermal person reidentification via dual-constrained top-ranking. In International Joint Conference on Artificial Intelligence (IJCAI), pages 1092–1099, 2018.
```

```
[4] Ye M, Shen J, Lin G, et al. Deep learning for person re-identification: A survey and outlook[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021
```

