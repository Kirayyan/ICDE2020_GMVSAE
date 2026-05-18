# Online Anomalous Trajectory Detection with Deep Generative Sequence Modeling (ICDE 2020)

## How To Use (the new version)

### Preprocessing
- Step1: Download Porto data (i.e., <tt>train.csv.zip</tt>) from https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data.
- Step2: Put the data file in <tt>./data/</tt>, and unzip it as <tt>porto.csv</tt>.
- Step3: Set the dataset info in <tt>./preprocess/preprocess.py</tt>), run preprocessing by <tt>sh preprocess.sh</tt>.

The processed data files (i.e., <tt>processed_porto_train.csv</tt> and <tt>processed_porto_val.csv</tt>) will be put in <tt>./data</tt>.


### Training 

#### Example of training on Porto dataset:
```python
python run_loop.py --mode=train --cluster_num=5 --num_epochs=5 --gpu_id=0 \ 
                   --model_dir=./ckpt --learning_rate=1e-4 --num_epochs=10 --pretrain_dir=./pretrain
```
More conveniently, we can run pretraining, training and evaluation via <tt>pretrain.sh</tt>, <tt>train.sh</tt> and <tt>eval.sh</tt>, respectively.

### Baseline workflow (train + stay/speed anomaly scores)

This repo implements **GMVSAE** from *Online Anomalous Trajectory Detection with Deep Generative Sequence Modeling* (ICDE 2020).

**1. Environment**

```bash
pip install -r requirements.txt
export TF_USE_LEGACY_KERAS=1
```

**2. Data**

- Real Porto: follow preprocessing above to obtain `data/processed_porto_train.csv` and `data/processed_porto_val.csv`.
- Smoke test without Kaggle data: `python3 scripts/create_synthetic_porto.py`

**3. Train checkpoints**

```bash
sh pretrain.sh   # saves ./pretrain/gmvsae_32_256_5/
sh train.sh      # saves ./ckpt/gmvsae_32_256_5/
```

**4. Generate stay / speed anomalies**

Built-in generators (repeat grid cell = stay; collapse middle segment = speed):

```bash
python3 scripts/generate_anomalies.py --split val --ratio 0.05 \
  --output ./data/anomalies_stay_speed.json
```

Or provide your own manifest (`records` list):

```json
{
  "records": [
    {"tid": 4, "trajectory": [1, 2, 2, 2, 2, 3], "label": 0, "type": "stay"}
  ]
}
```

- `label`: `0` = anomalous, `1` = normal  
- `score`: higher = more likely under the model (paper uses sequence likelihood)

**5. Output per-trajectory scores (CSV)**

```bash
sh score_stay_speed.sh
# or
python3 run_loop.py --mode=score --cluster_num=5 --model_dir=./ckpt \
  --eval_data=val --anomaly_path=./data/anomalies_stay_speed.json \
  --output_scores=./data/anomaly_scores.csv
```

Output columns: `tid, label, anomaly_type, score`.

**6. AUC evaluation (paper-style)**

```bash
OTYPE=stay sh eval.sh
OTYPE=speed sh eval.sh
```

### MST-OATD-aligned stay / speed anomalies (Porto + Chengdu)

If your anomalies follow [MST-OATD](https://github.com/chwang0721/MST-OATD) (`distance`, `fraction`, `observed_ratio`):

| Type | MST-OATD rule | GMVSAE input |
|------|----------------|--------------|
| **stay** | Temporal offset on a segment (grid unchanged) | Grid sequence with duplicated cells from timestamp gaps |
| **speed** | Spatial offset on a segment (MST `perturb_point`) | Grid IDs after spatial perturbation |

**Two datasets** (map sizes match MST-OATD): `porto` (51×119), `cd` (167×154).

```bash
# 1) Place MST-OATD npy under data/porto/ and data/cd/ (from their preprocess)
python3 scripts/convert_mst_npy_to_gmvsae.py --dataset porto
python3 scripts/convert_mst_npy_to_gmvsae.py --dataset cd

# 2) Generate stay & speed manifests (same params as MST-OATD paper: d=2, α=0.2, ρ=1.0)
python3 scripts/generate_mst_anomalies.py --dataset porto --anomaly_type both
python3 scripts/generate_mst_anomalies.py --dataset cd --anomaly_type both

# 3) Train + score both datasets (one command)
sh run_mst_oatd_baseline.sh
```

Outputs:

- `data/porto/scores_stay_val.csv`, `data/porto/scores_speed_val.csv`
- `data/cd/scores_stay_val.csv`, `data/cd/scores_speed_val.csv`

If you already ran official `generate_outliers.py`, import npy then score:

```bash
python3 scripts/import_mst_oatd_npy.py --dataset porto \
  --traj_npy ./data/porto/outliers_data_2_0.2_1.0.npy \
  --idx_npy ./data/porto/outliers_idx_2_0.2_1.0.npy --anomaly_type full
python3 run_loop.py --mode=score --dataset porto --anomaly_path ./data/porto/anomalies_full_val.json \
  --output_scores ./data/porto/scores_full_val.csv
```

#### Parameters:
| Name                  | Type            | Description   |
| :-------------        |:-------------   |:------------- |
| mode                  | enum(str)       | pretrain, train or evaluate. |
| data_filename         | str             | data file (e.g., ./data/processed_porto.csv). |
| map_size              | \(int, int\)    | size of the grid map. |
| token_dim             | int             | dimensionality of grid token. | 
| rnn_dim               | int             | dimensionality of rnn hidden state. |
| cluster_num           | int             | number of Gaussian components. |
| model_dir             | str             | directory to save/load a model during training or eval. |
| pretrain_dir          | str             | directory to save/load a model during pretraining.  |
| num_negs              | int             | number of negative samples during training.  |
| optimizer             | enum(str)       | training optimizer (e.g., adam or sgd). |
| learning_rate         | float           | learning rate for training. |
| num_epochs            | int             | number of passes over the training data. |
| log_steps             | int             | number of batches to print the log info. |

## Citation

Please kindly cite the paper if this repo is helpful :)

```
@inproceedings{liu2020online,
  title={Online anomalous trajectory detection with deep generative sequence modeling},
  author={Liu, Yiding and Zhao, Kaiqi and Cong, Gao and Bao, Zhifeng},
  booktitle={2020 IEEE 36th International Conference on Data Engineering (ICDE)},
  pages={949--960},
  year={2020},
  organization={IEEE}
}
```
