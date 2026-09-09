# Privacy-Preserving Fine-Tuning

This repository contains the code and experimental pipeline for the **pilot study of privacy-preserving fine-tuning on the SHDR dataset**.

The complete experimental workflow consists of four main stages:

1. **Generate privacy-preserving images using SAPP**
2. **Fine-tune the pretrained foundation model**
3. **Evaluate model utility**
4. **Evaluate privacy against membership inference and reconstruction attacks**

The main experimental settings include:

- **SAPP**: our privacy-preserving fine-tuning method
- **Baseline**: fine-tuning without privacy protection
- **DP ($\epsilon=1$)**
- **DP ($\epsilon=8$)**
- **DP ($\epsilon=32$)**

---

# 1. Generate Privacy-Preserving Images Using SAPP

The first stage generates the privacy-preserving image dataset used for SAPP fine-tuning.

Run:

```bash
attackFFM.sh
```

The main configuration file is:

```bash
DP/dp_fl_config_attack.yaml
```

Modify the required dataset paths, output paths, and other experiment-specific parameters in this configuration file before running the script.

The generated privacy-preserving images will be used as the training data in Stage 2.

---

# 2. Fine-Tune the Pretrained Foundation Model

## 2.1 SAPP Fine-Tuning

Fine-tune the pretrained foundation model using the privacy-preserving images generated in Stage 1.

Run:

```bash
trainDPBaselineEncrypt.sh
```

The main configuration file is:

```bash
DP/dp_fl_config_baseline_encrypt.yaml
```

Please modify the dataset path, pretrained checkpoint path, model save directory, and other experiment-specific parameters before training.

---

## 2.2 Baseline and DP Settings

For comparison, the pretrained foundation model is also fine-tuned under the following settings:

1. **Baseline**: without any privacy protection
2. **DP with $\epsilon=1$**
3. **DP with $\epsilon=8$**
4. **DP with $\epsilon=32$**

Detailed instructions are provided in:

- [Baseline Model Training](#5-baseline-model-training)
- [DP Model Training](#6-dp-model-training)

---

# 3. Utility Evaluation

Classification performance is used to evaluate the utility of the fine-tuned models.

## 3.1 Test the SAPP Model

The testing script is:

```bash
testDPEncrypt.sh
```

Use this script to evaluate the models fine-tuned using SAPP-generated privacy-preserving images.

---

## 3.2 Test the Baseline and DP Models

The testing script is:

```bash
testDP.sh
```

### 3.2.1 Modify the Log Name

Before each experiment, modify:

```bash
LOG_NAME="..."
```

Use a descriptive name containing the corresponding experimental setting, such as the DP parameter and random seed.

### 3.2.2 Set the Model Directory

Modify:

```bash
MODEL_DIR="..."
```

`MODEL_DIR` should point to the model directory generated during training, corresponding to:

```yaml
general:
  save_dir: <model save directory>
```

### 3.2.3 Select Epochs for Evaluation

The epochs to be evaluated are controlled by:

```bash
for EP in $(seq 0 7)
```

For example, to test epochs 0–7:

```bash
for EP in $(seq 0 7)
```


### 3.2.4 Modify the Test Configuration

The test configuration file is:

```bash
DP/dp_fl_config_test.yaml
```

Modify the test dataset path:

```yaml
dataset:
  base_path: <test dataset path>
```

Unless otherwise required, keep the remaining testing parameters unchanged.

### 3.2.5 Testing Checklist

Before each test:

1. Modify `LOG_NAME` in `testDP.sh`.
2. Set `MODEL_DIR` to the corresponding training output directory.
3. Set the required epoch range in `for EP in $(seq ...)`.
4. Set `dataset.base_path` in `dp_fl_config_test.yaml`.
5. Verify the model directory, test dataset, and log directory.
6. Submit the testing job.

---

# 4. Privacy Evaluation

Privacy is evaluated using two categories of attacks:

1. **Membership Inference Attacks (MIA)**
2. **Data Reconstruction Attack**

Four MIA evaluations are currently included:

- Confidence-based MIA
- Gradient-based MIA
- Robust Membership Inference Attack (RMIA)
- Patient-level RMIA

---

## 4.1 Membership Inference Attacks

### 4.1.1 Confidence-Based Membership Inference Attack

#### Step 1: Generate model output scores for training samples

Run:

```bash
genMIAData.sh
```

#### Step 2: Generate model output scores for test samples

Run:

```bash
genMIADataTest.sh
```

#### Step 3: Perform confidence-based MIA

Run:

```bash
MIA_attack/confidence_attack.py
```

This script generates and stores the confidence-based MIA results.

#### Step 4: Plot the ROC curves

Run:

```bash
MIA_attack/plot_confidence_loglog_all.py
```

---

### 4.1.2 Gradient-Based Membership Inference Attack

#### Step 1: Perform gradient-based MIA

Run:

```bash
MIA_attack/gradient_attack.py
```

This script generates the gradient-based membership inference results.

#### Step 2: Plot the ROC curves

Run:

```bash
MIA_attack/plot_gradient_loglog_all.py
```

---

### 4.1.3 Robust Membership Inference Attack (RMIA)

Run:

```bash
miaAttack_rmia_para.sh
```

In this job script, set the RMIA execution script to:

```bash
MIA_attack/LiRA/main_rmia_vit_val.py
```

This generates and saves the RMIA results.

After the attack is completed, generate the comparison figure using:

```bash
MIA_attack/LiRA/loadnpz_loglog_comparison.py
```

---

### 4.1.4 Patient-Level Robust Membership Inference Attack

Run:

```bash
miaAttack_rmia_para.sh
```

Set the execution script to:

```bash
MIA_attack/LiRA/main_rmia_patient.py
```

This performs the patient-level RMIA evaluation.

After the attack is completed, generate the patient-level comparison figure using:

```bash
MIA_attack/LiRA/loadnpz_loglog_comparison_patient.py
```

---

## 4.2 Data Reconstruction Attack

The reconstruction attack uses a U-Net trained with paired plaintext and privacy-preserving images.

### 4.2.1 Train the U-Net Reconstruction Model

Run:

```bash
trainUnet.sh
```

Set the execution script in `trainUnet.sh` to:

```bash
pairAttack/Pytorch-UNet/trainHierar.py
```

The U-Net is trained using paired plaintext images and their corresponding privacy-preserving images.

---

### 4.2.2 Reconstruct Plaintext Images

Run:

```bash
trainUnet.sh
```

Set the execution script to:

```bash
pairAttack/Pytorch-UNet/testHierar.py
```

This script uses the trained reconstruction model to infer plaintext images from the privacy-preserving images.

---

# 5. Baseline Model Training

## 5.1 Training Script

The Baseline model is trained using:

```bash
trainDPBaseline.sh
```

The Baseline corresponds to standard fine-tuning **without any privacy protection technique**.

---

## 5.2 Configuration

The Baseline configuration file is:

```bash
DP/dp_fl_config_baseline.yaml
```

Before each experiment, modify the following parameters.

### Log Name

Modify `LOG_NAME` in `trainDPBaseline.sh`:

```bash
LOG_NAME="..."
```

### Model Save Directory

```yaml
general:
  save_dir: <model save directory>
```

### Dataset Path

```yaml
dataset:
  base_path: <dataset path>
```

### Pretrained Model / Checkpoint

```yaml
model:
  chkpt_path: <checkpoint path>
```

Unless otherwise required, keep all other configuration parameters unchanged.

---

## 5.3 Baseline Training Checklist

Before each training run:

1. Modify `LOG_NAME` in `trainDPBaseline.sh`.
2. Modify `general.save_dir`.
3. Verify `dataset.base_path`.
4. Verify `model.chkpt_path`.
5. Keep all other configuration parameters unchanged.
6. Verify the log and model save directories.
7. Submit the training job.

---

# 6. DP Model Training

## 6.1 Training Script

The DP model is trained using:

```bash
DeepUWFAnsible/trainDPBaselineEncrypt.sh
```

The main DP configuration file is:

```bash
DP/dp_fl_config.yaml
```

---

## 6.2 Experiment Configuration

### Log Name

Before each experiment, modify:

```bash
LOG_NAME="..."
```

The log name should identify the corresponding `epsilon`, `seed`, and other relevant experimental settings.

### Model Save Directory

```yaml
general:
  save_dir: <model save directory>
```

### Random Seed

```yaml
general:
  seed: 0
```

Each DP setting is evaluated using five random seeds:

```text
0, 1, 2, 3, 4
```

### Differential Privacy Budget

Modify:

```yaml
dp:
  epsilon: <epsilon>
```

The evaluated privacy budgets are:

```text
epsilon = 1
epsilon = 8
epsilon = 32
```

### Dataset and Pretrained Model

Modify according to the local environment:

```yaml
dataset:
  base_path: <local dataset path>

model:
  chkpt_path: <local pretrained model/checkpoint path>
```

Unless otherwise required, keep the remaining parameters unchanged.

---

## 6.3 Batch Size and GPU Memory

The current experiments use:

```text
batch_size = 24
```

Please keep `batch_size = 24` whenever possible to ensure consistency with the previous experiments.

Approximately **60+ GB of GPU memory** is required for this batch size.

If the experiment cannot run normally even in the available 4-GPU environment, please report the issue rather than reducing the batch size, because changing the batch size may make the experimental settings inconsistent with previous runs.

---

## 6.4 Epochs and Early Stopping

The configuration specifies:

```text
epochs = 50
```

However, based on existing experiments, the best-performing checkpoint on the current evaluation set typically appears within the first five epochs, and sometimes as early as epoch 1.

Therefore, approximately **10 epochs** are usually sufficient to determine whether training can be stopped early.

For consistency, keep:

```text
epochs = 50
```

in the configuration file. The total epoch number may affect intermediate variables or training logic in the current implementation, and this dependency has not yet been fully verified.

If the model has clearly converged after approximately 10 epochs, the training job can be manually stopped after checking the evaluation results.

---

## 6.5 Current DP Experiment Status

| Epsilon | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 |
|---|---|---|---|---|---|
| 32 | Completed | Completed | Completed | Completed | Completed |
| 8 | To do | To do | To do | To do | To do |
| 1 | To do | To do | To do | To do | To do |

According to the current experiment status, **10 DP training runs remain**.

> Update this table as experiments are completed.

---

## 6.6 DP Training Checklist

Before each DP training run:

1. Modify `LOG_NAME` in `trainDPBaselineEncrypt.sh`.
2. Modify `general.save_dir` in `DP/dp_fl_config.yaml`.
3. Set `general.seed` to the required seed (`0`, `1`, `2`, `3`, or `4`).
4. Set `dp.epsilon` to the required privacy budget (`1`, `8`, or `32`).
5. Verify `dataset.base_path`.
6. Verify `model.chkpt_path`.
7. Keep all other configuration parameters unchanged.
8. Keep `batch_size = 24` whenever possible.
9. Keep `epochs = 50` in the configuration; manually stop training if appropriate after checking approximately 10 epochs.
10. Submit the training job and verify the log and model output directories.

---

# 7. Overall Experimental Workflow

The complete privacy-preserving experiment can be summarized as:

```text
SHDR Training Data
        |
        +-----------------------------+
        |                             |
        v                             v
     Baseline                       SAPP
   Fine-tuning               Privacy-preserving
        |                    Image Generation
        |                             |
        |                             v
        |                       SAPP Fine-tuning
        |                             |
        +-------------+---------------+
                      |
             +--------+--------+
             |                 |
             v                 v
      Utility Evaluation   Privacy Evaluation
      Classification       |
                           +-- Confidence-based MIA
                           +-- Gradient-based MIA
                           +-- RMIA
                           +-- Patient-level RMIA
                           +-- Reconstruction Attack

Additional comparison:
DP Fine-tuning (epsilon = 1, 8, 32)
        |
        +--> Utility Evaluation
        +--> Privacy Evaluation
```

The main comparison therefore evaluates the privacy–utility trade-off among:

```text
Baseline
SAPP
DP (epsilon = 1)
DP (epsilon = 8)
DP (epsilon = 32)
```
