# CIFAR-100 input data

Original Python-format files from the existing local experiment dataset.
The two large files are stored with Git LFS; run `git lfs pull` to obtain them.
The compact experiment uses repository-relative paths to this directory.

Dataset source: https://www.cs.toronto.edu/~kriz/cifar.html

| File | SHA-256 |
| --- | --- |
| train | `735e79b04f092ca3d2e6d07f368c0a7d70d48c48d28865950cc24454cf45129b` |
| test | `4b67687d9933c4db8f0831104447f15b93774f4f464bd0516f0f0f2ac83b7864` |
| meta | `a5d4786345c961390f865e93b434dbd5c6904ce880667e0cb888c97d449f28b9` |

Use the experiment's saved train/validation/test split. The test images and
labels must not be used to choose cells, calibration, weights or readout.
