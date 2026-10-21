# SVM

The public data contain only 10 format examples. For formal training or evaluation, first set `AGS_DATA_DIR` to the complete local dataset directory; see the repository-root README.

The seven features in Text S3 are used. StandardScaler and the RBF SVM are fitted only on the training set. The shared independent leave-one-reactor-out 80:20 rule is used.

Run `python src/train.py` from this directory, or `python "SVM + FD + GLCM texture features/SVM/train_svm.py"` from the repository root.
