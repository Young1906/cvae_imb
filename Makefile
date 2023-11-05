dev:
	python -m modules.baseline.samplers

# run all experiment
all: breast_cancer balance parkinsons connectionist ionosphere heart_2cl breast_tissue frogs ecoli

reset:
	rm -rf .log logs checkpoints results/*.csv
# Breast cancer dataset
# --------------------------------------------------
breast_cancer: train_breast_cancer infer_breast_cancer

train_breast_cancer:
	python -m modules.cvae.train --config config/cvae/breast_cancer_catboost.yml

infer_breast_cancer:
	python -m modules.cvae --config config/cvae/breast_cancer_catboost.yml
	python -m modules.cvae --config config/cvae/breast_cancer_decision_tree.yml
	python -m modules.cvae --config config/cvae/breast_cancer_gbc.yml
	python -m modules.cvae --config config/cvae/breast_cancer_knn.yml
	python -m modules.cvae --config config/cvae/breast_cancer_lr.yml
	python -m modules.cvae --config config/cvae/breast_cancer_mlp.yml
	python -m modules.cvae --config config/cvae/breast_cancer_svm.yml

# Balance dataset
# --------------------------------------------------
balance: train_balance infer_balance

train_balance:
	python -m modules.cvae.train --config config/cvae/balance_catboost.yml

infer_balance:
	python -m modules.cvae --config config/cvae/balance_catboost.yml
	python -m modules.cvae --config config/cvae/balance_decision_tree.yml
	python -m modules.cvae --config config/cvae/balance_gbc.yml
	python -m modules.cvae --config config/cvae/balance_knn.yml
	python -m modules.cvae --config config/cvae/balance_lr.yml
	python -m modules.cvae --config config/cvae/balance_mlp.yml
	python -m modules.cvae --config config/cvae/balance_svm.yml

# Parkinsons dataset
# --------------------------------------------------
parkinsons: train_parkinsons infer_parkinsons

train_parkinsons:
	python -m modules.cvae.train --config config/cvae/parkinsons_catboost.yml

infer_parkinsons:
	python -m modules.cvae --config config/cvae/parkinsons_catboost.yml
	python -m modules.cvae --config config/cvae/parkinsons_decision_tree.yml
	python -m modules.cvae --config config/cvae/parkinsons_gbc.yml
	python -m modules.cvae --config config/cvae/parkinsons_knn.yml
	python -m modules.cvae --config config/cvae/parkinsons_lr.yml
	python -m modules.cvae --config config/cvae/parkinsons_mlp.yml
	python -m modules.cvae --config config/cvae/parkinsons_svm.yml

# Connectionist dataset
# --------------------------------------------------
connectionist: train_connectionist infer_connectionist

train_connectionist:
	python -m modules.cvae.train --config config/cvae/connectionist_catboost.yml

infer_connectionist:
	python -m modules.cvae --config config/cvae/connectionist_catboost.yml
	python -m modules.cvae --config config/cvae/connectionist_decision_tree.yml
	python -m modules.cvae --config config/cvae/connectionist_gbc.yml
	python -m modules.cvae --config config/cvae/connectionist_knn.yml
	python -m modules.cvae --config config/cvae/connectionist_lr.yml
	python -m modules.cvae --config config/cvae/connectionist_mlp.yml
	python -m modules.cvae --config config/cvae/connectionist_svm.yml

# Breast Tissue dataset
# --------------------------------------------------
ionosphere: train_ionosphere infer_ionosphere

train_ionosphere:
	python -m modules.cvae.train --config config/cvae/ionosphere_catboost.yml

infer_ionosphere:
	python -m modules.cvae --config config/cvae/ionosphere_catboost.yml
	python -m modules.cvae --config config/cvae/ionosphere_decision_tree.yml
	python -m modules.cvae --config config/cvae/ionosphere_gbc.yml
	python -m modules.cvae --config config/cvae/ionosphere_knn.yml
	python -m modules.cvae --config config/cvae/ionosphere_lr.yml
	python -m modules.cvae --config config/cvae/ionosphere_mlp.yml
	python -m modules.cvae --config config/cvae/ionosphere_svm.yml

# Breast Tissue dataset
# --------------------------------------------------
heart_2cl: train_heart_2cl infer_heart_2cl

train_heart_2cl:
	python -m modules.cvae.train --config config/cvae/heart_2cl_catboost.yml

infer_heart_2cl:
	python -m modules.cvae --config config/cvae/heart_2cl_catboost.yml
	python -m modules.cvae --config config/cvae/heart_2cl_decision_tree.yml
	python -m modules.cvae --config config/cvae/heart_2cl_gbc.yml
	python -m modules.cvae --config config/cvae/heart_2cl_knn.yml
	python -m modules.cvae --config config/cvae/heart_2cl_lr.yml
	python -m modules.cvae --config config/cvae/heart_2cl_mlp.yml
	python -m modules.cvae --config config/cvae/heart_2cl_svm.yml

# Breast Tissue dataset
# --------------------------------------------------
breast_tissue: train_breast_tissue infer_breast_tissue

train_breast_tissue:
	python -m modules.cvae.train --config config/cvae/breast_tissue_catboost.yml

infer_breast_tissue:
	python -m modules.cvae --config config/cvae/breast_tissue_catboost.yml
	python -m modules.cvae --config config/cvae/breast_tissue_decision_tree.yml
	python -m modules.cvae --config config/cvae/breast_tissue_gbc.yml
	python -m modules.cvae --config config/cvae/breast_tissue_knn.yml
	python -m modules.cvae --config config/cvae/breast_tissue_lr.yml
	python -m modules.cvae --config config/cvae/breast_tissue_mlp.yml
	python -m modules.cvae --config config/cvae/breast_tissue_svm.yml

# Frog exp
# --------------------------------------------------
frogs : train_frogs infer_frogs

train_frogs:
	python -m modules.cvae.train --config config/cvae/frogs_catboost.yml

infer_frogs:
	python -m modules.cvae --config config/cvae/frogs_catboost.yml
	python -m modules.cvae --config config/cvae/frogs_decision_tree.yml
	python -m modules.cvae --config config/cvae/frogs_gbc.yml
	python -m modules.cvae --config config/cvae/frogs_knn.yml
	python -m modules.cvae --config config/cvae/frogs_lr.yml
	python -m modules.cvae --config config/cvae/frogs_mlp.yml
	python -m modules.cvae --config config/cvae/frogs_svm.yml


# Ecoli exp
# --------------------------------------------------
ecoli: train_ecoli infer_ecoli

train_ecoli:
	python -m modules.cvae.train --config config/ecoli.yml

infer_ecoli:
	python -m modules.cvae --config config/ecoli.yml


# DOWNLOAD DATASET 
# --------------------------------------------------
download: ds_init ds_download 

ds_init:
	mkdir -p datasets

ds_download:
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/00253/micromass.zip'\
		-o datasets/micromass.zip
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/00408/gastroenterology_dataset.zip'\
		-o datasets/gastroenterology_dataset.zip
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/00604/PersonGaitDataSet.mat'\
		-o datasets/personal_gait_dataset.mat
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran%20Calls%20(MFCCs).zip'\
		-o datasets/anuran_calls_mfccs.zip
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/00192/BreastTissue.xls'\
		-o datasets/breast_tissue.xls
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/00233/CNAE-9.data'\
		-o datasets/cnae9.data
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'\
		-o datasets/sonar.all-data
	curl 'http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'\
		-o datasets/ionosphere.data
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'\
		-o datasets/ecoli.data
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'\
		-o datasets/seeds_datasets.txt
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.data'\
		-o datasets/lung-cancer.data
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/00470/pd_speech_features.rar'\
		-o datasets/ph_speech_features.rar
	curl 'http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'\
		-o datasets/parkinsons.data
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.train'\
		-o datasets/spectf.train
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.test'\
		-o datasets/spectf.test
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data'\
		-o datasets/semeion.data
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'\
		-o datasets/wdbc.data
