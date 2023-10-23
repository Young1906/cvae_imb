# run all experiment
all: heart_2cl breast-tissue frogs ecoli


# Breast Tissue dataset
# --------------------------------------------------
heart_2cl: train_heart_2cl infer_heart_2cl

train_heart_2cl:
	python -m modules.cvae.train --config config/heart_2cl.yml

infer_heart_2cl:
	python -m modules.cvae --config config/heart_2cl.yml

# Breast Tissue dataset
# --------------------------------------------------
breast-tissue: train_breast_tissue infer_breast_tissue

train_breast_tissue:
	python -m modules.cvae.train --config config/breast-tissue.yml

infer_breast_tissue:
	python -m modules.cvae --config config/breast-tissue.yml

# Frog exp
# --------------------------------------------------
frogs : train_frogs infer_frogs

train_frogs:
	python -m modules.cvae.train --config config/frogs.yaml

infer_frogs:
	python -m modules.cvae --config config/frogs.yaml


# Ecoli exp
# --------------------------------------------------
ecoli: train_ecoli infer_ecoli

train_ecoli:
	python -m modules.cvae.train --config config/ecoli.yaml

infer_ecoli:
	python -m modules.cvae --config config/ecoli.yaml


# DOWNLOAD DATASET 
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
	curl 'https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data'\
		-o datasets/semeion.data
