dev:
	python -m modules.cvae \
		--ds_name ionosphere\
		--clf_name knn\
		--pth logs/dev/version_0/checkpoints/epoch=99-step=2000.ckpt\
		--encoder mlp_16_8_16\
		--decoder mlp_16_8_16\
		--z_dim 8\
		--n_class 2 
		 
train:
	python -m modules.cvae.train --config config/dev.yaml

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
