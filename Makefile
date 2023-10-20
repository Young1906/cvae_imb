download: ds_init ds_download 

ds_init:
	mkdir -p datasets

ds_download:
	wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/00253/micromass.zip'\
		-o datasets/micromass.zip
	wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/00408/gastroenterology_dataset.zip'\
		-o datasets/gastroenterology_dataset.zip
	wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/00604/PersonGaitDataSet.mat'\
		-o datasets/personal_gait_dataset.mat
	wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/00406/Anuran%20Calls%20(MFCCs).zip'\
		-o datasets/anuran_calls_mfccs.zip
	wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/00192/BreastTissue.xls'\
		-o datasets/breast_tissue.xls
	wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/00233/CNAE-9.data'\
		-o datasets/cnae9.data
	wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'\
		-o datasets/sonar.all-data
	wget 'http://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data'\
		-o datasets/ionosphere.data
	wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data'\
		-o datasets/ecoli.data
	wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'\
		-o datasets/seeds_datasets.txt
	wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/lung-cancer/lung-cancer.data'\
		-o datasets/lung-cancer.data
	wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/00470/pd_speech_features.rar'\
		-o datasets/ph_speech_features.rar
	wget 'http://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'\
		-o datasets/parkinsons.data
	wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECTF.train'\
		-o datasets/spectf.train
	wget 'https://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data'\
		-o datasets/semeion.data





