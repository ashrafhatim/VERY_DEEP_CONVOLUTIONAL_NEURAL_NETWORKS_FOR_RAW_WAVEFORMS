# VERY DEEP CONVOLUTIONAL NEURAL NETWORKS FOR RAW WAVEFORMS
> Implementation of ["VERY DEEP CONVOLUTIONAL NEURAL NETWORKS FOR RAW WAVEFORMS"](https://arxiv.org/pdf/1610.00087.pdf) paper during AMMI bootcamp 2021
## Download the dataset
URBANSOUND8K DATASET: [download link](https://goo.gl/8hY5ER)
```
!wget https://goo.gl/8hY5ER 
!tar -xf 8hY5ER 
```
## Download the requirements
```
pip install -r src/requirements.txt
```
## Run the model
```
python src/main.py --epochs 5 --val-fold 5 --exp-name "M3" --savePath "" --dataPath "/content/UrbanSound8K/audio"\
                                                    --metadataPath "/content/UrbanSound8K/metadata/UrbanSound8K.csv"
```


