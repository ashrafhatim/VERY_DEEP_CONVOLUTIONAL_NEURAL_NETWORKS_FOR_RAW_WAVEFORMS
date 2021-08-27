# VERY DEEP CONVOLUTIONAL NEURAL NETWORKS FOR RAW WAVEFORMS
Unofficial implementation of ["VERY DEEP CONVOLUTIONAL NEURAL NETWORKS FOR RAW WAVEFORMS"](https://arxiv.org/pdf/1610.00087.pdf) paper as part of African Master of Machine Intelligence's bootcamp 2021 ([AMMI](https://aimsammi.org/)).
### Download the dataset
URBANSOUND8K DATASET: [link](https://urbansounddataset.weebly.com/urbansound8k.html)
```
!wget https://goo.gl/8hY5ER 
!tar -xf 8hY5ER 
```
### Download the requirements
```
pip install -r src/requirements.txt
```
### Run the model
>Experiment Name (exp_name): M3, M5, M11, M18, M34_res
```                                                 
python /src/main.py  --exp-name "M5" --epoch 5 --val-fold 10 --dataset-path '/content/UrbanSound8K/' --save-path "/content/drive/MyDrive/MODELS/"
```


