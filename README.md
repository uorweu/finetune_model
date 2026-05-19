### Requirements
downloads data/ on drives and move it to fine_tune_yamnet/yamnet_finetune/
and cd to fine_tune_yamnet/yamnet_finetune

run
```powershell
python src/preprocess.py
```

wait for data/raw being processed

then run 
```powershell
python train.py
```

for training


