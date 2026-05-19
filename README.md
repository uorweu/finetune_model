### Requirements
- Python 3.12
- git clone this repository:
```powershell
git clone git@github.com:uorweu/finetune_model.git $HOME/
```
- downloads data/ on drives and move it to $HOME/fine_tune_yamnet/yamnet_finetune/
- and cd to $HOME/fine_tune_yamnet/yamnet_finetune

run 
```powershell
python -m venv venv
```
- activate virtual environment:
```powershell
$HOME/finetune_model/venv/Scripts/Activate.ps1
```

run
```powershell
pip install -r requirements.txt
```

run
```powershell
python src/preprocess.py
```

run
```powershell
python 
```

wait for data/raw being processed

run
```powershell
python extract_embeddings.py
```

run 
```powershell
python train.py
```

for training
and wait until it ends


