# myproj — minimal Colab + GitHub + local PyTorch kit

## Local
python -m venv .venv

.venv\Scripts\activate

pip install -e .

python main.py

## Colab
!pip install git+https://github.com/<USER>/<REPO>.git

from myproj.tiny_train import run_demo

run_demo()
