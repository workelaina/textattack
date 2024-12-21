# ragbba

```shell
sudo apt install build-essential
conda create -y -n bba python=3.9
conda activate bba

git clone https://github.com/workelaina/textattack.git
cd textattack
# pip install -e '.[tensorflow]' --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e '.[tensorflow]'

# scp train.py ubuntu@server.mil:~/ragbba/
cd ~/ragbba
python download.py
python train.py
```
