### Model
``ResNet18``

### Dataset
``Speech Commands V1`` 

### Environment

``conda = 4.10.3``

``python = 3.6``

### How to run the code:

```bash
python3 start.py
```
or
```bash
nohup python3 start.py > log &
```
* Run the code under nohup, and redirect output in to log

### How to change training setting:

``package`` >> ``config`` >> ``for_FL.py``

you can ***change setting*** (i.e., attack ratio) in **for_FL.py**

### How to download dataset:

1. Install sox
```bash
sudo apt install sox
sudo apt-get install libsox-fmt-all
```
2. Excute ``startup.sh``
```bash
bash startup.sh
```
3. Build the folder nameed ``data_json`` and put all json file(generate after excuting ``startup.sh``) in folder. Then put ``data_json`` folder in ``data`` folder. 
