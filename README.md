# ML_Hanabi
Machine Learning solution to the cooperative, imperfect-information game Hanabi

Currently only implemented as naive MLP in master, but will create branches for different models.

You can read the write-up to the naive MLP [here.](https://docs.google.com/document/d/14ZCLFKE28LajVXr6D0yMmC0yzk0cvF7OK9G9BU091rU/edit?usp=sharing)

The create_data script is written in Python 2. # hardcoded creates data for 1 random agent in pool of unique agents

To run create_data in Keystone shell:
```
fork/clone repo into your home folder
cd ~/ML_Hanabi/hanabi-env
cmake .
make
cd ..
source /data1/shared/venvg2/bin/activate
mkdir data
python create_data.py
```

Then, to run the experiment: # naive mlp model trained on 10 selfplay games, then tested on 100 adhoc games
```
source /data1/shared/venvg/bin/activate
python run_experiment.py
```
