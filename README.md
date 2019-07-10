# ML_Hanabi
Machine Learning solution to the cooperative, imperfect-information game Hanabi

The create_data script is written in Python 2. # hardcoded creates data for 1 agent in 10 games
To run create_data in Keystone shell:
```
fork/clone repo into your home folder
cd ~/ganabi/hanabi-env
cmake .
make
cd ..
source /data1/shared/venvg2/bin/activate
mkdir data
python create_data.py
```

Then, to run the experiment: # naive mlp model trained on 10 games, then tested on random adhoc games
```
source /data1/shared/venvg/bin/activate
python run_experiment.py
```
