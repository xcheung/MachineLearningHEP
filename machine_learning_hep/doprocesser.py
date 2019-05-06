import yaml
import pickle
import multiprocessing as mp
import uproot
import os
from processer import processer

with open("default_complete.yaml", 'r') as run_config:
    data_config = yaml.load(run_config)
with open("data/database_ml_parameters.yml", 'r') as param_config:
    data_param = yaml.load(param_config)
with open("data/config_model_parameters.yml", 'r') as mod_config:
    data_model = yaml.load(mod_config)
mcordata = "data"

indexp = 0
case = data_config["case"]
param_case = data_param[case]

myprocess = processer(data_param[case], mcordata, indexp, 10)
myprocess.activate_unpack()
myprocess.run()
