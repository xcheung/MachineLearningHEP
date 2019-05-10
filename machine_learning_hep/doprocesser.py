#############################################################################
##  © Copyright CERN 2018. All rights not expressly granted are reserved.  ##
##                 Author: Gian.Michele.Innocenti@cern.ch                  ##
## This program is free software: you can redistribute it and/or modify it ##
##  under the terms of the GNU General Public License as published by the  ##
## Free Software Foundation, either version 3 of the License, or (at your  ##
## option) any later version. This program is distributed in the hope that ##
##  it will be useful, but WITHOUT ANY WARRANTY; without even the implied  ##
##     warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    ##
##           See the GNU General Public License for more details.          ##
##    You should have received a copy of the GNU General Public License    ##
##   along with this program. if not, see <https://www.gnu.org/licenses/>. ##
#############################################################################

"""
main script for doing data processing, machine learning and analysis
"""
import time
import yaml
from multiprocesser import MultiProcesser  # pylint: disable=import-error

def doprocesser():

    with open("default_complete.yaml", 'r') as run_config:
        data_config = yaml.load(run_config)
    with open("data/database_ml_parameters.yml", 'r') as param_config:
        data_param = yaml.load(param_config)
    with open("data/database_run_list.yml", 'r') as runlist_config:
        run_param = yaml.load(runlist_config)
        case = data_config["case"]
    t0 = time.time()

    for i in range(data_param[case]["multi"]["mc"]["nperiods"]):
        mymultiprocess_mc = MultiProcesser(data_param[case], run_param, "mc")
        mymultiprocess_mc.multi_unpack(i)
        mymultiprocess_mc.multi_skim(i)
        mymultiprocess_mc.multi_merge(i)
    mymultiprocess_mc.multi_merge_allinone()
doprocesser()
