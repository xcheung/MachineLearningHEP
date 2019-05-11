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
import os
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

    mcordata = "mc"
    doconversion = 1
    doskim = 1
    domerge = 1
    domergeinone = 1

    dirpkl = data_param[case]["multi"][mcordata]["pkl"]
    dirpklsk = data_param[case]["multi"][mcordata]["pkl_skimmed"]
    dirpklml = data_param[case]["multi"][mcordata]["pkl_skimmed_merge_for_ml"]
    dirpklmltot = data_param[case]["multi"][mcordata]["pkl_skimmed_merge_for_ml_all"]

    for i in range(data_param[case]["multi"][mcordata]["nperiods"]):
        if not os.path.exists(dirpkl[i]):
            os.makedirs(dirpkl[i])
        if not os.path.exists(dirpklsk[i]):
            os.makedirs(dirpklsk[i])
        if not os.path.exists(dirpklml[i]):
            os.makedirs(dirpklml[i])
    if not os.path.exists(dirpklmltot):
        os.makedirs(dirpklmltot)

    mymultiprocess = MultiProcesser(data_param[case], run_param, mcordata)

    for i in range(data_param[case]["multi"][mcordata]["nperiods"]):
        if doconversion == 1:
            mymultiprocess.multi_unpack(i)
        if doskim == 1:
            mymultiprocess.multi_skim(i)
        if domerge == 1:
            mymultiprocess.multi_merge(i)
    if domergeinone == 1:
        mymultiprocess.multi_merge_allinone()
doprocesser()
