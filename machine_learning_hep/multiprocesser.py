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
from processer import Processer
from utilities import merge_method
class MultiProcesser: # pylint: disable=too-many-instance-attributes
    species = "multiprocesser"
    def __init__(self, datap, run_param, mcordata):
        self.datap = datap
        self.run_param = run_param
        self.mcordata = mcordata
        self.prodnumber = len(datap["multi"][self.mcordata]["unmerged_tree_dir"])
        #directories
        self.d_root_all = datap["multi"][self.mcordata]["unmerged_tree_dir"]
        self.d_pkl_all = datap["multi"][self.mcordata]["pkl"]
        self.d_pklsk_all = datap["multi"][self.mcordata]["pkl_skimmed"]
        self.d_pklml_all = datap["multi"][self.mcordata]["pkl_skimmed_merge_for_ml"]
        self.d_pklml_all_merged = datap["multi"][self.mcordata]["pkl_skimmed_merge_for_ml_all"]
        self.p_period = datap["multi"][self.mcordata]["period"]
        self.p_seedmerge = datap["multi"][self.mcordata]["seedmerge"]
        self.p_fracmerge = datap["multi"][self.mcordata]["fracmerge"]

        self.p_maxfiles = datap["multi"][self.mcordata]["maxfiles"]
        self.p_chunksizeunp = datap["multi"][self.mcordata]["chunksizeunp"]
        self.p_chunksizeskim = datap["multi"][self.mcordata]["chunksizeskim"]


        #namefiles pkl
        self.n_reco = datap["files_names"]["namefile_reco"]
        self.n_evt = datap["files_names"]["namefile_evt"]
        self.n_evtorig = datap["files_names"]["namefile_evtorig"]
        self.n_gen = datap["files_names"]["namefile_gen"]
        self.l_reco_all = None
        self.o_reco_merged = None

    def multi_unpack(self, indexp):
        myprocess = Processer(self.datap, self.run_param, self.mcordata,
                              self.p_maxfiles[indexp], self.d_root_all[indexp],
                              self.d_pkl_all[indexp], self.d_pklsk_all[indexp],
                              self.d_pklml_all[indexp],
                              self.p_period[indexp], self.p_chunksizeunp[indexp],
                              self.p_chunksizeskim[indexp], 30,
                              self.p_fracmerge[indexp], self.p_seedmerge[indexp])
        myprocess.process_unpack_par()

    def multi_unpack_allperiods(self):
        for i in range(self.prodnumber):
            self.multi_unpack(i)

    def multi_skim(self, indexp):
        myprocess = Processer(self.datap, self.run_param, self.mcordata,
                              self.p_maxfiles[indexp], self.d_root_all[indexp],
                              self.d_pkl_all[indexp], self.d_pklsk_all[indexp],
                              self.d_pklml_all[indexp],
                              self.p_period[indexp], self.p_chunksizeunp[indexp],
                              self.p_chunksizeskim[indexp], 30,
                              self.p_fracmerge[indexp], self.p_seedmerge[indexp])
        myprocess.process_skim_par()

    def multi_skim_allperiods(self):
        for i in range(self.prodnumber):
            self.multi_skim(i)

    def multi_merge(self, indexp):
        myprocess = Processer(self.datap, self.run_param, self.mcordata,
                              self.p_maxfiles[indexp], self.d_root_all[indexp],
                              self.d_pkl_all[indexp], self.d_pklsk_all[indexp],
                              self.d_pklml_all[indexp],
                              self.p_period[indexp], self.p_chunksizeunp[indexp],
                              self.p_chunksizeskim[indexp], 30,
                              self.p_fracmerge[indexp], self.p_seedmerge[indexp])
        myprocess.process_merge()

    def multi_merge_allperiods(self):
        for i in range(self.prodnumber):
            self.multi_merge(i)

    def multi_merge_allinone(self):
        self.l_reco_all = [os.path.join(direc, self.n_reco) for direc in self.d_pklml_all]
        self.o_reco_merged = os.path.join(self.d_pklml_all_merged, self.n_reco)
        merge_method(self.l_reco_all, self.o_reco_merged)
