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
import multiprocessing as mp
import pickle
import uproot
from machine_learning_hep.listfiles import list_files_dir_lev2

class Processer: # pylint: disable=too-many-instance-attributes
    # Class Attribute
    species = 'processer'

    # Initializer / Instance Attributes
    def __init__(self, datap, run_param, mcordata, indexp, maxfiles):

        self.datap = datap
        self.mcordata = mcordata
        self.index_period = indexp
        self.maxfiles = maxfiles

        #namefile root
        self.n_root = datap["files_names"]["namefile_unmerged_tree"]
        #troot trees names
        self.n_treereco = datap["files_names"]["treeoriginreco"]
        self.n_treegen = datap["files_names"]["treeorigingen"]
        self.n_treeevt = datap["files_names"]["treeoriginevt"]

        #namefiles pkl
        self.n_reco = datap["files_names"]["namefile_reco"]
        self.n_evt = datap["files_names"]["namefile_evt"]
        self.n_gen = datap["files_names"]["namefile_gen"]

        #namefiles pkl skimmed
        self.n_recosk = datap["files_names"]["namefile_reco_skim"]
        self.n_evtsk = datap["files_names"]["namefile_evt_skim"]
        self.n_gensk = datap["files_names"]["namefile_gen_skim"]

        #directories
        self.d_root = datap["inputs"][self.mcordata]["unmerged_tree_dir"][self.index_period]
        self.d_pkl = datap["output_folders"]["pkl_out"][self.mcordata][self.index_period]
        self.d_pklsk = datap["output_folders"]["pkl_skimmed"][self.mcordata][self.index_period]

        #selections
        self.s_reco_unp = datap["skimming_sel"]
        self.s_evt_unp = datap["skimming_sel_evt"]
        self.s_gen_unp = datap["skimming_sel_gen"]
        self.s_reco_skim = datap["skimming2_sel"]
        self.s_evt_skim = datap["skimming2_sel_evt"]
        self.s_gen_skim = datap["skimming2_sel_gen"]
        self.s_reco_trackpid = datap["skimming2_dotrackpid"]

        #variables name
        self.v_all = datap["variables"]["var_all"]
        self.v_evt = datap["variables"]["var_evt"][self.mcordata]
        self.v_gen = datap["variables"]["var_all"]

        #list of files names
        self.l_root = None
        self.l_reco = None
        self.l_gen = None
        self.l_evt = None
        self.l_recosk = None
        self.l_gensk = None
        self.l_evtsk = None

        #parameter names
        self.maxperchunk = 30
        self.maxprocess = 8
        self.indexsample = None
        #activators
        self.activateunpack = False

        print(run_param)
    def activate_unpack(self):
        self.activateunpack = True
        print("Unpacker activated")

    def set_maxperchunk(self, maxperchunk):
        self.maxperchunk = maxperchunk

    def buildlistpkl(self):
        self.l_root, self.l_reco = list_files_dir_lev2(self.d_root, \
                                        self.d_pkl, self.n_root, self.n_reco)
        _, self.l_gen = list_files_dir_lev2(self.d_root, self.d_pkl, \
                                        self.n_root, self.n_gen)
        _, self.l_evt = list_files_dir_lev2(self.d_root, self.d_pkl, \
                                                 self.n_root, self.n_evt)

    def buildlistpklskim(self):
        _, self.l_recosk = list_files_dir_lev2(self.d_pkl, self.d_pklsk, \
                                               self.n_reco, self.n_recosk)
        _, self.l_gensk = list_files_dir_lev2(self.d_pkl, self.d_pklsk, \
                                               self.n_gen, self.n_gensk)
        _, self.l_evtsk = list_files_dir_lev2(self.d_pkl, self.d_pklsk, \
                                               self.n_evt, self.n_evtsk)

    def unpack(self, file_index):
        treereco = uproot.open(self.l_root[file_index])[self.n_treereco]
        dfreco = treereco.pandas.df(branches=self.v_all)
        if self.s_reco_unp is not None:
            dfreco = dfreco.query(self.s_reco_unp)
        dfreco.to_pickle(self.l_reco[file_index])

        treeevt = uproot.open(self.l_root[file_index])[self.n_treeevt]
        dfevt = treeevt.pandas.df(branches=self.v_evt)
        if self.s_evt_unp is not None:
            dfevt = dfevt.query(self.s_evt_unp)
        dfevt.to_pickle(self.l_evt[file_index])

        if self.mcordata == "mc":
            treegen = uproot.open(self.l_root[file_index])[self.n_treegen]
            dfgen = treegen.pandas.df(branches=self.v_gen)
            if self.s_gen_unp is not None:
                dfgen = dfgen.query(self.s_gen_unp)
            dfgen.to_pickle(self.l_gen[file_index])


#    def unpack(self, file_index):
#        self.unpack_module(self.l_root[file_index], self.l_reco[file_index], \
#                           self.n_treereco, self.v_all, self.s_reco_unp)
#        self.unpack_module(self.l_root[file_index], self.l_evt[file_index], \
#                           self.n_treeevt, self.v_evt, self.s_evt_unp)
#        if self.mcordata == "mc":
#            self.unpack_module(self.l_root[file_index], self.l_gen[file_index], \
#                               self.n_treegen, self.v_gen, self.s_gen_unp)

    def skim(self, file_index):
        df = pickle.load(open(self.l_reco[file_index], "rb"))
        df = df.query(self.s_reco_skim)
        df.to_pickle(self.l_recosk[file_index])

        dfevt = pickle.load(open(self.l_evt[file_index], "rb"))
        dfevt = df.query(self.s_evt_skim)
        dfevt.to_pickle(self.l_evtsk[file_index])

        if self.mcordata == "mc":
            dfgen = pickle.load(open(self.l_gen[file_index], "rb"))
            dfgen = df.query(self.s_gen_skim)
            dfgen.to_pickle(self.l_gensk[file_index])

    def parallelizer(self, function, argument_list):
        chunks = [argument_list[x:x+self.maxperchunk] \
                  for x in range(0, len(argument_list), self.maxperchunk)]
        for chunk in chunks:
            pool = mp.Pool(self.maxprocess)
            _ = [pool.apply(function, args=chunk[i]) for i in range(len(chunk))]
            pool.close()

    def unpackparallel(self):
        self.buildlistpkl()
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.unpack, arguments)

    def skimparallel(self):
        self.buildlistpklskim()
        arguments = [(i,) for i in range(len(self.l_reco))]
        self.parallelizer(self.skim, arguments)

    def run(self):
        print(self.activateunpack)
        if self.activateunpack:
            self.unpackparallel()

#    @staticmethod
#    def unpack_module(filein, fileout, treename, varlist, selection):
#        tree = uproot.open(filein)[treename]
#        df = tree.pandas.df(branches=varlist)
#        if selection is not None:
#            df = df.query(selection)
#        df.to_pickle(fileout)
