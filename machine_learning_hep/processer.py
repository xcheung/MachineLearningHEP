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
import os
import random as rd
import uproot
import pandas as pd
import numpy as np
from machine_learning_hep.selectionutils import selectfidacc
from machine_learning_hep.bitwise import filter_bit_df, tag_bit_df
from utilities import selectdfquery, selectdfrunlist, merge_method
from utilities import list_folders, appendfiletolist, appendmainfoldertolist
from utilities import create_folder_struc
class Processer: # pylint: disable=too-many-instance-attributes
    # Class Attribute
    species = 'processer'

    # Initializer / Instance Attributes
    # pylint: disable=too-many-statements, too-many-arguments
    def __init__(self, datap, run_param, mcordata, p_maxfiles,
                 d_root, d_pkl, d_pklsk, d_pkl_ml, p_period,
                 p_chunksizeunp, p_chunksizeskim, p_maxprocess,
                 p_frac_merge, p_rd_merge):

        #directories
        self.d_root = d_root
        self.d_pkl = d_pkl
        self.d_pklsk = d_pklsk
        self.d_pkl_ml = d_pkl_ml

        self.datap = datap
        self.mcordata = mcordata
        self.p_frac_merge = p_frac_merge
        self.p_rd_merge = p_rd_merge
        self.period = p_period
        self.runlist = run_param[self.period]

        self.p_maxfiles = p_maxfiles
        self.p_chunksizeunp = p_chunksizeunp
        self.p_chunksizeskim = p_chunksizeskim

        #parameter names
        self.p_maxprocess = p_maxprocess
        self.indexsample = None

        #namefile root
        self.n_root = datap["files_names"]["namefile_unmerged_tree"]
        #troot trees names
        self.n_treereco = datap["files_names"]["treeoriginreco"]
        self.n_treegen = datap["files_names"]["treeorigingen"]
        self.n_treeevt = datap["files_names"]["treeoriginevt"]

        #namefiles pkl
        self.n_reco = datap["files_names"]["namefile_reco"]
        self.n_evt = datap["files_names"]["namefile_evt"]
        self.n_evtorig = datap["files_names"]["namefile_evtorig"]
        self.n_gen = datap["files_names"]["namefile_gen"]


        #selections
        self.s_reco_unp = datap["sel_reco_unp"]
        self.s_good_evt_unp = datap["sel_good_evt_unp"]
        self.s_cen_unp = datap["sel_cen_unp"]
        self.s_gen_unp = datap["sel_gen_unp"]
        self.s_reco_skim = datap["sel_reco_skim"]
        self.s_gen_skim = datap["sel_gen_skim"]

        #bitmap
        self.b_trackcuts = datap["sel_reco_singletrac_unp"]
        self.b_std = datap["bitmap_sel"]["isstd"]
        self.b_mcsig = datap["bitmap_sel"]["ismcsignal"]
        self.b_mcsigprompt = datap["bitmap_sel"]["ismcprompt"]
        self.b_mcsigfd = datap["bitmap_sel"]["ismcfd"]
        self.b_mcbkg = datap["bitmap_sel"]["ismcbkg"]

        #variables name
        self.v_all = datap["variables"]["var_all"]
        self.v_evt = datap["variables"]["var_evt"][self.mcordata]
        self.v_gen = datap["variables"]["var_gen"]
        self.v_evtmatch = datap["variables"]["var_evt_match"]
        self.v_bitvar = datap["bitmap_sel"]["var_name"]
        self.v_isstd = datap["bitmap_sel"]["var_isstd"]
        self.v_ismcsignal = datap["bitmap_sel"]["var_ismcsignal"]
        self.v_ismcprompt = datap["bitmap_sel"]["var_ismcprompt"]
        self.v_ismcfd = datap["bitmap_sel"]["var_ismcfd"]
        self.v_ismcbkg = datap["bitmap_sel"]["var_ismcbkg"]

        #list of files names

        self.l_path = None
        self.l_root = None
        self.l_reco = None
        self.l_gen = None
        self.l_evt = None
        self.l_evtorig = None
        self.l_recosk = None
        self.l_gensk = None
        self.l_evtsk = None

        #list of output names
        self.o_reco_ml = None
        self.o_gen_ml = None
        self.o_evt_ml = None
        self.o_evtorig_ml = None

        self.buildlists()

    def buildlists(self):
        self.l_path = list_folders(self.d_root, self.n_root, self.p_maxfiles)
        self.l_root = appendfiletolist(self.l_path, self.n_root)
        self.l_reco = appendfiletolist(self.l_path, self.n_reco)
        self.l_recosk = appendfiletolist(self.l_path, self.n_reco)
        self.l_evt = appendfiletolist(self.l_path, self.n_evt)
        self.l_evtorig = appendfiletolist(self.l_path, self.n_evtorig)
        self.l_root = appendmainfoldertolist(self.l_root, self.d_root)
        self.l_reco = appendmainfoldertolist(self.l_reco, self.d_pkl)
        self.l_recosk = appendmainfoldertolist(self.l_recosk, self.d_pklsk)
        self.l_evt = appendmainfoldertolist(self.l_evt, self.d_pkl)
        self.l_evtorig = appendmainfoldertolist(self.l_evtorig, self.d_pkl)

        if self.mcordata == "mc":
            self.l_gen = appendfiletolist(self.l_path, self.n_gen)
            self.l_gensk = appendfiletolist(self.l_path, self.n_gen)
            self.l_gen = appendmainfoldertolist(self.l_gen, self.d_pkl)
            self.l_gensk = appendmainfoldertolist(self.l_gensk, self.d_pklsk)

    def unpack(self, file_index):
        treeevtorig = uproot.open(self.l_root[file_index])[self.n_treeevt]
        dfevtorig = treeevtorig.pandas.df(branches=self.v_evt)
        dfevtorig = selectdfrunlist(dfevtorig, self.runlist, "run_number")
        dfevtorig = selectdfquery(dfevtorig, self.s_cen_unp)
        dfevtorig.to_pickle(self.l_evtorig[file_index])
        dfevt = selectdfquery(dfevtorig, self.s_good_evt_unp)
        dfevt.to_pickle(self.l_evt[file_index])

        treereco = uproot.open(self.l_root[file_index])[self.n_treereco]
        dfreco = treereco.pandas.df(branches=self.v_all)
        dfreco = selectdfrunlist(dfreco, self.runlist, "run_number")
        dfreco = selectdfquery(dfreco, self.s_reco_unp)
        dfreco = pd.merge(dfreco, dfevt, on=self.v_evtmatch)
        dfgen = selectdfquery(dfreco, self.s_cen_unp)
        dfreco = selectdfquery(dfreco, self.s_good_evt_unp)
        isselacc = selectfidacc(dfreco.pt_cand.values, dfreco.y_cand.values)
        dfreco = dfreco[np.array(isselacc, dtype=bool)]
        if self.b_trackcuts is not None:
            dfreco = filter_bit_df(dfreco, self.v_bitvar, self.b_trackcuts)
        if self.mcordata == "mc":
            dfreco[self.v_isstd] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                       self.b_std), dtype=int)
            dfreco[self.v_ismcsignal] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                            self.b_mcsig), dtype=int)
            dfreco[self.v_ismcprompt] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                            self.b_mcsigprompt), dtype=int)
            dfreco[self.v_ismcfd] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                        self.b_mcsigfd), dtype=int)
            dfreco[self.v_ismcbkg] = np.array(tag_bit_df(dfreco, self.v_bitvar,
                                                         self.b_mcbkg), dtype=int)
        dfreco.to_pickle(self.l_reco[file_index])

        if self.mcordata == "mc":
            treegen = uproot.open(self.l_root[file_index])[self.n_treegen]
            dfgen = treegen.pandas.df(branches=self.v_gen)
            dfgen = selectdfrunlist(dfgen, self.runlist, "run_number")
            dfgen = pd.merge(dfgen, dfevt, on=self.v_evtmatch)
            dfgen = selectdfquery(dfgen, self.s_gen_unp)
            dfgen = selectdfquery(dfgen, self.s_good_evt_unp)
            dfgen = selectdfquery(dfgen, self.s_cen_unp)
            dfgen.to_pickle(self.l_gen[file_index])

    def skim(self, file_index):
        dfreco = pickle.load(open(self.l_reco[file_index], "rb"))
        dfreco = selectdfquery(dfreco, self.s_reco_skim)
        dfreco.to_pickle(self.l_recosk[file_index])

        if self.mcordata == "mc":
            dfgen = pickle.load(open(self.l_gen[file_index], "rb"))
            dfgen = selectdfquery(dfgen, self.s_gen_skim)
            dfgen.to_pickle(self.l_gensk[file_index])

    def parallelizer(self, function, argument_list, maxperchunk):
        print(argument_list, maxperchunk)
        chunks = [argument_list[x:x+maxperchunk] \
                  for x in range(0, len(argument_list), maxperchunk)]
        for chunk in chunks:
            pool = mp.Pool(self.p_maxprocess)
            _ = [pool.apply(function, args=chunk[i]) for i in range(len(chunk))]
            pool.close()

    def process_unpack_par(self):
        create_folder_struc(self.d_pkl, self.l_path)
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.unpack, arguments, self.p_chunksizeunp)

    def process_skim_par(self):
        create_folder_struc(self.d_pklsk, self.l_path)
        arguments = [(i,) for i in range(len(self.l_reco))]
        self.parallelizer(self.skim, arguments, self.p_chunksizeskim)

    def merge(self):
        nfiles = len(self.l_recosk)
        ntomerge = (int)(nfiles * self.p_frac_merge)
        rd.seed(self.p_rd_merge)
        filesel = rd.sample(range(0, nfiles), ntomerge)
        list_sel_recosk = [self.l_recosk[j] for j in filesel]
        list_sel_evt = [self.l_evt[j] for j in filesel]
        list_sel_evtorig = [self.l_evtorig[j] for j in filesel]

        self.o_reco_ml = os.path.join(self.d_pkl_ml, self.n_reco)
        self.o_evt_ml = os.path.join(self.d_pkl_ml, self.n_evt)
        self.o_evtorig_ml = os.path.join(self.d_pkl_ml, self.n_evtorig)

        merge_method(list_sel_recosk, self.o_reco_ml)
        merge_method(list_sel_evt, self.o_evt_ml)
        merge_method(list_sel_evtorig, self.o_evtorig_ml)

        if self.mcordata == "mc":
            list_sel_gensk = [self.l_gensk[j] for j in filesel]
            self.o_gen_ml = os.path.join(self.d_pkl_ml, self.n_gen)
            merge_method(list_sel_gensk, self.o_gen_ml)

    def process_merge(self):
        self.merge()
