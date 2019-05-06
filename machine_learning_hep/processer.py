import yaml
import pickle
import multiprocessing as mp
import uproot
import os
   ##
class Processer:
    # Class Attribute
    species = 'processer'

    # Initializer / Instance Attributes
    def __init__(self, datap, mcordata, indexp, maxfiles):

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
        self.d_reco = datap["output_folders"]["pkl_out"][self.mcordata][self.index_period]
        self.d_evt = self.d_reco
        self.d_gen = self.d_reco
        self.d_recosk = datap["output_folders"]["pkl_skimmed"][self.mcordata][self.index_period]
        self.d_evtsk = self.d_recosk
        self.d_recosk = self.d_recosk

        #selections
        self.s_reco_unp = datap["skimming_sel"]
        self.s_evt_unp = datap["skimming_sel_evt"]
        self.s_gen_unp = datap["skimming_sel_gen"]
        self.s_reco_skim = datap["skimming2_sel"]
        self.s_evt_skim = datap["skimming2_sel_evt"]
        self.s_gen_skim = datap["skimming2_sel_gen"]
        self.s_reco_trackpid = datap["skimming2_dotrackpid"]
        self.s_centr = datap["sel_cent"]

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

    def activate_unpack(self):
        self.activateunpack = True
        print("Unpacker activated")

    def set_maxperchunk(self, maxperchunk):
        self.maxperchunk = maxperchunk

    def buildlistpkl(self):
        self.l_root, self.l_reco = self.list_files_dir_lev2(self.d_root, self.d_reco, self.n_root, self.n_reco)
        _, self.l_gen = self.list_files_dir_lev2(self.d_root, self.d_gen, self.n_root, self.n_gen)
        _, self.l_evt = self.list_files_dir_lev2(self.d_root, self.d_evt, self.n_root, self.n_evt)

    def buildlistskim(self):
        _, self.l_recosk = list_files_dir_lev2(self.d_reco, self.d_recosk, self.n_reco, self.n_recosk)

    def unpack_module(self, filein, fileout, treename, varlist, selection):
        tree = uproot.open(filein)[treename]
        df = tree.pandas.df(branches=varlist)
        if selection is not None:
            df = df.query(selection)
        df.to_pickle(fileout)

    def unpack(self, file_index):
        self.unpack_module(self.l_root[file_index], self.l_reco[file_index], self.n_treereco, self.v_all, self.s_reco_unp)
        self.unpack_module(self.l_root[file_index], self.l_evt[file_index], self.n_treeevt, self.v_evt, self.s_evt_unp)
        if (self.mcordata == "mc"):
            self.unpack_module(self.l_root[file_index], self.l_gen[file_index], self.n_treegen, self.v_gen, self.s_gen_unp)

    def skim(self, filein, fileout):
        df = pickle.load(open(filein, "rb"))
        df = df.query(self.s_reco_skim)
        df.to_pickle(fileout)

    def parallelizer(self, function, argument_list):
        chunks = [argument_list[x:x+self.maxperchunk] for x in range(0, len(argument_list), self.maxperchunk)]
        for chunk in chunks:
            pool = mp.Pool(self.maxprocess)
            _ = [pool.apply(function,args=chunk[i]) for i in range(len(chunk))]
            pool.close()

    def unpackparallel(self):
        self.buildlistpkl()
        arguments = [(i,) for i in range(len(self.l_root))]
        self.parallelizer(self.unpack, arguments)

    def run(self):
        print(self.activateunpack)
        if self.activateunpack:
            self.unpackparallel()

    # pylint: disable=too-many-nested-blocks
    def list_files_dir_lev2(self, main_dir, outdir, filenameinput, filenameoutput):
        list_subdir0 = os.listdir(main_dir)
        listfilespath = list()
        listfilesflatout = list()
        for subdir0 in list_subdir0:
            subdir0full = os.path.join(main_dir, subdir0)
            outdir0full = os.path.join(outdir, subdir0)
            if not os.path.exists(outdir0full) and os.path.isdir(subdir0full):
                os.makedirs(outdir0full)
            if os.path.isdir(subdir0full):
                list_subdir1 = os.listdir(subdir0full)
                for subdir1 in list_subdir1:
                    subdir1full = os.path.join(subdir0full, subdir1)
                    outdir1full = os.path.join(outdir0full, subdir1)
                    if not os.path.exists(outdir1full) and os.path.isdir(subdir1full):
                        os.makedirs(outdir1full)
                    if os.path.isdir(subdir1full):
                        list_files_ = os.listdir(subdir1full)
                        for myfile in list_files_:
                            filefull = os.path.join(subdir1full, myfile)
                            filefullout = os.path.join(outdir1full, myfile)
                            filefullout = filefullout.replace(filenameinput, \
                                                              filenameoutput)
                            if os.path.isfile(filefull) and \
                            myfile == filenameinput:
                                listfilespath.append(filefull)
                                listfilesflatout.append(filefullout)
        return listfilespath, listfilesflatout

#     def skimmer(self):
#         arguments = [(self.l_pkl[i], self.l_pklsk[i]) for i in range(len(self.l_pklsk))]
#         self.parallelizer(self.skim, arguments)

