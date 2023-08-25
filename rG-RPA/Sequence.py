##  Mike Phillips, 6/9/2022
##  * Sequence Class Definition *
##  Simple and convenient structure for loading a protein sequence from file.
##  - store the raw sequence and translated charges
##  - calculate characterization quantities: length, fractions of +/-, SCD, possibly asymmetric SCD
##  - simple accessors and information table


import csv
import numpy as np

# GLOBAL physical (charge) properties
Rcharge = 1
Kcharge = 1
Echarge = -1
Dcharge = -1
Hcharge = 0
Ccharge = 0
Ycharge = 0
Xcharge = -2        # X=fictitious (e.g. phosphorylation)
# GLOBAL dictionary: effective charge of each residue (complete ionization)
aminos = {"R":Rcharge, "K":Kcharge, "D":Dcharge, "E":Echarge, "X":Xcharge, \
          "H":Hcharge, "C":Ccharge,"Y":Ycharge, "G":0, "Q":0, "N":0, \
          "S":0, "F":0, "A":0, "B":0, "Z":0, "I":0, "L":0, "M":0, "P":0, "T":0, "W":0, "V":0}
hydrophobics = ("A","I","L","M","V","F","W","P")    # somewhat fluid
aromatics = ("F", "W", "Y")                         #

#####   Class Defnition : object for loading, holding, and inspecting a given sequence
class Sequence:
    def __init__(self, name="", alias="", info=True, \
                    file="./RG_tests.csv", headName="NAME", headSeq="SEQUENCE"):
        # load immediately if given, print info if desired
        if name:
            self.load(seqName=name, seqFile=file, hName=headName, hSeq=headSeq, seqAlias=alias)
            if info:
                self.info()
    #   #   #   #   #   #   #

    #   load sequence from file; save name, list of residue charges, key info
    def load(self, seqName="sv1", seqFile="./RG_tests.csv", \
                hName="NAME", hSeq="SEQUENCE", seqAlias=""):
        pname = seqName
        alias = seqAlias
        file = seqFile
        # read csv file, extract relevant sequence
        with open(file, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f, dialect="excel")
            line1 = reader.__next__()           # first line (column headings)
            if hName not in line1:
                print("\nNAME HEADING NOT FOUND - heading '{}' was not found in file '{}'.\n".format(hName,seqFile))
                raise NameError
                return
            if hSeq not in line1:
                print("\nSEQUENCE HEADING NOT FOUND - heading '{}' was not found in file '{}'.\n".format(hSeq,seqFile))
                raise NameError
                return
            name_i = line1.index(hName)        # index of protein names
            seq_i = line1.index(hSeq)      # index of sequence
            for row in reader:
                # check if row is for selected protein
                if pname == row[name_i]:
                    aminoseq = row[seq_i]
                    pseq = self.translate(aminoseq)
                    break
                else:
                    continue
            try:
                pseq
            except NameError:
                print("\nSEQUENCE NOT FOUND - seq. '{}' was not found in file '{}'.\n".format(seqName,seqFile))
                raise NameError
                return
        self.file = file        # store file for reference
        (self.headName, self.headSeq) = hName, hSeq     # store file headings for reference
        self.seqName = pname    # store sequence name - as reference to spreadsheet file
        if alias:
            self.seqAlias = alias   # store alternative name - display, plotting
        else:
            self.seqAlias = pname   # just use actual name, if no alias given
        self.sigma = pseq       # store charge sequence list (completely ionized charges)
        self.aminos = aminoseq  # store raw sequence (string of amino acid letters)
        self.characterize()     # store other values characterizing the sequence
        return pseq

    #   print sequence information
    def info(self, showSeq=False, showSCD=True, detailSCD=False):
        print("_____   "*5)
        print("\nSELECTED SEQUENCE:\t'{}'  ('{}')\n".format(self.seqAlias, self.seqName))
        if showSeq:
            print("{}\n".format(self.aminos))
            print("{}\n".format(self.sigma))
        print("Sequence values:")
        ALLVALS = ("N","Np","Nm","fracp","fracm","frac_hp","frac_ar")
        for val in ALLVALS:
            print("    {:7}   {:3.4g}".format(val, eval("self." + val)))
        if showSCD or detailSCD:
            print("    {:7}   {:3.4g}".format("SCD", self.SCD))
            if detailSCD:
                print("       {:5}\t{:3.4g}".format("SCD++", self.ogSCD("++")))
                print("       {:5}\t{:3.4g}".format("SCD--", self.ogSCD("--")))
                print("       {:5}\t{:3.4g}".format("SCD+-", self.ogSCD("+-")))
                print("       {:5}\t{:3.4g}".format("SCD(++)-(--)", (self.ogSCD("++")-self.ogSCD("--"))))
        print("_____   "*5)
        print("")
        return

    # characterize sequence with several important quantities
    def characterize(self):
        pseq = self.sigma
        aminoseq = self.aminos
        self.N = len(pseq)          # length of polymer chain (sum of residues) [i.e. molecule type B]
        self.Np = pseq.count(1)      # count all positive residues
        self.Nm = (pseq.count(-1) + pseq.count(-2))    # count all negative residues
        self.fracp, self.fracm = self.Np/self.N, self.Nm/self.N   # positive & negative fractions
        self.frac_hp = self.count_hydro(aminoseq)/self.N    # fraction of hydrophobic aminos
        self.frac_ar = self.count_aroma(aminoseq)/self.N    # fraction of aromatic aminos
#        self.refRange0 = np.array(range(self.N))    # full reference range
        self.refRange1 = np.array(range(1,self.N))  # save common reference range 1..N (for speed)
        self.refRange1_dif = self.N - self.refRange1    # difference from range list (0..N-1)
        self.SCD = self.scd_func()  # store value of 'sequence charge decoration' metric
        self.sumsig = self.listSig()        # list of sigma[i]*sigma[j] sums
        self.totalsig = self.sumsig[0]/self.N  # store total charge-squared, per monomer
        self.sumqoff = self.listQoff()      # list of sigma[i]+sigma[j] sums
        self.qtot = self.sumqoff[0]/2     # total charge on the chain
        self.sumsig1 = np.array(self.sumsig[1:])    # list of product sums after, first element
        self.sumqoff1 = np.array(self.sumqoff[1:])  # list of pair sums after, first element
        self.Xmat = []
#        self.Xmat = np.ones((self.N, self.N))   # placeholder for X matrix ('Xij') -- all ones by default
        return

    # translate given character sequence to list/tuple
    def translate(self, char_seq):
        lst = []
        for c in char_seq:
            lst.append( aminos[c] )
        return tuple(lst)

    # count number of hydrophobics (compositional only, no hydrophobic metric)
    def count_hydro(self, char_seq):
        tot = 0
        for c in char_seq:
            if c in hydrophobics:
                tot += 1
        return tot

    # count number of aromatics (compositional only)
    def count_aroma(self, char_seq):
        tot = 0
        for c in char_seq:
            if c in aromatics:
                tot += 1
        return tot

    #   list all sums of charge multiplications
    def listSig(self):
        sig = self.sigma
        N = self.N
        # includes charge-squared as '0' element, others offset (1 < d < N-1)
        lst = [ sum([sig[i]*sig[i+d] for i in range(N-d)]) for d in range(N) ]
        return lst

    #   list all sums of offset charge combination
    def listQoff(self):
        sig = self.sigma
        N = self.N
        # includes 2*qtot as '0' element, others offset
        lst = [ sum( [(sig[i]+sig[i+d]) for i in range(N-d)] ) for d in range(N) ]
        return lst

    # sequence charge decoration metric
    def scd_func(self):
        return self.ogSCD(mode="all")

    #   detailed SCD function (for asymmetry, etc)
    def ogSCD(self, mode="all"):
        N = self.N
        seq = self.sigma
        tot = 0
        for m in self.refRange1:
            for n in range(m):
                qm = seq[m]
                qn = seq[n]
                if mode == "++":
                    if qm < 0 or qn < 0:
                        continue
                elif mode == "--":
                    if qm > 0 or qn > 0:
                        continue
                elif mode in ("+-", "-+"):
                    if round(np.sign(qm)) == round(np.sign(qn)):
                        continue
                tot += qm * qn * ( (m-n)**(0.5) )
        return (tot/N)

    #   copy -- make identical copy for future reference (independent of original)
    def copy(self, newAlias=None, newInfo=False):
        # handle any new alias
        if not newAlias:
            newAlias = self.seqAlias
        # create new object 
        ob = Sequence(name=self.seqName, alias=newAlias, info=newInfo, \
                        file=self.file, headName=self.headName, headSeq=self.headSeq)
        return ob
#####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####
