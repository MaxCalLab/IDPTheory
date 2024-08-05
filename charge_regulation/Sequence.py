##  Mike Phillips, 6/9/2022
##  * Sequence Class Definition *
##  Simple and convenient structure for loading a protein sequence from file.
##  - store the raw sequence and translated charges
##  - calculate characterization quantities: length, fractions of +/-, SCD, possibly asymmetric SCD
##  - simple accessors and information table
##  - option to include pH effect on charge, with standard pKa values and choice of exclusions
##  ** cleaned / updated:  4/16/2024  [for working well with various single-chain models]

import pandas as pd
import numpy as np

# GLOBAL charge properties
Rcharge = 1
Kcharge = 1
Echarge = -1
Dcharge = -1
Hcharge = 0
Ccharge = 0
Ycharge = 0
Xcharge = -2        # X=fictitious (e.g. phosphorylation)
# GLOBAL dictionary: effective charge of each residue (complete ionization / ignoring pH)
amino_charges = {"R":Rcharge, "K":Kcharge, "D":Dcharge, "E":Echarge, "X":Xcharge, \
          "H":Hcharge, "C":Ccharge,"Y":Ycharge, "G":0, "Q":0, "N":0, \
          "S":0, "F":0, "A":0, "I":0, "L":0, "M":0, "P":0, "T":0, "W":0, "V":0, "B":0, "J":0, "O":0, "U":0, "Z":0}

# GLOBAL dictionary: pKa values for each amino acid side-chain
pKas = {"R":12.3, "K":10.5, "D":3.5, "E":4.2, "H":6.6, "C":6.8, "Y":10.3, "Nterm":7.7, "Cterm":3.3}     # cf. Pace et al, JBC 2009

# GLOBAL definitions of residue classes; stated below such that each of the 20 amino acids fall into distinct classifications
#   > those not listed below are presumed to be charged, i.e. (E D R K), or cf. 'charges' dictionary above
hydrophobics = ("A", "I", "L", "M", "V", "P")   # consider also: "F", "W"
aromatics = ("F", "W", "Y")
polars = ("S", "T", "C", "N", "Q", "H", "G")    # consider also: "Y", "W"


#####   Class Defnition : object for loading, holding, and inspecting a given sequence  #####
class Sequence:
    def __init__(self, name="", alias="", aminos=None, info=True, \
                    file="./example_sequences/SVseqs.csv", headName="NAME", headSeq="SEQUENCE"):
        # load immediately if given, print info if desired
        if name:
            pq = self.load(seqName=name, seqAminos=aminos, seqFile=file, hName=headName, hSeq=headSeq, seqAlias=alias)
            if pq and info:
                self.info()
    #   #   #   #   #   #   #

    #   load sequence from file; save name, list of residue charges, key info
    def load(self, seqName="sv1", seqAlias="", seqAminos=None, \
                seqFile="./example_sequences/SVseqs.csv", hName="NAME", hSeq="SEQUENCE"):
        # direct input of aminos
        if seqAminos:
            self.aminos = seqAminos
        # read csv file, extract relevant sequence
        else:
            sheet = pd.read_csv(seqFile)
            if hName not in sheet:
                print("\nNAME HEADING NOT FOUND - heading '{}' was not found in file '{}'.\n".format(hName,seqFile))
                raise NameError
                return
            if hSeq not in sheet:
                print("\nSEQUENCE HEADING NOT FOUND - heading '{}' was not found in file '{}'.\n".format(hSeq,seqFile))
                raise NameError
                return
            name_col = sheet[hName].astype(str)
            ind = (name_col == seqName)
            if not any(ind):
                print("\nSEQUENCE NOT FOUND - seq. '{}' was not found in file '{}'.\n".format(seqName,seqFile))
                raise NameError
                return
            self.aminos = sheet[hSeq][ind].iloc[0]  # store raw sequence (string of amino acid letters)
        self.charges = self.translate()     # store charge sequence list (completely ionized charges, or using pH & pKa)
        self.file = seqFile         # store file for reference
        (self.headName, self.headSeq) = hName, hSeq     # store file headings for reference
        self.seqName = seqName      # store sequence name - as reference to spreadsheet file
        if seqAlias:
            self.seqAlias = seqAlias    # store alternative name - display, plotting
        else:
            self.seqAlias = seqName   # use actual name, if no alias given
        self.characterize()     # store other values characterizing the sequence
        return self.charges

    #   print sequence information
    def info(self, showSeq=False, showSCD=True, detailSCD=False, showSCD_ls=False):
        print("_____   "*5)
        print("\nSELECTED SEQUENCE:\t'{}'  ('{}')\n".format(self.seqAlias, self.seqName))
        if showSeq:
            print("{}\n".format(self.aminos))
            print("{}\n".format(self.charges))
        print("Sequence values:")
        ALLVALS = ("N","Np","Nm","fracp","fracm","frac_hp","frac_ar","frac_pol")
        for val in ALLVALS:
            print("\t{:8}\t{:1.4g}".format(val, eval("self." + val)))
        if showSCD or detailSCD:
            print("\t{:8}\t{:1.4g}".format("SCD", self.SCD))
            if detailSCD:
                print("\t    {:12}   \t{:4.4g}".format("SCD++", self.scd_func(term="++")))
                print("\t    {:12}   \t{:4.4g}".format("SCD--", self.scd_func(term="--")))
                print("\t    {:12}   \t{:4.4g}".format("SCD+-", self.scd_func(term="+-")))
                print("\t    {:12}   \t{:4.4g}".format("SCD(++)-(--)", (self.scd_func(term="++")-self.scd_func(term="--"))))
        if showSCD_ls:
            print("\t{:8}\t{:4.4g}".format("SCD_ls", self.scd_func(low_salt=True)))
        print("_____   "*5 + "\n")
        return

    # characterize sequence with several important quantities
    def characterize(self):
        pseq = np.asarray(self.charges)
        self.N = len(pseq)          # length of polymer chain (residues)
        self.Np = pseq[(pseq > 0)].sum()    # charge of positive residues
        self.Nm = - pseq[(pseq < 0)].sum()  # charge of negative residues
        self.fracp, self.fracm = self.Np/self.N, self.Nm/self.N     # positive & negative fractions (NCPR given by fracp-fracm, FCR by fracp+fracm)
        self.Nhp = self.count_hydro()       # count of hydrophobics
        self.Nar = self.count_aroma()       # count of aromatics
        self.Npol = self.count_polar()      # count of polars
        self.Npro = self.aminos.count("P")  # count of prolines
        self.frac_hp, self.frac_ar, self.frac_pol, self.frac_pro = self.Nhp/self.N, self.Nar/self.N, self.Npol/self.N, self.Npro/self.N
        self.SCD = self.scd_func()  # store value of 'sequence charge decoration' metric
        return

    # translate given character sequence to list/tuple
    def translate(self):
        lst = [amino_charges[c] for c in self.aminos]
        return tuple(lst)

    # use specified pH value to obtain charge sequence, with optional exclusions
    def PHtranslate(self, pH=7, pKexclude=('Nterm','Cterm')):
        neg = ("D", "E", "Y", "C", "Cterm")      # aminos with acidic side-chains -> can become negatively charged
        pos = ("K", "R", "H", "Nterm")           # aminos with basic side-chains -> can become positively charged
        # grab charge & character sequences
        aminoseq = list(self.aminos)
        aminoseq = ["Nterm"] + aminoseq + ["Cterm"]     # add terminals at begin & end
        # initialize charge sequence
        chargeseq = []
        for c in aminoseq:
            if c in neg:
                q = -1
            elif c in pos:
                q = 1
            else:
                chargeseq.append(0)
                continue
            if c in pKexclude:
                chargeseq.append(0)
                continue
            else:
                div = 1 + np.power(10, (q*(pH-pKas[c])))
                chargeseq.append( q / div )
        chargeseq[0] = chargeseq[1] + chargeseq.pop(0)      # combine Nterminal charge with first residue charge
        chargeseq[-1] = chargeseq[-2] + chargeseq.pop(-1)   # combine Cterminal charge with last residue charge
        return tuple(chargeseq)

    #   function for 'structural charge' : total charge, accounting for pH and side-chain pKa values
    def strCharge(self, pH=7, pKexclude=('Nterm','Cterm'), show=True):
        charge = sum(self.PHtranslate(pH, pKexclude))
        if show:
            print("\nStructural Charge of '{}'\t@ pH={:1.1f}\t[neglecting {}]:".format(self.seqName, pH, str(pKexclude)))
            print("\tqstr = {:3.3g}\n".format(charge))
        return charge

    # count number of hydrophobics (compositional only, no hydrophobic metric)
    def count_hydro(self):
        csq = np.asarray(list(self.aminos))
        tot = 0
        for c in hydrophobics:
            tot += (csq == c).sum()
        return tot

    # count number of aromatics (compositional only)
    def count_aroma(self):
        csq = np.asarray(list(self.aminos))
        tot = 0
        for c in aromatics:
            tot += (csq == c).sum()
        return tot

    # count number of polars (compositional only)
    def count_polar(self):
        csq = np.asarray(list(self.aminos))
        tot = 0
        for c in polars:
            tot += (csq == c).sum()
        return tot

    #   detailed SCD function (for asymmetry, low-salt)
    def scd_func(self, term="all", low_salt=False):
        tot = 0
        for m in range(self.N):
            for n in range(m):
                qm = self.charges[m]
                qn = self.charges[n]
                if term == "++":
                    if qm < 0 or qn < 0:
                        continue
                elif term == "--":
                    if qm > 0 or qn > 0:
                        continue
                elif term in ("+-", "-+"):
                    if round(np.sign(qm)) == round(np.sign(qn)):
                        continue
                # use power = 1 for 'low_salt', power = 0.5 for regular
                if low_salt:
                    tot += qm * qn * (m-n)
                else:
                    tot += qm * qn * np.power(m-n,0.5)
        return (tot/self.N)

    #   copy -- make identical copy for future reference (independent of original)
    def copy(self, newAlias=None, newInfo=False):
        # handle any new alias
        if not newAlias:
            newAlias = self.seqAlias
        # create new object
        ob = Sequence(name=self.seqName, alias=newAlias, aminos=self.aminos, info=newInfo, \
                        file=self.file, headName=self.headName, headSeq=self.headSeq)
        if ob.charges != self.charges:
            ob.charges = self.charges
            ob.characterize()
        return ob

#####   #####   #####   #####   #####   #####   #####   #####   #####   #####   #####

