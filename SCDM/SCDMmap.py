##  Mike Phillips, 10/30/2021
##  For checking SCDM map of given sequence
##   - slow version (CPU)
##  Edited for clarity: Sept. 2023


import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm          # for colorbar zero / normalization
from time import perf_counter
from os import path


####    IMPORTANT SETTINGS      ####

# selected matrix type : 'normal' or 'low salt'
mat_type = "normal"

# input file and column headings
afile = "./SVset.csv"    # csv file of sequences
head_name = "NAME"             # column heading for names
head_seq = "SEQUENCE"          # column heading for amino sequences

# selected protein / sequence
seqname = "as1"
seq_suf = ""            # 'suffix' label, just for output filename

# output directory (if you want to save)
SaveDir = None
#SaveDir = "./SCDM out/"

# make plot?
PLOT = True

# want symmetric colorbar?  i.e. -max..0..max  instead of  min..0..max
symmetric_bar = True

# other plot options
clr = 'seismic'         # heatmaps: 'coolwarm'  'bwr'  'seismic'
interp = None           # interpolations: 'none', 'antialiased', 'nearest', 'bilinear', 'gaussian'

#####   #####   #####   #####   #####   #####   #####


##  KEY FUNCTIONS   ##

# translate given character sequence to list/tuple
def translate(char_seq):
    # physical (charge) properties
    Hcharge = 0
    # dictionary: effective charge of each residue
    aminos = {"H":Hcharge, "R":1, "K":1, "D":-1, "E":-1, "X":-2,        # X = model for phosphorylation
              "G":0, "Q":0, "N":0, "S":0, "F":0, "Y":0, "A":0, "C":0,
              "I":0, "L":0, "M":0, "P":0, "T":0, "W":0, "V":0, "B":0, "Z":0}
    lst = []
    for c in char_seq:
        lst += [ aminos[c] ]
    return tuple(lst)

# generalized SCD - piece '0' of SCDM
def gscd(seq, i, j):
    tot = 0
    main_exp = 0.5 + exp_shift
    for m in range(j+1,i+1):
        for n in range(j,m):
            tot += seq[m]*seq[n]*( (m-n)**(main_exp) )
    return (tot)

# other parts of SCDM
def gmat1(seq, i, j):
    tot = 0
    main_exp = -1.5 + exp_shift
    for m in range(j+1,i+1):
        for n in range(0,j):
            tot += seq[m]*seq[n]*( (m-j)**(2) )*( (m-n)**(main_exp) )
    return (tot)

def gmat2(seq, i, j):
    tot = 0
    main_exp = -1.5 + exp_shift
    for m in range(i+1,N):
        for n in range(0,j):
            tot += seq[m]*seq[n]*( (i-j)**(2) )*( (m-n)**(main_exp) )
    return (tot)

def gmat3(seq, i, j):
    tot = 0
    main_exp = -1.5 + exp_shift
    for m in range(i+1,N):
        for n in range(j,i+1):
            tot += seq[m]*seq[n]*( (i-n)**(2) )*( (m-n)**(main_exp) )
    return (tot)

def mat_term(seq, i, j):
    return ( gscd(seq,i,j) + gmat1(seq,i,j) + gmat2(seq,i,j) + gmat3(seq,i,j) )


##  EVALUATION  ##

# read csv file, extract relevant sequence
full_seqname = seqname
if seq_suf:
    full_seqname += "-" + seq_suf
with open(afile, newline="") as file:
    reader = csv.reader(file, dialect="excel")
    line1 = reader.__next__()           # first line (column headings)
    name_i = line1.index(head_name)        # index of protein names
    seq_i = line1.index(head_seq)      # index of sequence
    for row in reader:
        # check if row is for selected protein
        if seqname == row[name_i]:
            pseq = translate(row[seq_i])
            break
        else:
            continue

# shift common exponent appearing in matrix calculations
if mat_type.lower()[0] == 'n':
    exp_shift = 0
    mat_lbl = ""
elif mat_type.lower()[0] == 'l':
    exp_shift = 0.5
    mat_lbl = r"$_{low\,salt}$"
else:
    print("\nWARNING:  given matrix type '%s' is not found.\n" % mat_type)

# build sequence charge decoration matrix (SCDM)
print("\n" + full_seqname.upper())
print(mat_type)
N = len(pseq)   # number of residues
scdm = np.zeros((N,N))  # initial matrix of zeros
t1 = perf_counter()     # track time to build matrix
for i in range(1,N):
    for j in range(i):
        scdm[i,j] = mat_term(pseq,i,j) / (i-j)
t2 = perf_counter()     # final time
print("TIME to build matrix:\t%2.6f" % (t2-t1))
# sanity check
print("\nSCD value (extracted & rescaled from SCDM):\t%2.5f\n" % (scdm[N-1,0]*(N-1)/N))
# absolute maximum for symmetric plotting
mat_min = scdm.min()
mat_max = scdm.max()
abs_max = max(abs(mat_min), mat_max)


# save output file : SCDM numpy array
if SaveDir:
    # output filename - protein name, suffix label, low-salt option
    outfile = "SCDMarray_" + full_seqname
    if mat_type.lower()[0] == 'l':
        outfile += "_ls"
    outfile += ".npy"
    outpath = path.join(SaveDir,outfile)
    print(f"Saving SCDM as:\t'{outpath}'\n")
    np.save(outpath, scdm)


# plot SCDM map for selected protein / sequence
if PLOT:
    ##  Plot styles
    PSTY = "seaborn-notebook"
    plt.style.use(PSTY)

    ##  Plot font sizes
    SMALL_SIZE = 10
    MEDIUM_SIZE = 13
    LARGE_SIZE = 14

    plt.rc('font', size=MEDIUM_SIZE)        # controls default text sizes
    plt.rc('axes', titlesize=LARGE_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=LARGE_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

    ##  Make Plot
    fig, ax = plt.subplots(figsize=(7,7))
    if symmetric_bar:
        cnorm = TwoSlopeNorm(0, vmin=-abs_max, vmax=abs_max)
    else:
        cnorm = TwoSlopeNorm(0)
    img = ax.imshow(scdm, cmap=clr, interpolation=interp, norm=cnorm)
    fig.colorbar(img, ax=ax, shrink=0.75)
    diag = list(range(N))
    ax.plot(diag,diag,"k-")
    ax.set_xlim(0,N-1)
    ax.set_ylim(N-1,0)
    ax.set_xlabel(r"$j$")
    ax.set_ylabel(r"$i$")
    ax.set_title(r"SCDM" + mat_lbl + " : " + full_seqname.upper())
    plt.tight_layout()
    if SaveDir:
        outfig = outfile[:-4]
        outfig += ".pdf"
        outfpath = path.join(SaveDir,outfig)
        print(f"Saving SCDM plot as:\t'{outfpath}'\n")
        fig.savefig(outfpath)
    plt.show()
    plt.close()

