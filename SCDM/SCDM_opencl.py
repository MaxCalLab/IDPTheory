##  Austin Haider
##  SCDM calculation using GPU based on pyOpenCL package


import csv
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm          # for colorbar zero / normalization
from time import perf_counter
from os import path


####    IMPORTANT SETTINGS      ####

# Example sequence : protein region called RAM1
seq = 'DDRKRRRQHGQLWFPEGFKVSEASKKKRREDLEKTVVQELTWPALLANKESQTERNDLLLLGDFKDGEPNGMALDSMHVPAGPMFRDEQDARWDQHKDQD'
#seq = None      # to load from CSV file instead

# selected protein / sequence
seqname = "RAM1"
#seqname = "as1"
seq_suf = ""            # 'suffix' label, just for output filename

# input file and column headings  (only used if above 'seq = None')
afile = "./SVset.csv"    # csv file of sequences
head_name = "NAME"             # column heading for names
head_seq = "SEQUENCE"          # column heading for amino sequences

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

SCDM_prog = 'SCDM_program.cl'


def q_list(seq):
    """
    Converts a sequence of amino acids to a list of charge assignments for each residue.
    :param seq: String of single letter amino acids
    :return: list of integers corresponding to charge of each residue
    """
    N = len(seq)
    q_temp = np.zeros([N,1])
    for n in range(0, N):
        if seq[n] == 'K' or seq[n] == 'R':
            q_temp[n] = 1.0
        elif seq[n] == 'E' or seq[n] == 'D':
            q_temp[n] = -1.0
        elif seq[n] == 'X':
            q_temp[n] = -2.0
    return q_temp


def run_program(seq, prog_path):
    """
    Compute the SCD matrix for a given amino acid sequence
    :param seq: string of one-letter amino acids
    :param prog_path: string of path to .cl file used to calculate SCD matrix
    :return: NxN array of floats representing the SCD matrix for given sequence. N is the length of the sequence.
    """

    scdm_prog = open(prog_path).read()

    ctx = cl.create_some_context()

    queue = cl.CommandQueue(ctx)
    N = len(seq)

    q_array = np.empty((N*N)).astype(np.float32)
    cl_q_array = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, q_array.nbytes)

    charge_list = q_list(seq).astype(np.float32)
    cl_charge_list = cl.array.to_device(queue, charge_list)

    program = cl.Program(ctx, scdm_prog).build()
    q_array_res = program.SCDM_ij
    q_array_res.set_scalar_arg_dtypes([np.int32, None, None])
    q_array_res(queue, (N, N), None, N, cl_q_array, cl_charge_list.data)

    queue.finish()
    cl.enqueue_copy(queue, q_array, cl_q_array)

    q_temp_arr = q_array
    SCDM_array = np.reshape(q_temp_arr, (N,N))

    for i in range(N):
        for j in range(i):
            SCDM_array[i, j] /= (i-j)
            SCDM_array[j, i] = 0.0

    return SCDM_array


##  EVALUATION  ##

# read csv file, extract relevant sequence (if not specified explicitly)
full_seqname = seqname
if seq_suf:
    full_seqname += "-" + seq_suf
if not seq:
    with open(afile, newline="") as file:
        reader = csv.reader(file, dialect="excel")
        line1 = reader.__next__()           # first line (column headings)
        name_i = line1.index(head_name)        # index of protein names
        seq_i = line1.index(head_seq)      # index of sequence
        for row in reader:
            # check if row is for selected protein
            if seqname == row[name_i]:
                seq = row[seq_i]
                break
            else:
                continue

# RUN
t1 = perf_counter()
SCDM = run_program(seq, SCDM_prog)
t2 = perf_counter()
print("TIME to build matrix:\t%2.6f" % (t2-t1))

# save output file : SCDM numpy array
if SaveDir:
    # output filename - protein name, suffix label, low-salt option
    outfile = "SCDMarray_" + full_seqname
    outfile += ".npy"
    outpath = path.join(SaveDir,outfile)
    print(f"Saving SCDM as:\t'{outpath}'\n")
    np.save(outpath, SCDM)

# plot SCDM for selected protein / sequence
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
        SCDM = np.asarray(SCDM)
        mat_min = SCDM.min()
        mat_max = SCDM.max()
        abs_max = max(abs(mat_min), mat_max)
        cnorm = TwoSlopeNorm(0, vmin=-abs_max, vmax=abs_max)
    else:
        cnorm = TwoSlopeNorm(0)
    img = ax.imshow(SCDM, cmap=clr, interpolation=interp, norm=cnorm)
    fig.colorbar(img, ax=ax, shrink=0.75)
    N = len(seq)
    diag = list(range(N))
    ax.plot(diag,diag,"k-")
    ax.set_xlim(0,N-1)
    ax.set_ylim(N-1,0)
    ax.set_xlabel(r"$j$")
    ax.set_ylabel(r"$i$")
    ax.set_title(r"SCDM : " + full_seqname.upper())
    plt.tight_layout()
    if SaveDir:
        outfig = outfile[:-4]
        outfig += ".pdf"
        outfpath = path.join(SaveDir,outfig)
        print(f"Saving SCDM plot as:\t'{outfpath}'\n")
        fig.savefig(outfpath)
    plt.show()
    plt.close()

