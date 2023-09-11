# How to use this code

## To calculate SCDM, you should open one of:

* `SCDMmap` - CPU based calculation, using sequence stated explicitly or from CSV file. Also has basic plotter.
* `SCDM_opencl` - GPU based calculation with pyOpenCL, otherwise very similar.


## In either case, you should modify code to specify:

* `seq` - Either explicit sequence, or 'None' to use sequence from CSV file
* `seqname` - Name for the sequence of interest (if `seq=None`: as listed in the CSV file)
* `seq_suf` - (if saving) Extra string 'suffix' that is appended to the output filename
* `afile` - (if `seq=None`) CSV file where sequences are kept (containing a header row, followed by reference names and amino acid sequences)
* `head_name` & `head_seq` - (if `seq=None`) Heading labels correponding to the CSV file, for 'name' and 'sequence'
* `mat_type` - (CPU `SCDMmap` only) Type of matrix to calculate, either 'normal' or 'low salt'
* `SaveDir` - (Optional) Directory for saving results, if desired
* `PLOT` - (Optional) True/False indicating whether you want a plot shown. Will be saved to `SaveDir`, if specified.
    * `symmetric_bar` - True/False for symmetric colorbar values when plotting.
    * `clr` - Name of selected colormap, when plotting, a few options shown as comment in code.
    * `interp` - Name of selected interpolation to smoothen appearance, some options shown as comment in code.


### Generally, you should not need to modify any code below the settings!


### Run: `python SCDMmap.py` or `python SCDM_opencl.py`


### Note: CPU `SCDMmap` is slow, long sequences (1000+) can take some time! (maybe several hours)

### Note: GPU `SCDM_opencl` requires package 'pyopencl', and hardware (GPU) that supports OpenCL. Should work with most systems (Intel, AMD, Nvidia). May not work with newer Apple systems (M1+).


## Extra notes for using GPU calculation with pyOpenCL

- Code consists of one python script (`SCDM_opencl.py`) and one OpenCL script (`SCDM_program.cl`)
- Built and tested for macOS Big Sur Version 11.7 using Python 3.7+

### USAGE:
- Place `SCDM_program.cl` and `SCDM_opencl.py` in same directory.
- Enter the protein sequence you wish to compute the SCDM of directly into 
 `SCDM_opencl.py` script and use the `run_program` function, using the protein
 sequence and path to `SCDM_program.cl` as inputs, then run the script.
- The run_program function will return an NxN numpy array (the SCDM for that protein) 
 which will be calculated using the system's GPU.
- Using the GPU to calculate SCDM in this manner will dramatically improve calculation 
 time, especially for larger protein sequences, when compared to a standard CPU calculation
 with python.
