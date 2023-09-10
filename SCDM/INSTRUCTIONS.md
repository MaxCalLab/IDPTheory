# How to use this code

## To calculate SCDM, you should open:

* `SCDMmap` - CPU based calculation, using sequence loaded from CSV file. Also has basic plotter.


### You should modify the code to specify:

* `mat_type` - Type of matrix to calculate, either 'normal' or 'low salt'
* `afile` - CSV file where sequences are kept (containing a header row, followed by reference names and amino acid sequences)
* `head_name` & `head_seq` - Heading labels correponding to the CSV file, for 'name' and 'sequence'
* `seqname` - Name of the sequence of interest (as listed in the CSV file)
* `seq_suf` - (if saving) Extra string 'suffix' that is appended to the output filename
* `SaveDir` - (Optional) Directory for saving results, if desired
* `PLOT` - (Optional) True/False indicating whether you want a plot shown. Will be saved to `SaveDir`, if specified.
    * `symmetric_bar` - True/False for symmetric colorbar values when plotting.
    * `clr` - Name of selected colormap, when plotting, a few options shown as comment in code.
    * `interp` - Name of selected interpolation to smoothen appearance, some options shown as comment in code.

### Generally, you should not need to modify any code below the settings!

### Run: `python SCDMmap.py`

## Note: long sequences (1000+) can take some time! (maybe several hours)