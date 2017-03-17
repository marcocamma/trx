# xray

Different utilities to work with time resolved solution/powder scattering data
They should be extended to absorption soon

The idea is an hierarchical structure of modules

> user → beamline → workhorses

## Workhorses
azav.py:  uses pyFAI to convert images into 1d curves
dataReduction.py: works on the normalization, referencing and averaging.

Both modules uses some common utilities in utils.py including file based storage for persistency and convenience (numpy or hdf5 based);
Indeed the data_storage class wraps around dictionaries allowing accessing with .attribute syntax.
In utils plotfuncitons are presents

## beamline
this step is needed to read the diagnostic information (time delay, monitors, etc). An example for current id9 data collection macros is given (id9.py). Essentially it means creating a dictionary where each key will be added to the main data dictionary (that has keys like q, data, files, etc.)

## user
at the user level, a rather simple macro like the one provided as an example (example_main_tiox.py) should be sufficient for most needs.


