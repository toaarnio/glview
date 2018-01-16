#!/usr/bin/python -B

"""
A collection of simple command-line parsing functions.
"""

from __future__ import print_function as __print  # hide from help(argv)

import sys, os, glob

######################################################################################
#
#  P U B L I C   A P I
#
######################################################################################

def filenames(patterns, extensions=None, sort=False, allowAllCaps=False):
    """
    Examples:
      filenames, basenames = argv.filenames(sys.argv[1:])
      filenames, basenames = argv.filenames(sys.argv[1:], [".ppm", ".png"], sort=True)
      filenames, basenames = argv.filenames(sys.argv[1:], [".jpg"], allowAllCaps=True)
    """
    filenames = [glob.glob(filepattern) for filepattern in patterns]                # expand wildcards
    filenames = [item for sublist in filenames for item in sublist]                 # flatten nested lists
    filenames = [f for f in set(filenames) if os.path.exists(f)]                    # check file existence
    if extensions is not None:
        extensions += [e.upper() for e in extensions] if allowAllCaps else []       # jpg => [jpg, JPG]
        filenames = [f for f in filenames if os.path.splitext(f)[1] in extensions]  # filter by extension
    filenames = sorted(filenames) if sort else filenames                            # sort if requested
    basenames = [os.path.splitext(f)[0] for f in filenames]                         # strip extensions
    return filenames, basenames

def exists(argname):
    """
    Example:
      showHelp = argv.exists("--help")
    """
    if argname in sys.argv:
        argidx = sys.argv.index(argname)
        del sys.argv[argidx]
        return True
    else:
        return False

def validint(argname, default=None, validInts=None):
    """
    Example:
      numtiles = argv.validint("--split", 1, [1, 2, 3, 4])
    """
    argstr = _string(argname)
    useDefault = argstr is None
    if not useDefault:
        if not _isValid(argname, int(argstr), validInts):
            sys.exit(-1)
    return default if useDefault else int(argstr)

def exitIfAnyUnparsedOptions():
    isOptionArg = ["--" in arg for arg in sys.argv]
    if any(isOptionArg):
        argname = sys.argv[isOptionArg.index(True)]
        print("Unrecognized command-line option: %s"%(argname))
        sys.exit(-1)

######################################################################################
#
#  I N T E R N A L   F U N C T I O N S
#
######################################################################################

def _string(argname, default=None):
    if argname in sys.argv:
        argidx = sys.argv.index(argname)
        argstr = sys.argv[argidx + 1]
        del sys.argv[argidx:argidx+2]
        return argstr
    else:
        return default

def _isValid(argname, arg, validArgs=None):
    if validArgs is not None:
        if arg not in validArgs:
            print("Unrecognized value for command-line option '%s': '%s'"%(argname, arg))
            print("The available values include: %s"%str(validArgs)[1:-1])
            return False
    return True
