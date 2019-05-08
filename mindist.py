#!/usr/bin/env python

"""
Calculates the minimum pairwise distance between atoms in
different periodic images in a simulation trajectory.

Equivalent to g_mindist in GROMACS.

.2019. joaor@stanford.edu
"""

from __future__ import print_function, division

import argparse
import logging
import sys

# Format logger
logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='[%(asctime)s] %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')

# 3rd party libraries
import mdtraj as md
import numpy as np

try:
    import simtk.openmm.app as app
    has_openmm = True
except ImportError:
    logging.warning('WARNING: Could not import OpenMM.')
    logging.warning('Support for reading mmCIF files disabled.')
    has_openmm = False

# Local imports
import pbc_dist

##
# Parse user input and options
ap = argparse.ArgumentParser(description=__doc__)

# Mandatory
ap.add_argument('trajectory', help='Input trajectory file (.dcd)')
ap.add_argument('topology', help='Input topology file (.cif)')

# Options
ap.add_argument('--no-reimage', action='store_false', dest='reimage',
                help='Do *not* reimage the trajectory before calculations')
ap.add_argument('--stride', type=int, default=1,
                help='Analyze every n-th frame')

ap.add_argument('--serial', action='store_false', dest='parallel',
                help='Does not parallelize distance calculations')

sel = ap.add_mutually_exclusive_group()
sel.add_argument('--alpha_carbons', action='store_const', dest='atomsel',
                 const='alpha',
                 help='Uses only alpha carbons')
sel.add_argument('--backbone', action='store_const', dest='atomsel',
                 const='backbone',
                 help='Uses only backbone atoms')
sel.add_argument('--heavy_atoms', action='store_const', dest='atomsel',
                 const='heavy',
                 help='Uses only heavy atoms (default)')
sel.add_argument('--all_atoms', action='store_const', dest='atomsel',
                 const='all',
                 help='Uses all atoms (!! Might be slow !!)')
ap.set_defaults(atomsel='heavy')
ap.set_defaults(parallel=True)
cmd = ap.parse_args()

if cmd.parallel:
    dist_func = pbc_dist.pbc_mindist_parallel
else:
    dist_func = pbc_dist.pbc_mindist_serial

logging.info('Started')
logging.info('Using:')
logging.info(f'  trajectory: {cmd.trajectory}')
logging.info(f'  parallelization: {cmd.parallel}')

# Read topology
if cmd.topology.endswith('.cif'):
    mol = app.PDBxFile(cmd.topology)
    top = md.Topology.from_openmm(mol.topology)
else:  # assume PDB, TOP, etc
    top = cmd.topology

# Read trajectory
t = md.load(cmd.trajectory, top=top, stride=cmd.stride)

logging.info('Trajectory Details:')
logging.info('  no. of atoms: {}'.format(t.n_atoms))
logging.info('  no. of frames: {}'.format(t.n_frames))

# Select atoms
if cmd.atomsel == 'alpha':
    atomsel = t.top.select('protein and name CA')
    logging.info(f'Selecting protein alpha carbons (N={len(atomsel)})')
elif cmd.atomsel == 'heavy':
    atomsel = t.top.select('protein and not element H')
    logging.info(f'Selecting protein heavy atoms (N={len(atomsel)})')
elif cmd.atomsel == 'backbone':
    atomsel = t.top.select('protein and backbone')
    logging.info(f'Selecting protein backbone atoms (N={len(atomsel)})')
elif cmd.atomsel == 'all':
    atomsel = t.top.select('protein')
    logging.info(f'Selecting all protein atoms (N={len(atomsel)})')

t.atom_slice(atomsel, inplace=True)

assert t.n_atoms > 1, 'Trajectory contains only one atom?'

# Reimage trajectory
if cmd.reimage:
    logging.info(f'Reimaging trajectory')
    mols = t.top.find_molecules()
    if len(mols) < 2:
        logging.warning('WARNING: NUMBER OF MOLECULES IS 1')
        logging.warning('REIMAGING IS VERY LIKELY TO FAIL!')
        logging.warning('CHECK THE CONNECTIVITY OF YOUR SYSTEM')

    t.image_molecules(inplace=True, anchor_molecules=mols[:1])

# Calculate distance
logging.info(f'Calculating distances')
info, min_dist = dist_func(t.xyz.astype('float64'),
                           t.unitcell_vectors.astype('float64'))

# Get smallest distance
f = np.argmin(min_dist)
# Return info on minimum distance (and sqrt it)
d = np.sqrt(min_dist[f])
i, j = info[f]

logging.info((f'Minimum distance between periodic images is {d:6.3f} nm'
              f' between atoms {i} and {j} at frame {f}'))
