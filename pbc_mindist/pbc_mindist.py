#!/usr/bin/env python

# Copyright 2019 João Pedro Rodrigues
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# 3rd party libraries
import mdtraj as md
import numpy as np

try:
    import simtk.openmm.app as app
    has_openmm = True
except ImportError:
    print('WARNING: Could not import OpenMM.', file=sys.stderr)
    print('Support for reading mmCIF files disabled.', file=sys.stderr)
    has_openmm = False

# Local imports
from .dist import pbc_mindist_parallel, pbc_mindist_serial


def parse_args():
    """Parse user input and options"""
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

    ap.add_argument('-v', '--verbose', action='store_true',
                    help='Outputs diagnostic log messages')

    sel = ap.add_mutually_exclusive_group()
    sel.add_argument('--alpha', action='store_const', dest='atomsel',
                     const='alpha',
                     help='Uses only alpha carbons')
    sel.add_argument('--backbone', action='store_const', dest='atomsel',
                     const='backbone',
                     help='Uses only protein backbone atoms')
    sel.add_argument('--heavy', action='store_const', dest='atomsel',
                     const='heavy',
                     help='Uses only heavy atoms (default)')
    sel.add_argument('--all', action='store_const', dest='atomsel',
                     const='all',
                     help='Uses all atoms (!! Might be slow !!)')

    ap.set_defaults(atomsel='heavy')
    ap.set_defaults(parallel=True)

    return ap.parse_args()


def main():
    """Main code"""

    args = parse_args()

    # Format logger
    if args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        format='[%(asctime)s] %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S')

    if args.parallel:
        dist_func = pbc_mindist_parallel
    else:
        dist_func = pbc_mindist_serial

    logging.info('Started')
    logging.info('Using:')
    logging.info(f'  trajectory: {args.trajectory}')
    logging.info(f'  topology: {args.topology}')
    logging.info(f'  parallelization: {args.parallel}')
    logging.info(f'  atom selection: {args.atomsel}')

    # Read topology
    if args.topology.endswith('.cif'):
        mol = app.PDBxFile(args.topology)
        top = md.Topology.from_openmm(mol.topology)
    else:  # assume PDB, TOP, etc
        top = args.topology

    # Read trajectory
    logging.info('Reading trajectory:')
    t = md.load(args.trajectory, top=top, stride=args.stride)

    logging.info('  no. of atoms: {}'.format(t.n_atoms))
    logging.info('  no. of frames: {}'.format(t.n_frames))

    # Select atoms
    if args.atomsel == 'alpha':
        atomsel = t.top.select('protein and name CA')
        logging.info(f'Selecting protein alpha carbons (N={len(atomsel)})')
    elif args.atomsel == 'heavy':
        atomsel = [a.index for a in t.top.atoms if
                   not a.residue.is_water and  # no solvent
                   not a.residue.name.startswith('POP') and  # no lipids
                   a.element.symbol in set(('C', 'N', 'O', 'P', 'S'))]
        logging.info(f'Selecting protein heavy atoms (N={len(atomsel)})')
    elif args.atomsel == 'backbone':
        atomsel = t.top.select('protein and backbone')
        logging.info(f'Selecting protein backbone atoms (N={len(atomsel)})')
    elif args.atomsel == 'all':
        atomsel = t.top.select('protein')
        logging.info(f'Selecting all protein atoms (N={len(atomsel)})')

    t.atom_slice(atomsel, inplace=True)

    assert t.n_atoms > 1, 'Trajectory must contain more than one atom?'

    # Reimage trajectory
    if args.reimage:
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

    # Get smallest distance index
    f = np.argmin(min_dist)
    # Return info on minimum distance (and sqrt it)
    d = np.sqrt(min_dist[f])
    i, j = info[f]

    logging.info((f'Minimum distance between periodic images is {d:6.3f} nm'
                  f' between atoms {i} and {j} at frame {f}'))


if __name__ == '__main__':
    main()
