"""
Optimized Cython code to calculate distances
taking into account periodic boundary conditions.
"""

import cython
from cython.parallel import prange
cimport cython

import numpy as np
cimport numpy as np

# Define numpy array types
DTYPE = np.float
INT = np.int
ctypedef np.float_t DTYPE_t
ctypedef np.int_t INT_t

# Self-contained distance calculation function to use safely in parallel loop.
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef (int, int, float) frame_mindist(DTYPE_t [:, :] xyz, DTYPE_t [:, :] shift) nogil:
    """Returns the minimum periodic distance (excl self) in a sim. frame.

    Arguments
    ---------
    xyz (np.ndarray) : 2D numpy array with positions per atom
    shifts (np.ndarray) : 2D numpy array with displacement vectors

    Returns
    -------
    tuple : atom i, atom j, sq distance between i and j
    """

    cdef DTYPE_t dvec[3]
    cdef DTYPE_t pbc_dvec[3]
    # cdef DTYPE_t [:] dvec = np.zeros([3])  # xi-xj, .., vecs
    # cdef DTYPE_t [:] pbc_dvec = np.zeros([3])

    cdef Py_ssize_t i, j, s  # indices
    cdef Py_ssize_t na = xyz.shape[0]  # number of atoms
    cdef Py_ssize_t nshift = shift.shape[0]
    cdef INT_t min_i, min_j

    cdef DTYPE_t sq_dij  # sq distance
    cdef DTYPE_t min_dij = 99.9  # initialize min distance to large value

    for i in range(na):
        for j in range(i+1, na):

            dvec[0] = xyz[i][0] -xyz[j][0]
            dvec[1] = xyz[i][1] -xyz[j][1]
            dvec[2] = xyz[i][2] -xyz[j][2]

            for s in range(nshift):
                pbc_dvec[0] = dvec[0] + shift[s][0]
                pbc_dvec[1] = dvec[1] + shift[s][1]
                pbc_dvec[2] = dvec[2] + shift[s][2]

                sq_dij = pbc_dvec[0]*pbc_dvec[0] + \
                         pbc_dvec[1]*pbc_dvec[1] + \
                         pbc_dvec[2]*pbc_dvec[2]

                if sq_dij <= min_dij:
                    min_dij = sq_dij
                    min_i = <INT_t>i
                    min_j = <INT_t>j

    return (min_i, min_j, min_dij)

# Main function iterating over frames
@cython.boundscheck(False)
@cython.wraparound(False)
def pbc_mindist_parallel(DTYPE_t [:, :, :] xyz, DTYPE_t [:, :, :] bvec):
    """Calculates the minimum periodic distance between two atoms in the trajectory.

    Calculates the displacement vectors for each of the 26 (27 - self)
    periodic images (assuming 3D PBCs) and iteratively adds these to the
    distance calculated for the self image. 

    Outputs the minimum distance found in all pairs of atoms in all frames.

    Port of the algorithm implemented in g_mindist (GROMACS).

    Arguments
    ---------
    xyz (np.ndarray) : NxMx3 array with simulation data (frame/atom/pos)
    bvec (np.ndarray) : Nx3x3 array with box vectors (frame, box vector)

    Returns
    -------
    info (np.array) : atom i, atom j, frame k.
    min_dists (np.array) : distance dij
    """

    cdef Py_ssize_t nf = xyz.shape[0]  # number of frames

    cdef Py_ssize_t nshift, sx, sy, sz, dim, f  # indices
    cdef DTYPE_t [:, :, :] shift = np.zeros([nf, 26, 3], dtype=DTYPE)

    # Calculate displacement vectors (shifts)
    nshift = 0
    for sz in range(-1, 2):
        for sy in range(-1, 2):
            for sx in range(-1, 2):
                if sx or sy or sz:  # not self-image
                    for dim in range(3):  # x, y, z
                        for f in range(nf):
                            shift[f][nshift][dim] = bvec[f][0][dim]*sx + \
                                                    bvec[f][1][dim]*sy + \
                                                    bvec[f][2][dim]*sz
                    nshift += 1

    # Calculate distances
    cdef INT_t [:, :] info_v = np.zeros((nf, 2), dtype=INT)
    cdef DTYPE_t [:] min_dists_v = np.zeros((nf,), dtype=DTYPE)

    cdef INT_t f_min_i, f_min_j
    cdef DTYPE_t f_min_dij

    for f in prange(nf, nogil=True):
        f_min_i, f_min_j, f_min_dij = frame_mindist(xyz[f, :, :], shift[f, :, :])
        info_v[f][0] = f_min_i
        info_v[f][1] = f_min_j
        min_dists_v[f] = f_min_dij

    # Convert memviews back to full numpy arrays
    # so we can use np. functions
    info = np.asarray(info_v)
    min_dists = np.asarray(min_dists_v)

    return (info, min_dists)

def pbc_mindist_serial(DTYPE_t [:, :, :] xyz, DTYPE_t [:, :, :] bvec):
    """Calculates the minimum periodic distance between two atoms in the trajectory.

    Calculates the displacement vectors for each of the 26 (27 - self)
    periodic images (assuming 3D PBCs) and iteratively adds these to the
    distance calculated for the self image. 

    Outputs the minimum distance found in all pairs of atoms in all frames.

    Port of the algorithm implemented in g_mindist (GROMACS).

    Arguments
    ---------
    xyz (np.ndarray) : NxMx3 array with simulation data (frame/atom/pos)
    bvec (np.ndarray) : Nx3x3 array with box vectors (frame, box vector)

    Returns
    -------
    info (np.array) : atom i, atom j, frame k.
    min_dists (np.array) : distance dij
    """

    cdef Py_ssize_t nf = xyz.shape[0]  # number of frames

    cdef Py_ssize_t nshift, sx, sy, sz, dim, f  # indices
    cdef DTYPE_t [:, :, :] shift = np.zeros([nf, 26, 3], dtype=DTYPE)

    # Calculate displacement vectors (shifts)
    nshift = 0
    for sz in range(-1, 2):
        for sy in range(-1, 2):
            for sx in range(-1, 2):
                if sx or sy or sz:  # not self-image
                    for dim in range(3):  # x, y, z
                        for f in range(nf):
                            shift[f][nshift][dim] = bvec[f][0][dim]*sx + \
                                                    bvec[f][1][dim]*sy + \
                                                    bvec[f][2][dim]*sz
                    nshift += 1

    # Calculate distances
    cdef INT_t [:, :] info_v = np.zeros((nf, 2), dtype=INT)
    cdef DTYPE_t [:] min_dists_v = np.zeros((nf,), dtype=DTYPE)

    cdef INT_t f_min_i, f_min_j
    cdef DTYPE_t f_min_dij

    for f in range(nf):
        f_min_i, f_min_j, f_min_dij = frame_mindist(xyz[f, :, :], shift[f, :, :])
        info_v[f][0] = f_min_i
        info_v[f][1] = f_min_j
        min_dists_v[f] = f_min_dij

    # Convert memviews back to full numpy arrays
    # so we can use np. functions
    info = np.asarray(info_v)
    min_dists = np.asarray(min_dists_v)

    return (info, min_dists)