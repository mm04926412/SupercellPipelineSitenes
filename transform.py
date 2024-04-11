from pymatgen.transformations.standard_transformations import *
from pymatgen.transformations.advanced_transformations import *
import h5py
import numpy as np
import pickle as pk
import matminer.featurizers.structure as struc_feat
from multiprocessing import Process, Pool, cpu_count
from matbench.bench import MatbenchBenchmark
import scipy as sp
import argparse
import sys
from os.path import exists
from tqdm import tqdm
class OrthorhombicSupercellTransform(AbstractTransformation):
    """
    This transformation generates a combined scaled and shear of the provided unit cell to achieve a roughly
    OrthorhomicSupercell with roughly equal side lengths. Gets more accurate the larger the difference in size between primitive and supercell
    Robust alternative to the existing cubic supercell method that guarantees the inverse matrix is not singular when rounded. 
    """

    def __init__(self, N_Atoms):
        """
        Args:
            charge_balance_sp: The desired number of atoms in the supercell
        """
        self.N_Atoms = int(N_Atoms)

    def apply_transformation(self, structure):
        """
        Applies the transformation.

        Args:
            structure: Input Structure

        Returns:
            OrthorhombicSupercell
        """

        lattice_matrix = structure.lattice.as_dict()["matrix"]

        #RQ decomposition in this context provides a scale and shear matrix (R) that maps a Orthorhombic cell of unit volume to the current lattice parameters
        R, Q = sp.linalg.rq(lattice_matrix)
        #Invert R to get the scale+shear that maps the current unit cell to the Orthorhombic cell
        R1 = np.linalg.inv(R)
        #R1 is the inverse of R, we require the inverse of the diagonal component of R1 to remove the unwanted normalization included in the rq algorithm
        R1_Diagonal = np.zeros(R1.shape)
        np.fill_diagonal(R1_Diagonal,np.diagonal(R1))
        #S is the 'ideal' normalized shearing, it is not yet suitable due to its non-integer components
        S = sp.linalg.inv(R1_Diagonal) @ R1

        #The lattice parameters of Q are the "ideal" attained by directly applying S, we compute our scaling matrix by iteratively incrementing the shortest lattice parameter on Q
        #until any further increments breach the upper atom limit. These increments on Q are used to compute the scaling component of the transformation
        start_len = len(structure)
        Sheared_cell = S @ lattice_matrix
        Sheared_abc = [np.linalg.norm(Sheared_cell[0]),np.linalg.norm(Sheared_cell[1]),np.linalg.norm(Sheared_cell[2])]
        increments = (1,1,1)
        found_transform = False
        #Iteratively increment the shortest lattice parameters until doing so brings the number of atoms above the limit
        while not found_transform:
            new_increments = list(increments) #Deep copy
            shortest = np.argmin([i*j for i,j in zip(increments,Sheared_abc)]) #Return the shortest lattice parameter of Q post scaling
            new_increments[shortest] += 1
            if np.prod(new_increments)*start_len <= self.N_Atoms: #If this increment brings the total number of atoms above the ceiling then return the transformation matrix, otherwise repeat
                increments = new_increments
            else:
                found_transform=True
        cubic_upscale_approx = np.rint(np.diag(increments) @ S) # Create combined scale and shear matrix and round the off diagonals to the nearest integer, this provides an integer approximation of the shear, larger supercells will be more precise
        structure = SupercellTransformation(scaling_matrix=cubic_upscale_approx).apply_transformation(structure) # Apply the computed integer supercell transformation
        return structure

    @property
    def inverse(self):
        """Returns: None"""
        return None

    @property
    def is_one_to_many(self):
        """Returns: False"""
        return False
