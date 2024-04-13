"""
Script to process frames from a trajectory as PDB files in a directory. Identifies channel centres and can determine angle for rotation of PDBs to align each channel.
"""

from pprint import pprint
import gc
import argparse
import os
from Bio.PDB import *
from Bio import PDB
import numpy as np
import math
import csv
from numpy import pi, array
from Bio.PDB.vectors import Vector, rotmat
import pymol2 as pymol
from pymol import cmd

#outer = 166
#middle = 170
#inner = 174

def calculate_centre(list_of_coords):
    """Calculates the centre of a list of coordinates"""
    centre = np.array(list_of_coords).mean(axis=0)
    return(centre)

def parse_arguments():
    arg_parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
        )
    arg_parser.add_argument("-f", "--folder", nargs='?', help="Input folder containing pdbs")
    args = arg_parser.parse_args()
    return args


class Protein():

    def __init__(self, filename):
        self.filename = filename
        self.pdb = self.get_bio_pdb(filename)
        self.id = int(filename.split("pdb.",2)[-1]) #Gets the number from numbered pdbs (e.g. protein.pdb.2 would give 2)

    def get_bio_pdb(self, filename):
        ''' Takes a PDB filename and returns a BioPDB Structure object'''
        parser = PDB.PDBParser(QUIET=True)
        return(parser.get_structure('protein', filename))
    
    def to_origin(self, point):
        """ Translates the PDB coordinates, placing point at the origin"""
        rotation_identity_matrix = PDB.rotmat(PDB.Vector([0, 0, 0]), PDB.Vector([0, 0, 0]))
        for model in self.pdb:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom.transform(rotation_identity_matrix, -point)

class Apoferritin(Protein):
    
    def calculate_centre(self,residue):
        """Calculates the centre of a list of coordinates"""
        self.centre = np.array(list(a.cpoints[residue] for a in self.four_fold.values())).mean(axis=0)

    def __init__(self,file):
        super(Apoferritin, self).__init__(file)
        _FOUR_FOLD = [("H","I","K","P"),("G","M","A","S"),("V","E","C","B"),("X","L","R","F"),("U","T","J","W"),("O","N","D","Q")]
        _THREE_FOLD = [("S","U","B"),("T","I","G"),("A","N","C"),("E","F","D"),("J","L","K"),("O","M","H"),("P","Q","R"),("X","V","W")]
        self.four_fold = {key:FourFold(key, self) for key in _FOUR_FOLD}
        self.three_fold = {key:ThreeFold(key, self) for key in _THREE_FOLD}
        self.centre = calculate_centre(list(a.cpoints[166] for a in self.four_fold.values())) # Calculates the centre using the midpoints of all six 4-fold channels

class Pore():
    def __init__(self):
        self.z_angle = None
        self.x_angle = None
        self.y_angle = None

    def calculate_pore_centre(self, residue_number, chain_group, pdb):
        """Calculates the centre of residues surrounding a pore"""
        for model in pdb:
            coords_to_find_centre_of = []
            for chain in chain_group:
                coords_to_find_centre_of.append(model[chain][residue_number]["CA"].get_coord())
            cpoint = calculate_centre(coords_to_find_centre_of)
            return(cpoint)
    
    def get_angles_2(self):
        # Assign constants corresponding to list indices
        x,y,z=0,1,2

        # Calculate vectors
        pore_outer_point = self.cpoints[166]
        pore_inner_point = self.cpoints[174]
        pore_length = np.linalg.norm(np.array(pore_outer_point)-np.array(pore_inner_point))
        pore_x_component = pore_outer_point[x] - pore_inner_point[x]
        pore_y_component = pore_outer_point[y] - pore_inner_point[y]
        pore_z_component = pore_outer_point[z] - pore_inner_point[z]
        
        # Rotation angle around the x-axis
        z_unit_vector = pore_z_component/pore_length
        pore_y_angle = np.arccos(z_unit_vector) * 180/math.pi
        
        # Rotation around the z-axis
        pore_z_angle = math.atan2(pore_y_component,pore_x_component) * 180 /math.pi

       # PRINT FOR DEBUG AND TO MANUALLY PASTE INTO PYMOL ----       

       # print(f"pseudoatom inner, pos=[{','.join(str(a) for a in pore_inner_point)}],vdw=2.5")
       # print(f"pseudoatom outer, pos=[{','.join(str(a) for a in pore_outer_point)}],vdw=2.5")
       # print(f"rotate z, {-pore_z_angle}, origin=[{','.join(str(a) for a in pore_inner_point)}]")
       # print(f"rotate y, {-pore_y_angle}, origin=[{','.join(str(a) for a in pore_inner_point)}]")
        self.y_angle = -pore_y_angle
        self.z_angle = -pore_z_angle

    def get_angles_in_radians(self, aft):

        x,y,z=0,1,2
        point = self.cpoints[166] # Reduces 3D vector to 2D plane in certain quadrant (x,y or z,y)
        centre = aft.centre # Reduces 3D vector to 2D plane in certain quadrant (x,y or z,y)
        radius = np.linalg.norm(np.array(centre)-np.array(point))
        z_unit_vector = (centre[z] - point[z]) / radius
        angle_x = np.arccos(z_unit_vector)
        if (centre[y] < point[y]):
            angle_x = 0-(np.arccos(z_unit_vector))
        else:
            angle_x = np.arccos(z_unit_vector)
        x_unit_vector = (centre[x] - point[x]) / (centre[y] - point[y])
        angle_z = math.atan(x_unit_vector)
        self.x_angle = angle_x
        self.z_angle = angle_z

class FourFold(Pore):
    def __init__(self,pore_tuple, protein):
        self.cpoints = {residue:self.calculate_pore_centre(residue, pore_tuple, protein.pdb) for residue in [166,170,174]}
        super(FourFold, self).__init__()

class ThreeFold(Pore):
    def __init__(self,pore_tuple, protein):
        self.cpoints = {residue:self.calculate_pore_centre(residue, pore_tuple, protein.pdb) for residue in [166,170,174]}
        super(ThreeFold, self).__init__()


def get_pdb_files(folder):
    a = list(os.listdir(folder))
    a.sort(key=lambda x: list(map(int,x.split("pdb.",2)[1:2])))
    files = []
    for filename in a:
        file = os.path.join(folder,filename)
        if os.path.isfile(file) and "pdb" in filename:
            files.append(file)
    return(files)

def output_to_tsv(filename, list_to_output):
    with open(filename, 'a+') as file:
        writer=csv.writer(file, delimiter="\t",lineterminator="\n")
        for row in list_to_output:
            writer.writerow(row)

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def pymol_rotate(protein):
    list_to_output = []
    for channel_id, channel in protein.four_fold.items():
        channel.get_angles_2()
        a = pymol.PyMOL()
        a.start()
        import __main__
        __main__.pymol_argv = [ 'pymol', '-qc'] # Quiet and no GUI
        filename = "".join(channel_id)
        a.cmd.load(protein.filename,filename,format="pdb")
        a.cmd.rotate(axis = "z", angle=channel.z_angle, origin=channel.cpoints[174])
        a.cmd.rotate(axis = "y", angle=channel.y_angle, origin=channel.cpoints[174])
        a.cmd.save(filename+".pdb."+str(protein.id), filename,format="pdb")
        list_to_output.append([protein.id," ".join(str(a) for a in channel.cpoints[166])," ".join(str(a) for a in channel.cpoints[174]),filename])
        a.stop()
    return list_to_output

if __name__ == '__main__':
    args = parse_arguments()
    files = get_pdb_files(args.folder)
    with open("cpoints.dat", 'a+') as f:
        writer=csv.writer(f, delimiter="\t",lineterminator="\n") 
        for file in files:
            protein = Apoferritin(file)
            for line in pymol_rotate(protein):
                writer.writerow(line)
                print(line) 
