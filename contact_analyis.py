import MDAnalysis as mda
from prolif import Fingerprint, Molecule
import rdkit
from rdkit import Chem
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import pickle
import tempfile
import os
import warnings
import sys
import traceback
from MDAnalysis.lib.distances import distance_array

warnings.filterwarnings('ignore')

"""
Methodology:
1. iterate through the trajectory
2. find residues that are within a given distance of the base selection
3. see if there might be a valid interaction
3. Categorize interactions and save
"""

class contactAnalyzer:
    def __init__(self):
        """
        Initialize contact analyzer
        
        interaction_types:
        { interaction type : [ distance_cutoff, ] }
        
        """
        # Define interaction types to analyze
        """
        self.interaction_types = {
            'close_contact':[2.5],
            'hydrophobic' : [4.0],
            'HBond' : [4.0],
            'Ionic',
            'PiStacking' : [7.5, ],
            'PiCation',
            'Metal'
        }
        self.residues = {
            'cations' : ['ARG', 'HIS', 'LYS'],
            'anions' : ['ASP', 'GLU'],
            'polar' : 
        }
        """


    def load_system(self, struct:str, traj:str = None, sel1:str = None, sel2:str = None):
        """
        Load structures for protein-ligand interaction analysis
        
        Parameters
        ----------
        struct1 : str
            Path to first structure file
        traj : str, optional
            Path to trajectory file
        sel1 : str
            Selection string for first selection
        sel2 : str
            Selection string for second selection
        """
        print("Loading system...")
        try:
            # Create MDAnalysis universe
            if traj is not None:
                u = mda.Universe(struct, traj)
            else:
                u = mda.Universe(struct)

            print("\nSystem Details:")
            if struct:
                    print(f"Structure: {struct}")
                    
            if traj:
                    print(f"Trajectory: {traj}")
                    print(f"Number of frames: {len(u.trajectory)}")
            
            return u
            
        except Exception as e:
            raise Exception(f"Error loading system: {str(e)}")

    def analyze_interactions(self, base_selection:str, universe:mda.Universe, cutoff:float, freq:int):

        try:
            # Create atom groups
            base_atoms = universe.select_atoms(base_selection)
            all_atoms = universe.atoms
            
            # Initialize results dictionary
            results = {}
            
            # Iterate through trajectory
            for ts in universe.trajectory[::freq]:
                # Get current frame number
                frame = ts.frame
                
                # Get positions of base selection
                base_positions = base_atoms.positions
                
                # Calculate distances between base selection and all atoms
                distances = distance_array(base_positions, 
                                        all_atoms.positions,
                                        box=ts.dimensions)
                
                # Find atoms within cutoff of ANY atom in base selection
                nearby_mask = (distances < cutoff).any(axis=0)
                nearby_atoms = all_atoms[nearby_mask]
                
                # Get minimum distance to base selection for each nearby atom
                #min_distances = distances[:, nearby_mask].min(axis=0)

                # Add which base atom is closest to each nearby atom
                closest_base_atoms = []
                closest_base_atom_names = []
                closest_base_residues = []
                
                for i in range(len(nearby_atoms)):
                    closest_base_idx = np.argmin(distances[:, nearby_mask][:, i])
                    closest_base_atoms.append(base_atoms[closest_base_idx].id)
                    closest_base_atom_names.append(base_atoms[closest_base_idx].name)
                    closest_base_residues.append(f"{base_atoms[closest_base_idx].resname}{base_atoms[closest_base_idx].resnum}")

                info = pd.DataFrame({
                    'res1' : closest_base_residues,
                    'res2'
                })


                
                # Create detailed information for nearby atoms
                nearby_info = pd.DataFrame({
                    'residue_id': nearby_atoms.resids,
                    'residue_name': nearby_atoms.resnames,
                    'residue_number': nearby_atoms.resnums,
                    'atom_name': nearby_atoms.names,
                    'atom_type': nearby_atoms.types,
                    'segment_id': nearby_atoms.segids,
                    'min_distance': min_distances,
                    'x': nearby_atoms.positions[:, 0],
                    'y': nearby_atoms.positions[:, 1],
                    'z': nearby_atoms.positions[:, 2]
                })
                
                # Store results for this frame
                results[frame] = {
                    'time': universe.trajectory.time,
                    'base_positions': base_positions.copy(),
                    'nearby_atoms': nearby_atoms,
                    'nearby_info': nearby_info,
                    'n_nearby': len(nearby_atoms)
                }
                
                # Add which base atom is closest to each nearby atom
                closest_base_atoms = []
                closest_base_atom_names = []
                closest_base_residues = []
                
                for i in range(len(nearby_atoms)):
                    closest_base_idx = np.argmin(distances[:, nearby_mask][:, i])
                    closest_base_atoms.append(base_atoms[closest_base_idx].id)
                    closest_base_atom_names.append(base_atoms[closest_base_idx].name)
                    closest_base_residues.append(f"{base_atoms[closest_base_idx].resname}{base_atoms[closest_base_idx].resnum}")
                
                results[frame]['nearby_info']['closest_base_atom_id'] = closest_base_atoms
                results[frame]['nearby_info']['closest_base_atom_name'] = closest_base_atom_names
                results[frame]['nearby_info']['closest_base_residue'] = closest_base_residues
                
            return results
        
        except Exception as e:
            raise Exception(f"Error analyzing interactions: {str(e)}")
        

    def classify_interaction(self,):
        self.interaction_types
        pass
            
            

def main():
    parser = argparse.ArgumentParser(description='Contact Analysis')
    
    # Required argument
    parser.add_argument('struct1', help='First structure file')
    parser.add_argument('-s1', required=True,
                       help='Selection string for base (MDAnalysis syntax)')
    # Optional arguments with stored defaults
    parser.add_argument('-t', '--trajectory', default=None,
                       help='Trajectory file (optional)')
    

    
    args = parser.parse_args()
    
    try:
        analyzer = contactAnalyzer()
        
        # Load system with flexible input options
        universe = analyzer.load_system(
            args.struct1,
            args.trajectory
        )
        
        # Analyze interactions
        results = analyzer.analyze_interactions(args.s1, universe, 5, 10)

        print(results)
        
        # Generate outputs
        #analyzer.plot_interactions(results, args.prefix)
        
        #if args.report:
        #    analyzer.write_report(results)
        
        #if args.save:
        #    analyzer.save_results(results)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())