#!/usr/bin/env python

import argparse
import MDAnalysis as mda
import numpy as np
from typing import Tuple, List, Dict
from MDAnalysis.analysis import align, rms
import matplotlib.pyplot as plt
from termcolor import colored

class RMSDCalculator:
    def __init__(self):
        """Initialize RMSDCalculator."""
        pass
        
    def load_structures(self, pdb1: str, pdb2: str) -> Tuple[mda.Universe, mda.Universe]:
        """Load two PDB structures into MDAnalysis universes."""
        try:
            u1 = mda.Universe(pdb1)
            u2 = mda.Universe(pdb2)
            return u1, u2
        except Exception as e:
            raise Exception(f"Error loading PDB files: {str(e)}")
    
    def parse_residue_range(self, resid_range: str) -> Tuple[int, int]:
        """Parse residue range string 'xxx-xxx' into start and end residue numbers."""
        start, end = map(int, resid_range.split('-'))
        return start, end
    
    def get_selection_string(self, resid_range: str, mode: str) -> str:
        """Create MDAnalysis selection string based on residue range and mode."""
        start, end = self.parse_residue_range(resid_range)
        
        base_selection = f"resid {start}:{end}"
        if mode == 'c':
            return f"{base_selection} and name CA"
        elif mode in ['a', 'r']:
            return base_selection
            
        raise ValueError(f"Invalid mode: {mode}")
    
    def calculate_per_atom_rmsd(self, 
                              mobile_atoms: mda.AtomGroup, 
                              ref_atoms: mda.AtomGroup) -> Dict[str, float]:
        """Calculate RMSD for each atom."""
        rmsd_dict = {}
        for i, (mobile_atom, ref_atom) in enumerate(zip(mobile_atoms, ref_atoms)):
            dist = np.linalg.norm(mobile_atom.position - ref_atom.position)
            atom_name = f"{mobile_atom.resname}{mobile_atom.resid}-{mobile_atom.name}"
            rmsd_dict[atom_name] = dist
        return rmsd_dict
    
    def calculate_per_residue_rmsd(self, 
                                 mobile_atoms: mda.AtomGroup, 
                                 ref_atoms: mda.AtomGroup) -> Dict[str, float]:
        """Calculate RMSD for each residue."""
        rmsd_dict = {}
        for resid in np.unique(mobile_atoms.resids):
            mobile_res = mobile_atoms.select_atoms(f'resid {resid}')
            ref_res = ref_atoms.select_atoms(f'resid {resid}')
            
            if len(mobile_res) == len(ref_res):
                rmsd = rms.rmsd(mobile_res.positions, ref_res.positions, superposition=False)
                rmsd_dict[f"{mobile_res.residues[0].resname}{resid}"] = rmsd
        return rmsd_dict

    def plot_rmsd_values(self, rmsd_dict: Dict[str, float], mode: str, title: str, file_name: str):
        """Plot RMSD values as a line plot with improved readability."""
        # Create figure with appropriate size
        plt.figure(figsize=(15, 8))
        
        x = range(len(rmsd_dict))
        labels = list(rmsd_dict.keys())
        values = list(rmsd_dict.values())
        
        # Create the line plot with markers
        plt.plot(x, values, '-o', markersize=4, linewidth=1.5, alpha=0.7)
        
        # Determine label placement strategy based on number of data points
        n_labels = len(labels)
        
        if n_labels > 30:
            # For large datasets, show fewer labels
            step = max(1, n_labels // 20)  # Show ~20 labels
            shown_positions = x[::step]
            shown_labels = labels[::step]
            
            plt.xticks(shown_positions, shown_labels, rotation=45, ha='right')
        else:
            # For smaller datasets, show all labels
            plt.xticks(x, labels, rotation=45, ha='right')
        
        # Adjust layout
        plt.xlabel('Atom/Residue ID' if mode in ['a', 'r'] else 'CA ID')
        plt.ylabel('RMSD (Å)')
        plt.title(title)
        
        # Add mean line
        mean_rmsd = np.mean(values)
        plt.axhline(y=mean_rmsd, color='r', linestyle='--', 
                   label=f'Mean RMSD: {mean_rmsd:.3f} Å')
        
        # Add min/max annotations
        min_val = min(values)
        max_val = max(values)
        min_idx = values.index(min_val)
        max_idx = values.index(max_val)
        
        plt.annotate(f'Min: {min_val:.2f} Å\n{labels[min_idx]}', 
                    xy=(min_idx, min_val), xytext=(10, 10),
                    textcoords='offset points', ha='left',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.annotate(f'Max: {max_val:.2f} Å\n{labels[max_idx]}', 
                    xy=(max_idx, max_val), xytext=(10, -10),
                    textcoords='offset points', ha='left',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.legend()
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Adjust spacing and margins
        plt.subplots_adjust(bottom=0.2)  # Make room for labels
        plt.margins(x=0.02)  # Slight horizontal margins
        
        # Save plot with high DPI
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def align_and_calculate_rmsd(self, 
                               mobile: mda.Universe, 
                               reference: mda.Universe,
                               mobile_selection: str,
                               ref_selection: str,
                               mode: str) -> Tuple[float, Dict[str, float]]:
        """
        Align structures and calculate RMSD.
        Returns overall RMSD and per-atom/residue RMSD values.
        """
        # Create atom groups for alignment and RMSD calculation
        mobile_atoms = mobile.select_atoms(mobile_selection)
        ref_atoms = reference.select_atoms(ref_selection)
        
        if len(mobile_atoms) != len(ref_atoms):
            raise ValueError(f"Number of selected atoms differs: mobile={len(mobile_atoms)}, reference={len(ref_atoms)}")
        
        # Perform alignment
        align.alignto(mobile, reference, 
                     select=(mobile_selection, ref_selection),
                     weights="mass")
        
        # Calculate overall RMSD after alignment
        overall_rmsd = rms.rmsd(mobile_atoms.positions,
                              ref_atoms.positions,
                              superposition=False)
        
        # Calculate per-atom or per-residue RMSD
        if mode == 'a':
            rmsd_dict = self.calculate_per_atom_rmsd(mobile_atoms, ref_atoms)
        elif mode == 'r' or mode == 'c':
            rmsd_dict = self.calculate_per_residue_rmsd(mobile_atoms, ref_atoms)
        
        return overall_rmsd, rmsd_dict

def main():
    parser = argparse.ArgumentParser(description='Calculate RMSD between aligned PDB structure selections using MDAnalysis')
    parser.add_argument('pdb1', help='First PDB file')
    parser.add_argument('-s1', required=True, help='Residue selection for first PDB (format: xxx-xxx)')
    parser.add_argument('pdb2', help='Second PDB file')
    parser.add_argument('-s2', required=True, help='Residue selection for second PDB (format: xxx-xxx)')
    parser.add_argument('-m', choices=['a', 'r', 'c'], required=False, default='c', 
                        help='RMSD mode: a (all atoms), r (per residue), c (CA only)')
    parser.add_argument('-o', required=False, default='rmsd_plot.png', help='name of outputfile for RMSD plot. Default \'rmsd_plot.png\'')
    
    args = parser.parse_args()
    
    calculator = RMSDCalculator()
    
    try:
        # Load structures
        u1, u2 = calculator.load_structures(args.pdb1, args.pdb2)
        
        # Create selection strings
        sel1 = calculator.get_selection_string(args.s1, args.m)
        sel2 = calculator.get_selection_string(args.s2, args.m)

        # Output file name for RMSD plot
        #file_name = args.o
        
        # Print selection information
        atoms1 = u1.select_atoms(sel1)
        atoms2 = u2.select_atoms(sel2)
        print("\nStructure Analysis:")
        print(f"Structure 1: {args.pdb1}")
        print(f"Selection 1: {sel1}")
        print(f"Atoms selected: {len(atoms1)}")
        print(f"\nStructure 2: {args.pdb2}")
        print(f"Selection 2: {sel2}")
        print(f"Atoms selected: {len(atoms2)}")
        
        # Perform alignment and RMSD calculation
        overall_rmsd, rmsd_dict = calculator.align_and_calculate_rmsd(
            u1, u2, sel1, sel2, args.m
        )
        
        # Print results
        mode_desc = {'a': 'all atoms', 'r': 'per residue', 'c': 'CA only'}
        print(f"\nResults:")
        print(colored(f"Overall RMSD after alignment ({mode_desc[args.m]}): {overall_rmsd:.3f} Å", attrs=['bold']))
        print(f"Number of points compared: {len(atoms1)}")
        
        # Plot RMSD values
        title = f"RMSD Distribution ({mode_desc[args.m]})"
        calculator.plot_rmsd_values(rmsd_dict, args.m, title, args.o)
        print(f"\nRMSD plot has been saved as {args.o}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())
