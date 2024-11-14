#!/usr/bin/env python

import argparse
import MDAnalysis as mda
import numpy as np
from typing import Tuple, List, Dict, Union
from MDAnalysis.analysis import align, rms
import matplotlib.pyplot as plt
#from termcolor import colored

class RMSDCalculator:
    def __init__(self):
        """Initialize RMSDCalculator."""
        pass
        
    def load_structures(self, struct1: str, struct2: str, traj1: str = None, traj2: str = None) -> Tuple[mda.Universe, mda.Universe]:
        """
        Load structures and optional trajectories into MDAnalysis universes.
        
        Parameters:
        -----------
        struct1 : str
            Path to first structure file (PDB, GRO, etc.)
        struct2 : str
            Path to second structure file
        traj1 : str, optional
            Path to trajectory file for first structure
        traj2 : str, optional
            Path to trajectory file for second structure
        """
        try:
            u1 = mda.Universe(struct1, traj1) if traj1 else mda.Universe(struct1)
            u2 = mda.Universe(struct2, traj2) if traj2 else mda.Universe(struct2)
            return u1, u2
        except Exception as e:
            raise Exception(f"Error loading structure/trajectory files: {str(e)}")
    
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

    def plot_rmsd_values(self, rmsd_data: Union[Dict[str, float], np.ndarray], 
                        mode: str, title: str, file_name: str, 
                        is_trajectory: bool = False):
        """
        Plot RMSD values as either a line plot for single structure comparison
        or a time series for trajectory analysis.
        """
        plt.figure(figsize=(15, 8))
        
        if is_trajectory:
            # Plot trajectory RMSD
            times = np.arange(len(rmsd_data))
            plt.plot(times, rmsd_data, '-', linewidth=1.5, alpha=0.7)
            plt.xlabel('Frame')
            
            # Add mean and std lines
            mean_rmsd = np.mean(rmsd_data)
            std_rmsd = np.std(rmsd_data)
            plt.axhline(y=mean_rmsd, color='r', linestyle='--',
                       label=f'Mean RMSD: {mean_rmsd:.3f} ± {std_rmsd:.3f} Å')
            
            # Add min/max annotations
            min_val = np.min(rmsd_data)
            max_val = np.max(rmsd_data)
            min_idx = np.argmin(rmsd_data)
            max_idx = np.argmax(rmsd_data)
            
            plt.annotate(f'Min: {min_val:.2f} Å\nFrame {min_idx}',
                        xy=(min_idx, min_val), xytext=(10, 10),
                        textcoords='offset points', ha='left',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            plt.annotate(f'Max: {max_val:.2f} Å\nFrame {max_idx}',
                        xy=(max_idx, max_val), xytext=(10, -10),
                        textcoords='offset points', ha='left',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        else:
            # Original single structure plotting code
            x = range(len(rmsd_data))
            labels = list(rmsd_data.keys())
            values = list(rmsd_data.values())
            
            plt.plot(x, values, '-o', markersize=4, linewidth=1.5, alpha=0.7)
            
            n_labels = len(labels)
            if n_labels > 30:
                step = max(1, n_labels // 20)
                shown_positions = x[::step]
                shown_labels = labels[::step]
                plt.xticks(shown_positions, shown_labels, rotation=45, ha='right')
            else:
                plt.xticks(x, labels, rotation=45, ha='right')
            
            plt.xlabel('Atom/Residue ID' if mode in ['a', 'r'] else 'CA ID')
            
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
        
        plt.ylabel('RMSD (Å)')
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.subplots_adjust(bottom=0.2)
        plt.margins(x=0.02)
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()
    
    def align_and_calculate_rmsd(self, 
                               mobile: mda.Universe, 
                               reference: mda.Universe,
                               mobile_selection: str,
                               ref_selection: str,
                               mode: str) -> Tuple[Union[float, np.ndarray], Union[Dict[str, float], None]]:
        """
        Align structures and calculate RMSD.
        Handles both single structure comparison and trajectory analysis.
        """
        mobile_atoms = mobile.select_atoms(mobile_selection)
        ref_atoms = reference.select_atoms(ref_selection)
        
        if len(mobile_atoms) != len(ref_atoms):
            raise ValueError(f"Number of selected atoms differs: mobile={len(mobile_atoms)}, reference={len(ref_atoms)}")
        
        # Check if we're dealing with trajectories
        if len(mobile.trajectory) > 1 or len(reference.trajectory) > 1:
            # For trajectories, we'll calculate RMSD for each frame
            rmsd_trajectory = []
            
            # Ensure reference is at first frame
            reference.trajectory[0]
            ref_coords = ref_atoms.positions
            
            # Iterate through mobile trajectory
            for ts in mobile.trajectory:
                # Align current frame to reference
                mobile_coords = mobile_atoms.positions
                _, (rot, trans) = align.rotation_matrix(mobile_coords, ref_coords)
                
                # Apply transformation to all atoms
                mobile_atoms.translate(-trans)
                mobile_atoms.rotate(rot)
                
                # Calculate RMSD for current frame
                frame_rmsd = rms.rmsd(mobile_atoms.positions, ref_coords, superposition=False)
                rmsd_trajectory.append(frame_rmsd)
            
            return np.array(rmsd_trajectory), None
        else:
            # Single structure comparison (original functionality)
            align.alignto(mobile, reference, 
                         select=(mobile_selection, ref_selection),
                         weights="mass")
            
            overall_rmsd = rms.rmsd(mobile_atoms.positions,
                                  ref_atoms.positions,
                                  superposition=False)
            
            if mode == 'a':
                rmsd_dict = self.calculate_per_atom_rmsd(mobile_atoms, ref_atoms)
            elif mode in ['r', 'c']:
                rmsd_dict = self.calculate_per_residue_rmsd(mobile_atoms, ref_atoms)
            
            return overall_rmsd, rmsd_dict

def main():
    parser = argparse.ArgumentParser(description='Calculate RMSD between structures/trajectories using MDAnalysis')
    parser.add_argument('struct1', help='First structure file (PDB, GRO, etc.)')
    parser.add_argument('-s1', required=True, help='Residue selection for first structure (format: xxx-xxx)')
    parser.add_argument('struct2', help='Second structure file')
    parser.add_argument('-s2', required=True, help='Residue selection for second structure (format: xxx-xxx)')
    parser.add_argument('-t1', required=False, help='Trajectory file for first structure')
    parser.add_argument('-t2', required=False, help='Trajectory file for second structure')
    parser.add_argument('-m', choices=['a', 'r', 'c'], required=False, default='c',
                        help='RMSD mode: a (all atoms), r (per residue), c (CA only)')
    parser.add_argument('-o', required=False, default='rmsd_plot.png',
                       help='name of output file for RMSD plot. Default \'rmsd_plot.png\'')
    
    args = parser.parse_args()
    
    calculator = RMSDCalculator()
    
    try:
        # Load structures and trajectories
        u1, u2 = calculator.load_structures(args.struct1, args.struct2, args.t1, args.t2)
        
        # Create selection strings
        sel1 = calculator.get_selection_string(args.s1, args.m)
        sel2 = calculator.get_selection_string(args.s2, args.m)
        
        # Print selection information
        atoms1 = u1.select_atoms(sel1)
        atoms2 = u2.select_atoms(sel2)
        print("\nStructure Analysis:")
        print(f"Structure 1: {args.struct1}")
        if args.t1:
            print(f"Trajectory 1: {args.t1}")
            print(f"Number of frames: {len(u1.trajectory)}")
        print(f"Selection 1: {sel1}")
        print(f"Atoms selected: {len(atoms1)}")
        
        print(f"\nStructure 2: {args.struct2}")
        if args.t2:
            print(f"Trajectory 2: {args.t2}")
            print(f"Number of frames: {len(u2.trajectory)}")
        print(f"Selection 2: {sel2}")
        print(f"Atoms selected: {len(atoms2)}")
        
        # Perform alignment and RMSD calculation
        rmsd_result, rmsd_dict = calculator.align_and_calculate_rmsd(
            u1, u2, sel1, sel2, args.m
        )
        
        # Print results
        mode_desc = {'a': 'all atoms', 'r': 'per residue', 'c': 'CA only'}
        print(f"\nResults:")
        
        is_trajectory = isinstance(rmsd_result, np.ndarray)
        if is_trajectory:
            #print(colored(f"Trajectory RMSD analysis ({mode_desc[args.m]}):", attrs=['bold']))
            print(f"Trajectory RMSD analysis ({mode_desc[args.m]}):")
            print(f"Mean RMSD: {np.mean(rmsd_result):.3f} ± {np.std(rmsd_result):.3f} Å")
            print(f"Min RMSD: {np.min(rmsd_result):.3f} Å (Frame {np.argmin(rmsd_result)})")
            print(f"Max RMSD: {np.max(rmsd_result):.3f} Å (Frame {np.argmax(rmsd_result)})")
        else:
            print(colored(f"Overall RMSD after alignment ({mode_desc[args.m]}): {rmsd_result:.3f} Å", attrs=['bold']))
        
        print(f"Number of points compared: {len(atoms1)}")
        
                # Plot RMSD values
        title = f"RMSD Distribution ({mode_desc[args.m]})"
        calculator.plot_rmsd_values(rmsd_dict if rmsd_dict else rmsd_result,
                                  args.m, title, args.o, is_trajectory)
        
        print(f"\nRMSD plot has been saved as {args.o}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
        exit(main())