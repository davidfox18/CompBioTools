


import argparse
import MDAnalysis as mda
import numpy as np
from typing import Tuple, List, Dict, Union
from MDAnalysis.analysis import align, rms
import matplotlib.pyplot as plt
import sys
import time
import threading
import warnings
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=Warning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class RMSDCalculator:
    def __init__(self):
        """Initialize RMSDCalculator."""
        pass
    def show_progress(func):
        """
        Decorator that adds a progress bar for trajectory processing.
        Falls back to spinner for non-trajectory operations.
        """
        def wrapper(*args, **kwargs):
            self = args[0]  # Get instance reference
            
            # Check if we're dealing with a trajectory calculation
            if func.__name__.endswith('_trajectory'):
                # Get trajectory length from the first Universe object
                mobile = args[1]  # mobile Universe is the second argument
                n_frames = len(mobile.trajectory)
                
                with tqdm(total=n_frames, colour='green', desc="Processing frames", 
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} frames') as pbar:
                    # Store pbar in instance for use in the calculation functions
                    self.pbar = pbar
                    result = func(*args, **kwargs)
                    self.pbar = None
                return result
            else:
                # Use spinner for non-trajectory operations
                spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                busy = False
                abort = False

                def spinner_func():
                    nonlocal busy, abort
                    idx = 0
                    while busy and not abort:
                        sys.stdout.write(spinner[idx % len(spinner)])
                        sys.stdout.flush()
                        idx += 1
                        time.sleep(0.1)
                        sys.stdout.write("\b")
                        sys.stdout.flush()

                thread = threading.Thread(target=spinner_func)
                busy = True
                thread.start()

                try:
                    result = func(*args, **kwargs)
                except:
                    abort = True
                    raise
                finally:
                    busy = False
                    thread.join()

                return result
        return wrapper
        
    def spinner(func):
        """
        Decorator that adds a spinning indicator to a function that takes a while to run.

        Parameters:
        func (callable): The function to be decorated.

        Returns:
        callable: The decorated function.
        """
        def wrapper(*args, **kwargs):
            spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            busy = False
            abort = False

            def spinner_func():
                nonlocal busy, abort
                idx = 0
                while busy and not abort:
                    sys.stdout.write(spinner[idx % len(spinner)])
                    sys.stdout.flush()
                    idx += 1
                    time.sleep(0.1)
                    sys.stdout.write("\b")
                    sys.stdout.flush()

            thread = threading.Thread(target=spinner_func)
            busy = True
            thread.start()

            try:
                result = func(*args, **kwargs)
            except:
                abort = True
                raise
            finally:
                busy = False
                thread.join()

            return result
        return wrapper


    def load_structures(self, struct1: str, struct2: str, traj1: str = None, traj2: str = None) -> Tuple[mda.Universe, mda.Universe]:
        """Load structures and optional trajectories into MDAnalysis universes."""
        try:
            u1 = mda.Universe(struct1, traj1) if traj1 else mda.Universe(struct1)
            u2 = mda.Universe(struct2, traj2) if traj2 else mda.Universe(struct2)
            
            # Debug information
            print("\n\033[1mStructure Details:\033[0m")
            print(f"Structure 1: {struct1}")
            print(f"Total residues: {len(u1.residues)}")
            print(f"Residue numbers available: {u1.residues.resids[0]}-{u1.residues.resids[-1]}")
            
            print(f"\nStructure 2: {struct2}")
            print(f"Total residues: {len(u2.residues)}")
            print(f"Residue numbers available: {u2.residues.resids[0]}-{u2.residues.resids[-1]}")
            
            return u1, u2
        except Exception as e:
            raise Exception(f"Error loading structure/trajectory files: {str(e)}")
    
    """
    def parse_residue_range(self, resid_range: str) -> Tuple[int, int]:
        """"Parse residue range string 'xxx-xxx' into start and end residue numbers."""""
        start, end = map(int, resid_range.split('-'))
        return start, end
    """

    def get_selection_string(self, selection: str, mode: str) -> str:
        """Create MDAnalysis selection string based on selection and mode."""
        if mode == 'c':
            return f"({selection}) and name CA"
        elif mode in ['a', 'r']:
            return selection
            
        raise ValueError(f"Invalid mode: {mode}")
    
    @show_progress
    def calculate_per_atom_rmsd_trajectory(self,
                                mobile: mda.Universe,
                                reference: mda.Universe,
                                mobile_atoms: mda.AtomGroup,
                                ref_atoms: mda.AtomGroup,
                                mobile_align: mda.AtomGroup,
                                ref_align: mda.AtomGroup) -> Dict[str, float]:
        """Calculate RMSD between two trajectories for each atom."""
        rmsd_dict = {}
        atom_labels = [f"{atom.resname}{atom.resid}-{atom.name}" for atom in mobile_atoms]
        
        n_atoms = len(mobile_atoms)
        n_frames = len(mobile.trajectory)
        
        if n_atoms == 0 or n_frames == 0:
            raise ValueError("No atoms selected or empty trajectory")
            
        sum_squared_diff = np.zeros(n_atoms)
        
        for ts_idx, (ts_mobile, ts_ref) in enumerate(zip(mobile.trajectory, reference.trajectory)):
            align.alignto(mobile.atoms, reference.atoms,
                        select=(mobile_align, ref_align),
                        weights="mass")
            
            diff = mobile_atoms.positions - ref_atoms.positions
            squared_diff = np.sum(np.square(diff), axis=1)
            sum_squared_diff += squared_diff
            
            if hasattr(self, 'pbar') and self.pbar:
                self.pbar.update(1)
        
        rms_distances = np.sqrt(sum_squared_diff / n_frames)
        
        for atom_label, rms_dist in zip(atom_labels, rms_distances):
            rmsd_dict[atom_label] = rms_dist
            
        return rmsd_dict
    
    @show_progress
    def calculate_per_residue_rmsd_trajectory(self,
                                        mobile: mda.Universe,
                                        reference: mda.Universe,
                                        mobile_atoms: mda.AtomGroup,
                                        ref_atoms: mda.AtomGroup,
                                        mobile_align: mda.AtomGroup,
                                        ref_align: mda.AtomGroup) -> Dict[str, float]:
        """Calculate RMSD between two trajectories for each residue."""
        rmsd_dict = {}
        mobile_resids = np.unique(mobile_atoms.resids)
        ref_resids = np.unique(ref_atoms.resids)
        
        if len(mobile_resids) != len(ref_resids):
            raise ValueError("Number of residues differs between mobile and reference selections")
        
        residue_mapping = dict(zip(mobile_resids, ref_resids))
        residue_squared_diff = {resid: 0.0 for resid in mobile_resids}
        n_frames = len(mobile.trajectory)
        
        mobile_res_atoms = {}
        ref_res_atoms = {}
        for mobile_resid in mobile_resids:
            ref_resid = residue_mapping[mobile_resid]
            mobile_res_atoms[mobile_resid] = mobile_atoms.select_atoms(f'resid {mobile_resid}')
            ref_res_atoms[ref_resid] = ref_atoms.select_atoms(f'resid {ref_resid}')
            
            if len(mobile_res_atoms[mobile_resid]) != len(ref_res_atoms[ref_resid]):
                raise ValueError(f"Unequal number of atoms in residue pair {mobile_resid}/{ref_resid}")
        
        for ts_idx, (ts_mobile, ts_ref) in enumerate(zip(mobile.trajectory, reference.trajectory)):
            align.alignto(mobile.atoms, reference.atoms,
                        select=(mobile_align, ref_align),
                        weights="mass")
            
            for mobile_resid in mobile_resids:
                ref_resid = residue_mapping[mobile_resid]
                mob_atoms = mobile_res_atoms[mobile_resid]
                ref_atoms_res = ref_res_atoms[ref_resid]
                
                diff = mob_atoms.positions - ref_atoms_res.positions
                squared_diff = np.sum(np.square(diff))
                residue_squared_diff[mobile_resid] += squared_diff
            
            if hasattr(self, 'pbar') and self.pbar:
                self.pbar.update(1)
        
        for mobile_resid in mobile_resids:
            mobile_res = mobile_res_atoms[mobile_resid]
            n_atoms = len(mobile_res)
            rmsd = np.sqrt(residue_squared_diff[mobile_resid] / (n_frames * n_atoms))
            rmsd_dict[f"{mobile_res.residues[0].resname}{mobile_resid}"] = rmsd
        
        return rmsd_dict

    
    @spinner
    def calculate_per_residue_rmsd(self,
                          mobile_atoms: mda.AtomGroup,
                          ref_atoms: mda.AtomGroup) -> Dict[str, float]:
        """Calculate RMSD per residue for single structure comparison."""
        rmsd_dict = {}
        
        # Get unique residues
        mobile_resids = np.unique(mobile_atoms.resids)
        ref_resids = np.unique(ref_atoms.resids)
        
        if len(mobile_resids) != len(ref_resids):
            raise ValueError("Number of residues differs between mobile and reference selections")
            
        # Create mapping between mobile and reference residues
        residue_mapping = dict(zip(mobile_resids, ref_resids))
        
        for mobile_resid in mobile_resids:
            ref_resid = residue_mapping[mobile_resid]
            
            mobile_res = mobile_atoms.select_atoms(f'resid {mobile_resid}')
            ref_res = ref_atoms.select_atoms(f'resid {ref_resid}')
            
            if len(mobile_res) != len(ref_res):
                print(f"Warning: Unequal number of atoms in residue pair {mobile_resid}/{ref_resid}")
                continue
                
            # Calculate RMSD for this residue
            diff = mobile_res.positions - ref_res.positions
            squared_diff = np.sum(np.square(diff))
            rmsd = np.sqrt(squared_diff / len(mobile_res))
            rmsd_dict[f"{mobile_res.residues[0].resname}{mobile_resid}"] = rmsd
                
        return rmsd_dict
    
    @spinner
    def calculate_per_atom_rmsd(self,
                       mobile_atoms: mda.AtomGroup,
                       ref_atoms: mda.AtomGroup) -> Dict[str, float]:
        """Calculate RMSD per atom for single structure comparison."""
        rmsd_dict = {}
        
        if len(mobile_atoms) != len(ref_atoms):
            raise ValueError("Number of atoms differs between mobile and reference selections")
        
        # Calculate differences between corresponding atoms
        diff = mobile_atoms.positions - ref_atoms.positions
        squared_diff = np.sum(np.square(diff), axis=1)
        rmsd_values = np.sqrt(squared_diff)
        
        # Create dictionary with atom labels and their RMSD values
        for atom, rmsd in zip(mobile_atoms, rmsd_values):
            atom_label = f"{atom.resname}{atom.resid}-{atom.name}"
            rmsd_dict[atom_label] = rmsd
        
        return rmsd_dict
    
    def plot_rmsd_values(self, rmsd_data: Dict[str, float], 
                        mode: str, title: str, file_name: str):
        """Plot RMSD values for each atom/residue."""
        plt.figure(figsize=(15, 8))
        
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
        plt.ylabel('RMSD (Å)')
        
        mean_rmsd = np.mean(values)
        plt.axhline(y=mean_rmsd, color='r', linestyle='--',
                   label=f'Overall Mean RMSD: {mean_rmsd:.3f} Å')
        
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
                           mode: str) -> Tuple[float, Dict[str, float]]:
        """Align structures and calculate RMSD."""
        print("\n\033[1mSelection Details:\033[0m")
        print(f"Mobile selection string: {mobile_selection}")
        print(f"Reference selection string: {ref_selection}")
        
        mobile_atoms = mobile.select_atoms(mobile_selection)
        ref_atoms = reference.select_atoms(ref_selection)

        print(f"\nSelected Atoms:")
        print(f"Mobile atoms found: {len(mobile_atoms)}")
        if len(mobile_atoms) > 0:
            print(f"First few mobile residues: {[f'{r.resname}{r.resid}' for r in mobile_atoms.residues[:5]]}")
        
        print(f"Reference atoms found: {len(ref_atoms)}")
        if len(ref_atoms) > 0:
            print(f"First few reference residues: {[f'{r.resname}{r.resid}' for r in ref_atoms.residues[:5]]}")
        
        if len(mobile_atoms) == 0 or len(ref_atoms) == 0:
            raise ValueError("No atoms selected. Please check residue ranges and selection syntax.")
        
        if len(mobile_atoms) != len(ref_atoms):
            raise ValueError(f"Number of selected atoms differs: mobile={len(mobile_atoms)}, reference={len(ref_atoms)}")
        
        # Check if we're dealing with trajectories
        if len(mobile.trajectory) > 1:
            if mode == 'a':
                rmsd_dict = self.calculate_per_atom_rmsd_trajectory(
                    mobile, reference, mobile_atoms, ref_atoms, 
                    mobile_selection, ref_selection
                )
            elif mode == 'r' or mode == 'c':  # mode in ['r', 'c']
                rmsd_dict = self.calculate_per_residue_rmsd_trajectory(
                    mobile, reference, mobile_atoms, ref_atoms,
                    mobile_selection, ref_selection
                )
            else:
                raise ValueError("Invalid mode selection")
            
            overall_rmsd = np.mean(list(rmsd_dict.values()))
            return overall_rmsd, rmsd_dict
        else:
            # Single structure comparison
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
                
            else:
                raise ValueError("Invalid mode selection")
            
            return overall_rmsd, rmsd_dict
        
    def write_sorted_output(self, rmsd_dict: Dict[str, float], output_file: str = "rmsd_sorted.txt"):
        """Write a sorted table of RMSD values to a file."""
        # Convert dictionary items to sorted list
        sorted_items = sorted(rmsd_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate overall statistics
        rmsd_values = [v for k,v in sorted_items]
        mean_rmsd = np.mean(rmsd_values)
        std_rmsd = np.std(rmsd_values)
        
        # Write to file
        with open(output_file, 'w') as f:
            # Write header with statistics
            f.write(f"Overall Statistics:\n")
            f.write(f"Mean RMSD: {mean_rmsd:.3f} Å\n")
            f.write(f"Std Dev:   {std_rmsd:.3f} Å\n")
            f.write(f"Max RMSD:  {max(rmsd_values):.3f} Å ({sorted_items[0][0]})\n")
            f.write(f"Min RMSD:  {min(rmsd_values):.3f} Å ({sorted_items[-1][0]})\n\n")
            
            # Write table header
            f.write(f"{'Residue/Atom':<15} {'RMSD (Å)':<10} {'Z-score':<10}\n")
            f.write("-" * 35 + "\n")
            
            # Write sorted RMSD values with z-scores
            for label, rmsd in sorted_items:
                z_score = (rmsd - mean_rmsd) / std_rmsd if std_rmsd > 0 else 0
                f.write(f"{label:<15} {rmsd:10.3f} {z_score:10.3f}\n")
            
            # Write to console as well
            print(f"\nRMSD analysis written to {output_file}")


    @spinner
    def write_rmsd_to_pdb(self, mobile: mda.Universe, rmsd_dict: Dict[str, float], output_file: str = "rmsd_colored.pdb"):
        """Write a PDB file with RMSD values as B-factors."""
        # Ensure we're on first frame
        mobile.trajectory[0]
        
        # Create writer specifying the output format as PDB
        writer = mda.coordinates.PDB.PDBWriter(output_file)
        
        try:
            writer.write(mobile.atoms)
            writer.close()
            
            # Now read the file and modify B-factors
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
                for line in lines:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        resname = line[17:20].strip()
                        resid = line[22:26].strip()
                        atomname = line[12:16].strip()
                        
                        res_id = f"{resname}{resid}"
                        atom_id = f"{resname}{resid}-{atomname}"
                        
                        # Try atom ID first (for all-atom mode), then residue ID
                        rmsd_value = rmsd_dict.get(atom_id, rmsd_dict.get(res_id, 0.0))
                        
                        # Replace B-factor field (columns 61-66) with RMSD value
                        new_line = f"{line[:60]}{rmsd_value:6.2f}{line[66:]}"
                        f.write(new_line)
                    else:
                        f.write(line)
            
            print(f"\nStructure with RMSD as B-factors written to {output_file}")
            
        finally:
            writer.close()

        
    def save(self,
            rmsd_dict: Dict[str, float],
            output_file: str = "rmsd_dict.pickle"):
        """Function to save rmsd_dict to pickle file for further anlaysis"""
        
        with open(output_file, 'wb') as file:
            pickle.dump(rmsd_dict, file)

        file.close()



#@spinner
def main():
    parser = argparse.ArgumentParser(description='Calculate RMSD between structures/trajectories using MDAnalysis')
    parser.add_argument('struct1', help='First structure file (PDB, GRO, etc.)')
    parser.add_argument('-s1', required=True, type=str, help='Residue selection for first structure (format: xxx-xxx)')
    parser.add_argument('struct2', help='Second structure file')
    parser.add_argument('-s2', required=True, type=str, help='Residue selection for second structure (format: xxx-xxx)')
    parser.add_argument('-t1', required=False, help='Trajectory file for first structure')
    parser.add_argument('-t2', required=False, help='Trajectory file for second structure')
    parser.add_argument('-m', choices=['a', 'r', 'c'], required=False, default='c',
                        help='RMSD mode: a (all atoms), r (per residue), c (CA only)')
    parser.add_argument('--plot', nargs='?', const='rmsd_plot.png', default=None,
                       help='name of output file for RMSD plot. Default \'rmsd_plot.png\'')
    parser.add_argument('--table', nargs='?', const='rmsd_sorted.txt', default=None,
                  help='name of output file for sorted RMSD table. Default \'rmsd_sorted.txt\'')
    parser.add_argument('--pdb', nargs='?', const='rmsd_colored.pdb', default=None,
                  help='name of output PDB colored by RMSD. Default \'rmsd_colored.pdb\'')
    parser.add_argument('--save', nargs='?', const='rmsd_dict.pickle', default=None,
                       help='Write RMSF dictionary to pickle file (default: \'rmsd_dict.pickle\')')
    
    args = parser.parse_args()
    calculator = RMSDCalculator()
    
    try:
        # Load structures and trajectories
        u1, u2 = calculator.load_structures(args.struct1, args.struct2, args.t1, args.t2)
        
        # Create selection strings for both structures
        sel1 = calculator.get_selection_string(args.s1, args.m)
        sel2 = calculator.get_selection_string(args.s2, args.m)
        
        # Print selection information
        atoms1 = u1.select_atoms(sel1)
        atoms2 = u2.select_atoms(sel2)
        print("\n\033[1mStructure Analysis:\033[0m")
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
        overall_rmsd, rmsd_dict = calculator.align_and_calculate_rmsd(
            u1, u2, sel1, sel2, args.m
        )
        
        # Print results
        mode_desc = {'a': 'all atoms', 'r': 'per residue', 'c': 'CA only'}
        print(f"\nResults:")
        print(f"\033[1mOverall mean RMSD\033[0m ({mode_desc[args.m]}): {overall_rmsd:.3f} Å")
        
        # Write sorted RMSD table
        if args.table:
            calculator.write_sorted_output(rmsd_dict, args.table)
            
        
        # Write PDB with RMSD as B-factors
        if args.pdb:
            calculator.write_rmsd_to_pdb(u1, rmsd_dict, args.pdb)

        # Save rmsd_dict to pickle file
        if args.save is not None:
            calculator.save(rmsd_dict, args.save)
            print(f"\nDictionary file saved as {args.save}")
            
        
        # Plot RMSD values
        if args.plot is not None:
            title = f"RMSD Distribution ({mode_desc[args.m]})"
            calculator.plot_rmsd_values(rmsd_dict, args.m, title, args.plot)
            print(f"\nRMSD plot has been saved as {args.plot}")
        
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())