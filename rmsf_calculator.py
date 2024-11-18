import argparse
import MDAnalysis as mda
import numpy as np
from typing import Tuple, List, Dict, Union
from MDAnalysis.analysis import align
import matplotlib.pyplot as plt
import threading
import sys
import warnings
import time
from tqdm import tqdm
import pickle

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=Warning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

class RMSFCalculator:
    def __init__(self):
        """Initialize RMSFCalculator."""
        pass
        
    def show_progress(func):
        """Decorator that adds a progress bar for trajectory processing."""
        def wrapper(*args, **kwargs):
            self = args[0]  # Get instance reference
            
            # Get trajectory length from Universe object
            universe = args[1]  # Universe is the second argument
            n_frames = len(universe.trajectory)
            
            with tqdm(total=n_frames, colour='green', desc="Processing frames",
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} frames') as pbar:
                self.pbar = pbar
                result = func(*args, **kwargs)
                self.pbar = None
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

    def load_trajectory(self, struct: str, traj: str = None) -> mda.Universe:
        """Load structure and optional trajectory into MDAnalysis universe."""
        try:
            u = mda.Universe(struct, traj) if traj else mda.Universe(struct)
            
            print("\n\033[1mStructure Details:\033[0m")
            print(f"Structure: {struct}")
            if traj:
                print(f"Trajectory: {traj}")
                print(f"Number of frames: {len(u.trajectory)}")
            print(f"Total residues: {len(u.residues)}")
            print(f"Residue numbers available: {u.residues.resids[0]}-{u.residues.resids[-1]}")
            
            return u
        except Exception as e:
            raise Exception(f"Error loading structure/trajectory files: {str(e)}")
    
    """
    def parse_residue_range(self, resid_range: str) -> Tuple[int, int]:
        """"Parse residue range string 'xxx-xxx' into start and end residue numbers.""""
        start, end = map(int, resid_range.split('-'))
        if start > end:
            raise ValueError(f"Invalid range.")
        else:
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
    def calculate_rmsf(self, 
                      universe: mda.Universe,
                      selection: str,
                      reference: mda.Universe = None,
                      ref_selection: str = None,
                      mode: str = 'r') -> Dict[str, float]:
        """
        Calculate RMSF for selected atoms/residues over trajectory.
        
        Parameters:
        -----------
        universe : MDAnalysis.Universe
            Universe containing structure and trajectory
        selection : str
            Atom selection string
        reference : MDAnalysis.Universe, optional
            Reference structure for alignment
        ref_selection : str, optional
            Selection string for reference structure
        mode : str
            'a' for all-atom, 'r' for per-residue, 'c' for CA only
            
        Returns:
        --------
        Dict[str, float]
            Dictionary mapping residue/atom identifiers to RMSF values
        """
        # Select atoms
        mobile = universe.select_atoms(selection)
        
        if len(mobile) == 0:
            raise ValueError("No atoms selected. Please check selection syntax.")
            
        # Set up reference for alignment
        if reference is None:
            reference = universe
            ref_selection = selection
            
        ref_atoms = reference.select_atoms(ref_selection)
        
        # Initialize arrays for positions
        n_frames = len(universe.trajectory)
        positions = np.zeros((n_frames, len(mobile), 3))
        
        # Collect positions over trajectory
        for ts_idx, ts in enumerate(universe.trajectory):
            # Align to reference if provided
            if reference is not None:
                align.alignto(universe, reference,
                            select=(selection, ref_selection),
                            weights="mass")
            
            # Store positions
            positions[ts_idx] = mobile.positions
            
            if self.pbar:
                self.pbar.update(1)
        
        # Calculate mean positions
        mean_positions = np.mean(positions, axis=0)
        
        # Calculate RMSF
        rmsf_dict = {}
        
        if mode in ['r', 'c']:
            # Per-residue RMSF
            for residue in mobile.residues:
                res_indices = [i for i, atom in enumerate(mobile) 
                             if atom.resid == residue.resid]
                
                res_positions = positions[:, res_indices, :]
                res_mean = mean_positions[res_indices]
                
                # Calculate RMSF for this residue
                diff = res_positions - res_mean
                rmsf = np.sqrt(np.mean(np.sum(diff**2, axis=2), axis=0))
                rmsf_mean = np.mean(rmsf)
                
                rmsf_dict[f"{residue.resname}{residue.resid}"] = rmsf_mean
                
        else:
            # Per-atom RMSF
            diff = positions - mean_positions
            rmsf = np.sqrt(np.mean(np.sum(diff**2, axis=2), axis=0))
            
            for atom, value in zip(mobile, rmsf):
                atom_id = f"{atom.resname}{atom.resid}-{atom.name}"
                rmsf_dict[atom_id] = value
        
        return rmsf_dict

    def plot_rmsf_values(self, 
                        rmsf_dict: Dict[str, float],
                        mode: str,
                        title: str = "RMSF Analysis",
                        file_name: str = "rmsf_plot.png"):
        """Plot RMSF values for each atom/residue."""
        plt.figure(figsize=(15, 8))
        
        x = range(len(rmsf_dict))
        labels = list(rmsf_dict.keys())
        values = list(rmsf_dict.values())
        
        plt.plot(x, values, '-o', markersize=4, linewidth=1.5, alpha=0.7)
        
        n_labels = len(labels)
        if n_labels > 30:
            step = max(1, n_labels // 20)
            shown_positions = x[::step]
            shown_labels = labels[::step]
            plt.xticks(shown_positions, shown_labels, rotation=45, ha='right')
        else:
            plt.xticks(x, labels, rotation=45, ha='right')
        
        plt.xlabel('Atom/Residue ID' if mode == 'a' else 'Residue ID')
        plt.ylabel('RMSF (Å)')
        
        mean_rmsf = np.mean(values)
        plt.axhline(y=mean_rmsf, color='r', linestyle='--',
                   label=f'Mean RMSF: {mean_rmsf:.3f} Å')
        
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.subplots_adjust(bottom=0.2)
        plt.margins(x=0.02)
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close()

    def write_sorted_output(self, 
                          rmsf_dict: Dict[str, float],
                          output_file: str = "rmsf_sorted.txt"):
        """Write a sorted table of RMSF values to a file."""
        sorted_items = sorted(rmsf_dict.items(), key=lambda x: x[1], reverse=True)
        
        rmsf_values = [v for k,v in sorted_items]
        mean_rmsf = np.mean(rmsf_values)
        std_rmsf = np.std(rmsf_values)
        
        with open(output_file, 'w') as f:
            f.write(f"Overall Statistics:\n")
            f.write(f"Mean RMSF: {mean_rmsf:.3f} Å\n")
            f.write(f"Std Dev:   {std_rmsf:.3f} Å\n")
            f.write(f"Max RMSF:  {max(rmsf_values):.3f} Å ({sorted_items[0][0]})\n")
            f.write(f"Min RMSF:  {min(rmsf_values):.3f} Å ({sorted_items[-1][0]})\n\n")
            
            f.write(f"{'Residue/Atom':<15} {'RMSF (Å)':<10} {'Z-score':<10}\n")
            f.write("-" * 35 + "\n")
            
            for label, rmsf in sorted_items:
                z_score = (rmsf - mean_rmsf) / std_rmsf if std_rmsf > 0 else 0
                f.write(f"{label:<15} {rmsf:10.3f} {z_score:10.3f}\n")
            
        print(f"\nRMSF analysis written to {output_file}")

    @spinner
    def write_rmsf_to_pdb(self, 
                         universe: mda.Universe,
                         rmsf_dict: Dict[str, float],
                         output_file: str = "rmsf_colored.pdb"):
        """Write a PDB file with RMSF values as B-factors."""
        universe.trajectory[0]
        writer = mda.coordinates.PDB.PDBWriter(output_file)
        
        try:
            # Create mapping of residue numbers
            residue_mapping = {}
            for residue in universe.residues:
                residue_mapping[residue.resid] = {
                    'resname': residue.resname,
                    'orig_resid': residue.resid
                }
            
            writer.write(universe.atoms)
            writer.close()
            
            with open(output_file, 'r') as f:
                lines = f.readlines()
            
            with open(output_file, 'w') as f:
                atom_num = 1
                
                for line in lines:
                    if line.startswith('ATOM') or line.startswith('HETATM'):
                        curr_resid = int(line[22:26].strip())
                        
                        if curr_resid in residue_mapping:
                            mapped_info = residue_mapping[curr_resid]
                            resname = mapped_info['resname']
                            orig_resid = mapped_info['orig_resid']
                            
                            atomname = line[12:16].strip()
                            
                            res_id = f"{resname}{orig_resid}"
                            atom_id = f"{resname}{orig_resid}-{atomname}"
                            
                            rmsf_value = rmsf_dict.get(atom_id, rmsf_dict.get(res_id, 0.0))
                            
                            new_line = (f"{line[:6]}{atom_num:5d}{line[11:22]}"
                                      f"{orig_resid:4d}{line[26:60]}"
                                      f"{rmsf_value:6.2f}{line[66:]}")
                            
                            atom_num += 1
                            f.write(new_line)
                        else:
                            f.write(line)
                    else:
                        f.write(line)
                
            print(f"\nStructure with RMSF as B-factors written to {output_file}")
            
        finally:
            writer.close()

    def save(self,
            rmsf_dict: Dict[str, float],
            output_file: str = "rmsf_dict.pickle"):
        """Function to save rmsf_object to pickle file for further anlaysis"""
        
        with open(output_file, 'wb') as file:
            pickle.dump(rmsf_dict, file)

        file.close()

def main():
    parser = argparse.ArgumentParser(description='Calculate RMSF over trajectory using MDAnalysis')
    parser.add_argument('struct', help='Structure file (PDB, GRO, etc.)')
    parser.add_argument('traj', help='Trajectory file')
    parser.add_argument('-s', required=True, type=str, help='Residue selection (format: xxx-xxx)')
    parser.add_argument('-m', choices=['a', 'r', 'c'], default='c',
                       help='RMSF mode: a (all atoms), r (per residue), c (CA only)')
    parser.add_argument('--plot', nargs='?', const='rmsf_plot.png', default=None,
                       help='name of output file for RMSF plot. Default \'rmsf_plot.png\'')
    parser.add_argument('--table', nargs='?', const='rmsf_sorted.txt', default=None,
                       help='Write sorted RMSF values to file (default: rmsf_sorted.txt)')
    parser.add_argument('--pdb', nargs='?', const='rmsf_colored.pdb', default=None,
                       help='Write PDB with RMSF as B-factors (default: rmsf_colored.pdb)')
    parser.add_argument('--save', nargs='?', const='rmsf_dict.pickle', default=None,
                       help='Write RMSF dictionary to pickle file (default: rmsf_dict.pickle)')
    
    args = parser.parse_args()
    calculator = RMSFCalculator()
    
    try:
        # Load structure and trajectory
        u = calculator.load_trajectory(args.struct, args.traj)
        
        # Create selection string
        selection = calculator.get_selection_string(args.s, args.m)
        
        # Print selection information
        print("\n\033[1mSelection Details:\033[0m")
        print(f"Selection string: {selection}")
        atoms = u.select_atoms(selection)
        print(f"Atoms selected: {len(atoms)}")
        
        # Calculate RMSF
        rmsf_dict = calculator.calculate_rmsf(u, selection, mode=args.m)


        mode_desc = {'a': 'all atoms', 'r': 'per residue', 'c': 'CA only'}
        # Print overall statistics
        values = list(rmsf_dict.values())
        mean_rmsf = np.mean(values)
        print(f"\n\033[1mResults:\033[0m")
        print(f"Mean RMSF: {mean_rmsf:.3f} Å")
        
        # Optional outputs
        if args.table is not None:
            calculator.write_sorted_output(rmsf_dict, args.table)
            
        if args.pdb is not None:
            calculator.write_rmsf_to_pdb(u, rmsf_dict, args.pdb)

        if args.save is not None:
            calculator.save(rmsf_dict, args.save)
            print(f"\nDictionary file saved as {args.save}")
        
        # Plot RMSF values
        if args.plot is not None:
            title = f"RMSF Distribution ({mode_desc[args.m]})"
            calculator.plot_rmsf_values(rmsf_dict, args.m, title, args.plot)
            print(f"\nRMSF plot has been saved as {args.plot}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())