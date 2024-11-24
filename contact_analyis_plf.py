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

warnings.filterwarnings('ignore')

class contactAnalyzer:
    def __init__(self):
        """Initialize ProLIF analyzer"""
        # Define interaction types to analyze
        self.interaction_types = [
            'Hydrophobic',
            'HBond',
            'Halogen',
            'Ionic',
            'PiStacking',
            'PiCation',
            'Metal'
        ]

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
                u = mda.Universe(struct, traj, )
            else:
                u = mda.Universe(struct)

            # Create selections
            sel1 = f"({sel1}) and not resname WAT"
            sel2 = f"({sel2}) and not resname WAT"
            selection1 = u.select_atoms(sel1)
            selection2 = u.select_atoms(sel2)
            #mol1 = Molecule.from_mda(selection1)
            #mol2 = Molecule.from_mda(selection2)

            

            if selection1 is None or selection2 is None:
                raise ValueError("Failed to create selections")

            if struct:
                    print(f"Structure: {struct}")
                    
            if traj:
                    print(f"Trajectory: {traj}")
                    print(f"Number of frames: {len(u.trajectory)}")

            print("\nStructure Details:")
            print(f"Selection 1: {len(selection1)} atoms")
            print(f"Selection 2: {len(selection2)} atoms")
            
            
            return selection1, selection2, u
            
        except Exception as e:
            raise Exception(f"Error loading system: {str(e)}")

    def analyze_interactions(self, mol1:mda.AtomGroup, mol2:mda.AtomGroup, u:mda.Universe):
        """
        Analyze interactions using ProLIF
        
        Parameters
        ----------
        mol1 : prolif.Molecule
            Protein molecule
        mol2 : prolif.Molecule
            Ligand molecule
        n_frames : int, optional
            Number of frames in trajectory. If None, treats as single structure.
        
        Returns
        -------
        dict
            Dictionary containing interaction analysis results
        """
        print("Analyzing interactions...")
        try:

            # Initialize fingerprint
            fp = Fingerprint()

            # Get length of trajectory
            if u is not None:
                n_frames = len(u.trajectory)

            # Generate trajectory frames list
            traj = [(i, None) for i in range(n_frames)] if n_frames else [(0, None)]
            
            # Generate fingerprint - mol1 is protein, mol2 is ligand
            fp.run(u.trajectory[0:7], mol1, mol2,)
            
            # Get interaction details
            detailed_interactions = self._get_detailed_interactions(fp)
            interaction_summary = self._summarize_interactions(detailed_interactions)
            results = {
                'fingerprint': fp,
                'interactions': detailed_interactions,
                'summary': interaction_summary
            }
            
            return results
        
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            # Get the full traceback as a formatted string
            tb_str = ''.join(traceback.format_tb(exc_traceback))
            
            error_msg = (
                f"Error analyzing interactions:\n"
                f"Error type: {exc_type.__name__}\n"
                f"Error message: {str(e)}\n"
                f"Traceback:\n{tb_str}"
            )
            raise Exception(error_msg)

    def _get_detailed_interactions(self, fingerprint:Fingerprint):
        """Extract detailed interaction information from fingerprint"""
        
        interactions_dict = {}
        
        
        for idx, frame in fingerprint.ifp.items():
            interactions_dict[idx] = []
            
            # Iterate through each residue-atom pair in the first molecule
            for res_pair, interactions in frame.items():
                
                res1, res2 = res_pair[0], res_pair[1]
                
                for itype in interactions.items():
                    interaction_info = {
                        'residue1': str(res1),
                        'residue2': str(res2),
                        'type': itype[0]
                    }
                    interactions_dict[idx].append(interaction_info)

        
        return interactions_dict
                
    def _summarize_interactions(self, interactions:dict):
        """Generate summary statistics of interactions"""
        summary = dict(zip(self.interaction_types, values))
        values = 
        #summary.keys = self.interaction_types
        
        # Count interactions by type
        for idx in interactions:
                summary[interactions[idx]['type']] += 1
                    
        return summary

    def plot_interactions(self, results, output_prefix="prolif"):
        """Generate interaction plots"""
        summary = results['summary']
        
        # Plot interaction type distribution
        plt.figure(figsize=(10, 6))
        interaction_counts = {k: v for k, v in summary.items() if v > 0}
        if interaction_counts:
            sns.barplot(x=list(interaction_counts.keys()),
                       y=list(interaction_counts.values()))
            plt.title('Interaction Type Distribution')
            plt.xlabel('Interaction Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_distribution.png", dpi=300)
        plt.close()

        print(f"\nPlot saved as {output_prefix}_distribution.png")

    def write_report(self, results, output_file="prolif_report.txt"):
        """Write detailed interaction report"""
        with open(output_file, 'w') as f:
            f.write("ProLIF Interaction Analysis Report\n")
            f.write("================================\n\n")
            
            # Write summary
            f.write("Interaction Summary\n")
            f.write("-----------------\n")
            for itype, count in results['summary'].items():
                if count > 0:
                    f.write(f"{itype}: {count}\n")
            
            # Write detailed interactions
            f.write("\nDetailed Interactions\n")
            f.write("-------------------\n")
            
            for itype, interactions in results['interactions'].items():
                if interactions:
                    f.write(f"\n{itype.upper()} Interactions:\n")
                    for interaction in interactions:
                        f.write(f"\nResidue 1: {interaction['residue1']}")
                        f.write(f"\nResidue 2: {interaction['residue2']}")
                        f.write(f"\nAtoms 1: {', '.join(interaction['atoms1'])}")
                        f.write(f"\nAtoms 2: {', '.join(interaction['atoms2'])}")
                        if 'distance' in interaction:
                            f.write(f"\nDistance: {interaction['distance']:.2f} Å\n")

        print(f"\nReport written to {output_file}")

    def save_results(self, results, output_file="prolif_results.pickle"):
        """Save results to pickle file"""
        # Create a copy of results without the fingerprint object
        results_to_save = {
            'interactions': results['interactions'],
            'summary': results['summary']
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(results_to_save, f)
            
        print(f"\nResults saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Flexible protein interaction analysis using ProLIF')
    
    # Required argument
    parser.add_argument('struct1', help='First structure file')
    
    # Optional arguments with stored defaults
    parser.add_argument('-t', '--trajectory', default=None,
                       help='Trajectory file (optional)')
    parser.add_argument('-s1', required=True,
                       help='Selection string for first group (MDAnalysis syntax)')
    parser.add_argument('-s2', required=True,
                       help='Selection string for second group (MDAnalysis syntax)')
    parser.add_argument('--prefix', default='prolif',
                       help='Output file prefix (default: prolif)')
    parser.add_argument('--report', action='store_true',
                       help='Generate detailed report')
    parser.add_argument('--save', action='store_true',
                       help='Save results to pickle file')
    
    args = parser.parse_args()
    
    try:
        analyzer = contactAnalyzer()
        
        # Load system with flexible input options
        mol1, mol2, universe = analyzer.load_system(
            args.struct1,
            args.trajectory,
            args.s1,
            args.s2
        )
        
        # Analyze interactions
        results = analyzer.analyze_interactions(mol1, mol2, universe)
        print(results)
        
        # Generate outputs
        analyzer.plot_interactions(results, args.prefix)
        
        if args.report:
            analyzer.write_report(results)
        
        if args.save:
            analyzer.save_results(results)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())