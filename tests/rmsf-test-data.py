import numpy as np
import MDAnalysis as mda
from pathlib import Path

def create_minimal_dcd():
    # Load the structure
    u = mda.Universe("test_protein.pdb")
    
    # Create output path
    output_path = Path("test_trajectory.dcd")
    
    # Create a minimal trajectory (5 frames)
    with mda.coordinates.DCD.DCDWriter(str(output_path), n_atoms=len(u.atoms)) as w:
        # Write initial frame
        w.write(u)
        
        # Write 4 more frames with small random movements
        rng = np.random.default_rng(42)
        for _ in range(4):
            u.atoms.positions += rng.normal(0, 0.1, u.atoms.positions.shape)
            w.write(u)

if __name__ == "__main__":
    create_minimal_dcd()
