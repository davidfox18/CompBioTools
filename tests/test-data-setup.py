# tests/generate_test_data.py
import MDAnalysis as mda
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

def generate_test_data():
    """Generate test PDB and trajectory files for testing."""
    data_dir = Path(__file__).parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Write initial PDB
    with open(data_dir / 'sample.pdb', 'w') as f:
        f.write("""ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N  
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C  
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C  
ATOM      4  O   ALA A   1       1.593   2.339   0.699  1.00  0.00           O  
ATOM      5  N   ALA A   2       3.082   1.549  -0.766  1.00  0.00           N  
ATOM      6  CA  ALA A   2       3.691   2.860  -0.938  1.00  0.00           C  
ATOM      7  C   ALA A   2       4.167   3.428   0.388  1.00  0.00           C  
ATOM      8  O   ALA A   2       4.041   4.627   0.634  1.00  0.00           O  
END
""")
    
    # Create universe from the PDB
    u = mda.Universe(str(data_dir / 'sample.pdb'))
    
    # Generate XTC trajectory
    with mda.Writer(str(data_dir / 'sample.xtc'), u.atoms.n_atoms) as W:
        # Frame 1: Original positions
        W.write(u.atoms)
        
        # Frame 2: Shift by 1Å in x
        u.atoms.positions += np.array([1.0, 0.0, 0.0])
        W.write(u.atoms)
        
        # Frame 3: Additional shift in y
        u.atoms.positions += np.array([0.0, 1.0, 0.0])
        W.write(u.atoms)
    
    # Generate reference structure and trajectory
    with open(data_dir / 'ref.pdb', 'w') as f:
        f.write("""ATOM      1  N   ALA A   1       0.100   0.100   0.100  1.00  0.00           N  
ATOM      2  CA  ALA A   1       1.558   0.100   0.100  1.00  0.00           C  
ATOM      3  C   ALA A   1       2.109   1.520   0.100  1.00  0.00           C  
ATOM      4  O   ALA A   1       1.693   2.439   0.799  1.00  0.00           O  
ATOM      5  N   ALA A   2       3.182   1.649  -0.666  1.00  0.00           N  
ATOM      6  CA  ALA A   2       3.791   2.960  -0.838  1.00  0.00           C  
ATOM      7  C   ALA A   2       4.267   3.528   0.488  1.00  0.00           C  
ATOM      8  O   ALA A   2       4.141   4.727   0.734  1.00  0.00           O  
END
""")
    
    # Create reference trajectory
    u_ref = mda.Universe(str(data_dir / 'ref.pdb'))
    with mda.Writer(str(data_dir / 'ref.xtc'), u_ref.atoms.n_atoms) as W:
        # Frame 1: Original positions
        W.write(u_ref.atoms)
        
        # Frame 2: Different shift
        u_ref.atoms.positions += np.array([1.1, 0.1, 0.0])
        W.write(u_ref.atoms)
        
        # Frame 3: Another different shift
        u_ref.atoms.positions += np.array([0.1, 1.1, 0.1])
        W.write(u_ref.atoms)

if __name__ == "__main__":
    generate_test_data()
