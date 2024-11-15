# tests/test_rmsd_calculator.py
import pytest
import MDAnalysis as mda
import numpy as np
from pathlib import Path
import os
import tempfile
import warnings

from rmsd_calculator import RMSDCalculator

# Suppress warnings
warnings.filterwarnings('ignore')

class TestRMSDCalculator:
    @pytest.fixture
    def data_dir(self):
        """Fixture to provide path to test data directory."""
        return Path(__file__).parent / 'data'
    
    @pytest.fixture
    def calculator(self):
        """Fixture to provide RMSDCalculator instance."""
        return RMSDCalculator()

    @pytest.fixture
    def sample_files(self, data_dir):
        """Fixture to provide paths to sample files."""
        return {
            'pdb': str(data_dir / 'sample.pdb'),
            'ref_pdb': str(data_dir / 'ref.pdb'),
            'xtc': str(data_dir / 'sample.xtc'),
            'ref_xtc': str(data_dir / 'ref.xtc')
        }

    def test_load_structures(self, calculator, sample_files):
        """Test loading of structure files."""
        # Test loading just PDBs
        u1, u2 = calculator.load_structures(sample_files['pdb'], sample_files['ref_pdb'])
        assert isinstance(u1, mda.Universe)
        assert isinstance(u2, mda.Universe)
        assert len(u1.atoms) == len(u2.atoms)
        
        # Test loading with trajectories
        u1, u2 = calculator.load_structures(
            sample_files['pdb'], 
            sample_files['ref_pdb'],
            sample_files['xtc'],
            sample_files['ref_xtc']
        )
        assert len(u1.trajectory) == 3
        assert len(u2.trajectory) == 3

    @pytest.mark.parametrize("mode", ['a', 'r', 'c'])
    def test_rmsd_calculation_static(self, calculator, sample_files, mode):
        """Test RMSD calculation for static structures."""
        u1, u2 = calculator.load_structures(sample_files['pdb'], sample_files['ref_pdb'])
        
        overall_rmsd, rmsd_dict = calculator.align_and_calculate_rmsd(
            u1, u2,
            "resid 1:2",
            "resid 1:2",
            mode
        )
        
        assert overall_rmsd > 0
        assert isinstance(rmsd_dict, dict)
        assert len(rmsd_dict) > 0

    @pytest.mark.parametrize("mode", ['a', 'r', 'c'])
    def test_rmsd_calculation_trajectory(self, calculator, sample_files, mode):
        """Test RMSD calculation with trajectories."""
        u1, u2 = calculator.load_structures(
            sample_files['pdb'], 
            sample_files['ref_pdb'],
            sample_files['xtc'],
            sample_files['ref_xtc']
        )
        
        overall_rmsd, rmsd_dict = calculator.align_and_calculate_rmsd(
            u1, u2,
            "resid 1:2",
            "resid 1:2",
            mode
        )
        
        assert overall_rmsd > 0
        assert isinstance(rmsd_dict, dict)
        assert len(rmsd_dict) > 0
        
        # Check expected dictionary size
        if mode == 'a':
            assert len(rmsd_dict) == len(u1.atoms)
        elif mode == 'r':
            assert len(rmsd_dict) == len(u1.residues)
        else:  # mode == 'c'
            assert len(rmsd_dict) == len(u1.select_atoms("name CA"))

    def test_trajectory_frame_consistency(self, calculator, sample_files):
        """Test RMSD calculation consistency across trajectory frames."""
        u1, u2 = calculator.load_structures(
            sample_files['pdb'], 
            sample_files['ref_pdb'],
            sample_files['xtc'],
            sample_files['ref_xtc']
        )
        
        print(f"Number of frames in trajectory 1: {len(u1.trajectory)}")
        print(f"Number of frames in trajectory 2: {len(u2.trajectory)}")
        
        # Select only CA atoms for simpler testing
        selection = "name CA"
        mobile_atoms = u1.select_atoms(selection)
        ref_atoms = u2.select_atoms(selection)
        
        print(f"Number of selected atoms: {len(mobile_atoms)}")
        
        # Calculate RMSD
        overall_rmsd, rmsd_dict = calculator.align_and_calculate_rmsd(
            u1, u2,
            selection,
            selection,
            'c'
        )
        
        # Basic checks
        assert overall_rmsd > 0
        assert isinstance(rmsd_dict, dict)
        assert len(rmsd_dict) == len(mobile_atoms)
        assert all(isinstance(v, (int, float)) for v in rmsd_dict.values())

    def test_output_generation_trajectory(self, calculator, sample_files):
        """Test output generation with trajectory data."""
        # Load structures with trajectories
        u1, u2 = calculator.load_structures(
            sample_files['pdb'], 
            sample_files['ref_pdb'],
            sample_files['xtc'],
            sample_files['ref_xtc']
        )
        
        # Calculate RMSD with simple selection
        selection = "name CA"
        overall_rmsd, rmsd_dict = calculator.align_and_calculate_rmsd(
            u1, u2,
            selection,
            selection,
            'c'
        )
        
        # Basic validation of RMSD calculation
        assert overall_rmsd > 0, "Overall RMSD should be positive"
        assert len(rmsd_dict) == len(u1.select_atoms(selection)), "Should have RMSD for each selected atom"
        assert all(v >= 0 for v in rmsd_dict.values()), "All RMSD values should be non-negative"
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            output_file = f.name
            
        try:
            # Test text output
            calculator.write_sorted_output(rmsd_dict, output_file)
            assert os.path.exists(output_file), "Output file should exist"
            
            # Basic content validation
            with open(output_file, 'r') as f:
                content = f.read()
                assert "Mean RMSD" in content, "Output should contain mean RMSD"
                assert str(round(overall_rmsd, 3)) in content, "Output should contain the calculated RMSD"
            
            # Test plot
            plot_file = output_file.replace('.txt', '.png')
            calculator.plot_rmsd_values(rmsd_dict, 'c', "Test Plot", plot_file)
            assert os.path.exists(plot_file), "Plot file should exist"
            assert os.path.getsize(plot_file) > 100, "Plot file should contain actual data"
            os.unlink(plot_file)
        finally:
            os.unlink(output_file)

    def test_error_handling(self, calculator, sample_files):
        """Test error handling with common error cases."""
        u1, u2 = calculator.load_structures(
            sample_files['pdb'],
            sample_files['ref_pdb']
        )
        
        # Test invalid selection
        with pytest.raises(Exception):
            calculator.align_and_calculate_rmsd(
                u1, u2,
                "invalid selection",
                "invalid selection",
                'c'
            )
        
        # Test mismatched selections
        with pytest.raises(Exception):
            calculator.align_and_calculate_rmsd(
                u1, u2,
                "name CA",
                "name CB",  # Different atom selection
                'c'
            )
        
        # Test invalid mode
        with pytest.raises(ValueError):
            calculator.align_and_calculate_rmsd(
                u1, u2,
                "name CA",
                "name CA",
                'invalid_mode'
            )

    @pytest.mark.parametrize("mode", ['a', 'r', 'c'])
    def test_different_modes(self, calculator, sample_files, mode):
        """Test RMSD calculation in different modes."""
        u1, u2 = calculator.load_structures(
            sample_files['pdb'],
            sample_files['ref_pdb']
        )
        
        overall_rmsd, rmsd_dict = calculator.align_and_calculate_rmsd(
            u1, u2,
            "resid 1:2",
            "resid 1:2",
            mode
        )
        
        # Check mode-specific outputs
        if mode == 'a':
            assert len(rmsd_dict) == len(u1.select_atoms("resid 1:2")), "Should have RMSD for all atoms"
        elif mode == 'r':
            assert len(rmsd_dict) == len(u1.select_atoms("resid 1:2").residues), "Should have RMSD for each residue"
        else:  # mode == 'c'
            assert len(rmsd_dict) == len(u1.select_atoms("resid 1:2 and name CA")), "Should have RMSD for CA atoms"


        def test_selection_validation(self, calculator, sample_files):
            """Test selection string validation with trajectories."""
            u1, u2 = calculator.load_structures(
                sample_files['pdb'], 
                sample_files['ref_pdb'],
                sample_files['xtc'],
                sample_files['ref_xtc']
            )
            
            # Test invalid selection
            with pytest.raises(Exception):
                calculator.align_and_calculate_rmsd(
                    u1, u2,
                    "invalid selection",
                    "invalid selection",
                    'c'
                )
