import pytest
import numpy as np
import MDAnalysis as mda
from pathlib import Path
import matplotlib.pyplot as plt
from rmsf_calculator import RMSFCalculator
import os
#from MDAnalysis.tests.datafiles import PDB, XTC

@pytest.fixture
def data_dir():
    """Fixture for data directory path"""
    return Path("tests/data")

@pytest.fixture
def calculator():
    """Fixture for RMSFCalculator instance"""
    return RMSFCalculator()

@pytest.fixture
def universe(data_dir):
    """Fixture for MDAnalysis Universe with test data"""
    pdb_file = data_dir / "test_protein.pdb"
    dcd_file = data_dir / "test_trajectory.dcd"
    return mda.Universe(str(pdb_file), str(dcd_file))

@pytest.fixture
def expected_rmsf(data_dir):
    """Fixture to load expected RMSF values"""
    rmsf_dict = {}
    with open(data_dir / "expected_rmsf.txt") as f:
        next(f)  # Skip header
        for line in f:
            resid, rmsf = line.strip().split()
            rmsf_dict[resid] = float(rmsf)
    return rmsf_dict

def test_load_trajectory(calculator, data_dir):
    """Test loading of trajectory"""
    pdb_file = data_dir / "test_protein.pdb"
    dcd_file = data_dir / "test_trajectory.dcd"
    
    u = calculator.load_trajectory(str(pdb_file), str(dcd_file))
    
    assert isinstance(u, mda.Universe)
    assert len(u.trajectory) == 5  # We know we have 5 frames
    assert len(u.residues) == 10  # We know we have 10 residues
    assert len(u.atoms) == 30     # 3 atoms per residue

def test_parse_residue_range(calculator):
    """Test parsing of residue range strings"""
    # Test valid range
    start, end = calculator.parse_residue_range("1-10")
    assert start == 1
    assert end == 10
    
    # Test invalid format
    with pytest.raises(ValueError):
        calculator.parse_residue_range("invalid")
    
    # Test reversed range
    with pytest.raises(ValueError):
        calculator.parse_residue_range("10-1")

def test_get_selection_string(calculator):
    """Test creation of selection strings"""
    # Test CA selection
    sel_str = calculator.get_selection_string("1-10", 'c')
    assert sel_str == "(resid 1:10) and name CA"
    
    # Test all-atom selection
    sel_str = calculator.get_selection_string("1-10", 'a')
    assert sel_str == "resid 1:10"
    
    # Test residue selection
    sel_str = calculator.get_selection_string("1-10", 'r')
    assert sel_str == "resid 1:10"
    
    # Test protein selection
    sel_str = calculator.get_selection_string("protein", 'c')
    assert sel_str == "(protein) and name CA"
    
    # Test invalid mode
    with pytest.raises(ValueError):
        calculator.get_selection_string("1-10", 'invalid')

def test_calculate_rmsf_ca(calculator, universe, expected_rmsf):
    """Test RMSF calculation for CA atoms"""
    rmsf_dict = calculator.calculate_rmsf(
        universe,
        "(resid 1:10) and name CA",
        mode='c'
    )
    
    assert len(rmsf_dict) == 10  # Should have 10 residues
    
    # Compare with expected values
    for resid, expected in expected_rmsf.items():
        assert resid in rmsf_dict
        print(rmsf_dict[resid], expected)
        assert np.isclose(rmsf_dict[resid], expected, atol=1e-3)

def test_calculate_rmsf_all_atoms(calculator, universe):
    """Test RMSF calculation for all atoms"""
    rmsf_dict = calculator.calculate_rmsf(
        universe,
        "resid 1:10",
        mode='a'
    )
    
    assert len(rmsf_dict) == 30  # Should have all atoms
    assert all(isinstance(v, float) for v in rmsf_dict.values())
    
    # Check format of keys (should be ResidueAtom format)
    for key in rmsf_dict.keys():
        assert "-" in key  # Should be in format "ALAx-ATOM"
        resname, atom = key.split("-")
        assert resname.startswith("ALA")
        assert atom in ["N", "CA", "C"]

def test_calculate_rmsf_residue(calculator, universe):
    """Test RMSF calculation per residue"""
    rmsf_dict = calculator.calculate_rmsf(
        universe,
        "resid 1:10",
        mode='r'
    )
    
    assert len(rmsf_dict) == 10  # Should have 10 residues
    assert all(isinstance(v, float) for v in rmsf_dict.values())
    assert all(k.startswith("ALA") for k in rmsf_dict.keys())

def test_empty_selection(calculator, universe):
    """Test RMSF calculation with empty selection"""
    with pytest.raises(ValueError, match="No atoms selected"):
        calculator.calculate_rmsf(
            universe,
            "resid 999",
            mode='c'
        )

def test_plot_rmsf_values(calculator, expected_rmsf, tmp_path):
    """Test RMSF plotting"""
    output_file = tmp_path / "test_plot.png"
    
    calculator.plot_rmsf_values(
        expected_rmsf,
        mode='r',
        title="Test Plot",
        file_name=str(output_file)
    )
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0

def test_write_sorted_output(calculator, expected_rmsf, tmp_path):
    """Test writing sorted RMSF values"""
    output_file = tmp_path / "test_rmsf.txt"
    
    calculator.write_sorted_output(expected_rmsf, str(output_file))
    
    assert output_file.exists()
    with open(output_file) as f:
        content = f.read()
        assert "Overall Statistics:" in content
        assert "Mean RMSF:" in content
        assert "Std Dev:" in content
        assert "Max RMSF:" in content
        assert "Min RMSF:" in content
        
        # Check that all residues are listed
        for i in range(1, 11):
            assert f"ALA{i}" in content

def test_write_rmsf_to_pdb(calculator, universe, expected_rmsf, tmp_path):
    """Test writing RMSF values to PDB"""
    output_file = tmp_path / "test_rmsf.pdb"
    
    calculator.write_rmsf_to_pdb(universe, expected_rmsf, str(output_file))
    
    assert output_file.exists()
    assert output_file.stat().st_size > 0
    
    # Verify PDB format and content
    with open(output_file) as f:
        lines = f.readlines()
        
        # Check for essential PDB elements
        atom_lines = [l for l in lines if l.startswith('ATOM')]
        assert len(atom_lines) == 30  # Should have 30 atoms
        
        # Check atom order and residue numbering
        for i, line in enumerate(atom_lines, 1):
            assert line[0:4] == 'ATOM'
            residue_num = int(line[22:26])
            assert 1 <= residue_num <= 10
            
            # Verify atom names
            atom_name = line[12:16].strip()
            assert atom_name in ['N', 'CA', 'C']
            
            # Verify B-factor column exists
            assert len(line) >= 66  # B-factor column should exist

if __name__ == '__main__':
    pytest.main(['-v'])
