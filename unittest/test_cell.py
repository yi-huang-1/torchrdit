import pytest
import torch
from unittest.mock import MagicMock

from torchrdit.cell import Cell3D
from torchrdit.utils import create_material


@pytest.fixture
def cell_3d():
    return Cell3D()


def test_er1_property(cell_3d):
    assert cell_3d.er1.dtype == torch.complex128
    assert cell_3d.er1 == torch.tensor(1.0 + 0.0j)


def test_er2_property(cell_3d):
    assert cell_3d.er2.dtype == torch.complex128
    assert cell_3d.er2 == torch.tensor(1.0 + 0.0j)


def test_ur1_property(cell_3d):
    assert cell_3d.ur1.dtype == torch.complex128
    assert cell_3d.ur1 == torch.tensor(1.0 + 0.0j)


def test_ur2_property(cell_3d):
    assert cell_3d.ur2.dtype == torch.complex128
    assert cell_3d.ur2 == torch.tensor(1.0 + 0.0j)


def test_update_trn_material(cell_3d):
    material_sio = create_material(name="SiO", permittivity=3.0)
    cell_3d.update_trn_material(material_sio)
    assert cell_3d.er2 == torch.tensor(3.0)
    assert cell_3d.layer_manager._trn_material_name == "SiO"


def test_update_ref_material(cell_3d):
    material_sio = create_material(name="SiO", permittivity=3.0)
    cell_3d.update_ref_material(material_sio)
    assert cell_3d.er1 == torch.tensor(3.0)
    assert cell_3d.layer_manager._ref_material_name == "SiO"


def test_get_layout(cell_3d):
    layout = cell_3d.get_layout()
    assert layout[0].dtype == torch.float64
    assert layout[1].dtype == torch.float64
    assert layout[0].size() == torch.Size([512, 512])
    assert layout[1].size() == torch.Size([512, 512])


def test_add_layer(cell_3d):
    material_sio = create_material(name="SiO", permittivity=3.0)
    cell_3d.add_layer(material_name=material_sio, thickness=torch.tensor(3.0e-3))
    assert len(cell_3d.layer_manager.layers) == 1


def test_invalid_rdim():
    with pytest.raises(ValueError):
        Cell3D(rdim=[512])


def test_invalid_kdim():
    with pytest.raises(ValueError):
        Cell3D(kdim=[3])


def test_invalid_t1():
    with pytest.raises(ValueError):
        Cell3D(t1=None)


def test_invalid_t2():
    with pytest.raises(ValueError):
        Cell3D(t2=None)


def test_add_invalid_material(cell_3d):
    with pytest.raises(RuntimeError):
        cell_3d.add_layer(material_name="InvalidMaterial", thickness=torch.tensor(3.0e-3))


def test_mock_layer_manager(cell_3d):
    cell_3d.layer_manager = MagicMock()
    material_sio = create_material(name="SiO", permittivity=3.0)
    cell_3d.add_layer(material_name=material_sio, thickness=torch.tensor(3.0e-3))
    cell_3d.layer_manager.add_layer.assert_called_once()


def test_initialization_with_different_parameters():
    cell = Cell3D(lengthunit="nm", rdim=[256, 256], kdim=[5, 5])
    assert cell.lengthunit == "nm"
    assert cell.rdim == [256, 256]
    assert cell.kdim == [5, 5]


def test_property_methods(cell_3d):
    # Test that layers property returns a list
    assert isinstance(cell_3d.layers, list)
    assert len(cell_3d.layers) == 0  # No layers added yet


def test_material_management(cell_3d):
    material_sio = create_material(name="SiO", permittivity=3.0)
    cell_3d.add_materials([material_sio])
    assert "SiO" in cell_3d._matlib
    cell_3d._matlib.pop("SiO")
    assert "SiO" not in cell_3d._matlib


def test_lengthunit_micrometer_conversion():
    # Test the missing line 278: conversion of μm to um
    cell = Cell3D(lengthunit="μm")
    assert cell._lenunit == "um"


def test_add_materials_invalid_element():
    # Test missing lines 369-370: invalid material element
    cell = Cell3D()
    with pytest.raises(ValueError, match="The element of the argument should be the \\[MaterialClass\\] type."):
        cell.add_materials(["invalid_material"])


def test_add_materials_invalid_argument():
    # Test missing lines 372-373: invalid argument type
    cell = Cell3D()
    with pytest.raises(ValueError, match="Input argument should be a list."):
        cell.add_materials("not_a_list")


def test_add_layer_invalid_thickness():
    # Test missing lines 446-449: invalid thickness conversion
    cell = Cell3D()
    material_sio = create_material(name="SiO", permittivity=3.0)
    cell.add_materials([material_sio])
    with pytest.raises(ValueError, match="Invalid input thickness"):
        cell.add_layer(material_name="SiO", thickness=None)


def test_add_lattice_vectors_type_error():
    # Test missing lines 540-542: lattice vector type conversion error
    cell = Cell3D()
    # Create an invalid tensor that can't be converted to the required type
    # This is tricky to trigger, so let's mock the conversion
    from unittest.mock import patch

    invalid_tensor = torch.tensor([1.0, 2.0], dtype=torch.complex64)

    # Patch the to() method to raise an error
    with patch.object(invalid_tensor, "to", side_effect=TypeError("Mock conversion error")):
        with pytest.raises(TypeError, match="The element of the argument should be the"):
            cell._add_lattice_vectors(invalid_tensor)


def test_update_trn_material_air():
    # Test missing lines 589-590: air material case
    cell = Cell3D()
    cell.update_trn_material("air")  # Should pass without error
    cell.update_trn_material("Air")  # Should also pass (case insensitive)


def test_update_trn_material_invalid():
    # Test missing lines 592-593: invalid material error
    cell = Cell3D()
    with pytest.raises(RuntimeError, match="No materials named \\[invalid_material exists in the material lib.\\]"):
        cell.update_trn_material("invalid_material")


def test_update_ref_material_air():
    # Test missing lines 639-640: air material case
    cell = Cell3D()
    cell.update_ref_material("air")  # Should pass without error
    cell.update_ref_material("Air")  # Should also pass (case insensitive)


def test_update_ref_material_invalid():
    # Test missing lines 642-643: invalid material error
    cell = Cell3D()
    with pytest.raises(RuntimeError, match="No materials named \\[invalid_material exists in the material lib.\\]"):
        cell.update_ref_material("invalid_material")


def test_get_layer_structure(capsys):
    # Test missing lines 675-699: get_layer_structure print statements
    cell = Cell3D()
    material_sio = create_material(name="SiO", permittivity=3.0)
    cell.add_materials([material_sio])
    cell.add_layer(material_name="SiO", thickness=torch.tensor(0.5))

    cell.get_layer_structure()

    captured = capsys.readouterr()
    assert "Cell Type:" in captured.out
    assert "layer # Reflection" in captured.out
    assert "layer # Transmission" in captured.out
    assert "material name:" in captured.out
    assert "permittivity:" in captured.out
    assert "permeability:" in captured.out


def test_get_cell_type_cartesian_case():
    # Test missing line 825: Cartesian cell type detection
    from torchrdit.cell import CellType

    # Create a cell with specific lattice vectors that trigger the first Cartesian condition
    cell = Cell3D(t1=torch.tensor([0.0, 1.0]), t2=torch.tensor([1.0, 0.0]))
    assert cell.get_cell_type() == CellType.Cartesian
