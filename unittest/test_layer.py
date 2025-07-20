import pytest
import torch

from torchrdit.layers import Layer, HomogeneousLayer, GratingLayer, LayerBuilder, HomogeneousLayerBuilder, GratingLayerBuilder, LayerDirector, LayerManager

@pytest.fixture
def homogeneous_layer():
    return HomogeneousLayer(thickness=1.0, material_name='SiO2', is_optimize=True)

@pytest.fixture
def grating_layer():
    return GratingLayer(thickness=1.0, material_name='SiO2', is_optimize=True)

def test_layer_initialization(homogeneous_layer):
    assert homogeneous_layer.thickness == 1.0
    assert homogeneous_layer.material_name == 'SiO2'
    assert homogeneous_layer.is_optimize
    assert homogeneous_layer.is_homogeneous
    assert not homogeneous_layer.is_dispersive
    assert not homogeneous_layer.is_solved

def test_layer_setters(homogeneous_layer):
    homogeneous_layer.thickness = 2.0
    homogeneous_layer.material_name = 'Si'
    homogeneous_layer.is_optimize = False
    homogeneous_layer.is_dispersive = True
    homogeneous_layer.is_solved = True

    assert homogeneous_layer.thickness == 2.0
    assert homogeneous_layer.material_name == 'Si'
    assert not homogeneous_layer.is_optimize
    assert homogeneous_layer.is_dispersive
    assert homogeneous_layer.is_solved

def test_homogeneous_layer_str(homogeneous_layer):
    assert str(homogeneous_layer) == 'HomogeneousLayer(thickness=1.0, material_name=SiO2)'

def test_grating_layer_str(grating_layer):
    assert str(grating_layer) == 'GratingLayer(thickness=1.0, material_name=SiO2)'

def test_homogeneous_layer_builder():
    builder = HomogeneousLayerBuilder()
    builder.create_layer()
    builder.update_thickness(1.0)
    builder.update_material_name('SiO2')
    builder.set_optimize(True)
    builder.set_dispersive(False)

    layer = builder.get_layer_instance()
    assert layer.thickness == 1.0
    assert layer.material_name == 'SiO2'
    assert layer.is_optimize
    assert not layer.is_dispersive

def test_grating_layer_builder():
    builder = GratingLayerBuilder()
    builder.create_layer()
    builder.update_thickness(1.0)
    builder.update_material_name('SiO2')
    builder.set_optimize(True)
    builder.set_dispersive(False)

    layer = builder.get_layer_instance()
    assert layer.thickness == 1.0
    assert layer.material_name == 'SiO2'
    assert layer.is_optimize
    assert not layer.is_dispersive

def test_layer_director():
    director = LayerDirector()
    layer = director.build_layer('homogeneous', 1.0, 'SiO2', True, False)
    assert isinstance(layer, HomogeneousLayer)
    assert layer.thickness == 1.0
    assert layer.material_name == 'SiO2'
    assert layer.is_optimize
    assert not layer.is_dispersive

    layer = director.build_layer('grating', 1.0, 'SiO2', True, False)
    assert isinstance(layer, GratingLayer)
    assert layer.thickness == 1.0
    assert layer.material_name == 'SiO2'
    assert layer.is_optimize
    assert not layer.is_dispersive

@pytest.fixture
def layer_manager():
    lattice_t1 = torch.tensor([1.0, 0.0])
    lattice_t2 = torch.tensor([0.0, 1.0])
    vec_p = torch.linspace(-0.5, 0.5, 10)
    vec_q = torch.linspace(-0.5, 0.5, 10)
    return LayerManager(lattice_t1, lattice_t2, vec_p, vec_q)

def test_add_layer(layer_manager):
    layer_manager.add_layer('homogeneous', torch.tensor(1.0), 'SiO2', True, False)
    assert len(layer_manager.layers) == 1
    assert isinstance(layer_manager.layers[0], HomogeneousLayer)

def test_replace_layer_to_homogeneous(layer_manager):
    layer_manager.add_layer('grating', torch.tensor(1.0), 'SiO2', True, False)
    layer_manager.replace_layer_to_homogeneous(0)
    assert isinstance(layer_manager.layers[0], HomogeneousLayer)

def test_replace_layer_to_grating(layer_manager):
    layer_manager.add_layer('homogeneous', torch.tensor(1.0), 'SiO2', True, False)
    layer_manager.replace_layer_to_grating(0)
    assert isinstance(layer_manager.layers[0], GratingLayer)

def test_update_layer_thickness(layer_manager):
    layer_manager.add_layer('homogeneous', torch.tensor(1.0), 'SiO2', True, False)
    layer_manager.update_layer_thickness(0, torch.tensor(2.0))
    assert layer_manager.layers[0].thickness == 2.0

def test_update_trn_layer(layer_manager):
    layer_manager.update_trn_layer('SiO2', True)
    assert layer_manager.trn_material_name == 'SiO2'
    assert layer_manager.is_trn_dispersive

def test_update_ref_layer(layer_manager):
    layer_manager.update_ref_layer('SiO2', True)
    assert layer_manager.ref_material_name == 'SiO2'
    assert layer_manager.is_ref_dispersive

def test_gen_toeplitz_matrix(layer_manager):
    layer_manager.add_layer('homogeneous', torch.tensor(1.0), 'SiO2', True, False)
    layer_manager.layers[0].ermat = torch.ones((10, 10))
    layer_manager.gen_toeplitz_matrix(0, 3, 3, 'er', 'FFT')
    # Check that kermat is a tensor with expected shape
    assert isinstance(layer_manager.layers[0].kermat, torch.Tensor)
    assert layer_manager.layers[0].kermat.shape == (9, 9)  # 3*3 = 9