import numpy as np


from abc import ABCMeta, abstractmethod
from .utils import tensor_params_check

class Layer(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __str__(self) -> None:
        pass

class LayerBuilder(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self) -> None:
        pass
        
    @abstractmethod
    def create_layer(self):
        pass

    @abstractmethod
    def update_thickness(self, thickness):
        pass
    
    @abstractmethod
    def update_material_name(self, material_name):
        pass

    @abstractmethod
    def set_optimize(self, is_optimize):
        pass

    # read-only property
    @abstractmethod
    def get_layer_instance(self):
        pass

    @abstractmethod
    def set_dispersive(self, is_dispersive):
        pass

class HomogeneousLayer(Layer):
    def __init__(self, thickness = 0.0, material_name = '', is_optimize = False) -> None:
        self._thickness = thickness
        self._material_name = material_name 
        self._is_homogeneous = True
        self._is_optimize = is_optimize
        self._is_dispersive = False
        self._is_solved = False

        self.ermat = None 
        self.urmat = None 
        self.kermat = None
        self.kurmat = None

    def __str__(self) -> str:
        return f"HomogeneousLayer" 

    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        self._thickness = thickness

    @property
    def material_name(self):
        return self._material_name

    @material_name.setter
    def material_name(self, material_name):
        self._material_name = material_name

    # read-only property
    @property
    def is_homogeneous(self):
        return self._is_homogeneous

    @property
    def is_dispersive(self):
        return self._is_dispersive

    @is_dispersive.setter
    def is_dispersive(self, is_dispersive):
        self._is_dispersive = is_dispersive

    @property
    def is_optimize(self):
        return self._is_optimize

    @is_optimize.setter
    def is_optimize(self, is_optimize):
        self._is_optimize = is_optimize

    @property
    def is_solved(self):
        return self._is_solved

    @is_solved.setter
    def is_solved(self, is_solved):
        self._is_solved = is_solved

class GratingLayer(HomogeneousLayer):
    def __init__(self, thickness=0, material_name='', is_optimize=False) -> None:
        super().__init__(thickness, material_name, is_optimize)
        self._is_homogeneous = False

    def __str__(self) -> str:
        return f"GratingLayer"

class HomogeneousLayerBuilder(LayerBuilder):
    def __init__(self) -> None:
        pass

    def create_layer(self):
        self.layer = HomogeneousLayer()

    # @tensor_params_check
    def update_thickness(self, thickness):
        self.layer.thickness = thickness

    def update_material_name(self, material_name):
        self.layer.material_name = material_name

    def set_optimize(self, is_optimize = False):
        self.layer.is_optimize = is_optimize

    def get_layer_instance(self):
        return self.layer

    def set_dispersive(self, is_dispersive):
        self.layer.is_dispersive = is_dispersive

class GratingLayerBuilder(HomogeneousLayerBuilder):
    def __init__(self) -> None:
        pass

    def create_layer(self):
        self.layer = GratingLayer()

class LayerDirector:
    def __init__(self) -> None:
        pass
        
    def build_layer(self, layer_type, thickness, material_name, is_optimize = False, is_dispersive = False) -> Layer:
        if (layer_type == 'homogeneous'):
            layer_builder = HomogeneousLayerBuilder()
        elif (layer_type == 'grating'):
            layer_builder = GratingLayerBuilder()
        else:
            layer_builder = HomogeneousLayerBuilder()

        # layer_builder.create_layer(thickness=thickness, material_name=material_name, is_optimize=is_optimize)
        layer_builder.create_layer()
        layer_builder.update_thickness(thickness=thickness)
        layer_builder.update_material_name(material_name=material_name)
        layer_builder.set_optimize(is_optimize=is_optimize)
        layer_builder.set_dispersive(is_dispersive=is_dispersive)

        return layer_builder.get_layer_instance()

class LayerManager:
    """LayerManager.

    A class to manage layer instances.
    """
    
    def __init__(self, n_batches) -> None:
        self.n_batches = n_batches
        self.layers = []
        self.layer_director = LayerDirector()

        # semi-infinite layer
        self._ref_material_name = 'air'
        self._trn_material_name = 'air'

        self._is_ref_dispers = False
        self._is_trn_dispers = False

    @tensor_params_check(check_start_index=2, check_stop_index=2)
    def add_layer(self, layer_type, thickness, material_name, is_optimize = False, is_dispersive = False):
        new_layer = self.layer_director.build_layer(layer_type=layer_type,
                                                    thickness=thickness,
                                                    material_name=material_name,
                                                    is_optimize=is_optimize,
                                                    is_dispersive=is_dispersive)
        self.layers.append(new_layer)

    def replace_layer_to_homogeneous(self, layer_index):
        new_layer = self.layer_director.build_layer(layer_type='homogenous',
                                                    thickness=self.layers[layer_index].thickness,
                                                    material_name=self.layers[layer_index].material_name,
                                                    algorithm=self.layers[layer_index].sim_algorithm,
                                                    is_optimize=self.layers[layer_index].is_optimize,
                                                    is_dispersive=self.layers[layer_index].is_dispersive)
        self.layers[layer_index] = new_layer

    def replace_layer_to_grating(self, layer_index):
        new_layer = self.layer_director.build_layer(layer_type='grating',
                                                    thickness=self.layers[layer_index].thickness,
                                                    material_name=self.layers[layer_index].material_name,
                                                    is_optimize=self.layers[layer_index].is_optimize,
                                                    is_dispersive=self.layers[layer_index].is_dispersive)
        self.layers[layer_index] = new_layer

    @tensor_params_check(check_start_index=2, check_stop_index=2)
    def update_layer_thickness(self, layer_index, thickness):
        self.layers[layer_index].thickness = thickness

    def update_trn_layer(self, material_name: str, is_dispersive: bool):
        self._trn_material_name = material_name
        self._is_trn_dispers = is_dispersive

    def update_ref_layer(self, material_name: str, is_dispersive: bool):
        self._ref_material_name = material_name
        self._is_ref_dispers = is_dispersive

    @property
    def ref_material_name(self) -> str:
        return self._ref_material_name

    @property
    def trn_material_name(self) -> str:
        return self._trn_material_name

    @property
    def is_ref_dispersive(self) -> bool:
        return self._is_ref_dispers

    @property
    def is_trn_dispersive(self) -> bool:
        return self._is_trn_dispers

    @property
    def nlayer(self) -> int:
        return len(self.layers)


