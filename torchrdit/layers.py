""" This file defines all classes about layers. """
from abc import ABCMeta, abstractmethod
from .utils import tensor_params_check

class Layer(metaclass=ABCMeta):
    """Layer.
    This is the base class of Layer objects. This class is abstract.
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __str__(self) -> None:
        pass

class LayerBuilder(metaclass=ABCMeta):
    """LayerBuilder.
    This is the base class of the layer builder. This class is abstract.
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def create_layer(self):
        """create_layer.
        abstractmethod of creating layer.
        """

    @abstractmethod
    def update_thickness(self, thickness):
        """update_thickness.
        abstractmethod of updating layer thickness.

        Args:
            thickness:
        """

    @abstractmethod
    def update_material_name(self, material_name):
        """update_material_name.
        abstractmethod of updating material type.

        Args:
            material_name:
        """

    @abstractmethod
    def set_optimize(self, is_optimize):
        """set_optimize.
        abstractmethod of setting layer to be optimized.

        Args:
            is_optimize:
        """

    # read-only property
    @abstractmethod
    def get_layer_instance(self):
        """get_layer_instance.
        abstractmethod of getting layer object.
        """

    @abstractmethod
    def set_dispersive(self, is_dispersive):
        """set_dispersive.
        abstractmethod of set meterial dispersive.

        Args:
            is_dispersive:
        """

class HomogeneousLayer(Layer):
    """HomogeneousLayer.
    This class defines homogeneous layer.
    """

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
        return "HomogeneousLayer"

    @property
    def thickness(self):
        """thickness.
        returns thickness of the layer.
        """
        return self._thickness

    @thickness.setter
    def thickness(self, thickness):
        """thickness.
        update thickness.

        Args:
            thickness:
        """
        self._thickness = thickness

    @property
    def material_name(self):
        """material_name.
        returns material name.
        """
        return self._material_name

    @material_name.setter
    def material_name(self, material_name):
        """material_name.
        updates material name.

        Args:
            material_name:
        """
        self._material_name = material_name

    # read-only property
    @property
    def is_homogeneous(self):
        """is_homogeneous.
        returns whether homogenous.
        """
        return self._is_homogeneous

    @property
    def is_dispersive(self):
        """is_dispersive.
        returns whether dispersive.
        """
        return self._is_dispersive

    @is_dispersive.setter
    def is_dispersive(self, is_dispersive):
        """is_dispersive.
        set material dispersive.

        Args:
            is_dispersive:
        """
        self._is_dispersive = is_dispersive

    @property
    def is_optimize(self):
        """is_optimize.
        returns whether layer is to be optimized.
        """
        return self._is_optimize

    @is_optimize.setter
    def is_optimize(self, is_optimize):
        """is_optimize.
        set layer to be optimized.

        Args:
            is_optimize:
        """
        self._is_optimize = is_optimize

    @property
    def is_solved(self):
        """is_solved.

        returns whether layer is solved.
        """
        return self._is_solved

    @is_solved.setter
    def is_solved(self, is_solved):
        """is_solved.
        set layer as solved.

        Args:
            is_solved:
        """
        self._is_solved = is_solved

class GratingLayer(HomogeneousLayer):
    """GratingLayer.

    This class defines grating layer.
    """

    def __init__(self, thickness=0, material_name='', is_optimize=False) -> None:
        super().__init__(thickness, material_name, is_optimize)
        self._is_homogeneous = False

    def __str__(self) -> str:
        return "GratingLayer"

class HomogeneousLayerBuilder(LayerBuilder):
    """HomogeneousLayerBuilder.
    This class defines the builder for HomogeneousLayer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.layer = None

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
    """GratingLayerBuilder.
    This class defines the builder for GratingLayer.
    """
    def __init__(self) -> None:
        super().__init__()

    def create_layer(self):
        self.layer = GratingLayer()

class LayerDirector:
    """LayerDirector.
    This class defines the director class for layer builders.
    """

    def __init__(self) -> None:
        pass

    def build_layer(self, layer_type, thickness, material_name, is_optimize = False, is_dispersive = False) -> Layer:
        """build_layer.
        This function runs the actual building process for different types of layers.

        Args:
            layer_type:
            thickness:
            material_name:
            is_optimize:
            is_dispersive:

        Returns:
            Layer:
        """
        if layer_type == 'homogeneous':
            layer_builder = HomogeneousLayerBuilder()
        elif layer_type == 'grating':
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
        """add_layer.
        This function adds new layer object to the layer manager.

        Args:
            layer_type:
            thickness:
            material_name:
            is_optimize:
            is_dispersive:
        """
        new_layer = self.layer_director.build_layer(layer_type=layer_type,
                                                    thickness=thickness,
                                                    material_name=material_name,
                                                    is_optimize=is_optimize,
                                                    is_dispersive=is_dispersive)
        self.layers.append(new_layer)

    def replace_layer_to_homogeneous(self, layer_index):
        """replace_layer_to_homogeneous.
        This fucntion change the type of the layer to homogenous.

        Args:
            layer_index:
        """
        new_layer = self.layer_director.build_layer(layer_type='homogenous',
                                                    thickness=self.layers[layer_index].thickness,
                                                    material_name=self.layers[layer_index].material_name,
                                                    is_optimize=self.layers[layer_index].is_optimize,
                                                    is_dispersive=self.layers[layer_index].is_dispersive)
        self.layers[layer_index] = new_layer

    def replace_layer_to_grating(self, layer_index):
        """replace_layer_to_grating.
        This fucntion change the type of the layer to grating.

        Args:
            layer_index:
        """
        new_layer = self.layer_director.build_layer(layer_type='grating',
                                                    thickness=self.layers[layer_index].thickness,
                                                    material_name=self.layers[layer_index].material_name,
                                                    is_optimize=self.layers[layer_index].is_optimize,
                                                    is_dispersive=self.layers[layer_index].is_dispersive)
        self.layers[layer_index] = new_layer

    @tensor_params_check(check_start_index=2, check_stop_index=2)
    def update_layer_thickness(self, layer_index, thickness):
        """update_layer_thickness.
        This function updates the thickness of the layer.

        Args:
            layer_index:
            thickness:
        """
        self.layers[layer_index].thickness = thickness

    def update_trn_layer(self, material_name: str, is_dispersive: bool):
        """update_trn_layer.
        This function updates the transmission layer.

        Args:
            material_name (str): material_name
            is_dispersive (bool): is_dispersive
        """
        self._trn_material_name = material_name
        self._is_trn_dispers = is_dispersive

    def update_ref_layer(self, material_name: str, is_dispersive: bool):
        """update_ref_layer.
        This function updates the reflection layer.

        Args:
            material_name (str): material_name
            is_dispersive (bool): is_dispersive
        """
        self._ref_material_name = material_name
        self._is_ref_dispers = is_dispersive

    @property
    def ref_material_name(self) -> str:
        """ref_material_name.
        returns the material name of the reflection layer.

        Args:

        Returns:
            str:
        """
        return self._ref_material_name

    @property
    def trn_material_name(self) -> str:
        """trn_material_name.
        returns the material name of the transmission layer.

        Args:

        Returns:
            str:
        """
        return self._trn_material_name

    @property
    def is_ref_dispersive(self) -> bool:
        """is_ref_dispersive.
        returns whether reflection layer material is dispersive.

        Args:

        Returns:
            bool:
        """
        return self._is_ref_dispers

    @property
    def is_trn_dispersive(self) -> bool:
        """is_trn_dispersive.
        returns whether transmission layer material is dispersive.

        Args:

        Returns:
            bool:
        """
        return self._is_trn_dispers

    @property
    def nlayer(self) -> int:
        """nlayer.
        return number of layers in the model.

        Args:

        Returns:
            int:
        """
        return len(self.layers)
