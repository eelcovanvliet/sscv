from hivemind.structural import Structure
from hivemind.naval import Naval
from hivemind.abstracts import ParameterSet, Parameter, units


m = units.meter

class SSCVDesign(ParameterSet):
    Length:Parameter = 100 * m
    Width:Parameter = 40 * m
    Depth:Parameter = 40 * m


class SSCVStructure(Structure):
    
    def __init__(self, parameters:SSCVDesign) -> None:
        self._parameters = parameters

    def change_state(self):
        raise NotImplementedError()
        
    def get_inertia(self):
        raise NotImplementedError()
        
    def get_mesh(self):
        raise NotImplementedError()
    
    @property
    def parameters(self) -> SSCVDesign:
        return self._parameters
    
    @property
    def possible_states(self):
        raise NotImplementedError()
    
    @property
    def previous_state(self):
        raise NotImplementedError()
    
    @property
    def state(self):
        raise NotImplementedError()


class SSCVNavalParameters(ParameterSet):
    WaterDepth:Parameter = 100 * m
    WaterDensity:Parameter = 1025 * units.kg/m**3

class SSCVNaval(Naval):
    
    def __init__(self, parameters:SSCVNavalParameters, structure:SSCVStructure):
        self._parameters = parameters
        self._structure = structure

    def change_state(self):
        raise NotImplementedError()

    def get_natural_periods(self):
        raise NotImplementedError()

    def get_stability(self):
        raise NotImplementedError()

    @property
    def parameters(self) -> SSCVNavalParameters:
        return self._parameters

    @property
    def possible_states(self):
        raise NotImplementedError()

    @property
    def previous_state(self):
        raise NotImplementedError()

    @property
    def state(self):
        raise NotImplementedError()

    @property
    def structure(self) -> SSCVStructure:
        return self._structure


if __name__ == "__main__":

    design = SSCVDesign()
    my_struct = SSCVStructure(design)

    naval_parameters = SSCVNavalParameters()
    my_nav = SSCVNaval(parameters=naval_parameters, structure=my_struct)
    
    my_nav.structure.parameters