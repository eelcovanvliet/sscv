from hivemind.structural import Structure
from hivemind.naval import Naval
from hivemind.abstracts import ParameterSet, Parameter, units
from gmsh_utils import utils
from gmsh_utils import mesh
import gmsh
import numpy as np
from pathlib import Path
occ = gmsh.model.occ


m = units.meter

class SSCVDesign(ParameterSet):
    VesselLength:Parameter = 100 * m
    VesselWidth:Parameter = 70 * m
    DeckHeight:Parameter = 10*m
    
    PontoonLength:Parameter = 100 * m
    PontoonWidth:Parameter = 15 * m
    PontoonHeight:Parameter = 10 * m

    NumberOfColumnsPerPontoon: Parameter = 3 * units.dimensionless
    ColumnLength:Parameter = 20 * m
    ColumnWidth:Parameter = 10 * m
    ColumnHeight:Parameter = 15 * m


class SSCVStructure(Structure):
    
    mesh: str|None = None

    def __init__(self, parameters:SSCVDesign) -> None:
        self._parameters = parameters

    def change_state(self):
        raise NotImplementedError()
        
    def get_inertia(self):
        raise NotImplementedError()
        
    def get_mesh(self, file, show=False):
        file = Path(file)
        if file.suffix != '.msh':
            raise ValueError(f'Expected a .msh extesions, got {file.suffix}')
        file = file.with_suffix('.msh')
        
        utils.start('sscv')
        l = self.parameters.PontoonLength['m']
        w = self.parameters.PontoonWidth['m']
        h = self.parameters.PontoonHeight['m']
        pontoon_sb = utils.VolumeComponent(3, occ.add_box(0,0,0, l,w,h))
        pontoon_ps = pontoon_sb.copy()

        vw = self.parameters.VesselWidth['m']
        pontoon_ps.translate(dy=vw-w)

        cl = self.parameters.ColumnLength['m']
        cw = self.parameters.ColumnWidth['m']
        ch = self.parameters.ColumnHeight['m']
        column_template = utils.VolumeComponent(3, occ.add_box(0,0,0, cl,cw,ch))

        n = self.parameters.NumberOfColumnsPerPontoon[None]
        xlocations = np.linspace(0+2, l-2-cl, n)
        columns = []
        for y in [(w-cw)/2, vw-w+(w-cw)/2]:
            for x in xlocations:
                column = column_template.copy()
                column.translate(dx=x, dy=y, dz=h)
                columns.append(column)
        
        column_template.remove(recursive=True)
        sscv = pontoon_ps.fuse([pontoon_sb] + columns)
        
        
        # Mesh surfaces
        mesh.mesh_surfaces(1)
        
        # Write mesh file and store content in instance.
        gmsh.write(str(file))
        with open(file, 'r') as f:
            self.mesh = f.read()
        
        if show:
            utils.ui.start_ui(mode='mesh')
        return self.mesh




        
    
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

    my_struct.get_mesh(file='sscv.msh', show=True)

    naval_parameters = SSCVNavalParameters()
    my_nav = SSCVNaval(parameters=naval_parameters, structure=my_struct)
    
    my_nav.structure.parameters