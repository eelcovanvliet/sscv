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
    VesselLength: Parameter = 160 * m
    VesselWidth: Parameter = 92 * m
    DeckHeight: Parameter = 10*m

    NumberOfPontoons: Parameter = 2 * units.dimensionless
    PontoonLength: Parameter = 155 * m
    PontoonWidth: Parameter = 30 * m
    PontoonHeight: Parameter = 12.5 * m
    PontoonCb: Parameter = 0.85 * units.dimensionless
    PontoonVCB: Parameter = 0.5 * units.dimensionless
    PontoonLCB: Parameter = 0.5 * units.dimensionless

    NumberOfColumnsPerPontoon: Parameter = 3 * units.dimensionless
    ColumnLength: Parameter = 38 * m
    ColumnWidth: Parameter = 16 * m
    ColumnHeight: Parameter = 23 * m
    ColumnCb: Parameter = 0.97 * units.dimensionless
    ColumnVCB: Parameter = 0.5 * units.dimensionless
    ColumnLCB: Parameter = 0.5 * units.dimensionless

    # NumberOfCranes: Parameter = 2 * units.dimensionless
    # CraneCapacity: Parameter = 5000 * units.
    # CraneBoomMassFactor: Parameter = 0.25 * units.dimensionless
    # CraneLiftingHeight: Parameter = 120 * m
    # CraneRadius: Parameter = 40 * m

    LightshipWeightFactor: Parameter = 0.24 * units.dimensionless


class SSCVStructure(Structure):

    mesh: str | None = None

    def __init__(self, parameters: SSCVDesign) -> None:
        self._parameters = parameters
        self._hydrostatics = self.get_hydrostatics()

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
        pontoon_sb = utils.VolumeComponent(3, occ.add_box(0, 0, 0, l, w, h))
        pontoon_ps = pontoon_sb.copy()

        vw = self.parameters.VesselWidth['m']
        pontoon_ps.translate(dy=vw-w)

        cl = self.parameters.ColumnLength['m']
        cw = self.parameters.ColumnWidth['m']
        ch = self.parameters.ColumnHeight['m']
        column_template = utils.VolumeComponent(3, occ.add_box(0, 0, 0, cl, cw, ch))

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

    def get_waterplane_area(self, draft: float):
        # Todo add assessment if draft in range of drafts

        if (draft >= 0) and (draft <= self.parameters.PontoonHeight['m']):
            waterplane_area = self.parameters.PontoonLength['m'] * self.parameters.PontoonWidth['m'] * self.parameters.PontoonCb[None] * self.parameters.NumberOfPontoons[None] 
        elif (draft > self.parameters.PontoonHeight['m']) and (draft <= self.parameters.PontoonHeight['m'] + self.parameters.ColumnHeight['m']):
            waterplane_area = self.parameters.ColumnLength['m'] * self.parameters.ColumnWidth['m'] * self.parameters.ColumnCb[None] * self.parameters.NumberOfColumnsPerPontoon[None] * self.parameters.NumberOfPontoons[None]   
        else:
            raise NotImplementedError()

        return waterplane_area

    def get_displacement(self, draft: float):
        # Todo add assessment if draft in range of drafts

        if (draft >= 0) and (draft <= self.parameters.PontoonHeight['m']):
            displacement = self.parameters.PontoonLength['m'] * self.parameters.PontoonWidth['m'] * draft * self.parameters.PontoonCb[None] * self.parameters.NumberOfPontoons[None]
        elif (draft > self.parameters.PontoonHeight['m']) and (draft <= self.parameters.PontoonHeight['m'] + self.parameters.ColumnHeight['m']):
            displacement_submerged_pontoon = self.parameters.PontoonLength['m'] * self.parameters.PontoonWidth['m'] * self.parameters.PontoonCb[None] * self.parameters.PontoonHeight['m'] * self.parameters.NumberOfPontoons[None]
            displacement = displacement_submerged_pontoon + self.parameters.ColumnLength['m'] * self.parameters.ColumnWidth['m'] * (draft - self.parameters.PontoonHeight['m']) *self.parameters.ColumnCb[None] * self.parameters.NumberOfColumnsPerPontoon[None] * self.parameters.NumberOfPontoons[None]
        else:
            raise NotImplementedError()

        return displacement

    def get_center_of_buoyancy(self, draft: float):

        if (draft >= 0) and (draft <= self.parameters.PontoonHeight['m']):
            vertical_center_of_buoyancy = self.parameters.PontoonVCB[None] * draft
            longitudional_center_of_buoyancy = self.parameters.PontoonLCB[None] * self.parameters.PontoonLength['m']
        elif (draft > self.parameters.PontoonHeight['m']) and (draft <= self.parameters.PontoonHeight['m'] + self.parameters.ColumnHeight['m']):
            displacement_pontoons = self.get_displacement(self.parameters.PontoonHeight['m'])
            displacement_columns = self.get_displacement(draft) - displacement_pontoons
            vertical_center_of_buoyancy = (displacement_pontoons * self.parameters.PontoonVCB[None] * self.parameters.PontoonHeight['m'] + displacement_columns * (self.parameters.PontoonHeight['m'] + self.parameters.ColumnVCB[None] * (draft - self.parameters.PontoonHeight['m']))) / self.get_displacement(draft)
            longitudional_center_of_buoyancy = 0  # ToDo
        else:
            raise NotImplementedError()

        center_of_buoyancy = [longitudional_center_of_buoyancy, 0, vertical_center_of_buoyancy]

        return center_of_buoyancy

    def get_moment_of_waterplane_area(self, draft: float):

        if (draft >= 0) and (draft <= self.parameters.PontoonHeight['m']):
            It = (1 / 12 * self.parameters.PontoonCb[None] * self.parameters.PontoonLength['m'] * self.parameters.PontoonWidth['m'] ** 3
                  + self.parameters.PontoonCb[None] * self.parameters.PontoonWidth['m'] * self.parameters.PontoonLength['m'] * (self.parameters.VesselWidth['m'] / 2 - self.parameters.PontoonWidth['m']/2) ** 2) * self.parameters.NumberOfPontoons[None]
            Il = 1 / 12 * self.parameters.PontoonCb[None] * self.parameters.PontoonWidth['m'] * self.parameters.PontoonLength['m'] ** 3 * self.parameters.NumberOfPontoons[None]
        elif (draft > self.parameters.PontoonHeight['m']) and (draft <= self.parameters.PontoonHeight['m'] + self.parameters.ColumnHeight['m']):
            It = (1 / 12 * self.parameters.ColumnCb[None] * self.parameters.ColumnLength['m'] + self.parameters.ColumnWidth['m'] ** 3
                  + self.parameters.ColumnCb[None] * self.parameters.ColumnWidth['m'] * self.parameters.ColumnLength['m'] * (self.parameters.VesselWidth['m'] / 2 - self.parameters.ColumnWidth['m'] / 2) ** 2) * self.parameters.NumberOfPontoons[None] * self.parameters.NumberOfColumnsPerPontoon[None] # ToDo check -> add location of columns  
            Il = 0  # ToDo
        else:
            raise NotImplementedError()

        return It, Il

    def get_km(self, draft: float):
        It, Il = self.get_moment_of_waterplane_area(draft)
        displacement = self.get_displacement(draft)
        center_of_buoyancy = self.get_center_of_buoyancy(draft)

        KMt = It / displacement + center_of_buoyancy[2]
        KMl = Il / displacement + center_of_buoyancy[2]

        return KMt, KMl

    def get_hydrostatics(self):
        pass

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

    @property
    def hydrostatics(self):
        return self._hydrostatics


class SSCVNavalParameters(ParameterSet):
    WaterDepth: Parameter = 100 * m
    WaterDensity: Parameter = 1025 * units.kg/m**3


class SSCVNaval(Naval):

    def __init__(self, parameters: SSCVNavalParameters, structure: SSCVStructure):
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
    draft = 10

    # my_struct.get_mesh()

    my_struct.get_waterplane_area(draft)
    my_struct.get_displacement(draft)
    my_struct.get_moment_of_waterplane_area(draft)
    my_struct.get_center_of_buoyancy(draft)
    naval_parameters = SSCVNavalParameters()
    my_nav = SSCVNaval(parameters=naval_parameters, structure=my_struct)

    my_nav.structure.parameters
