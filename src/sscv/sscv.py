from hivemind.structural import Structure
from hivemind.naval import Naval
from hivemind.abstracts import ParameterSet, Parameter, units
from gmsh_utils import utils
from gmsh_utils import mesh
import gmsh
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
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
    

    

    def get_geometry(self, show=False) -> utils.VolumeComponent:
        """Start a new gmsh sessions and create the geometry of the sscv
        
        Parameters:
            show:bool
                If True, launches the gmsh gui to show the sscv. Default is False.

        Return:
            geometry:utils.VolumeComponent
                The VolumeComponent that contains all volumes that make up the sscv

        """
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
        geometry = pontoon_ps.fuse([pontoon_sb] + columns)
        

        if show:
            utils.ui.start_ui(mode='geo')
        return geometry
    
    def cut_geometry(self, geometry:utils.VolumeComponent, draft:float, roll:float):
        """Take the geometry and peforms a "cut" using a plane based on draft and roll
        
        Parameters:
            geometry:utils.VolumeComponent
                The geometry as returned by .get_geometry()

        Return:
            water_crossing_surfaces:List[Tuple]
                gmsh dimTags of the surfaces outlined by the sscv water line.

        """
        l = self.parameters.VesselLength['m']
        w = self.parameters.VesselWidth['m']

        geometry.translate(dy=-w/2, dz=-draft).rotate(0,0,0, 1,0,0, np.deg2rad(roll))

        # Create slicing plane
        size = max(l,w)*1.1
        plane = utils.make_polygon([
            [   0,  0,  0],
            [size,  0,  0],
            [size,size, 0],
            [   0, size,0],
        ])
        plane.translate(dy=-size/2)
        
        # Fragment (general fuse) geometry using plane
        mapping = geometry.fragment([plane])
    
        # Find the areas that represent the water plane
        plane_fragments = mapping[2]
        water_crossing_surfaces = []
        for area in plane_fragments:
            volumes, lines = gmsh.model.get_adjacencies(2, area[1])
            is_part_of_volume = bool(len(volumes))
            if is_part_of_volume:
                water_crossing_surfaces.append(area)
        return water_crossing_surfaces
    
    def get_area_moment_of_inertia(self, areas:List[tuple]) -> Tuple[float, np.array, np.array]:
        """Calculate the combined surface area, centroid and moment of inertia of a list
        of area dimTags.
        
        Parameters:
            areas:List[tuple]
                List of gmsh area dimTags, e.g. [(2,1), (2,2), (2,3)] for areas 1, 2 and 3.

        Returns:
            surface_area:float
                Total surface area of the water crossing surfaces
            surface_center:np.array[float] size (3,)
                The global position of the area centroid
            surface_moi: np.array[float] size (3,3)
                The area moment of inertia matrix wrt the center_of_surface

        """
        # Get the inertial properties of each surface
        II, masses, cogs = [], [], []
        for area in areas:
            mass = gmsh.model.occ.get_mass(2, area[1]) # NOTE: mass == surface area
            cog = gmsh.model.occ.get_center_of_mass(2, area[1])
            I = gmsh.model.occ.get_matrix_of_inertia(2, area[1]) # NOTE: wrt cog
            masses.append(mass)
            cogs.append(cog)
            II.append(I)

        # Compute the combined inertial properties using parallel axis theorem
        surface_area, surface_center, surface_moi = utils.parallel_axis_theorem(masses, cogs, II)
        # NOTE: waterplane_moi wrt waterplane_center
        return surface_area, surface_center, surface_moi
        

    def get_waterplane_properties(self, draft:float, roll:float, show=False) -> Tuple[float, np.array, np.array]:
        """Compute and return the inertial properties of the surfaces outlined by the vessel water line.
        
        Parameters:
            draft:float
                The draft of the vessel wrt keel in meters
            roll:
                The roll of the vessel in degrees
        
        Returns:
            area:float
                Total surface area of the water crossing surfaces
            center_of_surface:np.array[float] size (3,)
                The global position of the area centroid
            area_moment_of_inertia: np.array[float] size (3,3)
                The area moment of inertia matrix wrt the center_of_surface
        
        """
        geometry = self.get_geometry()
        water_crossing_surfaces = self.cut_geometry(geometry, draft, roll)
        area, center_of_surface, area_moment_of_inertia = self.get_area_moment_of_inertia(water_crossing_surfaces)
        if show:
            utils.ui.show_entities(water_crossing_surfaces, recursive=True, suppress_others=True)
            utils.ui.start_ui(mode='geo')
        return area, center_of_surface, area_moment_of_inertia


    def get_mesh(self,  file, show=False):
        file = Path(file)
        if file.suffix != '.msh':
            raise ValueError(f'Expected a .msh extesions, got {file.suffix}')
        file = file.with_suffix('.msh')

        sscv = self.get_geometry()

        
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
    
    # # Calculate water line area for different drafts
    # height = design.PontoonHeight['m']
    # data = []
    # for draft in np.linspace(height-2, height+2, 10):
    #     area, centroid, amoi = my_struct.get_waterplane_properties(draft, 0, show=False)
    #     data.append([draft, area])
    
    # df = pd.DataFrame(data, columns='draft area'.split())
    # df.set_index('draft', inplace=True)
    # plt = df.plot()
    # plt.set_ylabel('water line surface [m^2]')    


    # Calculate water line area for different roll
    height = design.PontoonHeight['m']
    data = []
    for roll in np.linspace(0, 10, 10):
        area, centroid, amoi = my_struct.get_waterplane_properties(15, roll, show=False)
        data.append([roll, area])
    
    df = pd.DataFrame(data, columns='roll area'.split())
    df.set_index('roll', inplace=True)
    plt = df.plot()
    plt.set_ylabel('water line surface [m^2]')    



    my_struct.get_mesh(file='sscv.msh', show=True)

    naval_parameters = SSCVNavalParameters()
    my_nav = SSCVNaval(parameters=naval_parameters, structure=my_struct)
    
    my_nav.structure.parameters