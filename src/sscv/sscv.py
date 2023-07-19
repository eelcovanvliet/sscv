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
d = units.dimensionless
mT = units.metric_ton

GRAVITYCONSTANT = 9.81
WATERDENSITY = 1.025


class SSCVDesign(ParameterSet):
    VesselLength: Parameter = 0 * m
    VesselWidth: Parameter = 0 * m
    DeckHeight: Parameter = 0 * m

    NumberOfPontoons: Parameter = 0 * d
    PontoonLength: Parameter = 0 * m
    PontoonWidth: Parameter = 0 * m
    PontoonHeight: Parameter = 0 * m
    PontoonCb: Parameter = 0 * d
    PontoonVCB: Parameter = 0 * d
    PontoonLCB: Parameter = 0 * d
    PontoonBallastCapacity: Parameter = 0 * d

    NumberOfColumnsPerPontoon: Parameter = 0 * d
    ColumnLength: Parameter = 0 * m
    ColumnWidth: Parameter = 0 * m
    ColumnHeight: Parameter = 0 * m
    ColumnCb: Parameter = 0 * d
    ColumnVCB: Parameter = 0 * d
    ColumnLCB: Parameter = 0 * d
    ColumnBallastCapacity: Parameter = 0 * d

    NumberOfCranes: Parameter = 0 * d
    CraneMaxCapacity: Parameter = 0 * mT
    CraneBoomMassFactor: Parameter = 0 * d
    CraneMaxLiftingHeight: Parameter = 0 * m
    CraneMaxRadius: Parameter = 0 * m

    LightshipWeightFactor: Parameter = 0 * d
    LightshipWeightLCG: Parameter = 0 * d
    LightshipWeightVCG: Parameter = 0 * d
    LightshipWeightKxx: Parameter = 0 * d
    LightshipWeightKyy: Parameter = 0 * d


class SSCVStructure(Structure):

    mesh: str | None = None

    def __init__(self, parameters: SSCVDesign) -> None:
        self._parameters = parameters
        self._hullcomponents = self.define_hull_components()
        self._hydrostatics = self.get_hydrostatics(draft_max=(self.parameters.PontoonHeight['m'] + self.parameters.ColumnHeight['m']))

    def change_state(self):
        raise NotImplementedError()

    def get_inertia(self):
        """Returns the inertia of the vessel  # ToDo, included added masses, and rzz -> return 6D matrix instead of separate values

        Returns:
            mass6D: mass matrix of vessel [mT]
            CoG: center of gravity of vessel w.r.t pontoon aft, centerline, keel [m, m, m]
        """
        vw = self.parameters.VesselWidth['m']
        pl = self.parameters.PontoonLength['m']
        ph = self.parameters.PontoonHeight['m']
        ch = self.parameters.ColumnHeight['m']
        dh = self.parameters.DeckHeight['m']

        # Mass matrix
        mass6D = np.eye(6)
        displacement = self.get_displacement(ph+ch+dh)
        Lightship = displacement * self.parameters.LightshipWeightFactor[None]

        kxx = self.parameters.LightshipWeightKxx[None] * vw
        kyy = self.parameters.LightshipWeightKyy[None] * pl
        kzz = 0  # Todo

        mass6D = np.diag([Lightship, Lightship, Lightship, kxx**2 * Lightship, kyy**2 * Lightship, kzz ** 2 * Lightship])

        # Center of Gravity
        longitudional_center_of_gravity = pl * self.parameters.LightshipWeightLCG[None]
        transverse_center_of_gravity = 0
        vertical_center_of_gravity = (ph + ch + dh) * self.parameters.LightshipWeightVCG[None]

        CoG = [longitudional_center_of_gravity, transverse_center_of_gravity, vertical_center_of_gravity]

        return mass6D, CoG

    def get_mesh(self, file, show=False):
        file = Path(file)
        if file.suffix != '.msh':
            raise ValueError(f'Expected a .msh extesions, got {file.suffix}')
        file = file.with_suffix('.msh')

    

    def define_hull_components(self) -> pd.DataFrame:  # ?? is df a usefull way to do this? Or use a class?
        vl = self.parameters.VesselLength['m']
        vw = self.parameters.VesselWidth['m']
        dh = self.parameters.DeckHeight['m']

        pl = self.parameters.PontoonLength['m']
        pw = self.parameters.PontoonWidth['m']
        ph = self.parameters.PontoonHeight['m']
        pcb = self.parameters.PontoonCb[None]
        plcb = self.parameters.PontoonLCB[None]
        pvcb = self.parameters.PontoonVCB[None]
        pballast = self.parameters.PontoonBallastCapacity[None]

        cl = self.parameters.ColumnLength['m']
        cw = self.parameters.ColumnWidth['m']
        ch = self.parameters.ColumnHeight['m']
        ccb = self.parameters.ColumnCb[None]
        clcb = self.parameters.ColumnLCB[None]
        cvcb = self.parameters.ColumnVCB[None]
        cballast = self.parameters.ColumnBallastCapacity[None]

        hull_components = pd.DataFrame()

        # Define pontoon locations
        y_pontoon_distance = []

        if self.parameters.NumberOfPontoons == 1:
            y_pontoon_distance = [0]
        else:
            y_pontoon_gaps = (vw - pw)/(self.parameters.NumberOfPontoons[None] - 1)

        for pontoon in range(self.parameters.NumberOfPontoons[None]):
            y_pontoon_distance.append(vw/2 - pw/2 - pontoon * y_pontoon_gaps)

        pontoonName = 1
        for pontoonIndex, _ in enumerate(range(self.parameters.NumberOfPontoons[None])):
            temp_df = pd.DataFrame({
                    'Name': [f'P{str(pontoonName)}'],
                    'Length': [pl],
                    'Width': [pw],
                    'Height': [ph],
                    'Cb': [pcb],
                    'LCB': [pvcb],
                    'VCB': [plcb],
                    'x': [pl/2],
                    'y': [y_pontoon_distance[pontoonIndex]],
                    'z': [0],
                    'ballastVolume': [pl*pw*ph*pballast],
                })

            pontoonName += 1
            hull_components = pd.concat([hull_components, temp_df], ignore_index=True)

        # Define column locations
        x_column_distance = []
        x_column_gaps = (pl - cl)/(self.parameters.NumberOfColumnsPerPontoon[None] - 1)
        for column in range(self.parameters.NumberOfColumnsPerPontoon[None]):
            x_column_distance.append(cl/2 + column * x_column_gaps)

        y_column_distance = [vw / 2 - cw / 2, -vw / 2 + cw / 2]  # !! Check how this works with using different amount of pontoons

        columnName = 1
        for columnIndex, _ in enumerate(range(self.parameters.NumberOfColumnsPerPontoon[None])):
            for pontoonIndex, _ in enumerate(range(self.parameters.NumberOfPontoons[None])):
                temp_df = pd.DataFrame({
                    'Name': [f'C{str(columnName)}'],
                    'Length': [cl],
                    'Width': [cw],
                    'Height': [ch],
                    'Cb': [ccb],
                    'LCB': [cvcb],
                    'VCB': [clcb],
                    'x': [x_column_distance[columnIndex]],
                    'y': [y_column_distance[pontoonIndex]],
                    'z': [ph],
                    'ballastVolume': [cl*cw*ch*cballast],
                })

                columnName += 1
                hull_components = pd.concat([hull_components, temp_df], ignore_index=True)

        # Define deckbox
        temp_df = pd.DataFrame({
                    'Name': [f'D1'],
                    'Length': [vl],
                    'Width': [vw],
                    'Height': [dh],
                    'Cb': [1],
                    'LCB': [0.5],
                    'VCB': [0.5],
                    'x': [pl - vl/2],
                    'y': [0],
                    'z': [ph + ch],
                    'ballastVolume': [0],
                })
        hull_components = pd.concat([hull_components, temp_df], ignore_index=True)

        return hull_components

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
        component3D = []
        for _, component in self.hullcomponents.iterrows():
            component3D.append(utils.VolumeComponent(3, occ.add_box(
                component['x'] - component['Length']/2,
                component['y'] - component['Width']/2,
                component['z'],
                component['Length'],
                component['Width'],
                component['Height'])))

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

        geometry = pontoon_ps.fuse([pontoon_sb] + columns)
        
        geometry = component3D[0].fuse(component3D[1:])

        if show:
            utils.ui.start_ui(mode='geo')
        return geometry

    def cut_geometry(self, geometry: utils.VolumeComponent, draft: float, roll: float):
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

        geometry.translate(dy=-w/2, dz=-draft).rotate(0, 0, 0, 1, 0, 0, np.deg2rad(roll))

        # Create slicing plane
        size = max(l, w)*1.1
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

    def get_area_moment_of_inertia(self, areas: List[tuple]) -> Tuple[float, np.array, np.array]:
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

    def get_waterplane_properties(self, draft: float, roll: float, show=False) -> Tuple[float, np.array, np.array]:
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

    def get_waterplane_area(self, draft: float):
        hullComponents = self.hullcomponents

        if (draft >= 0) and (draft <= self.parameters.PontoonHeight['m']):
            selected = hullComponents[hullComponents['Name'].str.contains('P')]
        elif (draft > self.parameters.PontoonHeight['m']) and (draft <= self.parameters.PontoonHeight['m'] + self.parameters.ColumnHeight['m']):
            selected = hullComponents[hullComponents['Name'].str.contains('C')]
        else:
            raise NotImplementedError()

        waterplane_area = np.sum(selected['Length'] * selected['Width'] * selected['Cb'])

        return waterplane_area

    def get_displacement_hull_components(self, draft: float):
        """Function that calculates the displacement per hull component:
          1. Checks local water level per hull component
          2. Uses local water level per hull component and shape to calculate displacement

        Args:
            draft (float): water depth with respect to vessel keel

        Returns:
            displacement (pd.Dataframe): pandas dataframe series with displacement per hull component
        """
        hullComponents = self.hullcomponents

        local_water_level_components = np.maximum(np.zeros(len(hullComponents)), np.minimum(draft - hullComponents['z'], hullComponents['Height']))
        displacement = hullComponents['Length'] * hullComponents['Width'] * hullComponents['Cb'] * local_water_level_components

        return displacement

    def get_displacement(self, draft: float):

        displacement = np.sum(self.get_displacement_hull_components(draft))

        return displacement

    def get_center_of_buoyancy(self, draft: float):

        if self.parameters.PontoonHeight['m'] + self.parameters.ColumnHeight['m'] < draft:
            raise ValueError('Draft exceeds vessel depth')

        selected = self.hullcomponents

        displacement = self.get_displacement_hull_components(draft)

        local_water_level_components = np.maximum(np.zeros(len(selected)), np.minimum(draft - selected['z'], selected['Height']))
        local_vcb_components = local_water_level_components * selected['VCB']
        global_vcb_components = local_vcb_components + selected['z']
        vertical_center_of_buoyancy = np.sum(displacement * global_vcb_components) / np.sum(displacement)

        local_lcb_components = selected['Length'] * selected['LCB'] - selected['Length'] / 2
        global_lcb_components = local_lcb_components + selected['x']
        longitudional_center_of_buoyancy = np.sum(displacement * global_lcb_components) / np.sum(displacement)

        center_of_buoyancy = [longitudional_center_of_buoyancy, 0, vertical_center_of_buoyancy]

        return center_of_buoyancy

    def get_moment_of_waterplane_area(self, draft: float):
        hullComponents = self.hullcomponents

        if (draft >= 0) and (draft <= self.parameters.PontoonHeight['m']):
            selected = hullComponents[hullComponents['Name'].str.contains('P')]
        elif (draft > self.parameters.PontoonHeight['m']) and (draft <= self.parameters.PontoonHeight['m'] + self.parameters.ColumnHeight['m']):
            selected = hullComponents[hullComponents['Name'].str.contains('C')]
        else:
            raise NotImplementedError()

        It = np.sum(1 / 12 * selected['Cb'] * selected['Length'] * selected['Width'] ** 3 + selected['Cb'] * selected['Length'] * selected['Width'] * selected['y']**2)
        Il = np.sum(1 / 12 * selected['Cb'] * selected['Width'] * selected['Length'] ** 3 + selected['Cb'] * selected['Length'] * selected['Width'] * (selected['x'] - self.parameters.PontoonLength['m'] / 2)**2)

        return It, Il

    def get_km(self, draft: float):
        It, Il = self.get_moment_of_waterplane_area(draft)
        displacement = self.get_displacement(draft)
        center_of_buoyancy = self.get_center_of_buoyancy(draft)

        KMt = It / displacement + center_of_buoyancy[2]
        KMl = Il / displacement + center_of_buoyancy[2]

        return KMt, KMl

    def get_hydrostatics(self, draft_max: float, draft_min: float = 0, delta_draft: float = 0.1):

        drafts = np.arange(draft_min, draft_max + delta_draft, delta_draft)

        hydrostatics = pd.DataFrame()
        for draft in drafts:
            displacement = self.get_displacement(draft)
            waterplane_area = self.get_waterplane_area(draft)
            lcb, tcb, vcb = self.get_center_of_buoyancy(draft)
            kmt, kml = self.get_km(draft)
            it, il = self.get_moment_of_waterplane_area(draft)

            temp_df = pd.DataFrame({
                'Draft': [draft],
                'Mass': [displacement*WATERDENSITY],
                'Displacement': [displacement],
                'WaterplaneArea': [waterplane_area],
                'LCB': [lcb],
                'TCB': [tcb],
                'VCB': [vcb],
                'KMt': [kmt],
                'KMl': [kml],
                'It': [it],
                'Il': [il],
            })

            hydrostatics = pd.concat([hydrostatics, temp_df], ignore_index=True)

        return hydrostatics

    @property
    def parameters(self) -> SSCVDesign:
        return self._parameters

    @property
    def hullcomponents(self) -> pd.DataFrame:
        return self._hullcomponents

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
    WaterDepth: Parameter = 0 * m
    WaterDensity: Parameter = WATERDENSITY * units.metric_ton/m**3

    VesselDraft: Parameter = 0 * m

    TypicalDeckLoad: Parameter = 0 * mT
    TypicalDeckLoadLCG: Parameter = 0 * d
    TypicalDeckLoadVCG: Parameter = 0 * m

    ProjectDeckLoad: Parameter = 0 * mT
    ProjectDeckLoadLCG: Parameter = 0 * d
    ProjectDeckLoadVCG: Parameter = 0 * m

    # ToDo improve use of VCG (some times height above x, sometimes percentage of x)

    Fuel: Parameter = 0 * mT
    FuelLCG: Parameter = 0 * d
    FuelVCG: Parameter = 0 * d

    FreshWater: Parameter = 0 * mT
    FreshWaterLCG: Parameter = 0 * d
    FreshWaterVCG: Parameter = 0 * d

    CraneRadius: Parameter = 0 * m
    CraneLoad: Parameter = 0 * mT
    CraneLiftingHeight: Parameter = 0 * m

    GMtreduction: Parameter = 0 * m
    GMlreduction: Parameter = 0 * m


class SSCVNaval(Naval):

    def __init__(self, parameters: SSCVNavalParameters, structure: SSCVStructure):
        self._parameters = parameters
        self._structure = structure
        self._mass_components = pd.DataFrame()

    def change_state(self):
        raise NotImplementedError()

    def get_natural_periods(self):
        raise NotImplementedError()

    def get_stability(self):
        raise NotImplementedError()

    def add_mass_component(self, mass_type: str, name: str, mass: float, lcg: float, tcg: float, vcg: float):
        temp_df = pd.DataFrame({
            'Type': mass_type,
            'Name': name,
            'Mass': [mass],
            'LCG': [lcg],
            'TCG': [tcg],
            'VCG': [vcg],
        })

        self._mass_components = pd.concat([self.mass_components, temp_df], ignore_index=True)

    def define_mass_components(self):

        pl = self.structure.parameters.PontoonLength['m']
        ph = self.structure.parameters.PontoonHeight['m']
        ch = self.structure.parameters.ColumnHeight['m']
        dh = self.structure.parameters.DeckHeight['m']

        craneMaxCap = self.structure.parameters.CraneMaxCapacity['metric_ton']
        craneBowDistance = 12
        craneRadius = self.parameters.CraneRadius['m']
        craneLiftingHeight = self.parameters.CraneLiftingHeight['m']

        deck_height = ph + ch + dh

        # Add vessel inertia
        vessel_mass6D, vesselCoG = self.structure.get_inertia()
        self.add_mass_component('Lightship', 'Lightweight', vessel_mass6D.diagonal()[0], vesselCoG[0], vesselCoG[1], vesselCoG[2])

        # Substract crane boom if load in crane
        if self.parameters.CraneLoad['metric_ton'] > 0:
            craneBoomMass = self.structure.parameters.CraneBoomMassFactor[None] * craneMaxCap
            self.add_mass_component('Lightship', 'CraneBoom', -craneBoomMass * 2, vesselCoG[0], vesselCoG[1], vesselCoG[2])

            # ToDo check VCG boom +-> factor 25 in parameters
            # ToDo split in amount of cranes/ps/sb side
            self.add_mass_component('CraneLoad', 'CraneBoom', craneBoomMass * 2, pl - craneBowDistance + craneRadius/2, 0, deck_height + 25 + (craneLiftingHeight - 25)/2)
            self.add_mass_component('CraneLoad', 'HookLoad', self.parameters.CraneLoad['metric_ton'] * 2, pl - craneBowDistance + craneRadius, 0, deck_height + craneLiftingHeight + 12)

        # Add consumables
        fuelmass = self.parameters.Fuel['metric_ton']
        fuellcg = self.parameters.FuelLCG[None] * pl
        fueltcg = 0
        fuelvcg = self.parameters.FuelVCG[None] * ph

        self.add_mass_component('Consumables', 'Fuel', fuelmass, fuellcg, fueltcg, fuelvcg)

        freshwatermass = self.parameters.FreshWater['metric_ton']
        freshwaterlcg = self.parameters.FreshWaterLCG[None] * pl
        freshwatertcg = 0
        freshwatervcg = self.parameters.FreshWaterVCG[None] * ph

        self.add_mass_component('Consumables', 'FreshWater', freshwatermass, freshwaterlcg, freshwatertcg, freshwatervcg)

        # Add deck loads

        typdeckloadmass = self.parameters.TypicalDeckLoad['metric_ton']
        typdeckloadlcg = self.parameters.TypicalDeckLoadLCG[None] * pl
        typdeckloadtcg = 0
        typdeckloadvcg = deck_height + self.parameters.TypicalDeckLoadVCG['m']

        self.add_mass_component('DeckLoad', 'Typical', typdeckloadmass, typdeckloadlcg, typdeckloadtcg, typdeckloadvcg)

        projdeckloadmass = self.parameters.ProjectDeckLoad['metric_ton']
        projdeckloadlcg = self.parameters.ProjectDeckLoadLCG[None] * self.structure.parameters.VesselLength['m']
        projdeckloadtcg = 0
        projdeckloadvcg = deck_height + self.parameters.ProjectDeckLoadVCG['m']

        self.add_mass_component('DeckLoad', 'Project', projdeckloadmass, projdeckloadlcg, projdeckloadtcg, projdeckloadvcg)


    def get_center_of_gravity(self):

        mass_components = self.mass_components

        longitudinal_center_of_gravity = np.sum(mass_components['Mass'] * mass_components['LCG']) / np.sum(mass_components['Mass'])
        transverse_center_of_gravity = 0
        vertical_center_of_gravity = np.sum(mass_components['Mass'] * mass_components['VCG']) / np.sum(mass_components['Mass'])

        center_of_gravity = [longitudinal_center_of_gravity, transverse_center_of_gravity, vertical_center_of_gravity]

        return center_of_gravity


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

    @property
    def mass_components(self) -> pd.DataFrame:
        return self._mass_components


if __name__ == "__main__":

    design = SSCVDesign()
    my_struct = SSCVStructure(design)
    my_struct.get_inertia()

    # my_struct.define_hull_components()

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
    # height = design.PontoonHeight['m']
    # data = []
    # for roll in np.linspace(0, 10, 10):
    #     area, centroid, amoi = my_struct.get_waterplane_properties(15, roll, show=False)
    #     data.append([roll, area])
    
    # df = pd.DataFrame(data, columns='roll area'.split())
    # df.set_index('roll', inplace=True)
    # plt = df.plot()
    # plt.set_ylabel('water line surface [m^2]')    

    # my_struct.get_mesh(file='sscv.msh', show=True)

    naval_parameters = SSCVNavalParameters()
    my_nav = SSCVNaval(parameters=naval_parameters, structure=my_struct)

    my_nav.define_mass_components()


    my_nav.structure.parameters
