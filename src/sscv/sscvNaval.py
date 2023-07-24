from hivemind.naval import Naval
from hivemind.abstracts import ParameterSet, Parameter, units
from gmsh_utils import utils
from gmsh_utils import mesh
import gmsh
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from .sscvStruct import SSCVDesign, SSCVStructure

m = units.meter
d = units.dimensionless
mT = units.metric_ton


class SSCVNavalParameters(ParameterSet):
    WaterDepth: Parameter = 0 * m
    WaterDensity: Parameter = 1.025 * units.metric_ton/m**3

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
        self._hydrostatics = self.get_hydrostatics(draft_max=(self.structure.parameters.PontoonHeight['m'] + self.structure.parameters.ColumnHeight['m']))
        self._mass_components = pd.DataFrame()

    def change_state(self):
        raise NotImplementedError()

    def get_natural_periods(self):
        raise NotImplementedError()

    def get_stability(self):
        raise NotImplementedError()

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
            mass = gmsh.model.occ.get_mass(2, area[1])  # NOTE: mass == surface area
            cog = gmsh.model.occ.get_center_of_mass(2, area[1])
            I = gmsh.model.occ.get_matrix_of_inertia(2, area[1])  # NOTE: wrt cog
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
        geometry = self.structure.get_geometry()
        water_crossing_surfaces = self.structure.cut_geometry(geometry, draft, roll)
        area, center_of_surface, area_moment_of_inertia = self.get_area_moment_of_inertia(water_crossing_surfaces)
        if show:
            utils.ui.show_entities(water_crossing_surfaces, recursive=True, suppress_others=True)
            utils.ui.start_ui(mode='geo')
        return area, center_of_surface, area_moment_of_inertia

    def get_waterplane_area(self, draft: float):
        hullComponents = self.structure.hullcomponents
        ph = self.structure.parameters.PontoonHeight['m']
        ch = self.structure.parameters.ColumnHeight['m']

        if (draft >= 0) and (draft <= ph):
            selected = hullComponents[hullComponents['Name'].str.contains('P')]
        elif (draft > ph) and (draft <= ph + ch):
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
        hullComponents = self.structure.hullcomponents

        local_water_level_components = np.maximum(np.zeros(len(hullComponents)), np.minimum(draft - hullComponents['z'], hullComponents['Height']))
        displacement = hullComponents['Length'] * hullComponents['Width'] * hullComponents['Cb'] * local_water_level_components

        return displacement

    def get_displacement(self, draft: float):

        displacement = np.sum(self.get_displacement_hull_components(draft))

        return displacement

    def get_center_of_buoyancy(self, draft: float):
        ph = self.structure.parameters.PontoonHeight['m']
        ch = self.structure.parameters.ColumnHeight['m']
        selected = self.structure.hullcomponents

        if ph + ch < draft:
            raise ValueError('Draft exceeds vessel depth')

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
        ph = self.structure.parameters.PontoonHeight['m']
        pl = self.structure.parameters.PontoonLength['m']
        ch = self.structure.parameters.ColumnHeight['m']
        hullComponents = self.structure.hullcomponents

        if (draft >= 0) and (draft <= ph):
            selected = hullComponents[hullComponents['Name'].str.contains('P')]
        elif (draft > ph) and (draft <= ph + ch):
            selected = hullComponents[hullComponents['Name'].str.contains('C')]
        else:
            raise NotImplementedError()

        It = np.sum(1 / 12 * selected['Cb'] * selected['Length'] * selected['Width'] ** 3 + selected['Cb'] * selected['Length'] * selected['Width'] * selected['y']**2)
        Il = np.sum(1 / 12 * selected['Cb'] * selected['Width'] * selected['Length'] ** 3 + selected['Cb'] * selected['Length'] * selected['Width'] * (selected['x'] - pl / 2)**2)

        return It, Il

    def get_km(self, draft: float):
        It, Il = self.get_moment_of_waterplane_area(draft)
        displacement = self.get_displacement(draft)
        center_of_buoyancy = self.get_center_of_buoyancy(draft)

        KMt = It / displacement + center_of_buoyancy[2]
        KMl = Il / displacement + center_of_buoyancy[2]

        return KMt, KMl

    def get_hydrostatics(self, draft_max: float, draft_min: float = 0, delta_draft: float = 0.1):
        waterdensity = self.parameters.WaterDensity['metric_ton / meter ** 3']
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
                'Mass': [displacement * waterdensity],
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
    def hydrostatics(self):
        return self._hydrostatics

    @property
    def mass_components(self) -> pd.DataFrame:
        return self._mass_components


if __name__ == "__main__":

    design = SSCVDesign()
    my_struct = SSCVStructure(design)

    naval_parameters = SSCVNavalParameters()
    my_nav = SSCVNaval(parameters=naval_parameters, structure=my_struct)
