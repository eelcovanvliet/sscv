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
from scipy.optimize import minimize

m = units.meter
d = units.dimensionless
mT = units.metric_ton


class SSCVNavalParameters(ParameterSet):
    WaterDepth: Parameter = 0 * m
    WaterDensity: Parameter = 1.025 * units.metric_ton/m**3
    GravityConstant: Parameter = 9.81 * m / units.s ** 2

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
        self.define_mass_components()
        self._ballast_condition, self.ballast_status = self.ballast_vessel(self.parameters.VesselDraft['m'])

    def change_state(self):
        raise NotImplementedError()

    def get_natural_periods(self):  # Todo implement added masses
        g = self.parameters.GravityConstant['m / s ** 2']
        water_density = self.parameters.WaterDensity['metric_ton / m ** 3']
        draft = self.parameters.VesselDraft['m']
        kxx_factor = self.structure.parameters.LightshipWeightKxx[None]
        kyy_factor = self.structure.parameters.LightshipWeightKyy[None]
        pl = self.structure.parameters.PontoonLength['m']
        vw = self.structure.parameters.VesselWidth['m']

        displacement = self.get_displacement(draft)
        waterplane_area = self.get_waterplane_area(draft)
        GMt, GMl = self.get_stability()

        # Heave
        stiffness_heave = waterplane_area * g * water_density
        mass_heave = displacement * water_density
        T_heave = self.calc_natural_period(stiffness_heave, mass_heave)

        # Roll
        stiffness_roll = displacement * water_density * g * GMt
        mass_roll = (kxx_factor * vw) ** 2 * displacement * water_density
        T_roll = self.calc_natural_period(stiffness_roll, mass_roll)

        # Pitch
        stiffness_pitch = displacement * water_density * g * GMl
        mass_pitch = (kyy_factor * pl) ** 2 * displacement * water_density
        T_pitch = self.calc_natural_period(stiffness_pitch, mass_pitch)

        return T_heave, T_roll, T_pitch

    @staticmethod
    def calc_natural_period(stiffness: float, mass: float):

        natural_period = (2 * np.pi) / np.sqrt((stiffness) / (mass))

        return natural_period

    def get_stability(self):  # Currently only initial stability GM (not GZ curve -> to be implemented and dynamic loss of hook load)

        # Get parameters
        draft = self.parameters.VesselDraft['m']
        GMt_reduction = self.parameters.GMtreduction['m']
        GMl_reduction = self.parameters.GMlreduction['m']

        # Get required information
        center_of_gravity = self.get_center_of_gravity_vessel_and_ballast()
        kmt, kml = self.get_km(draft)

        # GM calculation
        GMt = kmt - center_of_gravity[2]
        GMl = kml - center_of_gravity[2]

        # Free flooding effect
        GMt = GMt - GMt_reduction
        GMl = GMl - GMl_reduction

        return GMt, GMl

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

    def get_center_of_gravity_vessel_and_load(self):

        mass_components = self.mass_components

        longitudinal_center_of_gravity = np.sum(mass_components['Mass'] * mass_components['LCG']) / np.sum(mass_components['Mass'])
        transverse_center_of_gravity = 0
        vertical_center_of_gravity = np.sum(mass_components['Mass'] * mass_components['VCG']) / np.sum(mass_components['Mass'])

        center_of_gravity = [longitudinal_center_of_gravity, transverse_center_of_gravity, vertical_center_of_gravity]

        return center_of_gravity

    def get_center_of_gravity_ballast(self):

        ballast_condition = self.ballast_condition

        longitudinal_center_of_gravity = np.sum(ballast_condition['BallastMass'] * ballast_condition['x']) / np.sum(ballast_condition['BallastMass'])
        transverse_center_of_gravity = 0
        vertical_center_of_gravity = np.sum(ballast_condition['BallastMass'] * ballast_condition['z']) / np.sum(ballast_condition['BallastMass'])

        center_of_gravity = [longitudinal_center_of_gravity, transverse_center_of_gravity, vertical_center_of_gravity]

        return center_of_gravity

    def get_center_of_gravity_vessel_and_ballast(self):

        mass_components = self.mass_components
        ballast_condition = self.ballast_condition
        vessel_draft = self.parameters.VesselDraft['m']
        water_density = self.parameters.WaterDensity['metric_ton / m ** 3']

        mass_vessel_load = np.sum(mass_components['Mass'])
        mass_ballast = np.sum(ballast_condition['BallastMass'])

        total_mass = mass_vessel_load + mass_ballast
        displacement = self.get_displacement(vessel_draft)

        if total_mass != displacement * water_density:
            print(f'WARNING: vessel total mass of {total_mass:.2f} mT does not match displacment of {displacement * water_density :.2f} mT')

        center_of_gravity_vessel_and_load = np.array(self.get_center_of_gravity_vessel_and_load())
        center_of_gravity_ballast = np.array(self.get_center_of_gravity_ballast())

        center_of_gravity_total = (center_of_gravity_vessel_and_load * mass_vessel_load + center_of_gravity_ballast * mass_ballast) / total_mass

        return center_of_gravity_total

    def ballast_vessel(self, required_draft: float) -> pd.DataFrame:  # !! IMPROVE NAMING ETC!!!
        """Function to determine required ballast for level vessel at required draft

        Args:
            required_draft (float): _description_

        Returns:
            _type_: _description_
        """

        tolerance = 0.01

        water_density = self.parameters.WaterDensity['metric_ton / meter ** 3']
        ballastTanks = self.structure.ballasttanks
        ballastWater = np.zeros(ballastTanks.shape[0])
        pl = self.structure.parameters.PontoonLength['m']

        # Initialize ballast condition
        ballastCondition = ballastTanks.copy()
        ballastCondition['BallastVolume'] = np.zeros(ballastTanks.shape[0])
        ballastCondition['Perc_fil'] = np.zeros(ballastTanks.shape[0])
        ballastCondition['BallastMass'] = np.zeros(ballastTanks.shape[0])

        # Get vessel mass properties
        vessel_load_mass = np.sum(self.mass_components['Mass'])
        vessel_load_lcg = np.sum(self.mass_components['Mass'] * self.mass_components['LCG']) / vessel_load_mass

        # Get required displacement and LCG (above LCB)
        required_displacement_vol = self.get_displacement(required_draft)
        required_displacement_mass = required_displacement_vol * water_density
        required_lcg, _, _ = self.get_center_of_buoyancy(required_draft)

        # Calculate ballast mass and CoG
        required_ballast_mass = required_displacement_mass - vessel_load_mass
        required_ballast_lcg = (required_lcg * required_displacement_mass - vessel_load_lcg * vessel_load_mass) / required_ballast_mass

        if required_ballast_mass < 0:
            return ballastCondition, 'VesselAndLoadToHeavyForDraft'
        elif required_ballast_mass > np.sum(ballastTanks['ballastCapacityVolume'] * water_density):
            return ballastCondition, 'NotSufficientBallastWaterAvailableForDraft'

        # Fill ballast tanks -> for simplification reasoning only looking at PS (assumption symmetric ballast between both sides)
        selectedTanks = ballastTanks[ballastTanks['y'] >= 0]
        amount_of_tanks_selected = selectedTanks.shape[0]
        required_ballast_mass_one_side = required_ballast_mass / 2

        # Define objective and constraints for ballast function  # ToDo for future work implement free flooding effects in optimizer -> seperate optimizer from naval? (and upgrade for better results)
        objective = lambda x: np.sum((((x / (selectedTanks['ballastCapacityVolume'] * water_density)) * (selectedTanks['z_max'] - selectedTanks['z_min'])) / 2 + selectedTanks['z_min']) * x) / np.sum(x)

        cons1 = lambda x: np.sum(x) - required_ballast_mass_one_side
        cons2 = lambda x: np.sum(selectedTanks['x'] * x) / np.sum(x) - required_ballast_lcg

        cons = (
            {'type': 'eq', 'fun': cons1},
            {'type': 'eq', 'fun': cons2},
        )

        lower_bound = np.zeros(selectedTanks.shape[0]).reshape(amount_of_tanks_selected, 1)
        upper_bound = np.array(selectedTanks['ballastCapacityVolume'] * water_density).reshape(amount_of_tanks_selected, 1)
        bounds = np.concatenate((lower_bound, upper_bound), axis=1)

        # Define set of start conditions (as much mass at the aft, as much mass at the front, as much mass in pontoon) -> improve start condition feasibility (currently amount of requird ballast can be over max capacity)
        start_conditions = []

        divide_over_all_tanks = np.ones(amount_of_tanks_selected) * required_ballast_mass / amount_of_tanks_selected
        start_conditions.append(divide_over_all_tanks)

        divide_over_pontoons = np.zeros(amount_of_tanks_selected)
        pontoon_tanks = selectedTanks['HullComponent'].str.contains('P')
        divide_over_pontoons[np.where(pontoon_tanks)[0]] = (required_ballast_mass / np.sum(pontoon_tanks))
        start_conditions.append(divide_over_pontoons)

        # divide_over_aft_tanks = np.zeros(amount_of_tanks_selected)
        # aft_tanks = selectedTanks['x'] < 0.5 * pl
        # divide_over_aft_tanks[np.where(aft_tanks)[0]] = (required_ballast_mass / np.sum(aft_tanks))
        # start_conditions.append(divide_over_aft_tanks)

        ballastTankWaterMass = []
        ballastTankVCG = []

        solverresults = pd.DataFrame()
        for nr_start_conditions in range(len(start_conditions)):
            start_condition = start_conditions[nr_start_conditions]

            bestVCG = 100
            VCG_temp_results = []
            ballastTankWaterMass_temp_results = []
            for iteration in range(3):
                optimizer_result = minimize(objective, start_condition, constraints=cons, bounds=bounds)

                if np.sum(optimizer_result.x) - required_ballast_mass_one_side < tolerance:
                    if np.sum(selectedTanks['x'] * optimizer_result.x) / np.sum(optimizer_result.x) - required_ballast_lcg < tolerance:
                        temp_df = pd.DataFrame({
                            'initial_start_condition': [nr_start_conditions],
                            'iteration': [iteration],
                            'start_condition': [start_condition],
                            'BallastTankMass': [optimizer_result.x],
                            'VCG': [optimizer_result.fun],
                        })

                        solverresults = pd.concat([solverresults, temp_df], ignore_index=True)

                        VCG_temp_results.append(optimizer_result.fun)
                        ballastTankWaterMass_temp_results.append(optimizer_result.x)

                        if optimizer_result.fun < bestVCG:
                            start_condition = optimizer_result.x
                            bestVCG = optimizer_result.fun

            if len(VCG_temp_results) > 0:
                ballastTankWaterMass.append(ballastTankWaterMass_temp_results[np.argmin(VCG_temp_results)])
                ballastTankVCG.append(min(VCG_temp_results))

        ballastWater = ballastTankWaterMass[np.argmin(ballastTankVCG)]

        # Assign ballast to all tanks  # !! Not very robust but quick fix -> think about where to store ballast water
        ballastWaterAllTanks = np.zeros(ballastTanks.shape[0])
        ballastWaterAllTanks[selectedTanks.index] = ballastWater
        ballastWaterAllTanks[list(set(ballastTanks.index.values) - set(selectedTanks.index))] = ballastWater

        ballastCondition['BallastVolume'] = ballastWaterAllTanks / water_density
        ballastCondition['BallastMass'] = ballastWaterAllTanks
        ballastCondition['z'] = (ballastCondition['BallastVolume'] / ballastCondition['ballastCapacityVolume']) * (ballastCondition['z_max'] - ballastCondition['z_min']) / 2 + ballastCondition['z_min']
        ballastCondition['Perc_fil'] = ballastCondition['BallastVolume'] / ballastCondition['ballastCapacityVolume']

        return ballastCondition, 'VesselBallastedToEvenKeelAndDraft'

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

    @property
    def ballast_condition(self) -> pd.DataFrame:
        return self._ballast_condition


if __name__ == "__main__":

    design = SSCVDesign()
    my_struct = SSCVStructure(design)

    naval_parameters = SSCVNavalParameters()
    my_nav = SSCVNaval(parameters=naval_parameters, structure=my_struct)
