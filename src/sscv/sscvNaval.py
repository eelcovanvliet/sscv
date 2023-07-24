from hivemind.naval import Naval
from hivemind.abstracts import ParameterSet, Parameter, units
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
from .sscvStruct import SSCVDesign, SSCVStructure

m = units.meter
d = units.dimensionless
mT = units.metric_ton

GRAVITYCONSTANT = 9.81
WATERDENSITY = 1.025


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

    naval_parameters = SSCVNavalParameters()
    my_nav = SSCVNaval(parameters=naval_parameters, structure=my_struct)
