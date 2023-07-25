from hivemind.structural import Structure
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
    PontoonNumberOfBallastTanks: Parameter = 0 * d

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
        self._ballasttanks = self.define_ballast_tanks()

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
        displacement = np.sum(self.hullcomponents['Length'] * self.hullcomponents['Width'] * self.hullcomponents['Height'] * self.hullcomponents['Cb'])
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
                    'ballastCapacityVolume': [pl*pw*ph*pballast],
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
                    'ballastCapacityVolume': [cl*cw*ch*cballast],
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
                    'ballastCapacityVolume': [0],
                })
        hull_components = pd.concat([hull_components, temp_df], ignore_index=True)

        return hull_components

    def define_ballast_tanks(self):  # ToDo implement reduction of ballast capacity due to fuel/fresh water tanks!!
        pnumberoftanks = self.parameters.PontoonNumberOfBallastTanks[None]

        ballast_tanks = pd.DataFrame()

        for _, hullComponent in self.hullcomponents.iterrows():

            if ('P' in hullComponent['Name']) and (pnumberoftanks > 0):
                pl = hullComponent['Length']

                for ballasttank in range(pnumberoftanks):

                    x_tanks = [0.167 * pl, 0.5 * pl, 0.84 * pl]  # ?? why is this choice made?
                    x_tank = x_tanks[ballasttank]
                    # x_tank = (pl / (pnumberoftanks + 1)) * (ballasttank + 1)

                    temp_df = pd.DataFrame({
                        'Name': [f'BT_{hullComponent["Name"]}_{ballasttank + 1}'],
                        'HullComponent': [hullComponent['Name']],
                        'x': [x_tank],
                        'y': [hullComponent['y']],
                        'z_min': [hullComponent['z']],
                        'z_max': [hullComponent['z'] + hullComponent['Height']],
                        'ballastCapacityVolume': [hullComponent['ballastCapacityVolume'] / pnumberoftanks]
                    })

                    ballast_tanks = pd.concat([ballast_tanks, temp_df], ignore_index=True)

            if 'C' in hullComponent['Name']:

                temp_df = pd.DataFrame({
                    'Name': [f'BT_{hullComponent["Name"]}_{1}'],
                    'HullComponent': [hullComponent['Name']],
                    'x': [hullComponent['x']],
                    'y': [hullComponent['y']],
                    'z_min': [hullComponent['z']],
                    'z_max': [hullComponent['z'] + hullComponent['Height']],
                    'ballastCapacityVolume': [hullComponent['ballastCapacityVolume']]
                })

                ballast_tanks = pd.concat([ballast_tanks, temp_df], ignore_index=True)

        return ballast_tanks

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
        vl = self.parameters.VesselLength['m']
        vw = self.parameters.VesselWidth['m']

        geometry.translate(dy=-vw/2, dz=-draft).rotate(0, 0, 0, 1, 0, 0, np.deg2rad(roll))

        # Create slicing plane
        size = max(vl, vw)*1.1
        plane = utils.make_polygon([
            [0,  0,  0],
            [size,  0,  0],
            [size, size, 0],
            [0, size, 0],
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
    def hullcomponents(self) -> pd.DataFrame:
        return self._hullcomponents

    @property
    def ballasttanks(self) -> pd.DataFrame:
        return self._ballasttanks

    @property
    def possible_states(self):
        raise NotImplementedError()

    @property
    def previous_state(self):
        raise NotImplementedError()

    @property
    def state(self):
        raise NotImplementedError()
