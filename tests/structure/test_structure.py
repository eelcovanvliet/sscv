import pytest

from sscv.sscvStruct import SSCVDesign, SSCVStructure, Parameter, units

m = units.meter
mT = units.metric_ton


design = SSCVDesign()

design.VesselLength['m'] = 100 * m
design.VesselWidth['m'] = 60 * m
design.DeckHeight['m'] = 50 * m

design.NumberOfPontoons[None] = 8
design.PontoonLength['m'] = 90 * m
design.PontoonWidth['m'] = 20 * m
design.PontoonHeight['m'] = 15 * m
design.PontoonCb[None] = 0.5
design.PontoonVCB[None] = 0.1
design.PontoonLCB[None] = 0.1
design.PontoonBallastCapacity[None] = 10
design.NumberOfColumnsPerPontoon[None] = 10

design.ColumnLength['m'] = 10 * m
design.ColumnWidth['m'] = 10 * m
design.ColumnHeight['m'] = 10 * m
design.ColumnCb[None] = 10
design.ColumnVCB[None] = 10
design.ColumnLCB[None] = 10
design.ColumnBallastCapacity[None] = 10

design.NumberOfCranes[None] = 10
design.CraneMaxCapacity['tonne'] = 10 * mT
design.CraneBoomMassFactor[None] = 10
design.CraneMaxLiftingHeight['m'] = 10 * m
design.CraneMaxRadius['m'] = 10 * m
design.LightshipWeightFactor[None] = 10
design.LightshipWeightLCG[None] = 10
design.LightshipWeightVCG[None] = 10
design.LightshipWeightKxx[None] = 10
design.LightshipWeightKyy[None] = 10

def test_parameters_default():
    """Check default state of the SSCVDesign"""
    for parm in SSCVDesign():
        parm:Parameter
        assert parm.is_default
        assert parm.normalized_value == 0




    
def test_get_geometry():
    structure = SSCVStructure(design)
    geom = structure.get_geometry()
    









