import attractor
import numpy as np

def test_constructor():
    s=11
    p=22 
    b=2.66667
    start=0.0
    end=80.0
    points=1002222200
    # check if constructor correctly accept initial values
    x=attractor.Attractor(s, p, b, start, end, points)
    assert x.params[0] ==s
    assert x.params[1] ==p
    assert x.params[2] ==b
    assert x.start == start 
    assert  x.end == end
    assert  x.points == points
    
def test_attractor_evolve4():
    x=attractor.Attractor()
    x.evolve(order=4)
   # check if Attractor calculate all values for order=4
    assert x.solution['x'].count() >0
    assert x.solution['y'].count() >0
    assert x.solution['z'].count() >0
    
def test_attractor_evolve3():
    x=attractor.Attractor()
    x.evolve(order=3)
    # check if Attractor calculate all values for order=3
    assert x.solution['x'].count() >0
    assert x.solution['y'].count() >0
    assert x.solution['z'].count() >0
    
def test_attractor_evolve2():
    x=attractor.Attractor()
    x.evolve(order=2)
    # check if Attractor calculate all values for order=2
    assert x.solution['x'].count() >0
    assert x.solution['y'].count() >0
    assert x.solution['z'].count() >0
    
def test_attractor_evolve1():
    x=attractor.Attractor()
    x.evolve(order=1)
    # check if Attractor calculate all values for order=1
    assert x.solution['x'].count() >0
    assert x.solution['y'].count() >0
    assert x.solution['z'].count() >0
   