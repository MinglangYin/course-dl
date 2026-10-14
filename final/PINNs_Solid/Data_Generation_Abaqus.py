# -*- coding: mbcs -*-
# Do not delete the following import lines
from abaqus import *
from abaqusConstants import *
from odbAccess import *
from odbMaterial import *
from odbSection import *
import __main__
import numpy as np


def CS247(name = 'try', num_ele_cal = 40, num_ele_obs = 10, xnu = 0.0, func = 'X+1.0' ): #'0.1*X+0.2*Y+sin ( X )*0.1+log ( Y+2 )*0.2+exp ( X/10 )+sqrt(X+Y+1)'):
    import section
    import regionToolset
    import displayGroupMdbToolset as dgm
    import part
    import material
    import assembly
    import step
    import interaction
    import load
    import mesh
    import optimization
    import job
    import sketch
    import visualization
    import xyPlot
    import displayGroupOdbToolset as dgo
    import connectorBehavior
    modelname = name
    jobname = 'Job-' + modelname
    mdb.Model(name=modelname, modelType=STANDARD_EXPLICIT)
    session.viewports['Viewport: 1'].setValues(displayedObject=None)
    s1 = mdb.models[modelname].ConstrainedSketch(name='__profile__', 
        sheetSize=3.0)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=STANDALONE)
    s1.rectangle(point1=(-0.5, -0.5), point2=(0.5, 0.5))
    p = mdb.models[modelname].Part(name='Part-1', dimensionality=TWO_D_PLANAR, 
        type=DEFORMABLE_BODY)
    p = mdb.models[modelname].parts['Part-1']
    p.BaseShell(sketch=s1)
    s1.unsetPrimaryObject()
    p = mdb.models[modelname].parts['Part-1']
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models[modelname].sketches['__profile__']
    session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=ON, 
        engineeringFeatures=ON)
    session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
        referenceRepresentation=OFF)
    mdb.models[modelname].Material(name='Material-1')
    mdb.models[modelname].materials['Material-1'].Elastic(
        temperatureDependency=ON, table=((1.0, xnu, 1.0), (2.0, xnu, 2.0)))
    mdb.models[modelname].HomogeneousSolidSection(name='Section-1', 
        material='Material-1', thickness=None)
    p = mdb.models[modelname].parts['Part-1']
    f = p.faces
    faces = f.getSequenceFromMask(mask=('[#1 ]', ), )
    region = regionToolset.Region(faces=faces)
    p = mdb.models[modelname].parts['Part-1']
    p.SectionAssignment(region=region, sectionName='Section-1', offset=0.0, 
        offsetType=MIDDLE_SURFACE, offsetField='', 
        thicknessAssignment=FROM_SECTION)
    a = mdb.models[modelname].rootAssembly
    session.viewports['Viewport: 1'].setValues(displayedObject=a)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Initial')
    a1 = mdb.models[modelname].rootAssembly
    a1.DatumCsysByDefault(CARTESIAN)
    p = mdb.models[modelname].parts['Part-1']
    a1.Instance(name='Part-1-1', part=p, dependent=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(
        adaptiveMeshConstraints=ON)
    mdb.models[modelname].StaticStep(name='Step-1', previous='Initial')
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(step='Step-1')
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(loads=ON, bcs=ON, 
        predefinedFields=ON, connectors=ON, adaptiveMeshConstraints=OFF)
    a = mdb.models[modelname].rootAssembly
    v1 = a.instances['Part-1-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#1 ]', ), )
    region = regionToolset.Region(vertices=verts1)
    mdb.models[modelname].DisplacementBC(name='BC-0', createStepName='Step-1', 
        region=region, u1=0.0, u2=0.0, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
        distributionType=UNIFORM, fieldName='', localCsys=None)
    a = mdb.models[modelname].rootAssembly
    v1 = a.instances['Part-1-1'].vertices
    verts1 = v1.getSequenceFromMask(mask=('[#2 ]', ), )
    region = regionToolset.Region(vertices=verts1)
    mdb.models[modelname].DisplacementBC(name='BC-1', createStepName='Step-1', 
        region=region, u1=UNSET, u2=0.0, ur3=UNSET, amplitude=UNSET, fixed=OFF, 
        distributionType=UNIFORM, fieldName='', localCsys=None)
    a = mdb.models[modelname].rootAssembly
    s1 = a.instances['Part-1-1'].edges
    side1Edges1 = s1.getSequenceFromMask(mask=('[#2 ]', ), )
    region = regionToolset.Region(side1Edges=side1Edges1)
    mdb.models[modelname].SurfaceTraction(name='Load-1', createStepName='Step-1', 
        region=region, magnitude=1.0, directionVector=((0.0, 0.0, 0.0), (1.0, 
        0.0, 0.0)), distributionType=UNIFORM, field='', localCsys=None, 
        traction=GENERAL)
    a = mdb.models[modelname].rootAssembly
    s1 = a.instances['Part-1-1'].edges
    side1Edges1 = s1.getSequenceFromMask(mask=('[#8 ]', ), )
    region = regionToolset.Region(side1Edges=side1Edges1)
    mdb.models[modelname].SurfaceTraction(name='Load-2', createStepName='Step-1', 
        region=region, magnitude=1.0, directionVector=((0.0, 0.0, 0.0), (-1.0, 
        0.0, 0.0)), distributionType=UNIFORM, field='', localCsys=None, 
        traction=GENERAL)
    mdb.models[modelname].ExpressionField(name='AnalyticalField-1', 
        localCsys=None, description='', 
        expression=func)
    a = mdb.models[modelname].rootAssembly
    f1 = a.instances['Part-1-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#1 ]', ), )
    region = regionToolset.Region(faces=faces1)
    mdb.models[modelname].Temperature(name='Predefined Field-1', 
        createStepName='Step-1', region=region, distributionType=UNIFORM, 
        crossSectionDistribution=CONSTANT_THROUGH_THICKNESS, magnitudes=(1.0, 
        ))
    mdb.models[modelname].predefinedFields['Predefined Field-1'].setValues(
        distributionType=FIELD, field='AnalyticalField-1')
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=ON, loads=OFF, 
        bcs=OFF, predefinedFields=OFF, connectors=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=ON)
    a = mdb.models[modelname].rootAssembly
    partInstances =(a.instances['Part-1-1'], )
    a.seedPartInstance(regions=partInstances, size=1.0 / num_ele_cal, deviationFactor=0.1, 
        minSizeFactor=0.1)
    a = mdb.models[modelname].rootAssembly
    partInstances =(a.instances['Part-1-1'], )
    a.generateMesh(regions=partInstances)
    elemType1 = mesh.ElemType(elemCode=CPE4, elemLibrary=STANDARD)
    elemType2 = mesh.ElemType(elemCode=CPE3, elemLibrary=STANDARD)
    a = mdb.models[modelname].rootAssembly
    f1 = a.instances['Part-1-1'].faces
    faces1 = f1.getSequenceFromMask(mask=('[#1 ]', ), )
    pickedRegions =(faces1, )
    a.setElementType(regions=pickedRegions, elemTypes=(elemType1, elemType2))
    session.viewports['Viewport: 1'].assemblyDisplay.setValues(mesh=OFF)
    session.viewports['Viewport: 1'].assemblyDisplay.meshOptions.setValues(
        meshTechnique=OFF)
    mdb.Job(name=jobname, model=modelname, description='', type=ANALYSIS, 
        atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
        memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
        explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
        modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
        scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=1, 
        numGPUs=0)
    mdb.jobs[jobname].submit(consistencyChecking=OFF)

    mdb.jobs[jobname].waitForCompletion()
    # FEM completed. Store the output data.
    print('Abaqus Job Completed!')
    # Post-process
    odb = openOdb(path=jobname + '.odb')
    assembly = odb.rootAssembly
    instance = assembly.instances['PART-1-1']
    coords=[]
    for node in instance.nodes:
        coord = [node.label, node.coordinates[0], node.coordinates[1]]
        coords.append(coord)
    coords=np.array(coords)

    lastFrame = odb.steps['Step-1'].frames[-1]
    disp = lastFrame.fieldOutputs['U']
    disps = []
    for u in disp.values:
        disps.append(u.data)
    disps = np.array(disps)
    
    data_nodes_cal = np.hstack((coords, disps)) # 5 columns: node #, x-coord, y-coord, x-disp, y-disp
    
    num_nodes_cal = data_nodes_cal.shape[0]
    
    num_nodes_obs = (1+num_ele_obs)**2
    data_nodes_obs = np.zeros((num_nodes_obs,4))
    poi = 0
    for i in range(1+num_ele_obs):
        for j in range(1+num_ele_obs):
            xi = -0.5 + i*1.0 / num_ele_obs
            yi = -0.5 + j*1.0 / num_ele_obs
            dist2 = (data_nodes_cal[:,1]-xi)**2 + (data_nodes_cal[:,2]-yi)**2
            ind = np.argmin(dist2)
            if (dist2[ind]>1.0E-8):
                print('Error! > No matching!')
                exit()
            data_nodes_obs[poi,:] = data_nodes_cal[ind,1:]
            poi = poi + 1
    print(data_nodes_obs)
    np.save('Output_' + name, data_nodes_obs)
    session.odbs[jobname + '.odb'].close()
    pass
    #return data_nodes_obs  # n * 4: (x,y,u,v)
