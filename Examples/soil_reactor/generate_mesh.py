# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# generate_mesh.py : python script that integrates Gmsh library and generates the mesh file (mesh.msh2) of the desired 3D model (e.g., Reactor building, Reactor building + Auxiliary building,
#                    Reactor building + Auxiliary building + Seismic Resonant Metamaterials, etc) 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# author: Constantinos Kanellopoulos (2023)
# Chair of Structural Dynamics and Earthquake Engineering
# ETH, Zurich
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

###########################################################################################################################################################################
## import python modules and set options for gmsh
#
import gmsh
import sys
import numpy as np
import math

gmsh.initialize() # initialize gmsh
gmsh.model.add("NPP") # name of gmsh model
## set options for gmsh - refer to gmsh documentation for further information
gmsh.option.setNumber("Geometry.Tolerance", 1e-08)
gmsh.option.setNumber("Geometry.AutoCoherence", 1) # Remove all duplicate elementary entities (e.g., points having identical coordinates). Note that with the built-in geometry kernel Gmsh executes the Coherence command automatically after each geometrical transformation, unless Geometry.AutoCoherence is set to zero
gmsh.option.setNumber("Geometry.CopyMeshingMethod", 1)	#when copy geometrical entities, copy also the meshing properties of the original object
gmsh.option.setNumber("Mesh.ElementOrder", 1) # 1: 8-node bricks,  2: 27-node bricks 
#gmsh.option.setNumber("Mesh.SaveAll", 1)	# To force Gmsh to save all elements, whether they belong to physical groups or not, set the `Mesh.SaveAll' option to 1.)

###########################################################################################################################################################################
## activate (1) or deactivate (0) components of the model 
#
# activate or deactivate Reactor building
reactor_building = 1 
# activate or deactivate Auxiliary building
auxiliary_building = 0 
# activate or deactivate Metamaterials (each row or column looking at the plan view is devided in four groups (reading from left to right)) 
metamaterials = [(0, 0, 0, 0),	#top row, 
				(0, 0, 0, 0),	#bottom row
				(0, 0, 0, 0),	#left column
				(0, 0, 0, 0)]	#right column

metamaterials_exist = 0 # put 1 if at least one of the rows or columns above is activated

# activate or deactivate embedment
embedment = 1 # if embedment is activated, the last two soil layers with thickness 4.5 and 7 (see right below in SL) will be generated (note: metamaterials can be generated only if embedment is activated, so if metatamaterials are activated, the last 2 soil layers will be generated automatically)
			  # if embedment is deactivated, only the first 6 layers will be generated (the top of the 6th layer is the foundation level of the structures)

# activate or deactivate soil between the Reactor and the Auxiliary building
separation = 1 # 0 means gap, 1 means filled with soil (note: it will make a difference only when embedment is activated (i.e., the last 2 soil layers of thickness 4.5 and 7 in SL are generated))

# number of DRM layers (6 are used in the paper)
DRM_l = 0 # creating DRM layers takes time, try with 0 or 1 DRM layers first if you just want to visualize quickly the parts of the model

###########################################################################################################################################################################
## definition of variables
#
# variables related to soil geometry
SL = [4, 12, 4, 4, 6, 4.5, 4.5, 7]	# m, soil layers thickness from bot to top 
# The following variables are related to dimensions for Seismic Resonant Metamaterials that can be placed around the Auxiliary building. 
# Their geometic variables are still needed to create the geometry of the model WITHOUT them.
clear = 30 # m, distance between the edge of the model and the Metamaterials
n_meta_v = 3 # number of Metamaterials in the vertical direction (up to 5)
n_meta_h = 3 # number of Metamaterials in the horizontal direction
L_meta = 2 # m, length of each unit-cell Metamaterial (don't change) 
depth_meta = n_meta_v*L_meta # m, embedded depth of Metamaterials 
D = 1 # distance between the Metamaterials and the Auxiliary building

# variables related to Reactor building geometry
R_out = 20 # m, outside radius of the external wall of the Reactor building
R_in = 16.8 # m, outside radius of the internal wall of the Reactor building
R_vessel = 12.5 # m, outside radius of the Cylindrical Wall (CW) of the Reactor building
R_box = 6 # m, half side of the square in the center of the model (don't change so that foundation's and Reactor's Vessel (RV) base (pedestal) nodes coincide)
D_separation = 1 # m, distance between the Reactor and the Auxiliary building
t_out = 1.6 # m, thickness of the external wall
t_in = 1.6 # m, thickness of the internal wall
t_vessel = 1.2 # m, thickness of the Cylindrical Wall (CW)
width_vessel = 4.5 # m, width of the Water Pool (WP) (clear opening)
h_reactor = 47 # m, height of the Reactor building (without the foundation and the dome), total height of the Reactor building = 4.5 + 47 + 20 = 71.5 m
h_vessel = 27 # m, height of the Cylindrical Wall (CW) without the Water Pool (WP), total height of the Cylindrical Wall (CW) = 27 + 1.2 + 6.8 = 35 m
h_water_vessel = 8-t_vessel  # m, height of the Water Pool (WP)
h_reactor_vessel_base = 9 # m, height of the Reactor Vessel Pedestal (RVP)
h_reactor_vessel_walls = 16 # m, height of the Biological Shield Wall (BSW)

# variables related to Auxiliary building geometry
L_aux = 96 # m, length of the Auxiliary building (don't change)
floors_aux = 5 # number of floors of the Auxiliary building
h_floor_aux = 7 # m, height of each floor of the Auxiliary building

# different mesh sizes in m to be used later
mesh_lv1 = 0.5 # 0.5
mesh_lv2 = mesh_lv1*2
mesh_lv3 = mesh_lv1*3
mesh_lv4 = mesh_lv1*4
mesh_lv5 = mesh_lv1*5
mesh_lv6 = mesh_lv1*6
mesh_lv7 = mesh_lv1*7
mesh_lv8 = mesh_lv1*8
mesh_lv9 = mesh_lv1*9
mesh_lv10 = mesh_lv1*10

###########################################################################################################################################################################
## y coordinates of the model (without the circular part in the middle)
y0 = 0
y1 = clear
y2 = clear + n_meta_h*L_meta
y3 = clear + n_meta_h*L_meta + D
y4 = clear + n_meta_h*L_meta + D + L_aux*0.25
y5 = clear + n_meta_h*L_meta + D + L_aux*0.5
y6 = clear + n_meta_h*L_meta + D + L_aux*0.75
y7 = clear + n_meta_h*L_meta + D + L_aux*1.0
y8 = clear + n_meta_h*L_meta + D + L_aux*1.0 + D
y9 = clear + n_meta_h*L_meta + D + L_aux*1.0 + D + n_meta_h*L_meta
y10 = clear + n_meta_h*L_meta + D + L_aux*1.0 + D + n_meta_h*L_meta + clear

y = np.array(0)	#y0
y = np.append(y, [y1,y2,y3,y4,y5,y6,y7,y8,y9,y10]) #0 already in the matrix, dont forget
print("\n" + "y_coord_(without_circular_part) : " + str(y))

## x coordinates of the model (without the circular part in the middle)
x0 = 0
x1 = clear
x2 = clear + n_meta_h*L_meta
x3 = clear + n_meta_h*L_meta + D
x4 = clear + n_meta_h*L_meta + D + L_aux*0.25
x5 = clear + n_meta_h*L_meta + D + L_aux*0.5
x6 = clear + n_meta_h*L_meta + D + L_aux*0.75
x7 = clear + n_meta_h*L_meta + D + L_aux*1.0
x8 = clear + n_meta_h*L_meta + D + L_aux*1.0 + D
x9 = clear + n_meta_h*L_meta + D + L_aux*1.0 + D + n_meta_h*L_meta
x10 = clear + n_meta_h*L_meta + D + L_aux*1.0 + D + n_meta_h*L_meta + clear

x = np.array(0)	#x0
x = np.append(x, [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10])  #0 already in the matrix, dont forget
print("x_coord_(without_circular_part) : " + str(x))

## z coordinates of the soil
z0 = 0
z1 = SL[0]
z2 = SL[0]+SL[1]
z3 = SL[0]+SL[1]+SL[2]
z4 = SL[0]+SL[1]+SL[2]+SL[3]
z5 = SL[0]+SL[1]+SL[2]+SL[3]+SL[4]
z6 = SL[0]+SL[1]+SL[2]+SL[3]+SL[4]+SL[5]

if depth_meta > SL[7]:
	z7 = z6 + (SL[6]+SL[7]-10) #partition at -10
	z8 = z6 + (SL[6]+SL[7]-8) #partition at -8
	z9 = z6 + SL[6]
	z10 = z9 + (SL[7]-6) #partition at -6
	z11 = z9 + SL[7]
	z = np.array(0)	
	z = np.append(z, [z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11]) #0 already in the matrix, dont forget
	print("z_coord_soil : " + str(z))

else:
	z7 = z6 + SL[6]
	z8 = z7 + SL[7]-depth_meta
	z9 = z7 + SL[7]	
	z = np.array(0)	
	z = np.append(z, [z1,z2,z3,z4,z5,z6,z7,z8,z9]) #0 already in the matrix, dont forget
	print("z_coord_soil : " + str(z))


## z coordinates of the reactor
offset_reactor_building = 30
z_r1 = z6 + offset_reactor_building

if depth_meta > SL[7]:
	z_r2 = z_r1 + (SL[6]+SL[7]-10) #corresponding to partition at -10
	z_r3 = z_r1 + (SL[6]+SL[7]-8) #corresponding to partition at -8
	z_r4 = z_r1 + SL[6]
	z_r5 = z_r4 + (SL[7]-6) #corresponding to partition at -6
	z_r6 = z_r4 + SL[7]
	z_r7 = z_r4 + h_vessel
	z_r8 = z_r4 + h_reactor
	z_r9 = z_r8 + R_out
	z_r = np.empty(0)	
	z_r = np.append(z_r, [z_r1,z_r2,z_r3,z_r4,z_r5,z_r6,z_r7,z_r8,z_r9]) 
	print("z_coord_reactor : " + str(z_r))

else:
	z_r2 = z_r1 + SL[6]
	z_r3 = z_r2 + SL[7]-depth_meta
	z_r4 = z_r2 + SL[7]
	z_r5 = z_r2 + h_vessel
	z_r6 = z_r2 + h_reactor
	z_r7 = z_r6 + R_out	
	z_r = np.empty(0)	
	z_r = np.append(z_r, [z_r1,z_r2,z_r3,z_r4,z_r5,z_r6,z_r7]) 
	print("z_coord_reactor : " + str(z_r))



## z coordinates of the auxiliary building -- foundation 
offset_auxiliary_building_foundation = 150
z_aux_found1 = z6 + offset_auxiliary_building_foundation
z_aux_found = [z_aux_found1]

if depth_meta > SL[7]:
	z_aux_found2 = z_aux_found1 + (SL[6]+SL[7]-10) 	;	z_aux_found.append(z_aux_found2)	#corresponding to partition at -10
	z_aux_found3 = z_aux_found1 + (SL[6]+SL[7]-8) 	;	z_aux_found.append(z_aux_found3)	#corresponding to partition at -8
	z_aux_found4 = z_aux_found1 + SL[6]				;	z_aux_found.append(z_aux_found4)

else:
	z_aux_found2 = z_aux_found1 + SL[6]				;	z_aux_found.append(z_aux_found2)

print("z_coord_auxiliary building -- foundation : " + str(z_aux_found))



## z coordinates of the auxiliary building -- structure
offset_auxiliary_building_structure = 200
z_aux_str1 = z6 + offset_auxiliary_building_structure
z_aux_str = [z_aux_str1]

if depth_meta > SL[7]:
	z_aux_str2 = z_aux_str1 + (SL[6]+SL[7]-10) 	;	z_aux_str.append(z_aux_str2)	#corresponding to partition at -10
	z_aux_str3 = z_aux_str1 + (SL[6]+SL[7]-8) 	;	z_aux_str.append(z_aux_str3)	#corresponding to partition at -8
	z_aux_str4 = z_aux_str1 + SL[6]				;	z_aux_str.append(z_aux_str4)
	z_aux_str5 = z_aux_str4 + (SL[7]-6)			;	z_aux_str.append(z_aux_str5) 	#corresponding to partition at -6
	z_aux_str6 = z_aux_str4 + SL[7]				;	z_aux_str.append(z_aux_str6) 

else:
	z_aux_str2 = z_aux_str1 + SL[6]				;	z_aux_str.append(z_aux_str2)
	z_aux_str3 = z_aux_str2 + SL[7]-depth_meta	;	z_aux_str.append(z_aux_str3)
	z_aux_str4 = z_aux_str2 + SL[7]				;	z_aux_str.append(z_aux_str4)

for i in range(0,floors_aux-1):
	z_aux_str.append(z_aux_str1+SL[6]+SL[7]+(i+1)*h_floor_aux)

print("z_coord_auxiliary building -- structure : " + str(z_aux_str))



## z coordinates of the metamaterials
offset_metamaterials = 250
z_m_bot = z6 + SL[6] + SL[7] - depth_meta + offset_metamaterials
z_m_top = z6 + SL[6] + SL[7] + offset_metamaterials

print("z_coord_metamaterials (bottom, top) : " + str(z_m_bot) + ', ' + str(z_m_top))



## assign mesh level to each line in y and x directions
y_lines_mesh_lv = np.empty(0)
# print(y_lines_mesh_lv)
y_lines_mesh_lv = np.append(y_lines_mesh_lv,[mesh_lv5, mesh_lv4, mesh_lv4, mesh_lv4, mesh_lv4, mesh_lv4, mesh_lv4, mesh_lv4, mesh_lv4, mesh_lv5])
print("\n" + 'Mesh size of y lines: ' + str(y_lines_mesh_lv))

x_lines_mesh_lv = np.empty(0)
# print(x_lines_mesh_lv)
x_lines_mesh_lv = np.append(x_lines_mesh_lv,[mesh_lv5, mesh_lv4, mesh_lv4, mesh_lv4, mesh_lv4, mesh_lv4, mesh_lv4, mesh_lv4, mesh_lv4, mesh_lv5])
print('Mesh size of x lines: ' + str(x_lines_mesh_lv))

z_lines_mesh_lv = np.empty(0)
# print(z_lines_mesh_lv)
z_lines_mesh_lv = np.append(z_lines_mesh_lv,[mesh_lv9, mesh_lv9, mesh_lv5, mesh_lv5, mesh_lv4, mesh_lv4, mesh_lv4, mesh_lv4]) #the basic 8 layers
print('Mesh size of z lines: ' + str(z_lines_mesh_lv))

###########################################################################################################################################################################
## create points and put their tags in p matrix
p = np.zeros([len(x),len(y)], dtype=np.int64)
#print(p,type(p))

for i in range(0,len(x)):
	for j in range(0,len(y)):
		p[i,j] = gmsh.model.geo.addPoint(x[i], y[j], 0, 0, -1) #returns the tag of the point
		
print("\n" + "p_tags = " + "\n" + str(p))
gmsh.model.geo.synchronize()

###########################################################################################################################################################################
## create lines in y direction
for i in range(0,len(x)):

	for j in range(0,len(y)-1):

		xyz1 = gmsh.model.getValue(0, p[i,j], []) #returns x,y,z coord
		xyz2 = gmsh.model.getValue(0, p[i,j+1], [])

		if (j != len(y)-1) and not(i==5 and (j==4 or j==5)): #and not to exclude the circular part in the middle
			l = gmsh.model.geo.addLine(p[i,j], p[i,j+1], -1)
			#transfinite
			gmsh.model.geo.mesh.setTransfiniteCurve(l, int(math.ceil((xyz2[1]-xyz1[1])/y_lines_mesh_lv[j])+1)  )	#ceil returns int, but in format of float, so put int in front

gmsh.model.geo.synchronize()

###########################################################################################################################################################################
## create lines in x direction
for j in range(0,len(y)):

	for i in range(0,len(x)-1):

		xyz1 = gmsh.model.getValue(0, p[i,j], []) #returns x,y,z coord
		xyz2 = gmsh.model.getValue(0, p[i+1,j], [])

		if (i != len(x)-1) and not(j==5 and (i==4 or i==5)): #and not to exclude the circular part in the middle
			l = gmsh.model.geo.addLine(p[i,j], p[i+1,j], -1)
			#transfinite
			gmsh.model.geo.mesh.setTransfiniteCurve(l, int(math.ceil((xyz2[0]-xyz1[0])/x_lines_mesh_lv[i])+1)  )

gmsh.model.geo.synchronize()

###########################################################################################################################################################################
## create surfaces
e = 0.1
k = 1
s = 1

for j in range(0,len(y)-1):

	for i in range(0,len(x)-1):

		xyz1 = gmsh.model.getValue(0, p[i,j], [])
		xyz2 = gmsh.model.getValue(0, p[i+1,j+1], [])
		l_tags = gmsh.model.getEntitiesInBoundingBox(xyz1[0] - e, xyz1[1] - e, xyz1[2] - e,    xyz2[0] + e, xyz2[1] + e, xyz2[2] + e, 1)
		#print(l_tags)

		if len(l_tags) == 4 :  #if 4 lines are in l_tags, then create the surface
			gmsh.model.geo.addCurveLoop([ l_tags[0][1], l_tags[3][1], -l_tags[1][1], -l_tags[2][1] ], k)
			gmsh.model.geo.addPlaneSurface([k], s)
			gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
			gmsh.model.geo.mesh.setRecombine(2, s)
			k += 1
			s += 1

gmsh.model.geo.synchronize()

# ###########################################################################################################################################################################
## create the points in the centre

xyz_center = gmsh.model.getValue(0, p[5,5], []) #get coord of the center of the circles

R = np.array([ R_out+D_separation, R_out, R_out-t_out, R_in, R_in-t_in, R_vessel, R_vessel-t_vessel, R_box]) #create an array with all the radius
# print(R)

p_centre = np.zeros([len(R),8], dtype=int)

for i, Ri in enumerate(R) :
	# print(i,Ri)
	if Ri != R_box:

		for j in range(0,8):
			yy = xyz_center[1] + Ri * math.cos(j*math.pi/4)
			xx = xyz_center[0] + Ri * math.sin(j*math.pi/4)
			p_centre[i,j] = gmsh.model.geo.addPoint(xx, yy, 0, 0, -1)

	elif Ri == R_box: #the box helps to create a better mesh

		combination = np.array([[1,0],
								[1,1],
								[0,1],
								[-1,1],
								[-1,0],
								[-1,-1],
								[0,-1],
								[1,-1]
								])
		# print(combination)

		for j in range(0,8):
			yy = xyz_center[1] + R_box*combination[j,0]
			xx = xyz_center[0] + R_box*combination[j,1]
			p_centre[i,j] = gmsh.model.geo.addPoint(xx, yy, 0, 0, -1)

print("\n" + "p_centre_tags = " + "\n" + str(p_centre))

gmsh.model.geo.synchronize()

###########################################################################################################################################################################
## create lines, arcs and surfaces in the centre

#radial lines
l_radial = np.zeros([len(R)-1,8], dtype=int)

for j in range(0,8):
	for i in range(0,len(R)-1):
		l = gmsh.model.geo.addLine(p_centre[i,j], p_centre[i+1,j], -1)
		l_radial[i,j] = l

		if i != len(R)-2 :
			xyz1 = gmsh.model.getValue(0, p_centre[i,j], [])
			xyz2 = gmsh.model.getValue(0, p_centre[i+1,j], [])
			length = math.sqrt( (xyz2[0]-xyz1[0])**2 + (xyz2[1]-xyz1[1])**2 ) 
			# print(length)
			gmsh.model.geo.mesh.setTransfiniteCurve(l, int(math.ceil(length/mesh_lv4)+1)  )	#ceil returns int, but in format of float, so put int in front
			# print(int(math.ceil(length/mesh_lv4)+1))
		else:
			gmsh.model.geo.mesh.setTransfiniteCurve(l, 4 )

print("\n" + "l_radial_tags = " + "\n" + str(l_radial))


#arcs
l_arcs = np.zeros([len(R)-1,8], dtype=int)

for i in range(0,len(R)-1):
		for j in range(0,8):
			
			if j!=(8-1):
				c = gmsh.model.geo.addCircleArc(p_centre[i,j], p[5,5], p_centre[i,j+1])
				l_arcs[i,j] = c
			else:
				c = gmsh.model.geo.addCircleArc(p_centre[i,j], p[5,5], p_centre[i,0])
				l_arcs[i,j] = c

			xyz1 = gmsh.model.getValue(0, p[0,5], [])
			xyz2 = gmsh.model.getValue(0, p[0,6], [])
			gmsh.model.geo.mesh.setTransfiniteCurve(c, int(math.ceil((xyz2[1]-xyz1[1])/y_lines_mesh_lv[5])+1)  ) 

print("\n" + "l_arcs_tags = " + "\n" + str(l_arcs))


# create first surfaces (arcs-radial lines)
for i in range(0,len(l_radial)-1):
	for j in range(0,8):

		if j!=(8-1):
			loops = gmsh.model.geo.addCurveLoops([l_radial[i,j],l_radial[i,j+1],l_arcs[i,j],l_arcs[i+1,j]])
			s = gmsh.model.geo.addPlaneSurface(loops)
		else:
			loops = gmsh.model.geo.addCurveLoops([l_radial[i,j],l_radial[i,0],l_arcs[i,j],l_arcs[i+1,j]])
			s = gmsh.model.geo.addPlaneSurface(loops)
		
		gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
		gmsh.model.geo.mesh.setRecombine(2, s)	


#box
l_box_out = np.zeros([1,8], dtype=int)
l_box_in = np.zeros([1,8], dtype=int)

for j in range(0,8):

	if j!=(8-1):
		l_out = gmsh.model.geo.addLine(p_centre[len(R)-1,j], p_centre[len(R)-1,j+1], -1)
		l_box_out[0,j] = l_out
			
		if j%2==0: # 4 lines inside (cross)
			l_in = gmsh.model.geo.addLine(p_centre[len(R)-1,j], p[5,5], -1)
			l_box_in[0,j] = l_in
			gmsh.model.geo.mesh.setTransfiniteCurve(l_in, int(math.ceil((xyz2[1]-xyz1[1])/y_lines_mesh_lv[5])+1)  )

	else:
		l_out = gmsh.model.geo.addLine(p_centre[len(R)-1,j], p_centre[len(R)-1,0], -1)
		l_box_out[0,j] = l_out

	gmsh.model.geo.mesh.setTransfiniteCurve(l_out, int(math.ceil((xyz2[1]-xyz1[1])/y_lines_mesh_lv[5])+1)  )

print("\n" + "l_box_out_tags = " + "\n" + str(l_box_out))
print("\n" + "l_box_in_tags = " + "\n" + str(l_box_in))


# create surfaces between box and arcs-radial lines
for j in range(0,8):

		if j!=(8-1):
			loops = gmsh.model.geo.addCurveLoops([l_radial[len(l_radial)-1,j],l_radial[len(l_radial)-1,j+1],l_arcs[len(l_arcs)-1,j],l_box_out[0,j]])
			s = gmsh.model.geo.addPlaneSurface(loops)
		else:
			loops = gmsh.model.geo.addCurveLoops([l_radial[len(l_radial)-1,j],l_radial[len(l_radial)-1,0],l_arcs[len(l_arcs)-1,j],l_box_out[0,j]])
			s = gmsh.model.geo.addPlaneSurface(loops)
		
		gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
		gmsh.model.geo.mesh.setRecombine(2, s)

# create surfaces inside box
for j in range(0,8,2):

	if j!=(8-2):
		loops = gmsh.model.geo.addCurveLoops([ l_box_out[0,j],l_box_out[0,j+1],l_box_in[0,j],l_box_in[0,j+2] ])
		s = gmsh.model.geo.addPlaneSurface(loops)
	else:
		loops = gmsh.model.geo.addCurveLoops([ l_box_out[0,j],l_box_out[0,j+1],l_box_in[0,j],l_box_in[0,0] ])
		s = gmsh.model.geo.addPlaneSurface(loops)

	gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
	gmsh.model.geo.mesh.setRecombine(2, s)			
			

#lines from square to circle
l_square_to_circle = np.zeros([1,8], dtype=int)
l_square = np.zeros([1,8], dtype=int)

xy = np.array([[5,6],
			[6,6],
			[6,5],
			[6,4],
			[5,4],
			[4,4],
			[4,5],
			[4,6]
			])

for j in range(0,8):

	if j!=(8-1):

		l = gmsh.model.geo.addLine(p[xy[j,0],xy[j,1]], p_centre[0,j], -1)
		l_square_to_circle[0,j] = l
		xy1 = gmsh.model.getValue(0, p[xy[j,0],xy[j,1]], [])
		xy2 = gmsh.model.getValue(0, p[xy[j+1,0],xy[j+1,1]], [])
		# print(xy1)
		# print(xy2)
		if xy1[0]<xy2[0] and xy1[1]==xy2[1]:
			l_tag = gmsh.model.getEntitiesInBoundingBox(xy1[0] - e, xy1[1] - e, xy1[2] - e,    xy2[0] + e, xy2[1] + e, xy2[2] + e, 1)	
			# print(l_tag)	#e.g. [(1, 172)]
			l_square[0,j] = l_tag[0][1]
		elif xy1[0]==xy2[0] and xy1[1]>xy2[1]:
			l_tag = gmsh.model.getEntitiesInBoundingBox(xy1[0] - e, xy2[1] - e, xy1[2] - e,    xy2[0] + e, xy1[1] + e, xy2[2] + e, 1)	
			# print(l_tag)	#e.g. [(1, 172)]
			l_square[0,j] = l_tag[0][1]
		elif xy1[0]>xy2[0] and xy1[1]==xy2[1]:
			l_tag = gmsh.model.getEntitiesInBoundingBox(xy2[0] - e, xy1[1] - e, xy1[2] - e,    xy1[0] + e, xy2[1] + e, xy2[2] + e, 1)	
			# print(l_tag)	#e.g. [(1, 172)]
			l_square[0,j] = l_tag[0][1]	
		elif xy1[0]==xy2[0] and xy1[1]<xy2[1]:
			l_tag = gmsh.model.getEntitiesInBoundingBox(xy1[0] - e, xy1[1] - e, xy1[2] - e,    xy2[0] + e, xy2[1] + e, xy2[2] + e, 1)	
			# print(l_tag)	#e.g. [(1, 172)]
			l_square[0,j] = l_tag[0][1]		

	else:

		l = gmsh.model.geo.addLine(p[xy[j,0],xy[j,1]], p_centre[0,7], -1)
		l_square_to_circle[0,j] = l
		xy1 = gmsh.model.getValue(0, p[xy[j,0],xy[j,1]], [])
		xy2 = gmsh.model.getValue(0, p[xy[0,0],xy[0,1]], [])
		l_tag = gmsh.model.getEntitiesInBoundingBox(xy1[0] - e, xy1[1] - e, xy1[2] - e,    xy2[0] + e, xy2[1] + e, xy2[2] + e, 1)
		l_square[0,j] = l_tag[0][1]	
		

	gmsh.model.geo.mesh.setTransfiniteCurve(l, 5  )

print("\n" + "l_square_to_circle_tags = " + "\n" + str(l_square_to_circle))
print("\n" + "l_square_tags = " + "\n" + str(l_square))

# create surfaces between square and arcs-radial lines
for j in range(0,8):

	if j!=(8-1):
		loops = gmsh.model.geo.addCurveLoops([ l_square[0,j],l_arcs[0,j],l_square_to_circle[0,j],l_square_to_circle[0,j+1] ])
		s = gmsh.model.geo.addPlaneSurface(loops)
	else:
		loops = gmsh.model.geo.addCurveLoops([ l_square[0,j],l_arcs[0,j],l_square_to_circle[0,j],l_square_to_circle[0,0] ])
		s = gmsh.model.geo.addPlaneSurface(loops)
	
	gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
	gmsh.model.geo.mesh.setRecombine(2, s)

gmsh.model.geo.synchronize()

###########################################################################################################################################################################
# extrude soil layers
count_extrude = 1
#extrude first 6 layers,,,,, consumes time
H=0
for i in range(0,6):    #####################6 

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(0 - e, 0 - e, H - e,    x10 + e, y10 + e, H + e, 2)
	numElements = int(math.ceil(SL[i]/z_lines_mesh_lv[i]))
	# gmsh.option.setNumber("Geometry.AutoCoherence", 1) 	#didn't reduce running time
	gmsh.model.geo.extrude(s_dimtags, 0, 0, SL[i], [numElements], recombine = True)
	print("\n" + 'Extrude soil ' + str(count_extrude) + ' done !')
	count_extrude += 1
	# gmsh.option.setNumber("Geometry.AutoCoherence", 0)
	H = H + SL[i]
	gmsh.model.geo.synchronize()
	

#extrude last 2 soil layers
if embedment == 1 or metamaterials_exist == 1 :

	if depth_meta > SL[7]:	#meta are embedded into the second soil layer

		for i in range (6,len(z)-1):	#extrusion from z6 to z10, len(z)=12

			s_all_dimtags = gmsh.model.getEntitiesInBoundingBox(0 - e, 0 - e, z[i] - e,    x10 + e, y10 + e, z[i] + e, 2)	#select all the surfaces and start removing

			remove_dimtags = []

			if z[i] == z6:	#buildings are at level z6
				# auxiliary building dimtags
				if auxiliary_building == 1:
					s_aux_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z[i] - e,    x7 + e, y7 + e, z[i] + e, 2)
					xy = gmsh.model.getValue(0, p_centre[0,0], []) # first point of the first circle
					remove = gmsh.model.getEntitiesInBoundingBox(xy[0]-R_out-D_separation - e, xy[1]-2*R_out-2*D_separation - e, z[i] - e,    xy[0]+R_out+D_separation + e, xy[1] + e, z[i] + e, 2)
					s_aux_dimtags = list(set(s_aux_dimtags)^set(remove)) #remove common dimtags
					remove_dimtags += s_aux_dimtags

				# reactor building dimtags
				xy = gmsh.model.getValue(0, p_centre[1,0], [])
				s_reactor_dimtags = gmsh.model.getEntitiesInBoundingBox(xy[0]-R_out - e, xy[1]-2*R_out - e, z[i] - e,    xy[0]+R_out + e, xy[1] + e, z[i] + e, 2)
				if (reactor_building == 1) or (auxiliary_building == 1 and reactor_building == 0):
					# xy = gmsh.model.getValue(0, p_centre[1,0], [])
					# s_reactor_dimtags = gmsh.model.getEntitiesInBoundingBox(xy[0]-R_out - e, xy[1]-2*R_out - e, z[i] - e,    xy[0]+R_out + e, xy[1] + e, z[i] + e, 2)
					remove_dimtags += s_reactor_dimtags

				# separation dimtags
				if separation == 0: #opposite
					s_separation_dimtags = list(set(s_reactor_dimtags)^set(remove))
					remove_dimtags += s_separation_dimtags

			# metamaterial dimtags
			if depth_meta == (z11-z[i]):	#meta are at level z7 or z8
				s_meta_dimtags = [ [],[],[],[] ]
					
				for mi in range (0,4):
					s_meta_dimtags[0].append( gmsh.model.getEntitiesInBoundingBox(x[mi+3] - e, y8 - e, z[i] - e,    x[mi+4] + e, y9 + e, z[i] + e, 2) )
					s_meta_dimtags[1].append( gmsh.model.getEntitiesInBoundingBox(x[mi+3] - e, y1 - e, z[i] - e,    x[mi+4] + e, y2 + e, z[i] + e, 2) )
					s_meta_dimtags[2].append( gmsh.model.getEntitiesInBoundingBox(x1 - e, y[mi+3] - e, z[i] - e,    x2 + e, y[mi+4] + e, z[i] + e, 2) )
					s_meta_dimtags[3].append( gmsh.model.getEntitiesInBoundingBox(x8 - e, y[mi+3] - e, z[i] - e,    x9 + e, y[mi+4] + e, z[i] + e, 2) )
					# print(s_meta_dimtags)

				for mi in range (0,len(metamaterials)):
					for mj in range (0,len(metamaterials)):
						# print(metamaterials[mi][mj])
						if metamaterials[mi][mj] == 1:

							remove_dimtags += s_meta_dimtags[mi][mj]

				# print(s_meta_dimtags)

			#extrude
			surface_to_extrude = list(set(s_all_dimtags)^set(remove_dimtags))
			numElements = int(math.ceil((z[i+1]-z[i])/z_lines_mesh_lv[7])) #take 7 (top layer) conservatively
			# gmsh.option.setNumber("Geometry.AutoCoherence", 1)
			gmsh.model.geo.extrude(surface_to_extrude, 0, 0, (z[i+1]-z[i]), [numElements], recombine = True)
			print("\n" + 'Extrude soil ' + str(count_extrude) + ' done !')
			count_extrude += 1
			# gmsh.option.setNumber("Geometry.AutoCoherence", 0)

			gmsh.model.geo.synchronize()


	elif depth_meta < SL[7]: #meta are embedded only in the first soil layer

		for i in range (6,len(z)-1): #extrusion from z6 to z8, len(z)=10

			s_all_dimtags = gmsh.model.getEntitiesInBoundingBox(0 - e, 0 - e, z[i] - e,    x10 + e, y10 + e, z[i] + e, 2) #select all the surfaces and start removing

			remove_dimtags = []

			if z[i] == z6:	#buildings are at level z6
				# auxiliary building dimtags
				if auxiliary_building == 1:
					s_aux_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z[i] - e,    x7 + e, y7 + e, z[i] + e, 2)
					xy = gmsh.model.getValue(0, p_centre[0,0], []) # first point of the first circle
					remove = gmsh.model.getEntitiesInBoundingBox(xy[0]-R_out-D_separation - e, xy[1]-2*R_out-2*D_separation - e, z[i] - e,    xy[0]+R_out+D_separation + e, xy[1] + e, z[i] + e, 2)
					s_aux_dimtags = list(set(s_aux_dimtags)^set(remove)) #remove common dimtags
					remove_dimtags += s_aux_dimtags

				# reactor building dimtags
				xy = gmsh.model.getValue(0, p_centre[1,0], [])
				s_reactor_dimtags = gmsh.model.getEntitiesInBoundingBox(xy[0]-R_out - e, xy[1]-2*R_out - e, z[i] - e,    xy[0]+R_out + e, xy[1] + e, z[i] + e, 2)
				if (reactor_building == 1) or (auxiliary_building == 1 and reactor_building == 0):
					# xy = gmsh.model.getValue(0, p_centre[1,0], [])
					# s_reactor_dimtags = gmsh.model.getEntitiesInBoundingBox(xy[0]-R_out - e, xy[1]-2*R_out - e, z[i] - e,    xy[0]+R_out + e, xy[1] + e, z[i] + e, 2)
					remove_dimtags += s_reactor_dimtags

				# separation dimtags
				if separation == 0: #opposite
					xy = gmsh.model.getValue(0, p_centre[0,0], []) # first point of the first circle
					remove = gmsh.model.getEntitiesInBoundingBox(xy[0]-R_out-D_separation - e, xy[1]-2*R_out-2*D_separation - e, z[i] - e,    xy[0]+R_out+D_separation + e, xy[1] + e, z[i] + e, 2)
					s_separation_dimtags = list(set(s_reactor_dimtags)^set(remove))
					remove_dimtags += s_separation_dimtags

			# metamaterial dimtags
			if depth_meta == (z9-z[i]):	#meta are at level z8
				s_meta_dimtags = [ [],[],[],[] ]
					
				for mi in range (0,4):
					s_meta_dimtags[0].append( gmsh.model.getEntitiesInBoundingBox(x[mi+3] - e, y8 - e, z[i] - e,    x[mi+4] + e, y9 + e, z[i] + e, 2) )
					s_meta_dimtags[1].append( gmsh.model.getEntitiesInBoundingBox(x[mi+3] - e, y1 - e, z[i] - e,    x[mi+4] + e, y2 + e, z[i] + e, 2) )
					s_meta_dimtags[2].append( gmsh.model.getEntitiesInBoundingBox(x1 - e, y[mi+3] - e, z[i] - e,    x2 + e, y[mi+4] + e, z[i] + e, 2) )
					s_meta_dimtags[3].append( gmsh.model.getEntitiesInBoundingBox(x8 - e, y[mi+3] - e, z[i] - e,    x9 + e, y[mi+4] + e, z[i] + e, 2) )
					# print(s_meta_dimtags)

				for mi in range (0,len(metamaterials)):
					for mj in range (0,len(metamaterials)):
						# print(metamaterials[mi][mj])
						if metamaterials[mi][mj] == 1:

							remove_dimtags += s_meta_dimtags[mi][mj]

				# print(s_meta_dimtags)

			#extrude
			surface_to_extrude = list(set(s_all_dimtags)^set(remove_dimtags))
			numElements = int(math.ceil((z[i+1]-z[i])/z_lines_mesh_lv[7])) #take 7 (top layer) conservatively
			# gmsh.option.setNumber("Geometry.AutoCoherence", 1)
			gmsh.model.geo.extrude(surface_to_extrude, 0, 0, (z[i+1]-z[i]), [numElements], recombine = True)
			print("\n" + 'Extrude soil ' + str(count_extrude) + ' done !')
			count_extrude += 1
			# gmsh.option.setNumber("Geometry.AutoCoherence", 0)

			gmsh.model.geo.synchronize()


###########################################################################################################################################################################
## extrude DRM layers

DRM_mesh = mesh_lv5

x_DRM_min = - DRM_l * DRM_mesh
x_DRM_max = + x10 + DRM_l * DRM_mesh

y_DRM_min = - DRM_l * DRM_mesh
y_DRM_max = + y10 + DRM_l * DRM_mesh

z_DRM_min = - DRM_l * DRM_mesh


for i in range(0,DRM_l):	#x direction

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(-i*DRM_mesh - e, y0 - e, z0 - e,    -i*DRM_mesh + e, y10 + e, z[len(z)-1] + e, 2)
	gmsh.model.geo.extrude(s_dimtags, -DRM_mesh, 0, 0, [1], recombine = True)
	print("\n" + 'Extrude DRM x- ' + str(count_extrude) + ' done !')
	count_extrude += 1
	gmsh.model.geo.synchronize()

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x10+i*DRM_mesh - e, y0 - e, z0 - e,    x10+i*DRM_mesh + e, y10 + e, z[len(z)-1] + e, 2)
	gmsh.model.geo.extrude(s_dimtags, DRM_mesh, 0, 0, [1], recombine = True)
	print("\n" + 'Extrude DRM x+ ' + str(count_extrude) + ' done !')
	count_extrude += 1
	gmsh.model.geo.synchronize()


for i in range(0,DRM_l):	#y direction

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x_DRM_min - e, -i*DRM_mesh - e, z0 - e,    x_DRM_max + e, -i*DRM_mesh + e, z[len(z)-1] + e, 2)
	gmsh.model.geo.extrude(s_dimtags, 0, -DRM_mesh, 0, [1], recombine = True)
	print("\n" + 'Extrude DRM y- ' + str(count_extrude) + ' done !')
	count_extrude += 1
	gmsh.model.geo.synchronize()

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x_DRM_min - e, y10+i*DRM_mesh - e, z0 - e,    x_DRM_max + e, y10+i*DRM_mesh + e, z[len(z)-1] + e, 2)
	gmsh.model.geo.extrude(s_dimtags, 0, DRM_mesh, 0, [1], recombine = True)
	print("\n" + 'Extrude DRM y+ ' + str(count_extrude) + ' done !')
	count_extrude += 1
	gmsh.model.geo.synchronize()


for i in range(0,DRM_l):	#z direction

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x_DRM_min - e, y_DRM_min - e, -i*DRM_mesh - e,    x_DRM_max + e, y_DRM_max + e, -i*DRM_mesh + e, 2)
	gmsh.model.geo.extrude(s_dimtags, 0, 0, -DRM_mesh, [1], recombine = True)
	print("\n" + 'Extrude DRM z- ' + str(count_extrude) + ' done !')
	count_extrude += 1
	gmsh.model.geo.synchronize()


###########################################################################################################################################################################
## Physical groups
idx = 1

v_dimtags = gmsh.model.getEntitiesInBoundingBox(x_DRM_min - e, y_DRM_min - e, z_DRM_min - e,    x_DRM_max + e, y_DRM_max + e,  z[len(z)-1]+ e, 3)
# print(v_dimtags)
# print( [v_dimtags[i][1] for i in range(len(v_dimtags))] )

if (auxiliary_building == 1 and reactor_building == 1 and separation == 1):	#remove the separation from the soil layer
	v_dimtags_remove = gmsh.model.getEntitiesInBoundingBox(x4 - e, y4 - e,  z[len(z)-1]-SL[7]-SL[6] - e,    x6 + e, y6 + e, z[len(z)-1] + e, 3)
	v_dimtags = list(set(v_dimtags)^set(v_dimtags_remove))

gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
gmsh.model.setPhysicalName(3, idx, "soil_all")
idx += 1

h=0
for j,SLi in enumerate(SL):
	
	v_dimtags = gmsh.model.getEntitiesInBoundingBox(x0 - e, y0 - e, h - e,    x10 + e, y10 + e, h+SLi + e, 3)

	if (j == len(SL)-2 or j == len(SL)-1) and (auxiliary_building == 1 and reactor_building == 1 and separation == 1):	#remove the separation from the soil layer
		v_dimtags_remove = gmsh.model.getEntitiesInBoundingBox(x4 - e, y4 - e,  h - e,    x6 + e, y6 + e, h+SLi + e, 3)
		v_dimtags = list(set(v_dimtags)^set(v_dimtags_remove))
		
	h = h + SLi
	gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx)
	gmsh.model.setPhysicalName(3, idx, "soil_layer_"+str(j+1))
	idx += 1

h=0
for l in range(0,len(SL)):

	if l == 0 :	#first layer
		for i in range(0,DRM_l):
			v_dimtags1 = gmsh.model.getEntitiesInBoundingBox(x0-i*DRM_mesh - e, y0-i*DRM_mesh - e, h-i*DRM_mesh - e,    x10+i*DRM_mesh + e, y10+i*DRM_mesh + e, h+SL[l] + e, 3)
			v_dimtags2 = gmsh.model.getEntitiesInBoundingBox(x0-(i+1)*DRM_mesh - e, y0-(i+1)*DRM_mesh - e, h-(i+1)*DRM_mesh - e,    x10+(i+1)*DRM_mesh + e, y10+(i+1)*DRM_mesh + e, h+SL[l] + e, 3)

			# v_dimtags2 = gmsh.model.getEntitiesInBoundingBox(x[0]-(i+1)*cmesh - e, SL[0]-(i+1)*cmesh - e, 0 - e,    x[0]+L+(i+1)*cmesh + e, SL[1] + e, 1 + e, 3)
			cut = list(set(v_dimtags1)^set(v_dimtags2))
			gmsh.model.addPhysicalGroup(3, [cut[j][1] for j in range(len(cut))], idx)
			if i == 0:
				gmsh.model.setPhysicalName(3, idx, "DRM_layer" + "_soil_layer_1")
			else:	
				gmsh.model.setPhysicalName(3, idx, "damp_layer_" + str(i) + "_soil_layer_1")
			idx += 1

	else:

		for i in range(0,DRM_l):
			v_dimtags1 = gmsh.model.getEntitiesInBoundingBox(x0-i*DRM_mesh - e, y0-i*DRM_mesh - e, h - e,    x10+i*DRM_mesh + e, y10+i*DRM_mesh + e, h+SL[l] + e, 3)
			v_dimtags2 = gmsh.model.getEntitiesInBoundingBox(x0-(i+1)*DRM_mesh - e, y0-(i+1)*DRM_mesh - e, h - e,    x10+(i+1)*DRM_mesh + e, y10+(i+1)*DRM_mesh + e, h+SL[l] + e, 3)
			cut = list(set(v_dimtags1)^set(v_dimtags2))
			gmsh.model.addPhysicalGroup(3, [cut[j][1] for j in range(len(cut))], idx)
			if i == 0:
				gmsh.model.setPhysicalName(3, idx, "DRM_layer" + "_soil_layer_" + str(l+1))
			else:	
				gmsh.model.setPhysicalName(3, idx, "damp_layer_" + str(i) + "_soil_layer_" + str(l+1))
			idx += 1

	h = h + SL[l]


s_dimtags = gmsh.model.getEntitiesInBoundingBox(x_DRM_min - e, y_DRM_min - e, z_DRM_min - e,    x_DRM_max + e, y_DRM_max + e, z_DRM_min + e, 2)
gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
gmsh.model.setPhysicalName(2, idx, "bottom")
idx += 1

s_dimtags1 = gmsh.model.getEntitiesInBoundingBox(x_DRM_min - e, y_DRM_min - e, z_DRM_min - e,    x_DRM_min + e, y_DRM_max + e, z[len(z)-1] + e, 2)
s_dimtags2 = gmsh.model.getEntitiesInBoundingBox(x_DRM_max - e, y_DRM_min - e, z_DRM_min - e,    x_DRM_max + e, y_DRM_max + e, z[len(z)-1] + e, 2)
s_dimtags3 = gmsh.model.getEntitiesInBoundingBox(x_DRM_min - e, y_DRM_min - e, z_DRM_min - e,    x_DRM_max + e, y_DRM_min + e, z[len(z)-1] + e, 2)
s_dimtags4 = gmsh.model.getEntitiesInBoundingBox(x_DRM_min - e, y_DRM_max - e, z_DRM_min - e,    x_DRM_max + e, y_DRM_max + e, z[len(z)-1] + e, 2)
s_dimtags = s_dimtags1 + s_dimtags2 + s_dimtags3 + s_dimtags4
gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
gmsh.model.setPhysicalName(2, idx, "sides")
idx += 1


if reactor_building == 1:

	xy = gmsh.model.getValue(0, p_centre[1,0], [])
	s_reactor_dimtags = gmsh.model.getEntitiesInBoundingBox(xy[0]-R_out - e, xy[1]-2*R_out - e, z6 - e,    xy[0]+R_out + e, xy[1] + e, z6 + e, 2)
	gmsh.model.addPhysicalGroup(2, [s_reactor_dimtags[j][1] for j in range(len(s_reactor_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "soil-reactor_building_z")
	idx += 1

	if auxiliary_building == 0 or separation == 1:

		s_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_out - e, y5-R_out - e, z6 - e,    x5+R_out + e, y5+R_out + e, z[len(z)-1] + e, 2)
		s_dimtags_remove = gmsh.model.getEntitiesInBoundingBox(x5-R_out - e, y5-R_out - e, z6 - e,    x5+R_out + e, y5+R_out + e, z6 + e, 2)
		s_dimtags = list(set(s_dimtags)^set(s_dimtags_remove))
		gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
		gmsh.model.setPhysicalName(2, idx, "soil-reactor_building_radial")
		idx += 1


if auxiliary_building == 1:

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z6 - e,    x7 + e, y7 + e, z6 + e, 2)
	s_dimtags_remove = gmsh.model.getEntitiesInBoundingBox(x5-R_out-D_separation - e, y5-R_out-D_separation - e, z6 - e,    x5+R_out+D_separation + e, y5+R_out+D_separation + e, z6 + e, 2)
	s_dimtags = list(set(s_dimtags)^set(s_dimtags_remove))
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "soil-auxiliary_building_z")
	idx += 1

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z6 - e,    x3 + e, y7 + e, z[len(z)-1] + e, 2)
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "soil-auxiliary_building_x+")
	idx += 1

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x7 - e, y3 - e, z6 - e,    x7 + e, y7 + e, z[len(z)-1] + e, 2)
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "soil-auxiliary_building_x-")
	idx += 1

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z6 - e,    x7 + e, y3 + e, z[len(z)-1] + e, 2)
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "soil-auxiliary_building_y+")
	idx += 1

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y7 - e, z6 - e,    x7 + e, y7 + e, z[len(z)-1] + e, 2)
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "soil-auxiliary_building_y-")
	idx += 1

	if reactor_building == 0 or separation == 1:

		s_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_out-D_separation - e, y5-R_out-D_separation - e, z6 - e,    x5+R_out+D_separation + e, y5+R_out+D_separation + e, z[len(z)-1] + e, 2)
		s_dimtags_remove1 = gmsh.model.getEntitiesInBoundingBox(x5-R_out - e, y5-R_out - e, z6 - e,    x5+R_out + e, y5+R_out + e, z[len(z)-1] + e, 2)
		s_dimtags_remove2 = gmsh.model.getEntitiesInBoundingBox(x5-R_out-D_separation - e, y5-R_out-D_separation - e, z6 - e,    x5+R_out+D_separation + e, y5+R_out+D_separation + e, z6 + e, 2)
		s_dimtags_remove3 = gmsh.model.getEntitiesInBoundingBox(x5-R_out-D_separation - e, y5-R_out-D_separation - e, z[len(z)-1] - e,    x5+R_out+D_separation + e, y5+R_out+D_separation + e, z[len(z)-1] + e, 2)
		s_dimtags_remove = s_dimtags_remove1 + s_dimtags_remove2 + s_dimtags_remove3
		s_dimtags = list(set(s_dimtags)^set(s_dimtags_remove))
		gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
		gmsh.model.setPhysicalName(2, idx, "soil-auxiliary_building_radial")	# a few irrevelevant elements are contained in the group, but will not affect the interface 
		idx += 1


if metamaterials_exist == 1:

	s_dimtags1 = gmsh.model.getEntitiesInBoundingBox(x1 - e, y3 - e, z[len(z)-1]-depth_meta - e,    x1 + e, y7 + e, z[len(z)-1] + e, 2)	
	s_dimtags2 = gmsh.model.getEntitiesInBoundingBox(x3 - e, y1 - e, z[len(z)-1]-depth_meta - e,    x3 + e, y2 + e, z[len(z)-1] + e, 2)	
	s_dimtags3 = gmsh.model.getEntitiesInBoundingBox(x3 - e, y8 - e, z[len(z)-1]-depth_meta - e,    x3 + e, y9 + e, z[len(z)-1] + e, 2)	
	s_dimtags4 = gmsh.model.getEntitiesInBoundingBox(x8 - e, y3 - e, z[len(z)-1]-depth_meta - e,    x8 + e, y7 + e, z[len(z)-1] + e, 2)	
	s_dimtags = s_dimtags1 + s_dimtags2 + s_dimtags3 + s_dimtags4
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "soil-metamaterials_x+")	
	idx += 1

	s_dimtags1 = gmsh.model.getEntitiesInBoundingBox(x2 - e, y3 - e, z[len(z)-1]-depth_meta - e,    x2 + e, y7 + e, z[len(z)-1] + e, 2)	
	s_dimtags2 = gmsh.model.getEntitiesInBoundingBox(x7 - e, y1 - e, z[len(z)-1]-depth_meta - e,    x7 + e, y2 + e, z[len(z)-1] + e, 2)	
	s_dimtags3 = gmsh.model.getEntitiesInBoundingBox(x7 - e, y8 - e, z[len(z)-1]-depth_meta - e,    x7 + e, y9 + e, z[len(z)-1] + e, 2)	
	s_dimtags4 = gmsh.model.getEntitiesInBoundingBox(x9 - e, y3 - e, z[len(z)-1]-depth_meta - e,    x9 + e, y7 + e, z[len(z)-1] + e, 2)	
	s_dimtags = s_dimtags1 + s_dimtags2 + s_dimtags3 + s_dimtags4
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "soil-metamaterials_x-")	
	idx += 1

	s_dimtags1 = gmsh.model.getEntitiesInBoundingBox(x3 - e, y1 - e, z[len(z)-1]-depth_meta - e,    x7 + e, y1 + e, z[len(z)-1] + e, 2)	
	s_dimtags2 = gmsh.model.getEntitiesInBoundingBox(x1 - e, y3 - e, z[len(z)-1]-depth_meta - e,    x2 + e, y3 + e, z[len(z)-1] + e, 2)	
	s_dimtags3 = gmsh.model.getEntitiesInBoundingBox(x8 - e, y3 - e, z[len(z)-1]-depth_meta - e,    x9 + e, y3 + e, z[len(z)-1] + e, 2)	
	s_dimtags4 = gmsh.model.getEntitiesInBoundingBox(x3 - e, y8 - e, z[len(z)-1]-depth_meta - e,    x7 + e, y8 + e, z[len(z)-1] + e, 2)	
	s_dimtags = s_dimtags1 + s_dimtags2 + s_dimtags3 + s_dimtags4
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "soil-metamaterials_y+")	
	idx += 1

	s_dimtags1 = gmsh.model.getEntitiesInBoundingBox(x3 - e, y2 - e, z[len(z)-1]-depth_meta - e,    x7 + e, y2 + e, z[len(z)-1] + e, 2)	
	s_dimtags2 = gmsh.model.getEntitiesInBoundingBox(x1 - e, y7 - e, z[len(z)-1]-depth_meta - e,    x2 + e, y7 + e, z[len(z)-1] + e, 2)	
	s_dimtags3 = gmsh.model.getEntitiesInBoundingBox(x8 - e, y7 - e, z[len(z)-1]-depth_meta - e,    x9 + e, y7 + e, z[len(z)-1] + e, 2)	
	s_dimtags4 = gmsh.model.getEntitiesInBoundingBox(x3 - e, y9 - e, z[len(z)-1]-depth_meta - e,    x7 + e, y9 + e, z[len(z)-1] + e, 2)	
	s_dimtags = s_dimtags1 + s_dimtags2 + s_dimtags3 + s_dimtags4
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "soil-metamaterials_y-")	
	idx += 1

	s_dimtags1 = gmsh.model.getEntitiesInBoundingBox(x1 - e, y3 - e, z[len(z)-1]-depth_meta - e,    x2 + e, y7 + e, z[len(z)-1]-depth_meta + e, 2)	
	s_dimtags2 = gmsh.model.getEntitiesInBoundingBox(x8 - e, y3 - e, z[len(z)-1]-depth_meta - e,    x9 + e, y7 + e, z[len(z)-1]-depth_meta + e, 2)
	s_dimtags3 = gmsh.model.getEntitiesInBoundingBox(x3 - e, y1 - e, z[len(z)-1]-depth_meta - e,    x7 + e, y2 + e, z[len(z)-1]-depth_meta + e, 2)	
	s_dimtags4 = gmsh.model.getEntitiesInBoundingBox(x3 - e, y8 - e, z[len(z)-1]-depth_meta - e,    x7 + e, y9 + e, z[len(z)-1]-depth_meta + e, 2)	
	s_dimtags = s_dimtags1 + s_dimtags2 + s_dimtags3 + s_dimtags4
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "soil-metamaterials_z")	
	idx += 1

gmsh.model.geo.synchronize()


###########################################################################################################################################################################
####################################################################### Reactor building ##################################################################################
###########################################################################################################################################################################

if reactor_building == 1:

	# copy the geometry and the mesh from the circular soil
	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_out - e, y5-R_out - e, z6 - e,    x5+R_out + e, y5+R_out + e, z6 + e, 2)
	c = gmsh.model.geo.copy(s_dimtags)
	gmsh.model.geo.translate(c, 0, 0, offset_reactor_building)
	gmsh.model.geo.synchronize()

	# extrude foundation and walls
	count_extrude = 1

	if depth_meta > SL[7]:	#meta are embedded into the second soil layer

		for i in range (0,len(z_r)-2):	#extrusion from z_r1 to z_r7 , len(z_r)=9

			s_all_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_out - e, y5-R_out - e, z_r[i] - e,    x5+R_out + e, y5+R_out + e, z_r[i] + e, 2)	#select all the surfaces and start removing if needed

			remove_dimtags = []

			if z_r[i] == z_r4: #remove necessary s_dimtags at foundation level

				s_dimtags1 = gmsh.model.getEntitiesInBoundingBox(x5-R_out+t_out - e, y5-R_out+t_out - e, z_r[i] - e,    x5+R_out-t_out + e, y5+R_out-t_out + e, z_r[i] + e, 2)
				s_dimtags2 = gmsh.model.getEntitiesInBoundingBox(x5-R_in - e, y5-R_in - e, z_r[i] - e,    x5+R_in + e, y5+R_in + e, z_r[i] + e, 2)
				s_dimtags3 = gmsh.model.getEntitiesInBoundingBox(x5-R_in+t_in - e, y5-R_in+t_in - e, z_r[i] - e,    x5+R_in-t_in + e, y5+R_in-t_in + e, z_r[i] + e, 2)
				s_dimtags4 = gmsh.model.getEntitiesInBoundingBox(x5-R_vessel - e, y5-R_vessel - e, z_r[i] - e,    x5+R_vessel + e, y5+R_vessel + e, z_r[i] + e, 2)
				s_dimtags5 = gmsh.model.getEntitiesInBoundingBox(x5-R_vessel+t_vessel - e, y5-R_vessel+t_vessel - e, z_r[i] - e,    x5+R_vessel-t_vessel + e, y5+R_vessel-t_vessel + e, z_r[i] + e, 2)

				remove_dimtags1 = list(set(s_dimtags1)^set(s_dimtags2)) #remove common dimtags
				remove_dimtags2 = list(set(s_dimtags3)^set(s_dimtags4)) #remove common dimtags
				remove_dimtags3 = s_dimtags5

				remove_dimtags = remove_dimtags1 + remove_dimtags2 + remove_dimtags3

			if z_r[i] == z_r7:	#remove necessary s_dimtags at top vessel level

				s_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_vessel - e, y5-R_vessel - e, z_r[i] - e,    x5+R_vessel + e, y5+R_vessel + e, z_r[i] + e, 2)
				remove_dimtags = s_dimtags

			#extrude
			surface_to_extrude = list(set(s_all_dimtags)^set(remove_dimtags))
			numElements = int(math.ceil((z_r[i+1]-z_r[i])/z_lines_mesh_lv[7])) #take 7 (top layer) conservatively
			gmsh.model.geo.extrude(surface_to_extrude, 0, 0, (z_r[i+1]-z_r[i]), [numElements], recombine = True)
			print("\n" + 'Extrude reactor ' + str(count_extrude) + ' done !')
			count_extrude += 1

			gmsh.model.geo.synchronize()

	elif depth_meta < SL[7]:	#meta are embedded only in the first soil layer

		for i in range (0,len(z_r)-2):	#extrusion from z_r1 to z_r5 , len(z_r)=7

			s_all_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_out - e, y5-R_out - e, z_r[i] - e,    x5+R_out + e, y5+R_out + e, z_r[i] + e, 2)	#select all the surfaces and start removing if needed

			remove_dimtags = []

			if z_r[i] == z_r2: #remove necessary s_dimtags at foundation level

				s_dimtags1 = gmsh.model.getEntitiesInBoundingBox(x5-R_out+t_out - e, y5-R_out+t_out - e, z_r[i] - e,    x5+R_out-t_out + e, y5+R_out-t_out + e, z_r[i] + e, 2)
				s_dimtags2 = gmsh.model.getEntitiesInBoundingBox(x5-R_in - e, y5-R_in - e, z_r[i] - e,    x5+R_in + e, y5+R_in + e, z_r[i] + e, 2)
				s_dimtags3 = gmsh.model.getEntitiesInBoundingBox(x5-R_in+t_in - e, y5-R_in+t_in - e, z_r[i] - e,    x5+R_in-t_in + e, y5+R_in-t_in + e, z_r[i] + e, 2)
				s_dimtags4 = gmsh.model.getEntitiesInBoundingBox(x5-R_vessel - e, y5-R_vessel - e, z_r[i] - e,    x5+R_vessel + e, y5+R_vessel + e, z_r[i] + e, 2)
				s_dimtags5 = gmsh.model.getEntitiesInBoundingBox(x5-R_vessel+t_vessel - e, y5-R_vessel+t_vessel - e, z_r[i] - e,    x5+R_vessel-t_vessel + e, y5+R_vessel-t_vessel + e, z_r[i] + e, 2)

				remove_dimtags1 = list(set(s_dimtags1)^set(s_dimtags2)) #remove common dimtags
				remove_dimtags2 = list(set(s_dimtags3)^set(s_dimtags4)) #remove common dimtags
				remove_dimtags3 = s_dimtags5

				remove_dimtags = remove_dimtags1 + remove_dimtags2 + remove_dimtags3

			if z_r[i] == z_r5:	#remove necessary s_dimtags at top vessel level

				s_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_vessel - e, y5-R_vessel - e, z_r[i] - e,    x5+R_vessel + e, y5+R_vessel + e, z_r[i] + e, 2)
				remove_dimtags = s_dimtags

			#extrude
			surface_to_extrude = list(set(s_all_dimtags)^set(remove_dimtags))
			numElements = int(math.ceil((z_r[i+1]-z_r[i])/z_lines_mesh_lv[7])) #take 7 (top layer) conservatively
			gmsh.model.geo.extrude(surface_to_extrude, 0, 0, (z_r[i+1]-z_r[i]), [numElements], recombine = True)
			print("\n" + 'Extrude reactor ' + str(count_extrude) + ' done !')
			count_extrude += 1

			gmsh.model.geo.synchronize()



	##generate the Cylindrical Wall (CW)
	p_vessel_in = np.zeros([2,8], dtype=np.int64)
	r_vessel = [R_vessel, R_vessel-t_vessel, R_vessel-t_vessel-width_vessel, R_vessel-2*t_vessel-width_vessel]

	for i in range(0,2):
		for j in range(0,8):

			p_vessel_in[i,j] = gmsh.model.geo.addPoint(x5 + r_vessel[i+2]*math.cos(j*math.pi/4) , y5 + r_vessel[i+2]*math.sin(j*math.pi/4), z_r[len(z_r)-3], 0, -1) 

	print("\n" + "p_vessel_in = " + "\n" + str(p_vessel_in))
	gmsh.model.geo.synchronize()


 	#create a matrix with the points on top of the extruded wall
	p_on_vessel = np.zeros([1,8], dtype=np.int64)

	for j in range(0,8):

		p_dimtag = gmsh.model.getEntitiesInBoundingBox(x5 + r_vessel[1]*math.cos(j*math.pi/4) - e, y5 + r_vessel[1]*math.sin(j*math.pi/4) - e, z_r[len(z_r)-3] - e,    x5 + r_vessel[1]*math.cos(j*math.pi/4) + e, y5 + r_vessel[1]*math.sin(j*math.pi/4) + e, z_r[len(z_r)-3] + e, 0)
		p_on_vessel[0,j] = p_dimtag[0][1]

	print("\n" + "p_on_vessel = " + "\n" + str(p_on_vessel))


	#create radial lines for CW
	l_radial_vessel_in = np.zeros([2,8], dtype=np.int64)

	for j in range(0,8):
		l = gmsh.model.geo.addLine(p_on_vessel[0,j], p_vessel_in[0,j], -1)
		l_radial_vessel_in[0,j] = l
		gmsh.model.geo.mesh.setTransfiniteCurve(l, int(math.ceil(width_vessel/mesh_lv4)+1)  )	#ceil returns int, but in format of float, so put int in front

	for j in range(0,8):
		l = gmsh.model.geo.addLine(p_vessel_in[0,j], p_vessel_in[1,j], -1)
		l_radial_vessel_in[1,j] = l
		gmsh.model.geo.mesh.setTransfiniteCurve(l, int(math.ceil(t_vessel/mesh_lv4)+1)  )	#ceil returns int, but in format of float, so put int in front

	print("\n" + "l_radial_vessel_in = " + "\n" + str(l_radial_vessel_in))
	# gmsh.model.geo.synchronize()


	#create arc lines for CW
	l_arc_vessel = np.zeros([3,8], dtype=np.int64) #first one on the CW, next two inside
	p_dimtag = gmsh.model.getEntitiesInBoundingBox(x5 - e, y5 - e, z_r[len(z_r)-3] - e,    x5 + e, y5 + e, z_r[len(z_r)-3] + e, 0)
	p_centre_vessel = p_dimtag[0][1]

	xyz1 = gmsh.model.getValue(0, p[0,5], [])
	xyz2 = gmsh.model.getValue(0, p[0,6], [])

	for i in range(0,2): 

		for j in range(0,8):

			if j!=(8-1):
				c = gmsh.model.geo.addCircleArc(p_vessel_in[i,j], p_centre_vessel, p_vessel_in[i,j+1])
				l_arc_vessel[i+1,j] = c
				gmsh.model.geo.mesh.setTransfiniteCurve(c, int(math.ceil((xyz2[1]-xyz1[1])/y_lines_mesh_lv[5])+1)  ) 
			else:
				c = gmsh.model.geo.addCircleArc(p_vessel_in[i,j], p_centre_vessel, p_vessel_in[i,0])
				l_arc_vessel[i+1,j] = c
				gmsh.model.geo.mesh.setTransfiniteCurve(c, int(math.ceil((xyz2[1]-xyz1[1])/y_lines_mesh_lv[5])+1)  ) 

	print("\n" + "l_arc_vessel = " + "\n" + str(l_arc_vessel))
	gmsh.model.geo.synchronize()


	#fixed, is related to cos,sin
	mult = np.array([[1,0,0,1], 
					[2,1,1,2], 
					[3,3,2,2], 
					[4,4,3,3],
					[4,5,5,4],
					[5,6,6,5],
					[6,6,7,7],
					[7,7,8,8]], np.int32)

	for i in range(0,1):
		for j in range(0,8):
			l_dimtags = gmsh.model.getEntitiesInBoundingBox(x5+r_vessel[1]*math.cos(mult[j,0]*math.pi/4) - e, y5+r_vessel[1]*math.sin(mult[j,1]*math.pi/4) - e, z_r[len(z_r)-3] - e,    x5+r_vessel[1]*math.cos(mult[j,2]*math.pi/4) + e, y5+r_vessel[1]*math.sin(mult[j,3]*math.pi/4) + e, z_r[len(z_r)-3] + e, 1)
			remove_dimtags1 = gmsh.model.getEntitiesInBoundingBox(x5-r_vessel[1] - e, y5 - e, z_r[len(z_r)-3] - e,    x5+r_vessel[1] + e, y5 + e, z_r[len(z_r)-3] + e, 1) #horizontal
			remove_dimtags2 = gmsh.model.getEntitiesInBoundingBox(x5 - e, y5-r_vessel[1] - e, z_r[len(z_r)-3] - e,    x5 + e, y5+r_vessel[1] + e, z_r[len(z_r)-3] + e, 1) #vertical
			remove_dimtags3 = gmsh.model.getEntitiesInBoundingBox(x5-r_vessel[1]*math.cos(math.pi/4) - e, y5-r_vessel[1]*math.cos(math.pi/4) - e, z_r[len(z_r)-3] - e,    x5+r_vessel[1]*math.cos(math.pi/4) + e, y5+r_vessel[1]*math.cos(math.pi/4) + e, z_r[len(z_r)-3] + e, 1) #box for diagonal small lines
			remove_dimtags = remove_dimtags1 + remove_dimtags2 + remove_dimtags3

			elements_to_remove=[]
			for element in l_dimtags:
				if element in remove_dimtags:
					elements_to_remove.append(element)

			l_dimtags = list(set(l_dimtags)^set(elements_to_remove)) 
			# print(l_dimtags)

			l_arc_vessel[0,j] = l_dimtags[ 0 ][1]
			
	print("\n" + "l_arc_vessel = " + "\n" + str(l_arc_vessel))



	#create surfaces on CW
	s_arc_vessel_in = np.zeros([2,8], dtype=np.int64)

	for i in range(0,2):
		for j in range(0,8):

			if j!=(8-1):
				loops = gmsh.model.geo.addCurveLoops([l_arc_vessel[i,j],l_arc_vessel[i+1,j],l_radial_vessel_in[i,j],l_radial_vessel_in[i,j+1]])
				s_arc_vessel_in[i,j] = gmsh.model.geo.addSurfaceFilling(loops)
				gmsh.model.geo.mesh.setTransfiniteSurface(s_arc_vessel_in[i,j], "Left")
				gmsh.model.geo.mesh.setRecombine(2, s_arc_vessel_in[i,j])
			else:
				loops = gmsh.model.geo.addCurveLoops([l_arc_vessel[i,j],l_arc_vessel[i+1,j],l_radial_vessel_in[i,j],l_radial_vessel_in[i,0]])
				s_arc_vessel_in[i,j] = gmsh.model.geo.addSurfaceFilling(loops)
				gmsh.model.geo.mesh.setTransfiniteSurface(s_arc_vessel_in[i,j], "Left")
				gmsh.model.geo.mesh.setRecombine(2, s_arc_vessel_in[i,j])

	print("\n" + "s_arc_vessel_in = " + "\n" + str(s_arc_vessel_in))
	gmsh.model.geo.synchronize()



	#extrude CW
	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_vessel - e, y5-R_vessel - e, z_r[len(z_r)-3] - e,    x5+R_vessel + e, y5+R_vessel + e, z_r[len(z_r)-3] + e, 2)

	#extrude
	# surface_to_extrude = list(set(s_all_dimtags)^set(remove_dimtags))
	numElements = int(math.ceil(t_vessel/mesh_lv4)) 
	gmsh.model.geo.extrude(s_dimtags, 0, 0, t_vessel, [numElements], recombine = True)

	print("\n" + 'Extrude water pool base.. done !')
	gmsh.model.geo.synchronize()



	s_all_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_vessel - e, y5-R_vessel - e, z_r[len(z_r)-3]+t_vessel - e,    x5+R_vessel + e, y5+R_vessel + e, z_r[len(z_r)-3]+t_vessel + e, 2)
	remove_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-r_vessel[1] - e, y5-r_vessel[1] - e, z_r[len(z_r)-3]+t_vessel - e,    x5+r_vessel[1] + e, y5+r_vessel[1] + e, z_r[len(z_r)-3]+t_vessel + e, 2)
	s_dimtags = list(set(s_all_dimtags)^set(remove_dimtags))
	add_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-r_vessel[2] - e, y5-r_vessel[2] - e, z_r[len(z_r)-3]+t_vessel - e,    x5+r_vessel[2] + e, y5+r_vessel[2] + e, z_r[len(z_r)-3]+t_vessel + e, 2)
	s_dimtags = s_dimtags + add_dimtags
	numElements = int(math.ceil(h_water_vessel/mesh_lv4)) 
	gmsh.model.geo.extrude(s_dimtags, 0, 0, h_water_vessel, [numElements], recombine = True)

	print("\n" + 'Extrude water pool walls.. done !')
	gmsh.model.geo.synchronize()



	## generate the domes
	theta_deg = 45
	theta_rad = theta_deg * math.pi/180
	# print(theta_rad)

	# create a matrix with the points on top of the extruded 2 walls
	p_on_walls = np.zeros([4,8], dtype=np.int64)

	for i in range(1,5): #targeting the desired positions of array R

		for j in range(0,8):

			p_dimtag = gmsh.model.getEntitiesInBoundingBox(x5 + R[i]*math.cos(j*math.pi/4) - e, y5 + R[i]*math.sin(j*math.pi/4) - e, z_r[len(z_r)-2] - e,    x5 + R[i]*math.cos(j*math.pi/4) + e, y5 + R[i]*math.sin(j*math.pi/4) + e, z_r[len(z_r)-2] + e, 0)
			p_on_walls[i-1,j] = p_dimtag[0][1]

	print("\n" + "p_on_walls = " + "\n" + str(p_on_walls))


	# create points on the domes
	p_on_domes = np.zeros([4,9], dtype=np.int64) #9 is for the vertical points in the centre of dome

	k=1
	for i in range(1,5): #points 1,3,5,7, targeting the desired positions of array R

		p_on_domes[i-1,k] = gmsh.model.geo.addPoint(x5 + R[i]*math.cos(theta_rad) * math.cos(math.pi/4), y5 + R[i]*math.cos(theta_rad) * math.cos(math.pi/4), z_r[len(z_r)-2] + R[i]*math.sin(theta_rad), 0, -1) 
		k += 2
		p_on_domes[i-1,k] = gmsh.model.geo.addPoint(x5 - R[i]*math.cos(theta_rad) * math.cos(math.pi/4), y5 + R[i]*math.cos(theta_rad) * math.cos(math.pi/4), z_r[len(z_r)-2] + R[i]*math.sin(theta_rad), 0, -1) 
		k += 2
		p_on_domes[i-1,k] = gmsh.model.geo.addPoint(x5 - R[i]*math.cos(theta_rad) * math.cos(math.pi/4), y5 - R[i]*math.cos(theta_rad) * math.cos(math.pi/4), z_r[len(z_r)-2] + R[i]*math.sin(theta_rad), 0, -1) 
		k += 2
		p_on_domes[i-1,k] = gmsh.model.geo.addPoint(x5 + R[i]*math.cos(theta_rad) * math.cos(math.pi/4), y5 - R[i]*math.cos(theta_rad) * math.cos(math.pi/4), z_r[len(z_r)-2] + R[i]*math.sin(theta_rad), 0, -1) 
		k = 1

	gmsh.model.geo.synchronize()

	k=0
	for i in range(1,5): #points 0,2,4,6, targeting the desired positions of array R

		xyz1 = gmsh.model.getValue(0, p_on_domes[i-1,1], []) #returns x,y,z coord
		xyz2 = gmsh.model.getValue(0, p_on_domes[i-1,3], [])
		chord = math.sqrt( (xyz2[0]-xyz1[0])**2 + (xyz2[1]-xyz1[1])**2 + (xyz2[2]-xyz1[2])**2 )
		d1 = chord/2 #A1A3
		d2 = math.sqrt( R[i]**2 - d1**2 ) #OA3
		d3 = R[i]*math.sin(theta_rad) #OO1
		cos_angle = d3/d2 #OO1/OA3
		angle = math.acos(cos_angle) #theta1
		abs_projection_hor = R[i]*math.sin(angle)
		abs_projection_ver = R[i]*math.cos(angle)

		for j in range(0,4): #create the 4 points
			p_on_domes[i-1,k] = gmsh.model.geo.addPoint(x5 + abs_projection_hor*math.cos(j*math.pi/2), y5 + abs_projection_hor*math.sin(j*math.pi/2), z_r[len(z_r)-2] + abs_projection_ver, 0, -1)
			k += 2

		k=0

	for i in range (1,5): #create 4 vertical points at the centre of the dome for the cross

		p_on_domes[i-1,8] = gmsh.model.geo.addPoint(x5, y5, z_r[len(z_r)-2] + R[i], 0, -1)

	print("\n" + "p_on_domes = " + "\n" + str(p_on_domes))

	gmsh.model.geo.synchronize()



	# create arcs on domes
	p_dimtag = gmsh.model.getEntitiesInBoundingBox(x5 - e, y5 - e, z_r[len(z_r)-2] - e,    x5 + e, y5 + e, z_r[len(z_r)-2] + e, 0)
	p_centre_dome = p_dimtag[0][1]

	l_arcs_on_domes = np.zeros([4,8], dtype=np.int64)
	l_cup_on_domes = np.zeros([4,8], dtype=np.int64)
	l_cross_on_domes = np.zeros([4,8], dtype=np.int64)

	xyz1 = gmsh.model.getValue(0, p_on_walls[0,0], [])	#for transfinite
	xyz2 = gmsh.model.getValue(0, p_on_domes[0,0], [])
	length = math.sqrt( (xyz2[0]-xyz1[0])**2 + (xyz2[1]-xyz1[1])**2 + (xyz2[2]-xyz1[2])**2)

	for i in range(0,len(p_on_walls)): #(0,4)

		for j in range(0,8):

			c = gmsh.model.geo.addCircleArc(p_on_walls[i,j], p_centre_dome, p_on_domes[i,j])
			l_arcs_on_domes[i,j] = c
			gmsh.model.geo.mesh.setTransfiniteCurve(c, int(math.ceil(length/mesh_lv4)+1)  )

	print("\n" + "l_arcs_on_domes = " + "\n" + str(l_arcs_on_domes))


	#create radial lines on domes
	l_radial_on_domes = np.zeros([4,9], dtype=np.int64)

	for i in range(0,4):
		for j in range(0,9):

			if i%2==0:
				l = gmsh.model.geo.addLine(p_on_domes[i,j], p_on_domes[i+1,j], -1)
				l_radial_on_domes[i,j] = l
				gmsh.model.geo.mesh.setTransfiniteCurve(l, int(math.ceil(t_out/mesh_lv4)+1)  )	#ceil returns int, but in format of float, so put int in front

	print("\n" + "l_radial_on_domes = " + "\n" + str(l_radial_on_domes))


	xyz1 = gmsh.model.getValue(0, p[0,5], [])	#for transfinite
	xyz2 = gmsh.model.getValue(0, p[0,6], [])
	length = math.sqrt( (xyz2[0]-xyz1[0])**2 + (xyz2[1]-xyz1[1])**2 + (xyz2[2]-xyz1[2])**2)

	for i in range(0,len(p_on_walls)): #(0,4)

		for j in range(0,8):

			if j!=(8-1):
				c = gmsh.model.geo.addCircleArc(p_on_domes[i,j], p_centre_dome, p_on_domes[i,j+1])
				l_cup_on_domes[i,j] = c
			else:
				c = gmsh.model.geo.addCircleArc(p_on_domes[i,j], p_centre_dome, p_on_domes[i,0])
				l_cup_on_domes[i,j] = c

			gmsh.model.geo.mesh.setTransfiniteCurve(c, int(math.ceil(length/y_lines_mesh_lv[5])+1)  ) 

	print("\n" + "l_cup_on_domes = " + "\n" + str(l_cup_on_domes))


	for i in range(0,len(p_on_walls)): #(0,4)

		for j in range(0,8):

			if j%2==0: # 4 lines inside (cross)

				c = gmsh.model.geo.addCircleArc(p_on_domes[i,j], p_centre_dome, p_on_domes[i,8])
				l_cross_on_domes[i,j] = c
				gmsh.model.geo.mesh.setTransfiniteCurve(c, int(math.ceil(length/y_lines_mesh_lv[5])+1)  )

	print("\n" + "l_cross_on_domes = " + "\n" + str(l_cross_on_domes))
	gmsh.model.geo.synchronize()



	# l_arc_on_walls are needed to create the surfaces on domes
	l_arc_on_walls = np.zeros([4,8], dtype=np.int64)

	#fixed, is related to cos,sin
	mult = np.array([[1,0,0,1], 
					[2,1,1,2], 
					[3,3,2,2], 
					[4,4,3,3],
					[4,5,5,4],
					[5,6,6,5],
					[6,6,7,7],
					[7,7,8,8]], np.int32)

	for i in range(0,4):
		for j in range(0,8):
			l_dimtags = gmsh.model.getEntitiesInBoundingBox(x5+R[i+1]*math.cos(mult[j,0]*math.pi/4) - e, y5+R[i+1]*math.sin(mult[j,1]*math.pi/4) - e, z_r[len(z_r)-2] - e,    x5+R[i+1]*math.cos(mult[j,2]*math.pi/4) + e, y5+R[i+1]*math.sin(mult[j,3]*math.pi/4) + e, z_r[len(z_r)-2] + e, 1)
			remove_dimtags1 = gmsh.model.getEntitiesInBoundingBox(x5-R[i+1] - e, y5 - e, z_r[len(z_r)-2] - e,    x5+R[i+1] + e, y5 + e, z_r[len(z_r)-2] + e, 1) #horizontal
			remove_dimtags2 = gmsh.model.getEntitiesInBoundingBox(x5 - e, y5-R[i+1] - e, z_r[len(z_r)-2] - e,    x5 + e, y5+R[i+1] + e, z_r[len(z_r)-2] + e, 1) #vertical
			remove_dimtags3 = gmsh.model.getEntitiesInBoundingBox(x5-R[i+1]*math.cos(math.pi/4) - e, y5-R[i+1]*math.cos(math.pi/4) - e, z_r[len(z_r)-2] - e,    x5+R[i+1]*math.cos(math.pi/4) + e, y5+R[i+1]*math.cos(math.pi/4) + e, z_r[len(z_r)-2] + e, 1) #box for diagonal small lines
			remove_dimtags = remove_dimtags1 + remove_dimtags2 + remove_dimtags3

			elements_to_remove=[]
			for element in l_dimtags:
				if element in remove_dimtags:
					elements_to_remove.append(element)

			l_dimtags = list(set(l_dimtags)^set(elements_to_remove)) 
			# print(l_dimtags)

			l_arc_on_walls[i,j] = l_dimtags[ 0 ][1]
			
	print("\n" + "l_arc_on_walls = " + "\n" + str(l_arc_on_walls))



	l_radial_on_walls = np.zeros([4,8], dtype=np.int64)

	for i in range(0,4):

		for j in range(0,8):

			if i%2==0:
				l_dimtags = gmsh.model.getEntitiesInBoundingBox(x5 + ((R[i+1]+R[i+2])/2.0)*math.cos(j*math.pi/4) - t_out, y5 + ((R[i+1]+R[i+2])/2.0)*math.sin(j*math.pi/4) - t_out, z_r[len(z_r)-2] - e,    x5 + ((R[i+1]+R[i+2])/2.0)*math.cos(j*math.pi/4) + t_out, y5 + ((R[i+1]+R[i+2])/2.0)*math.sin(j*math.pi/4) + t_out, z_r[len(z_r)-2] + e, 1)
				l_radial_on_walls[i,j] = l_dimtags[0][1]

	print("\n" + "l_radial_on_walls = " + "\n" + str(l_radial_on_walls))



	# s_on_walls are needed to create the volumes on domes
	s_on_walls = np.zeros([4,8], dtype=np.int64)

	#fixed, is related to cos,sin
	mult = np.array([[1,0,0,1], 
					[2,1,1,2], 
					[3,3,2,2], 
					[4,4,3,3],
					[4,5,5,4],
					[5,6,6,5],
					[6,6,7,7],
					[7,7,8,8]], np.int32)

	oneortwo = np.array([[2,2,1,1], 
					[2,2,1,1], 
					[1,2,1,1], 
					[1,1,2,1],
					[1,1,2,1],
					[1,1,2,2],
					[1,1,1,2],
					[2,1,1,2]], np.int32)

	for i in range(0,4):

		if i%2==0:
			for j in range(0,8):
				s_dimtags = gmsh.model.getEntitiesInBoundingBox(x5+R[i+oneortwo[j,0]]*math.cos(mult[j,0]*math.pi/4) - e, y5+R[i+oneortwo[j,1]]*math.sin(mult[j,1]*math.pi/4) - e, z_r[len(z_r)-2] - e,    x5+R[i+oneortwo[j,2]]*math.cos(mult[j,2]*math.pi/4) + e, y5+R[i+oneortwo[j,3]]*math.sin(mult[j,3]*math.pi/4) + e, z_r[len(z_r)-2] + e, 2)
				# print(s_dimtags)
				s_on_walls[i,j] = s_dimtags[ 0 ][1]
			
	print("\n" + "s_on_walls = " + "\n" + str(s_on_walls))



	#create surfaces on domes
	s_arc_on_domes = np.zeros([4,8], dtype=np.int64)

	for i in range(0,4):
		for j in range(0,8):

			if j!=(8-1):
				loops = gmsh.model.geo.addCurveLoops([l_arc_on_walls[i,j],l_cup_on_domes[i,j],l_arcs_on_domes[i,j],l_arcs_on_domes[i,j+1]])
				# print([l_arc_on_walls[i,j],l_cup_on_domes[i,j],l_arcs_on_domes[i,j],l_arcs_on_domes[i,j+1]])
				s_arc_on_domes[i,j] = gmsh.model.geo.addSurfaceFilling(loops)
				gmsh.model.geo.mesh.setTransfiniteSurface(s_arc_on_domes[i,j], "Left")
				gmsh.model.geo.mesh.setRecombine(2, s_arc_on_domes[i,j])
			else:
				loops = gmsh.model.geo.addCurveLoops([l_arc_on_walls[i,j],l_cup_on_domes[i,j],l_arcs_on_domes[i,j],l_arcs_on_domes[i,0]])
				s_arc_on_domes[i,j] = gmsh.model.geo.addSurfaceFilling(loops)
				gmsh.model.geo.mesh.setTransfiniteSurface(s_arc_on_domes[i,j], "Left")
				gmsh.model.geo.mesh.setRecombine(2, s_arc_on_domes[i,j])

	print("\n" + "s_arc_on_domes = " + "\n" + str(s_arc_on_domes))
	gmsh.model.geo.synchronize()


	s_radial_on_domes = np.zeros([4,8], dtype=np.int64)

	for i in range(0,4):
		for j in range(0,8):

			if i%2==0:
				loops = gmsh.model.geo.addCurveLoops([l_radial_on_walls[i,j],l_radial_on_domes[i,j],l_arcs_on_domes[i,j],l_arcs_on_domes[i+1,j]])
				s = gmsh.model.geo.addSurfaceFilling(loops)
				s_radial_on_domes[i,j] = s
				gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
				gmsh.model.geo.mesh.setRecombine(2, s)

	print("\n" + "s_radial_on_domes = " + "\n" + str(s_radial_on_domes))
	gmsh.model.geo.synchronize()


	s_cup_on_domes = np.zeros([4,8], dtype=np.int64)

	for i in range(0,4):
		for j in range(0,8):

			if i%2==0 and j!=(8-1):
				loops = gmsh.model.geo.addCurveLoops([l_radial_on_domes[i,j],l_radial_on_domes[i,j+1],l_cup_on_domes[i,j],l_cup_on_domes[i+1,j]])
				s = gmsh.model.geo.addSurfaceFilling(loops)
				s_cup_on_domes[i,j] = s
				gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
				gmsh.model.geo.mesh.setRecombine(2, s)

			elif i%2==0 and j==(8-1):
				loops = gmsh.model.geo.addCurveLoops([l_radial_on_domes[i,j],l_radial_on_domes[i,0],l_cup_on_domes[i,j],l_cup_on_domes[i+1,j]])
				s = gmsh.model.geo.addSurfaceFilling(loops)
				s_cup_on_domes[i,j] = s
				gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
				gmsh.model.geo.mesh.setRecombine(2, s)

	print("\n" + "s_cup_on_domes = " + "\n" + str(s_cup_on_domes))
	gmsh.model.geo.synchronize()


	s_cross_on_domes = np.zeros([4,8], dtype=np.int64)

	for i in range(0,4): 
		for j in range(0,8):

			if i%2==0 and j%2==0: # 
				loop = gmsh.model.geo.addCurveLoops([l_cross_on_domes[i,j],l_cross_on_domes[i+1,j],l_radial_on_domes[i,j],l_radial_on_domes[i,8]])
				# print(loops) [int_number]
				s = gmsh.model.geo.addPlaneSurface(loop)
				s_cross_on_domes[i,j] = s
				gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
				gmsh.model.geo.mesh.setRecombine(2, s)

	print("\n" + "s_cross_on_domes = " + "\n" + str(s_cross_on_domes))
	gmsh.model.geo.synchronize()




	s_arc_top_on_domes = np.zeros([4,8], dtype=np.int64)

	for i in range(0,4): 
		for j in range(0,8):

			if j%2==0 and j!=(8-2): # 
				loop = gmsh.model.geo.addCurveLoops([l_cross_on_domes[i,j],l_cross_on_domes[i,j+2],l_cup_on_domes[i,j],l_cup_on_domes[i,j+1]])
				# print(l_cross_on_domes[i,j],l_cross_on_domes[i,j+2],l_cup_on_domes[i,j],l_cup_on_domes[i,j+1])
				# print(loops) [int_number]
				s = gmsh.model.geo.addSurfaceFilling(loop)
				s_arc_top_on_domes[i,j] = s
				gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
				gmsh.model.geo.mesh.setRecombine(2, s)

			elif j%2==0 and j==(8-2): # 
				loop = gmsh.model.geo.addCurveLoops([l_cross_on_domes[i,j],l_cross_on_domes[i,0],l_cup_on_domes[i,j],l_cup_on_domes[i,j+1]])
				# print(l_cross_on_domes[i,j],l_cross_on_domes[i,j+2],l_cup_on_domes[i,j],l_cup_on_domes[i,j+1])
				# print(loops) [int_number]
				s = gmsh.model.geo.addSurfaceFilling(loop)
				s_arc_top_on_domes[i,j] = s
				gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
				gmsh.model.geo.mesh.setRecombine(2, s)

	print("\n" + "s_arc_top_on_domes = " + "\n" + str(s_arc_top_on_domes))
	gmsh.model.geo.synchronize()


	#create volumes on domes
	v_sides_on_domes = np.zeros([4,8], dtype=np.int64)

	for i in range(0,4):
		for j in range(0,8):

			if i%2==0 and j!=(8-1):
				loop = gmsh.model.geo.addSurfaceLoop([s_on_walls[i,j], s_radial_on_domes[i,j], s_radial_on_domes[i,j+1], s_arc_on_domes[i,j], s_arc_on_domes[i+1,j], s_cup_on_domes[i,j] ])
				# print(loop)
				v = gmsh.model.geo.addVolume([loop])
				v_sides_on_domes[i,j] = v
				gmsh.model.geo.mesh.setTransfiniteVolume(v)

			elif i%2==0 and j==(8-1):
				loop = gmsh.model.geo.addSurfaceLoop([s_on_walls[i,j], s_radial_on_domes[i,j], s_radial_on_domes[i,0], s_arc_on_domes[i,j], s_arc_on_domes[i+1,j], s_cup_on_domes[i,j] ])
				# print(loop)
				v = gmsh.model.geo.addVolume([loop])
				v_sides_on_domes[i,j] = v
				gmsh.model.geo.mesh.setTransfiniteVolume(v)

	print("\n" + "v_sides_on_domes = " + "\n" + str(v_sides_on_domes))
	gmsh.model.geo.synchronize()



	v_top_on_domes = np.zeros([4,8], dtype=np.int64)

	for i in range(0,4):
		for j in range(0,8):

			if i%2==0 and j%2==0 and j!=(8-2):
				loop = gmsh.model.geo.addSurfaceLoop([s_cross_on_domes[i,j], s_cross_on_domes[i,j+2], s_cup_on_domes[i,j], s_cup_on_domes[i,j+1], s_arc_top_on_domes[i,j], s_arc_top_on_domes[i+1,j] ])
				# print(loop)
				v = gmsh.model.geo.addVolume([loop])
				v_top_on_domes[i,j] = v
				gmsh.model.geo.mesh.setTransfiniteVolume(v)

			elif i%2==0 and j%2==0 and j==(8-2):
				loop = gmsh.model.geo.addSurfaceLoop([s_cross_on_domes[i,j], s_cross_on_domes[i,0], s_cup_on_domes[i,j], s_cup_on_domes[i,j+1], s_arc_top_on_domes[i,j], s_arc_top_on_domes[i+1,j] ])
				# print(loop)
				v = gmsh.model.geo.addVolume([loop])
				v_top_on_domes[i,j] = v
				gmsh.model.geo.mesh.setTransfiniteVolume(v)

	print("\n" + "v_top_on_domes = " + "\n" + str(v_top_on_domes))
	gmsh.model.geo.synchronize()



	## Physical groups

	if depth_meta < SL[7]:
		v_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_out - e, y5-R_out - e, z_r1 - e,    x5+R_out + e, y5+R_out + e,  z_r2+ e, 3)
		gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(3, idx, "reactor_building_foundation")
		idx += 1

		v_dimtags_all = gmsh.model.getEntitiesInBoundingBox(x5-R_out - e, y5-R_out - e, z_r2 - e,    x5+R_out + e, y5+R_out + e,  z_r7+ e, 3)
		remove_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_in - e, y5-R_in - e, z_r2 - e,    x5+R_in + e, y5+R_in + e,  z_r6+R_in+ e, 3)
		v_dimtags = list(set(v_dimtags_all)^set(remove_dimtags))
		gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(3, idx, "reactor_building_outer_shell")
		idx += 1	

		v_dimtags_all = gmsh.model.getEntitiesInBoundingBox(x5-R_in - e, y5-R_in - e, z_r2 - e,    x5+R_in + e, y5+R_in + e,  z_r6+R_in+ e, 3)
		remove_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_vessel - e, y5-R_vessel - e, z_r2 - e,    x5+R_vessel + e, y5+R_vessel + e,  z_r6+R_in-t_in- e, 3)
		v_dimtags = list(set(v_dimtags_all)^set(remove_dimtags))
		gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(3, idx, "reactor_building_inner_shell")
		idx += 1

		v_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_vessel - e, y5-R_vessel - e, z_r2 - e,    x5+R_vessel + e, y5+R_vessel + e,  z_r5+t_vessel + e, 3)
		gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(3, idx, "reactor_building_vessel")
		idx += 1

		v_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_vessel - e, y5-R_vessel - e, z_r5+t_vessel - e,    x5+R_vessel + e, y5+R_vessel + e,  z_r5+t_vessel+h_water_vessel+ e, 3)
		gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(3, idx, "reactor_building_water_vessel")
		idx += 1

		s_dimtags_all = gmsh.model.getEntitiesInBoundingBox(x5-R_out - e, y5-R_out - e, z_r1 - e,    x5+R_out + e, y5+R_out + e,  z_r4+ e, 2)
		remove_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_out + e, y5-R_out + e, z_r1 - e,    x5+R_out - e, y5+R_out - e,  z_r4+ e, 2) #doesnt remove all but its okay
		s_dimtags = list(set(s_dimtags_all)^set(remove_dimtags))
		gmsh.model.addPhysicalGroup(2, [s_dimtags[i][1] for i in range(len(s_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(2, idx, "reactor_building-soil_radial")
		idx += 1

	else:
		v_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_out - e, y5-R_out - e, z_r1 - e,    x5+R_out + e, y5+R_out + e,  z_r4+ e, 3)
		gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(3, idx, "reactor_building_foundation")
		idx += 1	

		v_dimtags_all = gmsh.model.getEntitiesInBoundingBox(x5-R_out - e, y5-R_out - e, z_r4 - e,    x5+R_out + e, y5+R_out + e,  z_r9+ e, 3)
		remove_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_in - e, y5-R_in - e, z_r4 - e,    x5+R_in + e, y5+R_in + e,  z_r8+R_in+ e, 3)
		v_dimtags = list(set(v_dimtags_all)^set(remove_dimtags))
		gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(3, idx, "reactor_building_outer_shell")
		idx += 1	

		v_dimtags_all = gmsh.model.getEntitiesInBoundingBox(x5-R_in - e, y5-R_in - e, z_r4 - e,    x5+R_in + e, y5+R_in + e,  z_r8+R_in+ e, 3)
		remove_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_vessel - e, y5-R_vessel - e, z_r4 - e,    x5+R_vessel + e, y5+R_vessel + e,  z_r8+R_in-t_in- e, 3)
		v_dimtags = list(set(v_dimtags_all)^set(remove_dimtags))
		gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(3, idx, "reactor_building_inner_shell")
		idx += 1

		v_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_vessel - e, y5-R_vessel - e, z_r4 - e,    x5+R_vessel + e, y5+R_vessel + e,  z_r7+t_vessel + e, 3)
		gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(3, idx, "reactor_building_vessel")
		idx += 1

		v_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_vessel - e, y5-R_vessel - e, z_r7+t_vessel - e,    x5+R_vessel + e, y5+R_vessel + e,  z_r7+t_vessel+h_water_vessel+ e, 3)
		gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(3, idx, "reactor_building_water_vessel")
		idx += 1

		s_dimtags_all = gmsh.model.getEntitiesInBoundingBox(x5-R_out - e, y5-R_out - e, z_r1 - e,    x5+R_out + e, y5+R_out + e,  z_r6+ e, 2)
		remove_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_out + e, y5-R_out + e, z_r1 - e,    x5+R_out - e, y5+R_out - e,  z_r6+ e, 2)
		s_dimtags = list(set(s_dimtags_all)^set(remove_dimtags))
		gmsh.model.addPhysicalGroup(2, [s_dimtags[i][1] for i in range(len(s_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(2, idx, "reactor_building-soil_radial")
		idx += 1


	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R_out - e, y5-R_out - e, z_r1 - e,    x5+R_out + e, y5+R_out + e,  z_r1+ e, 2)
	gmsh.model.addPhysicalGroup(2, [s_dimtags[i][1] for i in range(len(s_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
	gmsh.model.setPhysicalName(2, idx, "reactor_building-soil_z")
	idx += 1


	###########################################################################################################################################################################
	## Reactor Vessel (RV)
	width_reactor_vessel = 10
	t_reactor_vessel = 2
	offset_reactor_vessel = 110

	## y coordinates of the reactor vessel
	y_rv0 = y5-width_reactor_vessel/2.
	y_rv1 = y_rv0 + t_reactor_vessel/4.
	y_rv2 = y_rv0 + t_reactor_vessel
	y_rv5 = y5+width_reactor_vessel/2.
	y_rv4 = y_rv5 - t_reactor_vessel/4.
	y_rv3 = y_rv5 - t_reactor_vessel

	y_rv = [y_rv0, y_rv1, y_rv2, y_rv3, y_rv4, y_rv5]
	print("\n" + "y_coord_reactor_vessel : " + str(y_rv))

	## x coordinates of the reactor vessel
	x_rv0 = x5-width_reactor_vessel/2.
	x_rv1 = x_rv0 + t_reactor_vessel/4.
	x_rv2 = x_rv0 + t_reactor_vessel
	x_rv5 = x5+width_reactor_vessel/2.
	x_rv4 = x_rv5 - t_reactor_vessel/4.
	x_rv3 = x_rv5 - t_reactor_vessel

	x_rv = [x_rv0, x_rv1, x_rv2, x_rv3, x_rv4, x_rv5]
	print("\n" + "x_coord_reactor_vessel : " + str(x_rv))

	## z coordinates of the reactor vessel
	z_rv0 = z6 + SL[6] + offset_reactor_vessel #SL[6] has the same length with the foundation
	z_rv1 = z_rv0 + h_reactor_vessel_base
	z_rv2 = z_rv1 + h_reactor_vessel_walls

	z_rv = [z_rv0, z_rv1, z_rv2]
	print("\n" + "z_coord_reactor_vessel : " + str(z_rv))

	## create points and put their tags in p_rv matrix
	p_rv = np.zeros([len(x_rv),len(y_rv)], dtype=np.int64)
	#print(p,type(p))

	for i in range(0,len(x_rv)):
		for j in range(0,len(y_rv)):
			p_rv[i,j] = gmsh.model.geo.addPoint(x_rv[i], y_rv[j], z_rv0, 0, -1) #returns the tag of the point
			
	print("\n" + "p_rv_tags = " + "\n" + str(p_rv))
	gmsh.model.geo.synchronize()


	## create lines in y direction
	for i in range(0,len(x_rv)):

		for j in range(0,len(y_rv)-1):

			xyz1 = gmsh.model.getValue(0, p_rv[i,j], []) #returns x,y,z coord
			xyz2 = gmsh.model.getValue(0, p_rv[i,j+1], [])

			# if (j != len(y)-1): #and not(i==5 and (j==4 or j==5)): #and not to exclude the circular part in the middle
			l = gmsh.model.geo.addLine(p_rv[i,j], p_rv[i,j+1], -1)
			#transfinite
			gmsh.model.geo.mesh.setTransfiniteCurve(l, int((xyz2[1]-xyz1[1])/0.5)+1)	#ceil returns int, but in format of float, so put int in front

	## create lines in x direction
	for j in range(0,len(y_rv)):

		for i in range(0,len(x_rv)-1):

			xyz1 = gmsh.model.getValue(0, p_rv[i,j], []) #returns x,y,z coord
			xyz2 = gmsh.model.getValue(0, p_rv[i+1,j], [])

			# if (i != len(x)-1): # and not(j==5 and (i==4 or i==5)): #and not to exclude the circular part in the middle
			l = gmsh.model.geo.addLine(p_rv[i,j], p_rv[i+1,j], -1)
			#transfinite
			gmsh.model.geo.mesh.setTransfiniteCurve(l, int((xyz2[0]-xyz1[0])/0.5)+1)

	gmsh.model.geo.synchronize()

	## create surfaces
	for j in range(0,len(y_rv)-1):

		for i in range(0,len(x_rv)-1):

			xyz1 = gmsh.model.getValue(0, p_rv[i,j], [])
			xyz2 = gmsh.model.getValue(0, p_rv[i+1,j+1], [])
			l_tags = gmsh.model.getEntitiesInBoundingBox(xyz1[0] - e, xyz1[1] - e, xyz1[2] - e,    xyz2[0] + e, xyz2[1] + e, xyz2[2] + e, 1)
			#print(l_tags)

			if len(l_tags) == 4 and not(i==2 and j==2):  #if 4 lines are in l_tags, then create the surface, not to leave the hole in the middle
				loop = gmsh.model.geo.addCurveLoop([ l_tags[0][1], l_tags[3][1], -l_tags[1][1], -l_tags[2][1] ], -1)
				s = gmsh.model.geo.addPlaneSurface([loop], -1)
				gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
				gmsh.model.geo.mesh.setRecombine(2, s)

	gmsh.model.geo.synchronize()


	#extrude Reactor Vessel (RV)
	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x_rv0 - e, y_rv0 - e, z_rv0 - e,    x_rv5 + e, y_rv5 + e, z_rv0 + e, 2)

	numElements = int(math.ceil(h_reactor_vessel_base/mesh_lv1)) 
	gmsh.model.geo.extrude(s_dimtags, 0, 0, h_reactor_vessel_base, [numElements], recombine = True)

	print("\n" + 'Extrude reactor vessel base.. done !')
	gmsh.model.geo.synchronize()


	s_all_dimtags = gmsh.model.getEntitiesInBoundingBox(x_rv0 - e, y_rv0 - e, z_rv1 - e,    x_rv5 + e, y_rv5 + e, z_rv1 + e, 2)
	remove_dimtags = gmsh.model.getEntitiesInBoundingBox(x_rv1 - e, y_rv1 - e, z_rv1 - e,    x_rv4 + e, y_rv4 + e, z_rv1 + e, 2)
	s_dimtags = list(set(s_all_dimtags)^set(remove_dimtags))

	numElements = int(math.ceil(h_reactor_vessel_walls/mesh_lv1)) 
	gmsh.model.geo.extrude(s_dimtags, 0, 0, h_reactor_vessel_walls, [numElements], recombine = True)

	print("\n" + 'Extrude reactor vessel walls.. done !')
	gmsh.model.geo.synchronize()


	# Physical groups
	v_dimtags = gmsh.model.getEntitiesInBoundingBox(x_rv0 - e, y_rv0 - e, z_rv0 - e,    x_rv5 + e, y_rv5 + e, z_rv1 + e, 3)
	gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
	gmsh.model.setPhysicalName(3, idx, "reactor_vessel_base")
	idx += 1

	v_dimtags = gmsh.model.getEntitiesInBoundingBox(x_rv0 - e, y_rv0 - e, z_rv1 - e,    x_rv5 + e, y_rv5 + e, z_rv2 + e, 3)
	gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
	gmsh.model.setPhysicalName(3, idx, "reactor_vessel_walls")
	idx += 1



###########################################################################################################################################################################
###################################################################### Auxiliary building #################################################################################
###########################################################################################################################################################################

if auxiliary_building == 1:

	# copy the geometry and the mesh from the soil
	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z6 - e,    x7 + e, y7 + e, z6 + e, 2)
	s_dimtags_remove = gmsh.model.getEntitiesInBoundingBox(x5-R_out-D_separation - e, y5-R_out-D_separation - e, z6 - e,    x5+R_out+D_separation + e, y5+R_out+D_separation + e, z6 + e, 2)
	s_dimtags = list(set(s_dimtags)^set(s_dimtags_remove))
	c = gmsh.model.geo.copy(s_dimtags)
	gmsh.model.geo.translate(c, 0, 0, offset_auxiliary_building_foundation)
	gmsh.model.geo.synchronize()



	## Foundation
	count_extrude = 1

	if depth_meta > SL[7]:	#meta are embedded into the second soil layer

		for i in range (0,3):	#foundation is consisted of 3 extruded layers (at -10,-8,-7)

			s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_found[i] - e,    x7 + e, y7 + e, z_aux_found[i] + e, 2)	

			#extrude
			# surface_to_extrude = list(set(s_all_dimtags)^set(remove_dimtags))
			numElements = int(math.ceil((z_aux_found[i+1]-z_aux_found[i])/z_lines_mesh_lv[7])) #take 7 (top layer) conservatively
			gmsh.model.geo.extrude(s_dimtags, 0, 0, (z_aux_found[i+1]-z_aux_found[i]), [numElements], recombine = True)
			print("\n" + 'Extrude the foundation of the auxiliary_building ' + str(count_extrude) + ' done !')
			count_extrude += 1

			gmsh.model.geo.synchronize()

	elif depth_meta < SL[7]:	#meta are embedded only in the first soil layer

		for i in range (0,1):	#foundation is consisted of 1 extruded layer (at -7)

			s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_found[i] - e,    x7 + e, y7 + e, z_aux_found[i] + e, 2)

			#extrude
			# surface_to_extrude = list(set(s_all_dimtags)^set(remove_dimtags))
			numElements = int(math.ceil((z_aux_found[i+1]-z_aux_found[i])/z_lines_mesh_lv[7])) #take 7 (top layer) conservatively
			gmsh.model.geo.extrude(s_dimtags, 0, 0, (z_aux_found[i+1]-z_aux_found[i]), [numElements], recombine = True)
			print("\n" + 'Extrude the foundation of the auxiliary_building ' + str(count_extrude) + ' done !')
			count_extrude += 1

			gmsh.model.geo.synchronize()



	## Structure
	## y coordinates of the structure
	y_aux0 = y5-4*L_aux/8.
	y_aux1 = y5-3*L_aux/8.
	y_aux2 = y5-2*L_aux/8.
	# y_aux3 = y5-1*L_aux/8.
	y_aux3 = y5
	# y_aux5 = y5+1*L_aux/8.
	y_aux4 = y5+2*L_aux/8.
	y_aux5 = y5+3*L_aux/8.
	y_aux6 = y5+4*L_aux/8.

	y_aux = [y_aux0, y_aux1, y_aux2, y_aux3, y_aux4, y_aux5, y_aux6]
	print("\n" + "y_coord_auxiliary_building : " + str(y_aux))

	## x coordinates of the structure
	x_aux0 = y5-4*L_aux/8.
	x_aux1 = y5-3*L_aux/8.
	x_aux2 = y5-2*L_aux/8.
	# x_aux3 = y5-1*L_aux/8.
	x_aux3 = y5
	# x_aux5 = y5+1*L_aux/8.
	x_aux4 = y5+2*L_aux/8.
	x_aux5 = y5+3*L_aux/8.
	x_aux6 = y5+4*L_aux/8.

	x_aux = [x_aux0, x_aux1, x_aux2, x_aux3, x_aux4, x_aux5, x_aux6]
	print("\n" + "x_coord_auxiliary_building : " + str(x_aux))


	## create points and put their tags in p_aux matrix
	p_aux = np.zeros([len(x_aux),len(y_aux)], dtype=np.int64)
	p_centre_aux = gmsh.model.geo.addPoint(x5, y5, z_aux_str[0]+SL[6]+SL[7], 0, -1) #returns the tag of the point
	#print(p,type(p))

	for i in range(0,len(x_aux)):
		for j in range(0,len(y_aux)):

			# if not ((i>=3 and i<=5) and (j>=3 and j<=5)):  
			p_aux[i,j] = gmsh.model.geo.addPoint(x_aux[i], y_aux[j], z_aux_str[0]+SL[6]+SL[7], 0, -1) #returns the tag of the point
			
	print("\n" + "p_aux_tags = " + "\n" + str(p_aux))
	gmsh.model.geo.synchronize()



	## create lines in y direction
	l_aux_y = np.zeros([len(x_aux),len(y_aux)-1], dtype=np.int64)

	for i in range(0,len(x_aux)):

		for j in range(0,len(y_aux)-1):

			# if (p_aux[i,j] and p_aux[i,j+1]) != 0: # to exclude the circular part in the middle
			if not ( (i == 3) and (j >= 2 and j<=3) ) :
				xyz1 = gmsh.model.getValue(0, p_aux[i,j], []) #returns x,y,z coord
				xyz2 = gmsh.model.getValue(0, p_aux[i,j+1], [])
				l_aux_y[i,j] = gmsh.model.geo.addLine(p_aux[i,j], p_aux[i,j+1], -1)
				#transfinite
				gmsh.model.geo.mesh.setTransfiniteCurve(l_aux_y[i,j], int(math.ceil((xyz2[1]-xyz1[1])/mesh_lv4))+1)	#ceil returns int, but in format of float, so put int in front

	print("\n" + "l_aux_y = " + "\n" + str(l_aux_y))
	# gmsh.model.geo.synchronize()

	## create lines in x direction
	l_aux_x = np.zeros([len(y_aux)-1,len(x_aux)], dtype=np.int64)

	for j in range(0,len(y_aux)):

		for i in range(0,len(x_aux)-1):

			# if (p_aux[i,j] and p_aux[i+1,j]) != 0: # to exclude the circular part in the middle
			if not ( (j == 3) and (i >= 2 and i<=3) ) :
				xyz1 = gmsh.model.getValue(0, p_aux[i,j], []) #returns x,y,z coord
				xyz2 = gmsh.model.getValue(0, p_aux[i+1,j], [])
				l_aux_x[i,j] = gmsh.model.geo.addLine(p_aux[i,j], p_aux[i+1,j], -1)
				#transfinite
				gmsh.model.geo.mesh.setTransfiniteCurve(l_aux_x[i,j], int(math.ceil((xyz2[0]-xyz1[0])/mesh_lv4))+1)	#ceil returns int, but in format of float, so put int in front

	print("\n" + "l_aux_x = " + "\n" + str(l_aux_x))
	gmsh.model.geo.synchronize()



	## create rectangular surfaces
	for j in range(0,len(y_aux)-1):

		for i in range(0,len(x_aux)-1):

			xyz1 = gmsh.model.getValue(0, p_aux[i,j], [])
			# print(xyz1)
			xyz2 = gmsh.model.getValue(0, p_aux[i+1,j+1], [])
			# print(xyz2)
			l_tags = gmsh.model.getEntitiesInBoundingBox(xyz1[0] - e, xyz1[1] - e, xyz1[2] - e,    xyz2[0] + e, xyz2[1] + e, xyz2[2] + e, 1)
			# print(l_tags)

			if len(l_tags) == 4 and not ( (j >= 2 and j<=3) and (i >= 2 and i<=3) ) :  #if 4 lines are in l_tags, then create the surface, not to leave the hole in the middle
				loop = gmsh.model.geo.addCurveLoop([ l_tags[0][1], l_tags[3][1], -l_tags[1][1], -l_tags[2][1] ], -1)
				s = gmsh.model.geo.addPlaneSurface([loop], -1)
				# print(s)
				gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
				gmsh.model.geo.mesh.setRecombine(2, s)

	gmsh.model.geo.synchronize()




	## create points in the centre (circle)
	p_circle_aux = np.zeros([1,8], dtype=int)
	xyz_center = gmsh.model.getValue(0, p_centre_aux, []) #get coord of the center of the circle

	for j in range(0,8):
		yy = xyz_center[1] + R[0] * math.sin(j*math.pi/4)	# R[0] = R_out+D_separation
		xx = xyz_center[0] + R[0] * math.cos(j*math.pi/4)
		p_circle_aux[0,j] = gmsh.model.geo.addPoint(xx, yy, z_aux_str[0]+SL[6]+SL[7], 0, -1)

	print("\n" + "p_circle_aux_tags = " + "\n" + str(p_circle_aux))
	gmsh.model.geo.synchronize()



	#arcs
	l_arc_aux = np.zeros([1,8], dtype=int)

	for j in range(0,8):
		
		if j!=(8-1):
			c = gmsh.model.geo.addCircleArc(p_circle_aux[0,j], p_centre_aux, p_circle_aux[0,j+1])
			l_arc_aux[0,j] = c
		else:
			c = gmsh.model.geo.addCircleArc(p_circle_aux[0,j], p_centre_aux, p_circle_aux[0,0])
			l_arc_aux[0,j] = c

		xyz1 = gmsh.model.getValue(0, p[0,5], [])
		xyz2 = gmsh.model.getValue(0, p[0,6], [])
		gmsh.model.geo.mesh.setTransfiniteCurve(c, int(math.ceil((xyz2[1]-xyz1[1])/y_lines_mesh_lv[5])+1)  ) 

	print("\n" + "l_arc_aux_tags = " + "\n" + str(l_arc_aux))
	# gmsh.model.geo.synchronize()



	#lines from square to circle
	l_square_to_circle_aux = np.zeros([1,8], dtype=int)

	xy = np.array([[4,3],
				[4,4],
				[3,4],
				[2,4],
				[2,3],
				[2,2],
				[3,2],
				[4,2]
				])

	for j in range(0,8):
		l = gmsh.model.geo.addLine(p_aux[xy[j,0],xy[j,1]], p_circle_aux[0,j], -1)
		l_square_to_circle_aux[0,j] = l

		gmsh.model.geo.mesh.setTransfiniteCurve(l, 5  )

	print("\n" + "l_square_to_circle_aux_tags = " + "\n" + str(l_square_to_circle_aux))
	gmsh.model.geo.synchronize()


	# create surfaces between square and arc-radial lines
	l_square_aux = np.zeros([1,8], dtype=int)
	l_square_aux[0,0] = l_aux_y[4,3]
	l_square_aux[0,1] = l_aux_x[3,4]
	l_square_aux[0,2] = l_aux_x[2,4]
	l_square_aux[0,3] = l_aux_y[2,3]
	l_square_aux[0,4] = l_aux_y[2,2]
	l_square_aux[0,5] = l_aux_x[2,2]
	l_square_aux[0,6] = l_aux_x[3,2]
	l_square_aux[0,7] = l_aux_y[4,2]

	for j in range(0,8):

		if j!=(8-1):
			loops = gmsh.model.geo.addCurveLoops([ l_square_aux[0,j],l_arc_aux[0,j],l_square_to_circle_aux[0,j],l_square_to_circle_aux[0,j+1] ])
			s = gmsh.model.geo.addPlaneSurface(loops)
		else:
			loops = gmsh.model.geo.addCurveLoops([ l_square_aux[0,j],l_arc_aux[0,j],l_square_to_circle_aux[0,j],l_square_to_circle_aux[0,0] ])
			s = gmsh.model.geo.addPlaneSurface(loops)
		
		gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
		gmsh.model.geo.mesh.setRecombine(2, s)

	gmsh.model.geo.synchronize()



	# extrude walls from lines
	#remove diagonals for the walls (for the mesh of the plates they are needed though)
	l_dimtags_all = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str[0]+SL[6]+SL[7] - e,    x7 + e, y7 + e, z_aux_str[0]+SL[6]+SL[7] + e, 1)
	l_dimtags_remove = gmsh.model.getEntitiesInBoundingBox(x5 + e, y5 + e, z_aux_str[0]+SL[6]+SL[7] - e,    x6 + e, y6 + e, z_aux_str[0]+SL[6]+SL[7] + e, 1)
	l_dimtags_remove += gmsh.model.getEntitiesInBoundingBox(x4 - e, y5 + e, z_aux_str[0]+SL[6]+SL[7] - e,    x5 - e, y6 + e, z_aux_str[0]+SL[6]+SL[7] + e, 1)
	l_dimtags_remove += gmsh.model.getEntitiesInBoundingBox(x4 - e, y4 - e, z_aux_str[0]+SL[6]+SL[7] - e,    x5 - e, y5 - e, z_aux_str[0]+SL[6]+SL[7] + e, 1)
	l_dimtags_remove += gmsh.model.getEntitiesInBoundingBox(x5 + e, y4 - e, z_aux_str[0]+SL[6]+SL[7] - e,    x6 + e, y5 - e, z_aux_str[0]+SL[6]+SL[7] + e, 1)
	# print(l_tags_remove)
	l_dimtags_to_extrude = list(set(l_dimtags_all)^set(l_dimtags_remove)) #remove common dimtags


	# extrude walls from lines
	if depth_meta > SL[7]:	#meta are embedded into the second soil layer

		l_dimtags = l_dimtags_to_extrude
		count_extrude = 1
		for i in range (5,0,-1):	#extrusion from z_aux_str6 to z_aux_str1,  z_aux_str[5]..[0]
			#extrude
			# surface_to_extrude = list(set(s_all_dimtags)^set(remove_dimtags))
			numElements = int(math.ceil((z_aux_str[i]-z_aux_str[i-1])/z_lines_mesh_lv[7])) #take 7 (top layer) conservatively
			gmsh.model.geo.extrude(l_dimtags, 0, 0, -(z_aux_str[i]-z_aux_str[i-1]), [numElements], recombine = True)
			print("\n" + 'Extrude auxiliary building -' + str(count_extrude) + ' done !')
			count_extrude += 1
			#update l_dimtags
			print(z_aux_str[i],z_aux_str[i-1])
			gmsh.model.geo.synchronize()
			l_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str[i-1] - e,    x7 + e, y7 + e, z_aux_str[i-1] + e, 1)

		l_dimtags = l_dimtags_to_extrude
		count_extrude = 1
		for i in range (0,floors_aux-1):	#extrusion from z_aux_str6 to the last floor
			#extrude
			# surface_to_extrude = list(set(s_all_dimtags)^set(remove_dimtags))
			numElements = int(math.ceil((z_aux_str[i+6]-z_aux_str[i+5])/z_lines_mesh_lv[7])) #take 7 (top layer) conservatively
			gmsh.model.geo.extrude(l_dimtags, 0, 0, +(z_aux_str[i+6]-z_aux_str[i+5]), [numElements], recombine = True)
			print("\n" + 'Extrude auxiliary building +' + str(count_extrude) + ' done !')
			count_extrude += 1
			#update l_dimtags
			print(z_aux_str[i+6],z_aux_str[i+5])
			gmsh.model.geo.synchronize()
			l_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str[i+6] - e,    x7 + e, y7 + e, z_aux_str[i+6] + e, 1)

			
	elif depth_meta < SL[7]:	#meta are embedded only in the first soil layer

		l_dimtags = l_dimtags_to_extrude
		count_extrude = 1
		for i in range (3,0,-1):	#extrusion from z_aux_str4 to z_aux_str1,  z_aux_str[3]..[0]
			#extrude
			# surface_to_extrude = list(set(s_all_dimtags)^set(remove_dimtags))
			numElements = int(math.ceil((z_aux_str[i]-z_aux_str[i-1])/z_lines_mesh_lv[7])) #take 7 (top layer) conservatively
			gmsh.model.geo.extrude(l_dimtags, 0, 0, -(z_aux_str[i]-z_aux_str[i-1]), [numElements], recombine = True)
			print("\n" + 'Extrude auxiliary building -' + str(count_extrude) + ' done !')
			count_extrude += 1
			#update l_dimtags
			print(z_aux_str[i],z_aux_str[i-1])
			gmsh.model.geo.synchronize()
			l_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str[i-1] - e,    x7 + e, y7 + e, z_aux_str[i-1] + e, 1)

		l_dimtags = l_dimtags_to_extrude
		count_extrude = 1
		for i in range (0,floors_aux-1):	#extrusion from z_aux_str4 to the last floor
			#extrude
			# surface_to_extrude = list(set(s_all_dimtags)^set(remove_dimtags))
			numElements = int(math.ceil((z_aux_str[i+4]-z_aux_str[i+3])/z_lines_mesh_lv[7])) #take 7 (top layer) conservatively
			gmsh.model.geo.extrude(l_dimtags, 0, 0, +(z_aux_str[i+4]-z_aux_str[i+3]), [numElements], recombine = True)
			print("\n" + 'Extrude auxiliary building +' + str(count_extrude) + ' done !')
			count_extrude += 1
			#update l_dimtags
			print(z_aux_str[i+4],z_aux_str[i+3])
			gmsh.model.geo.synchronize()
			l_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str[i+4] - e,    x7 + e, y7 + e, z_aux_str[i+4] + e, 1)



	#copy and create the rest floors from 1st floor
	count_floor = 2
	for i in range (0,floors_aux-1):	
		
		s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str[len(z_aux_str)-floors_aux + i] - e,    x7 + e, y7 + e, z_aux_str[len(z_aux_str)-floors_aux + i] + e, 2)
		c = gmsh.model.geo.copy(s_dimtags)
		gmsh.model.geo.translate(c, 0, 0, h_floor_aux)
		print("\n" + 'Copy -- auxiliary building floor #' + str(count_floor) + ' done !')
		count_floor += 1
		gmsh.model.geo.synchronize()



	## Physical groups
	if depth_meta < SL[7]:
		v_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_found1 - e,    x7 + e, y7 + e,  z_aux_found2 + e, 3)
		gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(3, idx, "auxiliary_building_foundation")
		idx += 1

		s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str1 - e,    x7 + e, y7 + e,  z_aux_str2 + e, 2)
		gmsh.model.addPhysicalGroup(2, [s_dimtags[i][1] for i in range(len(s_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(2, idx, "auxiliary_building_embedment")
		idx += 1

		s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str2 - e,    x3 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x7 - e, y3 - e, z_aux_str2 - e,    x7 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str2 - e,    x7 + e, y3 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y7 - e, z_aux_str2 - e,    x7 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		gmsh.model.addPhysicalGroup(2, [s_dimtags[i][1] for i in range(len(s_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(2, idx, "auxiliary_building_outer_walls")
		idx += 1

		s_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R[0] - e, y5-R[0] - e, z_aux_str2 - e,    x5+R[0] + e, y5+R[0] + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		gmsh.model.addPhysicalGroup(2, [s_dimtags[i][1] for i in range(len(s_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(2, idx, "auxiliary_building_inner_circular_wall")
		idx += 1

		for k in range(3,len(z_aux_str)):
			s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str[k] - e,    x7 + e, y7 + e,  z_aux_str[k] + e, 2)
			gmsh.model.addPhysicalGroup(2, [s_dimtags[i][1] for i in range(len(s_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
			gmsh.model.setPhysicalName(2, idx, "auxiliary_building_floor" + str(k-2))
			idx += 1

		s_dimtags = gmsh.model.getEntitiesInBoundingBox(x_aux1 - e, y3 - e, z_aux_str2 - e,    x_aux1 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x_aux2 - e, y3 - e, z_aux_str2 - e,    x_aux2 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x_aux3 - e, y3 - e, z_aux_str2 - e,    x_aux3 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x_aux4 - e, y3 - e, z_aux_str2 - e,    x_aux4 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x_aux5 - e, y3 - e, z_aux_str2 - e,    x_aux5 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y_aux1 - e, z_aux_str2 - e,    x7 + e, y_aux1 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y_aux2 - e, z_aux_str2 - e,    x7 + e, y_aux2 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y_aux3 - e, z_aux_str2 - e,    x7 + e, y_aux3 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y_aux4 - e, z_aux_str2 - e,    x7 + e, y_aux4 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y_aux5 - e, z_aux_str2 - e,    x7 + e, y_aux5 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		gmsh.model.addPhysicalGroup(2, [s_dimtags[i][1] for i in range(len(s_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(2, idx, "auxiliary_building_inner_walls")
		idx += 1

	else:

		v_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_found1 - e,    x7 + e, y7 + e,  z_aux_found4 + e, 3)
		gmsh.model.addPhysicalGroup(3, [v_dimtags[i][1] for i in range(len(v_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(3, idx, "auxiliary_building_foundation")
		idx += 1

		s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str1 - e,    x7 + e, y7 + e,  z_aux_str4 + e, 2)
		gmsh.model.addPhysicalGroup(2, [s_dimtags[i][1] for i in range(len(s_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(2, idx, "auxiliary_building_embedment")
		idx += 1

		s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str4 - e,    x3 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x7 - e, y3 - e, z_aux_str4 - e,    x7 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str4 - e,    x7 + e, y3 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y7 - e, z_aux_str4 - e,    x7 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		gmsh.model.addPhysicalGroup(2, [s_dimtags[i][1] for i in range(len(s_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(2, idx, "auxiliary_building_outer_walls")
		idx += 1

		s_dimtags = gmsh.model.getEntitiesInBoundingBox(x5-R[0] - e, y5-R[0] - e, z_aux_str4 - e,    x5+R[0] + e, y5+R[0] + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		gmsh.model.addPhysicalGroup(2, [s_dimtags[i][1] for i in range(len(s_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(2, idx, "auxiliary_building_inner_circular_wall")
		idx += 1

		for k in range(5,len(z_aux_str)):
			s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str[k] - e,    x7 + e, y7 + e,  z_aux_str[k] + e, 2)
			gmsh.model.addPhysicalGroup(2, [s_dimtags[i][1] for i in range(len(s_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
			gmsh.model.setPhysicalName(2, idx, "auxiliary_building_floor" + str(k-4))
			idx += 1

		s_dimtags = gmsh.model.getEntitiesInBoundingBox(x_aux1 - e, y3 - e, z_aux_str4 - e,    x_aux1 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x_aux2 - e, y3 - e, z_aux_str4 - e,    x_aux2 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x_aux3 - e, y3 - e, z_aux_str4 - e,    x_aux3 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x_aux4 - e, y3 - e, z_aux_str4 - e,    x_aux4 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x_aux5 - e, y3 - e, z_aux_str4 - e,    x_aux5 + e, y7 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y_aux1 - e, z_aux_str4 - e,    x7 + e, y_aux1 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y_aux2 - e, z_aux_str4 - e,    x7 + e, y_aux2 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y_aux3 - e, z_aux_str4 - e,    x7 + e, y_aux3 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y_aux4 - e, z_aux_str4 - e,    x7 + e, y_aux4 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y_aux5 - e, z_aux_str4 - e,    x7 + e, y_aux5 + e,  z_aux_str[len(z_aux_str)-1] + e, 2)
		gmsh.model.addPhysicalGroup(2, [s_dimtags[i][1] for i in range(len(s_dimtags))], idx) #requires input [1,2,3,4...], but v_dimtags is [(3,1),(3,2),..]
		gmsh.model.setPhysicalName(2, idx, "auxiliary_building_inner_walls")
		idx += 1



	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_found1 - e,    x7 + e, y7 + e, z_aux_found1 + e, 2)
	s_dimtags_remove = gmsh.model.getEntitiesInBoundingBox(x5-R_out-D_separation - e, y5-R_out-D_separation - e, z_aux_found1 - e,    x5+R_out+D_separation + e, y5+R_out+D_separation + e, z_aux_found1 + e, 2)
	s_dimtags = list(set(s_dimtags)^set(s_dimtags_remove))
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "auxiliary_building_foundation-soil_z")
	idx += 1

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_found1 - e,    x3 + e, y7 + e, z_aux_found1+SL[6] + e, 2)
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "auxiliary_building_foundation-soil_x+")
	idx += 1

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x7 - e, y3 - e, z_aux_found1 - e,    x7 + e, y7 + e, z_aux_found1+SL[6] + e, 2)
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "auxiliary_building_foundation-soil_x-")
	idx += 1

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_found1 - e,    x7 + e, y3 + e, z_aux_found1+SL[6] + e, 2)
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "auxiliary_building_foundation-soil_y+")
	idx += 1

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y7 - e, z_aux_found1 - e,    x7 + e, y7 + e, z_aux_found1+SL[6] + e, 2)
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "auxiliary_building_foundation-soil_y-")
	idx += 1



	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str1+SL[6] - e,    x3 + e, y7 + e, z_aux_str1+SL[6]+SL[7] + e, 2)
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "auxiliary_building_structure-soil_x+")
	idx += 1

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x7 - e, y3 - e, z_aux_str1+SL[6] - e,    x7 + e, y7 + e, z_aux_str1+SL[6]+SL[7] + e, 2)
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "auxiliary_building_structure-soil_x-")
	idx += 1

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str1+SL[6] - e,    x7 + e, y3 + e, z_aux_str1+SL[6]+SL[7] + e, 2)
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "auxiliary_building_structure-soil_y+")
	idx += 1

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y7 - e, z_aux_str1+SL[6] - e,    x7 + e, y7 + e, z_aux_str1+SL[6]+SL[7] + e, 2)
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "auxiliary_building_structure-soil_y-")
	idx += 1




###########################################################################################################################################################################
########################################################################### Metamaterials #################################################################################
###########################################################################################################################################################################

if metamaterials_exist == 1:

	#create an array of metamaterials to copy and paste later according to metamaterials list (this is not going to be used, it is only for copy and paste)
	offset_meta_copy_paste = 20

	# initialize lists for points and lines of the unit cell
	p_cell = [ [ [0 for i in range(3)] for j in range(3) ] for z in range(3) ]	 # [[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
	l_cell_x = [ [ [0 for i in range(2)] for j in range(3) ] for z in range(3) ]	# [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]]	[z][j][i]
	l_cell_y = [ [ [0 for j in range(2)] for i in range(3) ] for z in range(3) ]	# [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]]	[z][i][j]
	l_cell_z = [ [ [0 for z in range(2)] for i in range(3) ] for y in range(3) ]	# [[[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]], [[0, 0], [0, 0], [0, 0]]]	[i][j][z]


	## create 27 points of a unit cell and put their tags in p_cell list
	for z in range(0,3):
		for j in range(0,3):
			for i in range(0,3):

				p = gmsh.model.geo.addPoint(x3+L_meta/2.0*i, y8+L_meta/2.0*j, z_m_bot+ offset_meta_copy_paste +L_meta/2.0*z, 0, -1) #returns the tag of the point
				p_cell[z][j][i] = p


	print("\n" + "p_cell_tags = " + "\n" + str(p_cell))
	gmsh.model.geo.synchronize()


	## create lines in x direction
	for z in range(0,3):
		for j in range(0,3):
			for i in range(0,2):

				l = gmsh.model.geo.addLine(p_cell[z][j][i] , p_cell[z][j][i+1], -1)
				l_cell_x[z][j][i] = l
				#transfinite
				gmsh.model.geo.mesh.setTransfiniteCurve(l, 2)	


	print("\n" + "l_cell_x_tags = " + "\n" + str(l_cell_x))
	# gmsh.model.geo.synchronize()

	## create lines in y direction
	for z in range(0,3):
		for i in range(0,3):
			for j in range(0,2):

				l = gmsh.model.geo.addLine(p_cell[z][j][i] , p_cell[z][j+1][i], -1)
				l_cell_y[z][i][j] = l
				gmsh.model.geo.mesh.setTransfiniteCurve(l, 2)


	print("\n" + "l_cell_y_tags = " + "\n" + str(l_cell_y))
	# gmsh.model.geo.synchronize()

	## create lines in z direction
	for i in range(0,3):
		for j in range(0,3):
			for z in range(0,2):

				l = gmsh.model.geo.addLine(p_cell[z][j][i] , p_cell[z+1][j][i], -1)
				l_cell_z[i][j][z] = l
				gmsh.model.geo.mesh.setTransfiniteCurve(l, 2)


	print("\n" + "l_cell_z_tags = " + "\n" + str(l_cell_z))
	gmsh.model.geo.synchronize()



	## create surfaces
	#bottom and top surfaces
	for z in range(0,3,2): #skip the middle one
		for i in range(0,2):
			for j in range(0,2):
				loops = gmsh.model.geo.addCurveLoops([ l_cell_x[z][j][i], l_cell_x[z][j+1][i], l_cell_y[z][i][j], l_cell_y[z][i+1][j] ])
				s = gmsh.model.geo.addPlaneSurface(loops)
				gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
				gmsh.model.geo.mesh.setRecombine(2, s)

	#left and right surfaces
	for i in range(0,3,2): #skip the middle one
		for j in range(0,2):
			for z in range(0,2):
				loops = gmsh.model.geo.addCurveLoops([ l_cell_y[z][i][j], l_cell_y[z+1][i][j], l_cell_z[i][j][z], l_cell_z[i][j+1][z] ])
				s = gmsh.model.geo.addPlaneSurface(loops)
				gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
				gmsh.model.geo.mesh.setRecombine(2, s)

	#front and rear surfaces
	for j in range(0,3,2): #skip the middle one
		for i in range(0,2):
			for z in range(0,2):
				loops = gmsh.model.geo.addCurveLoops([ l_cell_x[z][j][i], l_cell_x[z+1][j][i], l_cell_z[i][j][z], l_cell_z[i+1][j][z] ])
				s = gmsh.model.geo.addPlaneSurface(loops)
				gmsh.model.geo.mesh.setTransfiniteSurface(s, "Left")
				gmsh.model.geo.mesh.setRecombine(2, s)

	gmsh.model.geo.synchronize()



	#create the ref set of metamaterials to be copied and pasted later
	for nx in range(0,11):
		# copy the geometry and the mesh from the unit cell
		e_dimtags = gmsh.model.getEntitiesInBoundingBox(x3+L_meta*nx - e, y8 - e, z_m_bot+offset_meta_copy_paste - e,    x3+L_meta*(nx+1) + e, y8+L_meta + e, z_m_top+offset_meta_copy_paste + e, -1)
		# s_dimtags = list(set(s_dimtags)^set(s_dimtags_remove))
		c = gmsh.model.geo.copy(e_dimtags)
		gmsh.model.geo.translate(c, L_meta, 0, 0)
		gmsh.model.geo.synchronize()

	for ny in range(0,n_meta_h-1):
		# copy the geometry and the mesh from the unit cell
		e_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y8+L_meta*ny - e, z_m_bot+offset_meta_copy_paste - e,    x4 + e, y8+L_meta*(ny+1) + e, z_m_top+offset_meta_copy_paste + e, -1)
		# s_dimtags = list(set(s_dimtags)^set(s_dimtags_remove))
		c = gmsh.model.geo.copy(e_dimtags)
		gmsh.model.geo.translate(c, 0, L_meta, 0)
		gmsh.model.geo.synchronize()

	for nz in range(0,n_meta_v-1):
		# copy the geometry and the mesh from the unit cell
		e_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y8 - e, z_m_bot+offset_meta_copy_paste  + L_meta*nz - e,    x4 + e, y9 + e, z_m_top+offset_meta_copy_paste + L_meta*(nz+1) + e, -1)
		# s_dimtags = list(set(s_dimtags)^set(s_dimtags_remove))
		c = gmsh.model.geo.copy(e_dimtags)
		gmsh.model.geo.translate(c, 0, 0, L_meta)
		gmsh.model.geo.synchronize()

	#dimtags of the ref set
	e_dimtags_copy_paste = gmsh.model.getEntitiesInBoundingBox(x3 - e, y8 - e, z_m_bot+offset_meta_copy_paste - e,    x4 + e, y9 + e, z_m_top+offset_meta_copy_paste + L_meta*n_meta_v + e, -1)



	#copy_paste metamaterials from the ref set
	print("\n" + 'Copy-paste metamaterials..')

	for i,im in enumerate(metamaterials): #1st layer of the list

		if i == 0: #top row

			for j,jm in enumerate(im): #2nd layer of the list
				
				if jm==1: # is_metamaterial
					c = gmsh.model.geo.copy(e_dimtags_copy_paste)
					gmsh.model.geo.translate(c, (x4-x3)*j, 0, -offset_meta_copy_paste)

					print("\n" + 'Copy-paste metamaterials -- top row ' + str(j+1) + ' done !')


		elif i == 1: #bottom row

			for j,jm in enumerate(im): #2nd layer of the list
				
				if jm==1: # is_metamaterial
					c = gmsh.model.geo.copy(e_dimtags_copy_paste)
					gmsh.model.geo.translate(c, (x4-x3)*j, -(y8-y1), -offset_meta_copy_paste)

					print("\n" + 'Copy-paste metamaterials -- bottom row ' + str(j+1) + ' done !')



	#rotate the initial metamaterial array to copy-paste the left and right rows
	gmsh.model.geo.rotate(e_dimtags_copy_paste, x3, y8, 0, 0, 0, 1, math.pi / 2)	#dimTags, x, y, z, ax, ay, az, angle


	for i,im in enumerate(metamaterials): #1st layer of the list

		if i == 2: #left row

			for j,jm in enumerate(im): #2nd layer of the list
				
				if jm==1: # is_metamaterial
					c = gmsh.model.geo.copy(e_dimtags_copy_paste)
					gmsh.model.geo.translate(c, -(x3-x2), -(y8-y3)+(y4-y3)*j, -offset_meta_copy_paste)

					print("\n" + 'Copy-paste metamaterials -- left row ' + str(j+1) + ' done !')


		elif i == 3: #right row

			for j,jm in enumerate(im): #2nd layer of the list
				
				if jm==1: # is_metamaterial
					c = gmsh.model.geo.copy(e_dimtags_copy_paste)
					gmsh.model.geo.translate(c, (x9-x3), -(y8-y3)+(y4-y3)*j, -offset_meta_copy_paste)

					print("\n" + 'Copy-paste metamaterials -- right row ' + str(j+1) + ' done !')




	#remove the initial metamaterial array
	gmsh.model.geo.remove(e_dimtags_copy_paste, recursive=True)	#If recursive is true, remove all the entities on their boundaries, down to dimension 0.
	gmsh.model.geo.synchronize()

	#dont know why, but the points were not removed
	final_remove = gmsh.model.getEntitiesInBoundingBox(x0 - e, y0 - e, z_m_bot+offset_meta_copy_paste - e,    x10 + e, y10 + e, z_m_top+offset_meta_copy_paste + L_meta*n_meta_v + e, -1)
	gmsh.model.geo.remove(final_remove, recursive=True)	

	gmsh.model.geo.synchronize()



	###########################################################################################################################################################################
	## Physical groups

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x1 - e, y1 - e, z_m_bot - e,    x9 + e, y9 + e, z_m_top + e, 2)	
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "meta_casings")	
	idx += 1

	############## kelvin-voigt_elements_x
	l_dimtags=[]
	#top row
	for z in range(0,n_meta_v):
		for j in range(0,n_meta_h):
			l_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y8+L_meta*j + e, z_m_bot+L_meta*z + e,    x7 + e, y8+L_meta*(j+1) - e, z_m_bot+L_meta*(z+1) - e, 1)

	#bottom row
	for z in range(0,n_meta_v):
		for j in range(0,n_meta_h):
			l_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y1+L_meta*j + e, z_m_bot+L_meta*z + e,    x7 + e, y1+L_meta*(j+1) - e, z_m_bot+L_meta*(z+1) - e, 1)

	#left row
	for z in range(0,n_meta_v):
		for j in range(0,L_aux/L_meta): #48
			l_dimtags += gmsh.model.getEntitiesInBoundingBox(x1 - e, y3+L_meta*j + e, z_m_bot+L_meta*z + e,    x2 + e, y3+L_meta*(j+1) - e, z_m_bot+L_meta*(z+1) - e, 1)

	#right row
	for z in range(0,n_meta_v):
		for j in range(0,L_aux/L_meta): #48
			l_dimtags += gmsh.model.getEntitiesInBoundingBox(x8 - e, y3+L_meta*j + e, z_m_bot+L_meta*z + e,    x9 + e, y3+L_meta*(j+1) - e, z_m_bot+L_meta*(z+1) - e, 1)

	gmsh.model.addPhysicalGroup(1, [l_dimtags[j][1] for j in range(len(l_dimtags))], idx)
	gmsh.model.setPhysicalName(1, idx, "kelvin_x")	
	idx += 1

	############## kelvin-voigt_elements_y
	l_dimtags=[]
	#top row
	for z in range(0,n_meta_v):
		for i in range(0,L_aux/L_meta): #48
			l_dimtags += gmsh.model.getEntitiesInBoundingBox(x3+L_meta*i + e, y8 - e, z_m_bot+L_meta*z + e,    x3+L_meta*(i+1) - e, y9 + e, z_m_bot+L_meta*(z+1) - e, 1)

	#bottom row
	for z in range(0,n_meta_v):
		for i in range(0,L_aux/L_meta): #48
			l_dimtags += gmsh.model.getEntitiesInBoundingBox(x3+L_meta*i + e, y1 - e, z_m_bot+L_meta*z + e,    x3+L_meta*(i+1) - e, y2 + e, z_m_bot+L_meta*(z+1) - e, 1)

	#left row
	for z in range(0,n_meta_v):
		for i in range(0,n_meta_h):
			l_dimtags += gmsh.model.getEntitiesInBoundingBox(x1+L_meta*i + e, y3 - e, z_m_bot+L_meta*z + e,    x1+L_meta*(i+1) - e, y7 + e, z_m_bot+L_meta*(z+1) - e, 1)

	#right row
	for z in range(0,n_meta_v):
		for i in range(0,n_meta_h):
			l_dimtags += gmsh.model.getEntitiesInBoundingBox(x8+L_meta*i + e, y3 - e, z_m_bot+L_meta*z + e,    x8+L_meta*(i+1) - e, y7 + e, z_m_bot+L_meta*(z+1) - e, 1)

	gmsh.model.addPhysicalGroup(1, [l_dimtags[j][1] for j in range(len(l_dimtags))], idx)
	gmsh.model.setPhysicalName(1, idx, "kelvin_y")	
	idx += 1

	############## kelvin-voigt_elements_z
	l_dimtags=[]
	#top row
	for j in range(0,n_meta_h):
		for i in range(0,L_aux/L_meta): #48
			l_dimtags += gmsh.model.getEntitiesInBoundingBox(x3+L_meta*i + e, y8+L_meta*j + e, z_m_bot - e,    x3+L_meta*(i+1) - e, y8+L_meta*(j+1) - e, z_m_top + e, 1)

	#bottom row
	for j in range(0,n_meta_h):
		for i in range(0,L_aux/L_meta): #48
			l_dimtags += gmsh.model.getEntitiesInBoundingBox(x3+L_meta*i + e, y1+L_meta*j + e, z_m_bot - e,    x3+L_meta*(i+1) - e, y1+L_meta*(j+1) - e, z_m_top + e, 1)

	#left row
	for i in range(0,n_meta_h):
		for j in range(0,L_aux/L_meta): #48
			l_dimtags += gmsh.model.getEntitiesInBoundingBox(x1+L_meta*i + e, y3+L_meta*j + e, z_m_bot - e,    x1+L_meta*(i+1) - e, y3+L_meta*(j+1) - e, z_m_top + e, 1)

	#right row
	for i in range(0,n_meta_h):
		for j in range(0,L_aux/L_meta): #48
			l_dimtags += gmsh.model.getEntitiesInBoundingBox(x8+L_meta*i + e, y3+L_meta*j + e, z_m_bot - e,    x8+L_meta*(i+1) - e, y3+L_meta*(j+1) - e, z_m_top + e, 1)


	gmsh.model.addPhysicalGroup(1, [l_dimtags[j][1] for j in range(len(l_dimtags))], idx)
	gmsh.model.setPhysicalName(1, idx, "kelvin_z")	
	idx += 1


	############## masses
	p_dimtags=[]
	#top row
	for z in range(0,n_meta_v):
		for j in range(0,n_meta_h):
			for i in range(0,L_aux/L_meta): #48
				p_dimtags += gmsh.model.getEntitiesInBoundingBox(x3+L_meta*i + e, y8+L_meta*j + e, z_m_bot+L_meta*z + e,    x3+L_meta*(i+1) - e, y8+L_meta*(j+1) - e, z_m_bot+L_meta*(z+1) - e, 0)

	#bottom row
	for z in range(0,n_meta_v):
		for j in range(0,n_meta_h):
			for i in range(0,L_aux/L_meta): #48
				p_dimtags += gmsh.model.getEntitiesInBoundingBox(x3+L_meta*i + e, y1+L_meta*j + e, z_m_bot+L_meta*z + e,    x3+L_meta*(i+1) - e, y1+L_meta*(j+1) - e, z_m_bot+L_meta*(z+1) - e, 0)

	#left row
	for z in range(0,n_meta_v):
		for j in range(0,L_aux/L_meta): #48
			for i in range(0,n_meta_h):
				p_dimtags += gmsh.model.getEntitiesInBoundingBox(x1+L_meta*i + e, y3+L_meta*j + e, z_m_bot+L_meta*z + e,    x1+L_meta*(i+1) - e, y3+L_meta*(j+1) - e, z_m_bot+L_meta*(z+1) - e, 0)

	#right row
	for z in range(0,n_meta_v):
		for j in range(0,L_aux/L_meta): #48
			for i in range(0,n_meta_h):
				p_dimtags += gmsh.model.getEntitiesInBoundingBox(x8+L_meta*i + e, y3+L_meta*j + e, z_m_bot+L_meta*z + e,    x8+L_meta*(i+1) - e, y3+L_meta*(j+1) - e, z_m_bot+L_meta*(z+1) - e, 0)

	gmsh.model.addPhysicalGroup(0, [p_dimtags[j][1] for j in range(len(p_dimtags))], idx)
	gmsh.model.setPhysicalName(0, idx, "masses")	
	idx += 1


	############## metamaterial-soil interface
	#x+
	s_dimtags=[]
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y8 - e, z_m_bot - e,    x3 + e, y9 + e, z_m_top + e, 2)	#top
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y1 - e, z_m_bot - e,    x3 + e, y2 + e, z_m_top + e, 2)	#bottom
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x1 - e, y3 - e, z_m_bot - e,    x1 + e, y7 + e, z_m_top + e, 2)	#left
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x8 - e, y3 - e, z_m_bot - e,    x8 + e, y7 + e, z_m_top + e, 2)	#right
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "metamaterials-soil_x+")	
	idx += 1

	#x-
	s_dimtags=[]
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x7 - e, y8 - e, z_m_bot - e,    x7 + e, y9 + e, z_m_top + e, 2)	#top
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x7 - e, y1 - e, z_m_bot - e,    x7 + e, y2 + e, z_m_top + e, 2)	#bottom
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x2 - e, y3 - e, z_m_bot - e,    x2 + e, y7 + e, z_m_top + e, 2)	#left
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x9 - e, y3 - e, z_m_bot - e,    x9 + e, y7 + e, z_m_top + e, 2)	#right
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "metamaterials-soil_x-")	
	idx += 1

	#y+
	s_dimtags=[]
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y8 - e, z_m_bot - e,    x7 + e, y8 + e, z_m_top + e, 2)	#top
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y1 - e, z_m_bot - e,    x7 + e, y1 + e, z_m_top + e, 2)	#bottom
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x1 - e, y3 - e, z_m_bot - e,    x2 + e, y3 + e, z_m_top + e, 2)	#left
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x8 - e, y3 - e, z_m_bot - e,    x9 + e, y3 + e, z_m_top + e, 2)	#right
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "metamaterials-soil_y+")	
	idx += 1

	#y-
	s_dimtags=[]
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y9 - e, z_m_bot - e,    x7 + e, y9 + e, z_m_top + e, 2)	#top
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y2 - e, z_m_bot - e,    x7 + e, y2 + e, z_m_top + e, 2)	#bottom
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x1 - e, y7 - e, z_m_bot - e,    x2 + e, y7 + e, z_m_top + e, 2)	#left
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x8 - e, y7 - e, z_m_bot - e,    x9 + e, y7 + e, z_m_top + e, 2)	#right
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "metamaterials-soil_y-")	
	idx += 1

	#z+
	s_dimtags=[]
	s_dimtags += gmsh.model.getEntitiesInBoundingBox(x1 - e, y1 - e, z_m_bot - e,    x9 + e, y9 + e, z_m_bot + e, 2)	#all
	# s_dimtags += gmsh.model.getEntitiesInBoundingBox(x3 - e, y2 - e, z_m_bot - e,    x7 + e, y2 + e, z_m_top + e, 2)	#bottom
	# s_dimtags += gmsh.model.getEntitiesInBoundingBox(x1 - e, y7 - e, z_m_bot - e,    x2 + e, y7 + e, z_m_top + e, 2)	#left
	# s_dimtags += gmsh.model.getEntitiesInBoundingBox(x8 - e, y7 - e, z_m_bot - e,    x9 + e, y7 + e, z_m_top + e, 2)	#right
	gmsh.model.addPhysicalGroup(2, [s_dimtags[j][1] for j in range(len(s_dimtags))], idx)
	gmsh.model.setPhysicalName(2, idx, "metamaterials-soil_z")	
	idx += 1



###########################################################################################################################################################################
## put everything in place
gmsh.option.setNumber("Geometry.AutoCoherence", 0) # assign 0 to AutoCoherence so when for example the Reactor building is translated on top of the soil, the soil-Reactor building nodes with the same coordinates are not merged into one node (this is needed to generate the interface elements later)

if reactor_building == 1:
	v_dimtags1 = gmsh.model.getEntitiesInBoundingBox(x5-R_out - e, y5-R_out - e, z_r1 - e,    x5+R_out + e, y5+R_out + e, z_r[len(z_r)-1] + e, 3)
	gmsh.model.geo.translate(v_dimtags1, 0, 0, -offset_reactor_building)

	v_dimtags2 = gmsh.model.getEntitiesInBoundingBox(x_rv0 - e, y_rv0 - e, z_rv0 - e,    x_rv5 + e, y_rv5 + e, z_rv2 + e, 3)
	gmsh.model.geo.translate(v_dimtags2, 0, 0, -offset_reactor_vessel)


if auxiliary_building == 1:
	v_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_found1 - e,    x7 + e, y7 + e, z_aux_found[len(z_aux_found)-1] + e, 3)
	gmsh.model.geo.translate(v_dimtags, 0, 0, -offset_auxiliary_building_foundation)

	s_dimtags = gmsh.model.getEntitiesInBoundingBox(x3 - e, y3 - e, z_aux_str1 - e,    x7 + e, y7 + e, z_aux_str[len(z_aux_str)-1] + e, 2)
	gmsh.model.geo.translate(s_dimtags, 0, 0, -offset_auxiliary_building_structure)


if metamaterials_exist == 1:
	e_dimtags = gmsh.model.getEntitiesInBoundingBox(x1 - e, y1 - e, z_m_bot - e,    x9 + e, y9 + e, z_m_top + e, -1)
	gmsh.model.geo.translate(e_dimtags, 0, 0, -offset_metamaterials)

###########################################################################################################################################################################
## generate 3D mesh and save it to mesh.msh2
#
# set color options
white = (255, 255, 255)
black = (0, 0, 0)
grey = (64, 64, 64)
green = (153, 255, 153)
gmsh.option.setNumber("Mesh.ColorCarousel", 0) #Mesh coloring (0: by element type, 1: by elementary entity, 2: by physical group, 3: by mesh partition) Default value: 1
gmsh.option.setColor("Mesh.Color.Quadrangles", green[0], green[1], green[2]) #(if Mesh.ColorCarousel=0)
gmsh.option.setColor("Geometry.Color.Curves", grey[0], grey[1], grey[2])

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(3) # generate the 3D mesh
gmsh.write("mesh.msh2") # save the generated mesh to mesh.msh2 file

###########################################################################################################################################################################
## open gmsh
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()


gmsh.finalize()

