# Python libraries
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import optimize
import numpy as np
import pickle

# Shapely libraries
from shapely.geometry import LineString, Point, Polygon, MultiPolygon
from descartes import PolygonPatch
from shapely.ops import cascaded_union
from shapely.geometry import  JOIN_STYLE

# Pedro Libraries
from figures import SIZE, BLUE, GRAY
from airfoil_module import CST
from CST_module import calculate_arc_length, dxi_u

# Functions
def calculate_c(length_0, A, TE_displacement, N1, N2):
    """Equations in the New_CST.pdf. Calculates the upper chord in order for
       the cruise and landing airfoils ot have the same length."""
    
    def integrand(psi, Au, delta_xi, N1, N2):
        return np.sqrt(1 + dxi_u(psi, Au, delta_xi, N1, N2)**2)
    
    def f(current_c):
        """Function dependent of c_C and that outputs c_C.""" 
        current_length, err = quad(integrand, 0, 1, args=(A, TE_displacement/current_c, N1, N2))
        print current_c, current_length
        return length_0/current_length

    current_c = optimize.fixed_point(f, length_0, maxiter=100)
    #In case the calculated chord is really close to the original, but the
    #algorithm was not able to make them equal
    if abs(length_0 - current_c) < 1e-7:
        return length_0
    #The output is an array so it needs the extra [0]
    return current_c

def calculate_deformed_psi(original_psi, length_0, length_deformed, A, TE_displacement):
    """Equations in the New_CST.pdf. Calculates the upper chord in order for
       the cruise and landing airfoils ot have the same length."""

    def integrand(psi, Au, delta_xi ):
        return np.sqrt(1 + dxi_u(psi, Au, delta_xi,N1,N2)**2)
    
    def f(current_psi):
        """Function dependent of c_C and that outputs c_C."""
        global result_psi
        result_psi = current_psi
        current_length, err = quad(integrand, 0, current_psi, args=(A, TE_displacement/length_deformed))
        return abs(length_0*original_psi - length_deformed*current_length)/length_0*original_psi

    optimize.minimize(f, [original_psi],method='L-BFGS-B',bounds=((0.00001,.999999),), options={'gtol':1e-9, 'maxiter':100000, 'ftol':1e-12, 'maxfun':100000})
    
    return result_psi

def extract_poly_coords(input):
    def main(geom):
        if geom.type == 'Polygon':
            exterior_coords = [geom.exterior.coords[:]]
            interior_coords = []
            for interior in geom.interiors:
                interior_coords.append(interior.coords[:])
        elif geom.type == 'MultiPolygon':
            exterior_coords = []
            interior_coords = []
            for part in geom:
                epc = extract_poly_coords(part)  # Recursive call
                exterior_coords += epc['exterior_coords']
                interior_coords += epc['interior_coords']
        else:
            raise ValueError('Unhandled geometry type: ' + repr(geom.type))
        return {'exterior_coords': exterior_coords,
                'interior_coords': interior_coords}
    if type(input) == list:
        output = []
        for i in range(len(input)):
            output.append(main(input[i]))
    else:
        output = main(input)
    return output


def find_point_inside(object):
    (minx, miny, maxx, maxy) = object.bounds
    step_size = 0.001
    x = minx
    y = miny
    done = False
    while not done:
        x += (maxx - minx)*step_size
        y += (maxy - miny)*step_size
        point = Point(x,y)
        if object.contains(point) and object.exterior.distance(point)>1e-4:
            done = True
        elif x>=maxx:
            done = True
        else:
            x1 = minx + (x - minx)
            y1 = maxy - (y - miny)
            point = Point(x1,y1)
            if object.contains(point) and object.exterior.distance(point)>1e-4:
                done = True
                x = x1
                y = y1
            elif x>=maxx:
                done = True 
    return x,y

def plot_line(ax, ob, color=GRAY):
    parts = hasattr(ob, 'geoms') and ob or [ob]
    for part in parts:
        x, y = part.xy
        ax.plot(x, y, color=color, linewidth=3, solid_capstyle='round', zorder=1)

def generate_geometries(length_0, A, TE_displacement, N1, N2, thickness, x_to_track = []): 
    # new chord
    current_chord = calculate_c(length_0, A, TE_displacement, N1, N2)
    
    # Get coordinates for specific points in the neutral line
    tracked = {'x':[],'y':[]}
    for x in x_to_track:
        tracked['y'].append(current_chord*calculate_deformed_psi(x/length_0, length_0, current_chord, A, TE_displacement)[0])
    tracked['x'] = CST(tracked['y'], current_chord, TE_displacement, Au=A, N1 = N1, N2 = N2)
    
    # Get offset values for left side
    left_tracked = {'x':[],'y':[]}
    for i in range(len(x_to_track)):
        dxi = dxi_u(tracked['y'][i]/current_chord, A, TE_displacement/current_chord,N1,N2)
        xi_component = -dxi/(dxi**2 + 1)**.5
        psi_component = 1/(dxi**2 + 1)**.5

        left_tracked['x'].append(tracked['x'][i] - thickness/2*psi_component)
        left_tracked['y'].append(tracked['y'][i] - thickness/2*xi_component)

    # Get offset values for right side
    right_tracked = {'x':[],'y':[]}
    for i in range(len(x_to_track)):
        dxi = dxi_u(tracked['y'][i]/current_chord, A, TE_displacement/current_chord,N1,N2)
        xi_component = -dxi/(dxi**2 + 1)**.5
        psi_component = 1/(dxi**2 + 1)**.5

        right_tracked['x'].append(tracked['x'][i] + thickness/2*psi_component)
        right_tracked['y'].append(tracked['y'][i] + thickness/2*xi_component)
    
    # non-dimensional y coordinates (extend it a little on the bottom to guarantee nice shape)
    psi = np.linspace(-.1,current_chord,1000)

    # non-dimensional x coordinates
    xi = CST(psi, current_chord, TE_displacement, Au=A, N1 = N1, N2 = N2)
   
    fig = plt.figure(1, figsize=SIZE, dpi=90)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    
    # Genrate neutral line
    line = LineString(zip(xi,psi))

    # Create offsets to neutral line
    offset_left = line.parallel_offset(thickness/2., 'left', join_style=1)
    offset_right = line.parallel_offset(thickness/2., 'right', join_style=1)

    # Create baffle
    coords = (offset_left.coords[:] + offset_right.coords[::] + [offset_left.coords[0]])
    main = Polygon(coords)
    
    # Remove base material
    coords = ((-1,0),(-1,-1),(1,-1),(1,0),(-1,0))
    root = Polygon(coords)
    main = main.difference(root)
    
    # Plot tracked points
    plt.scatter(right_tracked['x'],right_tracked['y'],c='c')
    plt.scatter(left_tracked['x'],left_tracked['y'],c='g')
    plt.scatter([0]*len(x_to_track),x_to_track, c='b')
    plt.scatter(tracked['x'],tracked['y'], c='r')

    # Plot geometry
    skin_patch = PolygonPatch(main, facecolor=	'#808080', edgecolor='#808080', alpha=0.5, zorder=2)
    ax.add_patch(skin_patch)
    plt.xlabel('x', fontsize = 14)
    plt.ylabel('y', fontsize = 14)
    plt.grid()
    plt.show() 
   
    # Get the flag
    flags = []
    x,y = find_point_inside(main)
    flags.append((x,y))  
    # Export data
    data_main = extract_poly_coords(main)
    
    data = {'model':data_main, 'flags':flags, 'left':left_tracked, 'right':right_tracked}
    
    output_file = 'curves.p'
    fileObject = open(output_file,'wb')
    pickle.dump(data, fileObject)
    fileObject.close()
    
def generate_geometries_animate(length_0, A, TE_displacement, N1, N2, thickness, x_to_track = []): 
    # new chord
    current_chord = calculate_c(length_0, A, TE_displacement, N1, N2)
    
    # Get coordinates for specific points in the neutral line
    tracked = {'x':[],'y':[]}
    for x in x_to_track:
        tracked['y'].append(current_chord*calculate_deformed_psi(x/length_0, length_0, current_chord, A, TE_displacement)[0])
    tracked['x'] = CST(tracked['y'], current_chord, TE_displacement, Au=A, N1 = N1, N2 = N2)
    
    # Get offset values for left side
    left_tracked = {'x':[],'y':[]}
    for i in range(len(x_to_track)):
        dxi = dxi_u(tracked['y'][i]/current_chord, A, TE_displacement/current_chord)
        xi_component = -dxi/(dxi**2 + 1)**.5
        psi_component = 1/(dxi**2 + 1)**.5

        left_tracked['x'].append(tracked['x'][i] - thickness/2*psi_component)
        left_tracked['y'].append(tracked['y'][i] - thickness/2*xi_component)

    # Get offset values for right side
    right_tracked = {'x':[],'y':[]}
    for i in range(len(x_to_track)):
        dxi = dxi_u(tracked['y'][i]/current_chord, A, TE_displacement/current_chord)
        xi_component = -dxi/(dxi**2 + 1)**.5
        psi_component = 1/(dxi**2 + 1)**.5

        right_tracked['x'].append(tracked['x'][i] + thickness/2*psi_component)
        right_tracked['y'].append(tracked['y'][i] + thickness/2*xi_component)
    
    # non-dimensional y coordinates (extend it a little on the bottom to guarantee nice shape)
    psi = np.linspace(-.1,current_chord,1000)

    # non-dimensional x coordinates
    xi = CST(psi, current_chord, TE_displacement, Au=A, N1 = N1, N2 = N2)
    
    # Genrate neutral line
    line = LineString(zip(xi,psi))

    # Create offsets to neutral line
    offset_left = line.parallel_offset(thickness/2., 'left', join_style=1)
    offset_right = line.parallel_offset(thickness/2., 'right', join_style=1)

    # Create baffle
    coords = (offset_left.coords[:] + offset_right.coords[::] + [offset_left.coords[0]])
    main = Polygon(coords)
    
    # Remove base material
    coords = ((-1,0),(-1,-1),(1,-1),(1,0),(-1,0))
    root = Polygon(coords)
    main = main.difference(root)
    
    # 3Plot tracked points
    # plt.scatter(right_tracked['x'],right_tracked['y'],c='c')
    # plt.scatter(left_tracked['x'],left_tracked['y'],c='g')
    # plt.scatter([0]*len(x_to_track),x_to_track, c='b')
    # plt.scatter(tracked['x'],tracked['y'], c='r')

    # Plot geometry
    skin_patch = PolygonPatch(main, facecolor=	'#909090', edgecolor='#909090', alpha=0.5, zorder=2)

    return skin_patch

if __name__ == '__main__':
    # Class coefficients (fixed)
    N1 = 1.
    N2 = 1.

    # Initial length
    length_0 = 1.

    # Coordinates to track
    x_to_track = [.05,.1,.2,.3,.4,.5,.6,.7,.8,.9]
    # Delta x
    TE_displacement = 0.5

    # Shape coefficients
    A = [-TE_displacement,-.9, .8]

    # Beam thickness
    beam_thickness = 0.05
    generate_geometries(length_0, A, TE_displacement, N1, N2, beam_thickness, x_to_track)