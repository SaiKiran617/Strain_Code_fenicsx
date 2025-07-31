# -*- coding: iso-8859-1 -*-
import numpy as np
import dolfinx
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import XDMFFile
from dolfinx.geometry import bb_tree, compute_collisions_points
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import meshio
import gmsh
import sys
import time
import psutil
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt



# Create communicator
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# ------ Material Properties ------
# Aluminum properties (Gates)
E_Al = 70e3      # Young's modulus (MPa)
nu_Al = 0.35     # Poisson's ratio
alpha_Al = 23.1e-6  # Thermal expansion coefficient (1/K)

# Germanium properties (Well)
E_Ge = 103e3     # Young's modulus (MPa)
nu_Ge = 0.26     # Poisson's ratio
alpha_Ge = 5.9e-6  # Thermal expansion coefficient (1/K)

# GeSi properties (Ge0.8Si0.2 barrier)
E_GeSi = 115e3   # Young's modulus (MPa)
nu_GeSi = 0.27   # Poisson's ratio
alpha_GeSi = 4.5e-6  # Thermal expansion coefficient (1/K) - between Si and Ge

# Aluminum Oxide properties (Insulation)
E_Al2O3 = 370e3  # Young's modulus (MPa)
nu_Al2O3 = 0.22  # Poisson's ratio
alpha_Al2O3 = 8.1e-6  # Thermal expansion coefficient (1/K)

# ------ Temperature Settings ------
# Temperature change (from room temp to near zero Kelvin)
T_room = 300  # Room temperature (K)
T_al2o3 = 550  # AL2O3 temperature (K)
T_cold = 0     # Near zero Kelvin (K)
delta_T = T_cold - T_room  # Temperature change (negative)
delta_T2 = T_cold - T_al2o3

# ------ Geometry Parameters ------


# Layer thicknesses
thickness_GeSi = 50       # GeSi barrier thickness (50 nm)
thickness_GeSi_l = 42       # GeSi barrier thickness (50 nm)
thickness_Ge = 16         # Ge well thickness (16 nm)
thickness_Al = 20         # Al gates thickness (20 nm)
thickness_Al2O3 = 5       # Al2O3 insulation thickness (5 nm)

# Gate dimensions
c_gate_diameter = 100     # Central gate diameter (100 nm)
gate_separation = 20      # Gate separation (20 nm)
side_gate_width = 80      # Side gate width (estimated)

# Total thickness of the model
total_thickness = thickness_GeSi + thickness_Ge + thickness_Al + thickness_GeSi_l +2*thickness_Al2O3


if rank == 0:
    print("Thermal Stress Analysis of Ge/GeSi Heterostructure Device")
    print(f"Temperature change: {delta_T:.1f} K (from {T_room:.1f} K to {T_cold:.1f} K)")




def create_and_save_mesh(device_length, multiplier,mesh_multipiler):
    """Create the quantum device mesh as a single unified object and save it directly as XDMF."""
    # Initialize gmsh
    gmsh.initialize(sys.argv)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("quantum_device_3d")
    
    # Dimensions
    lw = 16  # Ge well thickness
    barrier_thickness = 50  # Upper GeSi barrier thickness
    gate_thickness = 20  # Al gate thickness
    insulator_thickness = 5  # Al2O3 insulator thickness
    central_gate_diameter = 100  # Central gate diameter
    gate_separation = 20  # Separation between central and side gates
    device_extent = device_length  # Total width/length of the model domain
    lower_barrier = 42  # lower GeSi barrier thickness

    # Key interface z-coordinates
    bottom_interface_z = -(barrier_thickness + lw)  # z = -66
    middle_interface_z = -barrier_thickness  # z = -50
    top_interface_z = 0  # z = 0
    
    #gate dimensions
    gate_length = (device_extent - central_gate_diameter - 2 * insulator_thickness - 2 * gate_separation) / 2 
    gate_width = (300 - central_gate_diameter - 2 * insulator_thickness - 2 * gate_separation) / 2 - insulator_thickness

    # Mesh size
    coarse_mesh_size = 20  # Increased for areas away from interfaces
    standard_mesh_size = 10  # Standard mesh size
    fine_mesh_size = 5  # Finer mesh at interfaces
    ultra_fine_mesh_size = 3  # Ultra fine mesh at critical regions
    central_fine_mesh_size = 1*mesh_multipiler  # Very fine mesh for central region
    central_refinement_radius = 600  # Radius of central fine region in nm
    


    # Create substrate (Ge/GeSi heterostructure) - Z=0 at the top of the substrate
    # GeSi barrier
    barrier = gmsh.model.occ.addBox(-device_extent/2, -device_extent/2, -barrier_thickness, 
                                    device_extent, device_extent, barrier_thickness)

    # GeSi lower barrier
    lower_barrier_vol = gmsh.model.occ.addBox(-device_extent/2, -device_extent/2, -(barrier_thickness+lw+lower_barrier), 
                                    device_extent, device_extent, lower_barrier)
                                    
    # Ge well
    well = gmsh.model.occ.addBox(-device_extent/2, -device_extent/2, -(barrier_thickness + lw), 
                                device_extent, device_extent, lw)

    # Add insulator layer (Al2O3)
    insulator = gmsh.model.occ.addBox(-device_extent/2, -device_extent/2, 0, 
                                    device_extent, device_extent, insulator_thickness)

    # Create central circular gate (C)
    central_gate_radius = (central_gate_diameter + 2*insulator_thickness) / 2
    central_circle = gmsh.model.occ.addDisk(0, 0, insulator_thickness, 
                                          central_gate_radius, central_gate_radius)
    central_gate = gmsh.model.occ.extrude([(2, central_circle)], 0, 0, gate_thickness + insulator_thickness)[1][1]

    # Create side gates
    # Left gate (L)
    l_gate_box = gmsh.model.occ.addBox(-device_extent/2, -gate_width/2, insulator_thickness, 
                                      gate_length, gate_width, gate_thickness+insulator_thickness)

    # Right gate (R)
    r_gate_box = gmsh.model.occ.addBox(device_extent/2 - gate_length, -gate_width/2, insulator_thickness,
                                      gate_length, gate_width, gate_thickness+insulator_thickness)

    # Top gate (T)
    t_gate_box = gmsh.model.occ.addBox(-gate_width/2, device_extent/2 - gate_length, insulator_thickness, 
                                      gate_width, gate_length, gate_thickness+insulator_thickness)

    # Bottom gate (B)
    b_gate_box = gmsh.model.occ.addBox(-gate_width/2, -device_extent/2, insulator_thickness, 
                                      gate_width, gate_length, gate_thickness+insulator_thickness)

    # Synchronize before boolean operations
    gmsh.model.occ.synchronize()
    
    # Fuse all components into a single coherent object while preserving volume identities
    # First, fuse the substrate components (GeSi barrier and Ge well)
    substrate_parts = [(3, barrier), (3, well)]
    substrate, substrate_map = gmsh.model.occ.fuse(substrate_parts, [(3, lower_barrier_vol)], removeObject=True, removeTool=True)
    
    # Then fuse the insulator to the substrate
    device, device_map = gmsh.model.occ.fuse(substrate, [(3, insulator)], removeObject=True, removeTool=True)
    
    # Finally, fuse all gate structures to the device
    gate_parts = [(3, central_gate), (3, l_gate_box), (3, r_gate_box), (3, t_gate_box), (3, b_gate_box)]
    unified_device, unified_map = gmsh.model.occ.fuse(device, gate_parts, removeObject=True, removeTool=True)
    
    # Synchronize again after boolean operations
    gmsh.model.occ.synchronize()
    
    # ===== Enhanced mesh refinement strategy =====
    
    # 1. Create distance field from interfaces (keeping original strategy)
    gmsh.model.mesh.field.add("Distance", 1)
    surfaces = gmsh.model.getEntities(2)
    internal_surfaces = []
    
    # Find internal surfaces (interfaces between different materials)
    for s in surfaces:
        # Get adjacent volumes to this surface
        adjacent_volumes = gmsh.model.getAdjacencies(2, s[1])[1]
        
        # If this surface is adjacent to more than one volume, it's an interface
        if len(adjacent_volumes) > 1:
            internal_surfaces.append(s[1])
    
    # Set the surfaces for distance calculation
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", internal_surfaces)

    # 2. Create distance fields specifically for each critical z-plane (keeping original strategy)
    # Distance field for bottom interface (z = -66)
    gmsh.model.mesh.field.add("MathEval", 2)
    gmsh.model.mesh.field.setString(2, "F", f"abs(z - ({bottom_interface_z}))")
    
    # Distance field for middle interface (z = -50)
    gmsh.model.mesh.field.add("MathEval", 3)
    gmsh.model.mesh.field.setString(3, "F", f"abs(z - ({middle_interface_z}))")
    
    # Distance field for top interface (z = 0)
    gmsh.model.mesh.field.add("MathEval", 4)
    gmsh.model.mesh.field.setString(4, "F", f"abs(z - ({top_interface_z}))")
    
    # Distance field for area above top interface
    gmsh.model.mesh.field.add("MathEval", 5)
    gmsh.model.mesh.field.setString(5, "F", f"max(0, -z)")  # Equals 0 when z = 0, and |z| when z < 0
    
    # 3. NEW: Create radial distance field from center (0,0,0)
    gmsh.model.mesh.field.add("MathEval", 13)
    gmsh.model.mesh.field.setString(13, "F", f"sqrt(x^2 + y^2 + z^2)")  # Radial distance from origin
    
    # 4. Create threshold fields for each critical zone (keeping original strategy)
    # Threshold for bottom interface
    gmsh.model.mesh.field.add("Threshold", 6)
    gmsh.model.mesh.field.setNumber(6, "IField", 2)
    gmsh.model.mesh.field.setNumber(6, "LcMin", standard_mesh_size)
    gmsh.model.mesh.field.setNumber(6, "LcMax", standard_mesh_size)
    gmsh.model.mesh.field.setNumber(6, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(6, "DistMax", standard_mesh_size)
    
    # Threshold for middle interface
    gmsh.model.mesh.field.add("Threshold", 7)
    gmsh.model.mesh.field.setNumber(7, "IField", 3)
    gmsh.model.mesh.field.setNumber(7, "LcMin", standard_mesh_size)
    gmsh.model.mesh.field.setNumber(7, "LcMax", standard_mesh_size)
    gmsh.model.mesh.field.setNumber(7, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(7, "DistMax", standard_mesh_size)
    
    # Threshold for top interface
    gmsh.model.mesh.field.add("Threshold", 8)
    gmsh.model.mesh.field.setNumber(8, "IField", 4)
    gmsh.model.mesh.field.setNumber(8, "LcMin", fine_mesh_size)  
    gmsh.model.mesh.field.setNumber(8, "LcMax", fine_mesh_size)
    gmsh.model.mesh.field.setNumber(8, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(8, "DistMax", standard_mesh_size)
    
    # Threshold for area above top interface
    gmsh.model.mesh.field.add("Threshold", 9)
    gmsh.model.mesh.field.setNumber(9, "IField", 5)
    gmsh.model.mesh.field.setNumber(9, "LcMin", ultra_fine_mesh_size)
    gmsh.model.mesh.field.setNumber(9, "LcMax", fine_mesh_size)
    gmsh.model.mesh.field.setNumber(9, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(9, "DistMax", 10)  # Refine up to 10 units above
    
    # 5. NEW: Threshold for central radial refinement (very fine mesh within 200nm radius)
    gmsh.model.mesh.field.add("Threshold", 14)
    gmsh.model.mesh.field.setNumber(14, "IField", 13)  # Use radial distance field
    gmsh.model.mesh.field.setNumber(14, "LcMin", central_fine_mesh_size)  # 0.5nm mesh size
    gmsh.model.mesh.field.setNumber(14, "LcMax", fine_mesh_size)  # Transition to 6nm
    gmsh.model.mesh.field.setNumber(14, "DistMin", 0)  # Start refinement at center
    gmsh.model.mesh.field.setNumber(14, "DistMax", central_refinement_radius)  # End at 200nm radius
    
    # 6. Combine interface threshold fields (keeping original strategy)
    gmsh.model.mesh.field.add("Min", 10)
    gmsh.model.mesh.field.setNumbers(10, "FieldsList", [6, 7, 8, 9])
    
    # 7. Use standard distance field for general internal features (keeping original strategy)
    gmsh.model.mesh.field.add("Threshold", 11)
    gmsh.model.mesh.field.setNumber(11, "IField", 1)
    gmsh.model.mesh.field.setNumber(11, "LcMin", standard_mesh_size)
    gmsh.model.mesh.field.setNumber(11, "LcMax", coarse_mesh_size)
    gmsh.model.mesh.field.setNumber(11, "DistMin", 0)
    gmsh.model.mesh.field.setNumber(11, "DistMax", 25)
    
    # 8. Final minimum field combining ALL refinements (including new central refinement)
    gmsh.model.mesh.field.add("Min", 12)
    gmsh.model.mesh.field.setNumbers(12, "FieldsList", [10, 11, 14])  # Added field 14 for central refinement
    
    # Set the background field
    gmsh.model.mesh.field.setAsBackgroundMesh(12)

    # Generate mesh
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D meshes
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay for 3D meshes
    gmsh.model.mesh.generate(3)  # 3D mesh

    # Save the mesh in Gmsh format
    gmsh.write(f"device{multiplier}.msh")
    
    # Finalize gmsh
    gmsh.finalize()
    
    # Convert to XDMF format using meshio
    convert_to_xdmf(f"device{multiplier}.msh", multiplier)
    
    print(f"Unified mesh created and saved as quantum_device{multiplier}.xdmf")
    print(f"Central region (within {central_refinement_radius}nm radius) has {central_fine_mesh_size}nm mesh size")


def convert_to_xdmf(msh_file,multiplier):
    """Convert the Gmsh mesh to XDP0 = fem.functionspace(domaMF format with proper physical markers."""
    # Read the mesh
    mesh = meshio.read(msh_file)
    
    # Extract cells and points
    cells = mesh.cells_dict
    points = mesh.points
    
    # Create and write the mesh
    if "tetra" in cells:
        # For the main mesh with tetrahedral elements
        tetra_mesh = meshio.Mesh(points=points, cells={"tetra": cells["tetra"]})
        meshio.write(f"device{multiplier}.xdmf", tetra_mesh)
        
    os.remove(f"device{multiplier}.msh")
    print(f"Successfully deleted {msh_file}")
   
    
    
        
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 / 1024


def print_memory_usage(rank, stage="", comm=None):
    """Print total memory usage across all ranks"""
    comm = MPI.COMM_WORLD

    # Get memory usage for current rank
    local_mem_usage = get_memory_usage()
    total_mem_usage = comm.allreduce(local_mem_usage, op=MPI.SUM)
    
        
    if rank == 0:
      print(f"Total memory usage {stage}: {total_mem_usage:.2f} GB")
    
    return 0
    
 
    

def solve_thermal_stress(device_length,multiplier,mesh_multi):
    """Calculate the FEM solution for thermal stress in the quantum device with multi-stage fabrication"""
    
    print_memory_usage(rank, "1")
    
    if rank == 0:
        print("\n----- Multi-Stage FEM Solution for Thermal Stress in Quantum Device -----")
    
    # Read the mesh (collective operation)
    with XDMFFile(comm, f"device{multiplier}.xdmf", "r") as xdmf:
        domain = xdmf.read_mesh(name="Grid")
        tdim = domain.topology.dim
        
    print_memory_usage(rank, "2")
    
    comm.Barrier()
    if rank == 0:
       os.remove(f"device{multiplier}.xdmf")
       os.remove(f"device{multiplier}.h5")
       print(f"Successfully deleted device file")
    
    # Ensure all topology connectivities are created (collective operation)
    domain.topology.create_connectivity(tdim, tdim)
    
    del tdim
    
    # Get local cell indices (cells owned by this processor)
    cell_indices = np.arange(domain.topology.index_map(domain.topology.dim).size_local, dtype=np.int32)
    
    # Get cell midpoints (only for local cells)
    midpoints = mesh.compute_midpoints(domain, domain.topology.dim, cell_indices)
    
    # Define subdomain bounds
    x_min, x_max = -152.0, 152.0
    y_min, y_max = -152.0, 152.0
    # z bounds are complete (no restriction)
    
    # Function to check if a point is within subdomain bounds
    def in_subdomain_bounds(x):
        return (x_min <= x[0] <= x_max) & (y_min <= x[1] <= y_max)
    
    # Find cells that belong to the subdomain (local to each process)
    subdomain_mask = np.array([in_subdomain_bounds(midpoint) for midpoint in midpoints])
    subdomain_cells = cell_indices[subdomain_mask]
    
    del subdomain_mask
    
    # Create subdomain using mesh entities
    # Create cell tags for subdomain
    subdomain_cell_tags = mesh.meshtags(domain, domain.topology.dim, subdomain_cells, 
                                          np.ones(len(subdomain_cells), dtype=np.int32))
        
    print_memory_usage(rank, "mid subdomain mesh")
    
    # Extract subdomain mesh
    subdomain_mesh, subdomain_cell_map, subdomain_vertex_map, _ = mesh.create_submesh(
            domain, domain.topology.dim, subdomain_cell_tags.find(1)
        )

    del subdomain_cell_tags
    del subdomain_cells
    del subdomain_vertex_map
    
    if rank == 0:
        print(f"Created subdomain mesh with {subdomain_mesh.topology.index_map(subdomain_mesh.topology.dim).size_global} cells")
        
    print_memory_usage(rank, "after subdomain mesh")
    
    # Create a function space for material markers (using DG0 elements)
    P0 = fem.functionspace(domain, ("DG", 0)) 
    material_function = fem.Function(P0)
    material_function.name = "material"
    
    print_memory_usage(rank, "4")
    
    # Initialize material values array 
    material_values = np.ones_like(cell_indices, dtype=np.int32)
    
    print_memory_usage(rank, "5")
    
    outterpoint = device_length/2
    # Assign material markers based on z-coordinate of midpoints
    for i, midpoint in enumerate(midpoints):
        x = midpoint[0]
        y = midpoint[1]
        z = midpoint[2]
        
        # GeSi barrier (bottom layer)
        if -108 <= z < -66:
            material_values[i] = 1  # GeSi
        # Ge well (middle layer)  
        elif -66 <= z < -50:
            material_values[i] = 2  # Ge
        # GeSi upper barrier 
        elif -50 <= z < 0:
            material_values[i] = 1  # GeSi
        # Al2O3 insulator layer (bottom)
        elif 0 <= z < 5:
            material_values[i] = 3  # Al2O3
        # Al gates and Al2O3 regions
        elif 5 <= z < 25:
            if np.sqrt(x**2 + y**2) < 50:
                material_values[i] = 4  # Al
            elif -outterpoint < x < -80 and -30 < y < 30:
                material_values[i] = 4  # Al
            elif 80 < x < outterpoint and -30 < y < 30:
                material_values[i] = 4  # Al
            elif -30 < x < 30 and 80 < y < outterpoint:
                material_values[i] = 4  # Al
            elif -30 < x < 30 and -outterpoint < y < -80:
                material_values[i] = 4  # Al
            else:
                material_values[i] = 5  # Al2O3
        # Upper Al2O3 layer        
        elif 25 <= z <= 30:
                material_values[i] = 5  # Al2O3
                
    print_memory_usage(rank, "6")
        
    # Create mesh tags with assigned material values (local to each process)
    material_markers = mesh.meshtags(domain, domain.topology.dim, cell_indices, material_values)
    
    # Copy markers to our material function (local operation)
    material_function.x.array[cell_indices] = material_values
    
    print_memory_usage(rank, "7")
    
    # Create function space for displacements
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    
    # Define strain tensor
    def eps(u):
        return ufl.sym(ufl.grad(u))
    
    # Define fixed point constraint (common for all stages)
    def fixed_point(x):
        return np.isclose(x[0], 0.0, atol=1e-5) & np.isclose(x[1], 0.0, atol=1e-5) & np.isclose(x[2], -66.0, atol=1e-5)

    # Locate boundary vertices at fixed point (collective operation)
    vertices = mesh.locate_entities_boundary(domain, 0, fixed_point)
    point_dofs = fem.locate_dofs_topological(V, 0, vertices)
    bc_point = fem.dirichletbc(np.zeros(3), point_dofs, V)
    bcs = [bc_point]
    
    # Initialize cumulative displacement
    u_cumulative = fem.Function(V)
    u_cumulative.name = "cumulative_displacement"
    
    
    
    if rank == 0:
        print("\n=== STAGE 0: GeSi, Ge, GeSi (300K → 550K) ===")
        
    
    # STAGE 0: Only bottom layers (GeSi, Ge, GeSi) - Heat from 300 to 550K
    stage0_materials = [1, 2]  # GeSi, Ge
    u_stage0 = solve_stage(domain, V, P0, cell_indices, material_values, stage0_materials, 
                          bcs, 250.0, "Stage 0", rank)  # ΔT = 250K (300K to 550K)
    
    # Add stage 0 displacement to cumulative
    u_cumulative.x.array[:] += u_stage0.x.array[:]
    u_cumulative.x.scatter_forward()
    del u_stage0
    
    
    if rank == 0:
        print("\n=== STAGE 1: Bottom Al2O3 deposition (550K → 300K) ===")
        
    
    # STAGE 1: Only bottom layers (GeSi, Ge, GeSi, bottom Al2O3) - Cool from 550K to 300K
    stage1_materials = [1, 2, 3]  # GeSi, Ge, Al2O3 (bottom layer only)
    u_stage1 = solve_stage(domain, V, P0, cell_indices, material_values, stage1_materials, 
                          bcs, -250.0, "Stage 1", rank)  # ΔT = -250K (550K to 300K)
    
    # Add stage 1 displacement to cumulative
    u_cumulative.x.array[:] += u_stage1.x.array[:]
    u_cumulative.x.scatter_forward()
    del u_stage1
    
    if rank == 0:
        print("\n=== STAGE 2: Al gates deposition and heating (300K → 550K) ===")
    
    # STAGE 2: Include Al gates - Heat from 300K to 550K
    stage2_materials = [1, 2, 3, 4]  # GeSi, Ge, Al2O3 (bottom), Al
    u_stage2 = solve_stage(domain, V, P0, cell_indices, material_values, stage2_materials, 
                          bcs, 250.0, "Stage 2", rank)  # ΔT = +250K (300K to 550K)
    
    # Add stage 2 displacement to cumulative
    u_cumulative.x.array[:] += u_stage2.x.array[:]
    u_cumulative.x.scatter_forward()
    del u_stage2
    
    if rank == 0:
        print("\n=== STAGE 3: Upper Al2O3 deposition and final cooling (550K → 0K) ===")
    
    # STAGE 3: Include all materials - Cool from 550K to 0K
    stage3_materials = [1, 2, 3, 4, 5]  # All materials (upper Al2O3 same as material 3)
    u_stage3 = solve_stage(domain, V, P0, cell_indices, material_values, stage3_materials, 
                          bcs, -550.0, "Stage 3", rank)  # ΔT = -550K (550K to 0K)
    
    # Add stage 3 displacement to cumulative
    u_cumulative.x.array[:] += u_stage3.x.array[:]
    u_cumulative.x.scatter_forward()
    del u_stage3
    
    
    
    if rank == 0:
        print("\n=== Multi-stage fabrication completed ===")
        print("Final cumulative displacement calculated")
    
    # Use cumulative displacement for final analysis
    u_sol = u_cumulative
    
    del cell_indices
    del midpoints
    
    print_memory_usage(rank, "after all stages")
    
    # Continue with strain and stress calculations using the cumulative displacement...
    # [Rest of the original code for strain calculations and output remains the same]
    
    # Calculate strains and stresses using cumulative displacement
    S = fem.functionspace(subdomain_mesh, ("DG", 0))
    V_sub = fem.functionspace(subdomain_mesh, ("Lagrange", 1, (subdomain_mesh.geometry.dim,)))
    
    # Extract individual strain components
    strain_xx = fem.Function(P0)
    strain_xx_expr = fem.Expression(eps(u_sol)[0, 0], P0.element.interpolation_points())
    strain_xx.interpolate(strain_xx_expr)
    strain_xx.name = "strain_xx"
    
    strain_yy = fem.Function(P0)
    strain_yy_expr = fem.Expression(eps(u_sol)[1, 1], P0.element.interpolation_points())
    strain_yy.interpolate(strain_yy_expr)
    strain_yy.name = "strain_yy"
    
    strain_xz = fem.Function(P0)
    strain_xz_expr = fem.Expression(eps(u_sol)[0, 2], P0.element.interpolation_points())
    strain_xz.interpolate(strain_xz_expr)
    strain_xz.name = "strain_xz"
    
    # Calculate combined strain components
    strain_xx_plus_yy = fem.Function(P0)
    strain_xx_plus_yy_expr = fem.Expression(0.5*(eps(u_sol)[0, 0] + eps(u_sol)[1, 1]), P0.element.interpolation_points())
    strain_xx_plus_yy.interpolate(strain_xx_plus_yy_expr)
    strain_xx_plus_yy.name = "strain_xx_plus_yy"    

    # Synchronize all functions
    material_function.x.scatter_forward()
    u_sol.x.scatter_forward()
    strain_xz.x.scatter_forward()
    strain_xx_plus_yy.x.scatter_forward()
    
    # Create functions on subdomain
    material_sub = fem.Function(S)
    material_sub.name = "material"
        
    u_sol_sub = fem.Function(V_sub)
    u_sol_sub.name = "displacement"
        
    strain_xz_sub = fem.Function(S)
    strain_xz_sub.name = "strain_xz"

    strain_xx_plus_yy_sub = fem.Function(S)
    strain_xx_plus_yy_sub.name = "strain_xx_plus_yy"

    # Map data from original functions to subdomain functions
    material_sub.x.array[:] = material_function.x.array[subdomain_cell_map]
    strain_xz_sub.x.array[:] = strain_xz.x.array[subdomain_cell_map]
    strain_xx_plus_yy_sub.x.array[:] = strain_xx_plus_yy.x.array[subdomain_cell_map]
        
    # Interpolate displacement to subdomain
    u_sol_sub.interpolate(u_sol)
        
    # Synchronize subdomain functions
    material_sub.x.scatter_forward()
    u_sol_sub.x.scatter_forward()
    strain_xz_sub.x.scatter_forward()
    strain_xx_plus_yy_sub.x.scatter_forward()
        
    #print_memory_usage(rank, "before xdmf file")
        
    # Write subdomain output (collective operation)
    #with XDMFFile(comm, f"results_{multiplier}x.xdmf", "w") as xdmf:
    #        xdmf.write_mesh(subdomain_mesh)
    #        xdmf.write_function(material_sub)
    #        xdmf.write_function(u_sol_sub)
    #        xdmf.write_function(strain_xz_sub)
    #        xdmf.write_function(strain_xx_plus_yy_sub)
    
    #if rank == 0:
    #    print("Multi-stage thermal strain analysis completed.")
    #    print(f"Results saved in results{multiplier}.xdmf for visualization in ParaView.")
    
    # Generate 2D plots using subdomain data
    print_memory_usage(rank, "before plotting")
    
    # Get cell midpoints for subdomain mesh
    subdomain_cell_indices = np.arange(subdomain_mesh.topology.index_map(subdomain_mesh.topology.dim).size_local, dtype=np.int32)
    subdomain_midpoints = mesh.compute_midpoints(subdomain_mesh, subdomain_mesh.topology.dim, subdomain_cell_indices)
    
    # Extract coordinates and strain values
    x_coords_local = subdomain_midpoints[:, 0]
    y_coords_local = subdomain_midpoints[:, 1]
    z_coords_local = subdomain_midpoints[:, 2]
    strain_xz_values_local = strain_xz_sub.x.array[:]
    strain_xx_plus_yy_values_local = strain_xx_plus_yy_sub.x.array[:]
    
    # Gather data from all ranks to rank 0
    x_coords_all = comm.gather(x_coords_local, root=0)
    y_coords_all = comm.gather(y_coords_local, root=0)
    z_coords_all = comm.gather(z_coords_local, root=0)
    strain_xz_values_all = comm.gather(strain_xz_values_local, root=0)
    strain_xx_plus_yy_values_all = comm.gather(strain_xx_plus_yy_values_local, root=0)
    
    if rank == 0:
        print("\n----- Generating 2D Plots -----")
        
        # Concatenate data from all ranks
        x_coords = np.concatenate(x_coords_all)
        y_coords = np.concatenate(y_coords_all)
        z_coords = np.concatenate(z_coords_all)
        strain_xz_values = np.concatenate(strain_xz_values_all)
        strain_xx_plus_yy_values = np.concatenate(strain_xx_plus_yy_values_all)
        
        print(f"Total points for plotting: {len(x_coords)}")
        
        # Plot generation code (same as original)...
        # [Include the original plotting code here]
        
        try:
            import matplotlib.pyplot as plt
            from scipy.interpolate import griddata
            
            # Plot 1: XZ plane at y = 0, with strain 1/2*(xx + yy)
            y_tolerance = 5.0
            y_zero_mask = np.abs(y_coords) <= y_tolerance
            
            if np.any(y_zero_mask):
                x_plot1 = x_coords[y_zero_mask]
                z_plot1 = z_coords[y_zero_mask]
                strain_plot1 = 0.5 * strain_xx_plus_yy_values[y_zero_mask]
                
                print(f"Plot 1 points: {len(x_plot1)}")
                
                x_grid = np.linspace(-150, 150, 200)
                z_grid = np.linspace(np.min(z_plot1), np.max(z_plot1), 100)
                X_grid, Z_grid = np.meshgrid(x_grid, z_grid)
                
                strain_grid1 = griddata((x_plot1, z_plot1), strain_plot1, (X_grid, Z_grid), method='linear')
                
                plt.figure(figsize=(10, 6))
                contour1 = plt.contourf(X_grid, Z_grid, strain_grid1, levels=50, cmap='RdBu_r')
                plt.colorbar(contour1, label='1/2*(eps_xx+eps_yy)')
                plt.xlabel('X coordinate')
                plt.ylabel('Z coordinate')
                plt.title('XZ plane at y=0, 1/2*(eps_xx+eps_yy) - Multi-stage')
                plt.xlim(-150, 150)
                plt.tight_layout()
                plt.savefig(f'plot1_xz_eps_xx_plus_yy_d{multiplier}x_m{mesh_multi}x.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                # Plot 2: XZ plane at y = 0, with strain xz
                strain_plot2 = strain_xz_values[y_zero_mask]
                strain_grid2 = griddata((x_plot1, z_plot1), strain_plot2, (X_grid, Z_grid), method='linear')
                
                plt.figure(figsize=(10, 6))
                contour2 = plt.contourf(X_grid, Z_grid, strain_grid2, levels=50, cmap='RdBu_r')
                plt.colorbar(contour2, label='eps_xz')
                plt.xlabel('X coordinate')
                plt.ylabel('Z coordinate')
                plt.title('XZ plane at y=0, eps_xz - Multi-stage')
                plt.xlim(-150, 150)
                plt.tight_layout()
                plt.savefig(f'plot2_xz_eps_xz_d{multiplier}x_m{mesh_multi}x.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Plot 3: XY plane at z = -58, with strain xz
            z_target = -58.0
            z_tolerance = 2.0
            z_target_mask = np.abs(z_coords - z_target) <= z_tolerance
            
            if np.any(z_target_mask):
                x_plot3 = x_coords[z_target_mask]
                y_plot3 = y_coords[z_target_mask]
                strain_plot3 = strain_xz_values[z_target_mask]
                
                print(f"Plot 3 points: {len(x_plot3)}")
                
                x_grid_xy = np.linspace(-150, 150, 200)
                y_grid_xy = np.linspace(-150, 150, 200)
                X_grid_xy, Y_grid_xy = np.meshgrid(x_grid_xy, y_grid_xy)
                
                strain_grid3 = griddata((x_plot3, y_plot3), strain_plot3, (X_grid_xy, Y_grid_xy), method='linear')
                
                plt.figure(figsize=(8, 8))
                contour3 = plt.contourf(X_grid_xy, Y_grid_xy, strain_grid3, levels=50, cmap='RdBu_r')
                plt.colorbar(contour3, label='eps_xz')
                plt.xlabel('X coordinate')
                plt.ylabel('Y coordinate')
                plt.title('XY plane at z=-58, eps_xz - Multi-stage')
                plt.xlim(-150, 150)
                plt.ylim(-150, 150)
                plt.gca().set_aspect('equal')
                plt.tight_layout()
                plt.savefig(f'plot3_xy_eps_xz_d{multiplier}x_m{mesh_multi}x.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            print("Multi-stage matplotlib plots saved as PNG files.")
            
        except ImportError:
            print("Matplotlib not available. Only numpy data files were saved.")
    
    # Clean up local arrays to save memory
    del x_coords_local, y_coords_local, z_coords_local
    del strain_xz_values_local, strain_xx_plus_yy_values_local
    del subdomain_cell_indices, subdomain_midpoints
    
    print_memory_usage(rank, "after plotting")
    
    # Sync all processes before returning
    comm.Barrier()
    
    return


def solve_stage(domain, V, P0, cell_indices, material_values, active_materials, bcs, delta_T, stage_name, rank):
    """
    Solve thermal stress for a specific fabrication stage
    
    Parameters:
    - domain: FEniCS mesh domain
    - V: Function space for displacements  
    - P0: Function space for material properties
    - cell_indices: Local cell indices
    - material_values: Array of material IDs for each cell
    - active_materials: List of material IDs to include in this stage
    - bcs: Boundary conditions
    - delta_T: Temperature change for this stage
    - stage_name: Name for logging
    - rank: MPI rank
    """
    
    if rank == 0:
        print(f"Solving {stage_name} with ΔT = {delta_T}K")
        print(f"Active materials: {active_materials}")
    
    # Create material property functions for this stage
    E_func = fem.Function(P0)
    nu_func = fem.Function(P0)
    alpha_func = fem.Function(P0)
    
    # Temperature function for this stage
    T_func = fem.Function(P0)
    T_func.name = f"temperature_change_{stage_name}"
    
    # Assign material properties and temperature only for active materials
    for i, material_id in enumerate(material_values):
        if material_id in active_materials:
            # Assign material properties
            if material_id == 1:  # GeSi
                E_func.x.array[i] = E_GeSi
                nu_func.x.array[i] = nu_GeSi
                alpha_func.x.array[i] = alpha_GeSi
                T_func.x.array[i] = delta_T
            elif material_id == 2:  # Ge
                E_func.x.array[i] = E_Ge
                nu_func.x.array[i] = nu_Ge
                alpha_func.x.array[i] = alpha_Ge
                T_func.x.array[i] = delta_T
            elif material_id == 3:  # Al2O3
                E_func.x.array[i] = E_Al2O3
                nu_func.x.array[i] = nu_Al2O3
                alpha_func.x.array[i] = alpha_Al2O3
                T_func.x.array[i] = delta_T
            elif material_id == 4:  # Al
                E_func.x.array[i] = E_Al
                nu_func.x.array[i] = nu_Al
                alpha_func.x.array[i] = alpha_Al
                T_func.x.array[i] = delta_T
            elif material_id == 5:  # Al2O3
                E_func.x.array[i] = E_Al2O3
                nu_func.x.array[i] = nu_Al2O3
                alpha_func.x.array[i] = alpha_Al2O3
                T_func.x.array[i] = delta_T
        else:
            # Inactive materials: set zero temperature change
            if material_id == 3:  # Al2O3
                E_func.x.array[i] = E_Al2O3
                nu_func.x.array[i] = nu_Al2O3
                alpha_func.x.array[i] = 0.0
                T_func.x.array[i] = 0.0
            elif material_id == 4:  # Al
                E_func.x.array[i] = E_Al
                nu_func.x.array[i] = nu_Al
                alpha_func.x.array[i] = 0.0
                T_func.x.array[i] = 0.0
            elif material_id == 5:  # Al2O3
                E_func.x.array[i] = E_Al2O3
                nu_func.x.array[i] = nu_Al2O3
                alpha_func.x.array[i] = 0.0
                T_func.x.array[i] = 0.0
            
    
    # Synchronize function values across processes
    E_func.x.scatter_forward()
    nu_func.x.scatter_forward()
    alpha_func.x.scatter_forward()
    T_func.x.scatter_forward()
    
    # Calculate Lamé parameters
    mu = E_func / (2.0 * (1.0 + nu_func))
    lmbda = E_func * nu_func / ((1.0 + nu_func) * (1.0 - 2.0 * nu_func))
    
    # Define strain tensor
    def eps(u):
        return ufl.sym(ufl.grad(u))
    
    # Define mechanical stress
    def sigma_mech(e):
        return 2.0 * mu * e + lmbda * ufl.tr(e) * ufl.Identity(domain.topology.dim)
    
    # Define variational problem
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = ufl.inner(sigma_mech(eps(u)), eps(v)) * ufl.dx
    
    # Thermal stress term
    L = ufl.inner((3 * lmbda + 2 * mu) * alpha_func * T_func * ufl.Identity(domain.topology.dim), eps(v)) * ufl.dx
    
    # Create function for solution
    u_sol = fem.Function(V)
    u_sol.name = f"displacement_{stage_name}"
    
    # Set up and solve problem
    problem = LinearProblem(a, L, bcs=bcs, u=u_sol, petsc_options={
        "ksp_type": "gmres", 
        "pc_type": "gamg",
        "ksp_rtol": 1e-10,
        "ksp_atol": 1e-10,
        "ksp_max_it": 10000000000
    })
    
    u_sol = problem.solve()
    
    if rank == 0:
        print(f"{stage_name} solved successfully")
    
    # Clean up
    del problem, a, L, u, v
    del E_func, nu_func, alpha_func, T_func, mu, lmbda
    
    return u_sol


    
    

# ------ Main function ------
def main():
    """Main function to run the thermal stress analysis and visualization"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("\n========== Thermal Stress Analysis of Quantum Device with Multiple Domain Sizes ==========")
    
    # Define domain size multipliers to test
    domain_multipliers = [30]
    mesh_multipiler = 1

    
    # Lists to store results for each domain size
    all_results = []
    
    # Loop through different domain sizes
    for multiplier in domain_multipliers:
        start_time = time.perf_counter()
        initial_length = 300
        # Base dimensions
        device_length = initial_length*multiplier   # Length of the device in nm

        if rank == 0:
           print(f"\n--- Testing domain size multiplier: {multiplier} ---")
           print(f"Model dimensions: {device_length:.1f} nm x {device_length:.1f} nm x {total_thickness:.1f} nm")

        # Create mesh (only rank 0 should call gmsh)
        if rank == 0:
          create_and_save_mesh(device_length,multiplier,mesh_multipiler)
        
        # Make sure all processes have the mesh file before proceeding
        comm.barrier()
        
        # Solve the thermal stress problem with the new mesh
        solve_thermal_stress(device_length,multiplier,mesh_multipiler)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        if rank == 0:
          print(f"Execution time: {execution_time:.4f} seconds")

if __name__ == "__main__":
    main()
