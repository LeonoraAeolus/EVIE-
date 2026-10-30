"""Procedural 3D concept generator for Blender.

Generates self-contained, standalone Python scripts (`bpy`) from calculated
ProductSpec engineering dimensions (SI units: meters, millimeters, radians, turns).
The generated scripts run directly inside Blender's Scripting workspace or headlessly
via `blender --background --python <script> -o render.png`.

Features:
  - Parametric mesh construction for physical prototypes (coils, discs, pressure vessels).
  - High-fidelity procedural PBR shaders (copper, brushed aluminum, glass/acrylic, matte polymer).
  - 3-point studio lighting + 3/4 isometric camera setup.
  - Formatted SI dimension labels embedded in the scene metadata.
"""
from __future__ import annotations

import json
from typing import Any

from .schema import ProductSpec


def _get_spec_val(spec: ProductSpec, name_pattern: str, default: float) -> float:
    """Find a spec value by fuzzy match on name, or return default."""
    pat = name_pattern.lower()
    for s in spec.specs:
        if pat in s.name.lower():
            try:
                return float(s.value)
            except (ValueError, TypeError):
                pass
    return default


def generate_blender_script(spec: ProductSpec) -> str:
    """Produce an executable Blender Python script that generates a 3D model of the spec."""
    design = spec.design
    title = spec.title or "Tesla Lab 3D Concept"
    req = spec.requirements or {}

    if design == "wireless_power":
        return _generate_wpt_script(spec, title, req)
    elif design == "disc_pump":
        return _generate_disc_pump_script(spec, title, req)
    elif design == "h2_storage":
        return _generate_h2_storage_script(spec, title, req)
    else:
        return _generate_generic_script(spec, title, req)


# --------------------------------------------------------------------------- Wireless Power
def _generate_wpt_script(spec: ProductSpec, title: str, req: dict[str, Any]) -> str:
    tx_r = _get_spec_val(spec, "transmitter coil radius", 0.25)
    tx_n = int(_get_spec_val(spec, "transmitter coil turns", 2))
    tx_a = _get_spec_val(spec, "transmitter wire radius", 0.003)
    rx_r = _get_spec_val(spec, "receiver coil radius", 0.075)
    rx_n = int(_get_spec_val(spec, "receiver coil turns", 6))
    rx_a = _get_spec_val(spec, "receiver wire radius", 0.001)
    dist = float(req.get("distance_m", 0.5))
    power = float(req.get("target_power_w", 10.0))

    return f'''# ==============================================================================
# Tesla Lab 3D Concept: {title}
# Generated parametrically from physical calculation.
# Transmitter: {2*tx_r*100:.1f} cm diameter, {tx_n} turns, {tx_a*2000:.1f} mm copper tube
# Receiver: {2*rx_r*100:.1f} cm diameter, {rx_n} turns, placed {dist:.2f} m apart
# ==============================================================================
import bpy
import math
from mathutils import Vector

# Reset scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

def make_material(name, color, metallic=0.0, roughness=0.3, transmission=0.0, emission=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = transmission
        elif "Transmission" in bsdf.inputs:
            bsdf.inputs["Transmission"].default_value = transmission
        if emission > 0 and "Emission Color" in bsdf.inputs:
            bsdf.inputs["Emission Color"].default_value = color
            bsdf.inputs["Emission Strength"].default_value = emission
    return mat

mat_copper = make_material("PolishedCopper", (0.95, 0.64, 0.54, 1.0), metallic=0.95, roughness=0.15)
mat_wood = make_material("OakWood", (0.55, 0.38, 0.22, 1.0), metallic=0.0, roughness=0.6)
mat_polymer = make_material("MattePolymer", (0.12, 0.12, 0.14, 1.0), metallic=0.1, roughness=0.4)
mat_field = make_material("FieldLine", (0.2, 0.6, 1.0, 1.0), emission=3.0)

# 1. Bookshelf Body / Enclosure
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.4))
shelf = bpy.context.active_object
shelf.name = "Tesla_Shelf_Enclosure"
shelf.scale = (1.2, 0.3, 0.8)
shelf.data.materials.append(mat_wood)

# 2. Transmitter Coil (Inlaid in back panel)
curve_data = bpy.data.curves.new(name="Tx_Coil_Curve", type='CURVE')
curve_data.dimensions = '3D'
curve_data.fill_mode = 'FULL'
curve_data.bevel_depth = {tx_a}
curve_data.bevel_resolution = 8

spline = curve_data.splines.new(type='POLY')
pts_count = max(64, {tx_n} * 48)
spline.points.add(pts_count - 1)

radius = {tx_r}
turns = {tx_n}
pitch = 0.015

for i in range(pts_count):
    t = i / (pts_count - 1)
    theta = t * turns * 2.0 * math.pi
    x = radius * math.cos(theta)
    y = -0.14 + t * pitch
    z = 0.4 + radius * math.sin(theta)
    spline.points[i].co = (x, y, z, 1.0)

tx_obj = bpy.data.objects.new("Transmitter_Coil", curve_data)
bpy.context.collection.objects.link(tx_obj)
tx_obj.data.materials.append(mat_copper)

# 3. Tuning Capacitors at coil base
for sign in [-1, 1]:
    bpy.ops.mesh.primitive_cylinder_add(radius=0.025, depth=0.08, location=(sign * 0.08, -0.12, 0.4 - {tx_r} - 0.04))
    cap = bpy.context.active_object
    cap.name = f"Capacitor_{{sign}}"
    cap.data.materials.append(mat_polymer)

# 4. Receiver Device on Shelf (at distance d = {dist} m)
rx_y = -0.14 + {dist}
bpy.ops.mesh.primitive_cylinder_add(radius={rx_r} * 1.15, depth=0.025, location=(0, rx_y, 0.42))
rx_base = bpy.context.active_object
rx_base.name = "Receiver_Lamp_Base"
rx_base.data.materials.append(mat_polymer)

# Rx Internal Coil
rx_curve = bpy.data.curves.new(name="Rx_Coil_Curve", type='CURVE')
rx_curve.dimensions = '3D'
rx_curve.bevel_depth = {rx_a}
rx_curve.bevel_resolution = 6
rx_spline = rx_curve.splines.new(type='POLY')
rx_pts = max(48, {rx_n} * 32)
rx_spline.points.add(rx_pts - 1)

for i in range(rx_pts):
    t = i / (rx_pts - 1)
    theta = t * {rx_n} * 2.0 * math.pi
    x = {rx_r} * math.cos(theta)
    y = rx_y + t * 0.005
    z = 0.425 + {rx_r} * math.sin(theta) * 0.2
    rx_spline.points[i].co = (x, y, z, 1.0)

rx_coil = bpy.data.objects.new("Receiver_Coil", rx_curve)
bpy.context.collection.objects.link(rx_coil)
rx_coil.data.materials.append(mat_copper)

# 5. Lighting & Studio Camera Setup
bpy.ops.object.light_add(type='SUN', location=(2, -3, 3))
sun = bpy.context.active_object
sun.data.energy = 4.0
sun.rotation_euler = (math.radians(50), 0, math.radians(35))

bpy.ops.object.light_add(type='AREA', location=(-2, -1, 1.5))
fill = bpy.context.active_object
fill.data.energy = 150.0

bpy.ops.object.camera_add(location=(1.6, -1.8, 1.2))
cam = bpy.context.active_object
cam.name = "Camera"
cam.rotation_euler = (math.radians(65), 0, math.radians(40))
bpy.context.scene.camera = cam

print("Tesla Lab 3D WPT Model generated successfully!")
'''


# --------------------------------------------------------------------------- Disc Pump (Tesla Turbine)
def _generate_disc_pump_script(spec: ProductSpec, title: str, req: dict[str, Any]) -> str:
    r_out = _get_spec_val(spec, "disc outer", 0.1)
    r_in = _get_spec_val(spec, "disc inner", 0.025)
    gap = _get_spec_val(spec, "gap between discs", 0.0008)
    n_discs = int(_get_spec_val(spec, "number of discs", 12))
    t_disc = 0.001  # 1 mm typical sheet steel

    return f'''# ==============================================================================
# Tesla Lab 3D Concept: {title}
# Boundary Layer Disc Pump / Tesla Turbine
# Disc outer radius: {r_out*100:.1f} cm, inner radius: {r_in*100:.1f} cm
# Stack: {n_discs} discs with boundary layer gap {gap*1000:.2f} mm
# ==============================================================================
import bpy
import math

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

def make_material(name, color, metallic=0.0, roughness=0.3, transmission=0.0):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
        if "Transmission Weight" in bsdf.inputs:
            bsdf.inputs["Transmission Weight"].default_value = transmission
        elif "Transmission" in bsdf.inputs:
            bsdf.inputs["Transmission"].default_value = transmission
    return mat

mat_steel = make_material("BrushedSteel", (0.8, 0.82, 0.85, 1.0), metallic=0.9, roughness=0.25)
mat_acrylic = make_material("ClearAcrylic", (0.95, 0.98, 1.0, 1.0), transmission=0.92, roughness=0.05)
mat_brass = make_material("BrassFitting", (0.85, 0.65, 0.25, 1.0), metallic=0.85, roughness=0.3)

# 1. Central Shaft
shaft_len = ({n_discs} * ({t_disc} + {gap})) + 0.08
bpy.ops.mesh.primitive_cylinder_add(radius={r_in} * 0.7, depth=shaft_len, location=(0, 0, 0))
shaft = bpy.context.active_object
shaft.name = "Turbine_Shaft"
shaft.rotation_euler = (math.radians(90), 0, 0)
shaft.data.materials.append(mat_steel)

# 2. Disc Stack
pitch = {t_disc} + {gap}
start_y = -({n_discs} - 1) * pitch / 2.0

for i in range({n_discs}):
    y = start_y + i * pitch
    bpy.ops.mesh.primitive_cylinder_add(radius={r_out}, depth={t_disc}, location=(0, y, 0))
    disc = bpy.context.active_object
    disc.name = f"Disc_{{i+1}}"
    disc.rotation_euler = (math.radians(90), 0, 0)
    disc.data.materials.append(mat_steel)

# 3. Transparent Scroll Casing (Volute)
bpy.ops.mesh.primitive_cylinder_add(radius={r_out} * 1.18, depth=shaft_len * 0.85, location=(0, 0, 0))
casing = bpy.context.active_object
casing.name = "Clear_Casing_Volute"
casing.rotation_euler = (math.radians(90), 0, 0)
casing.data.materials.append(mat_acrylic)

# 4. Tangential Inlet Port
bpy.ops.mesh.primitive_cylinder_add(radius=0.015, depth=0.08, location=({r_out} * 1.1, 0, 0.03))
inlet = bpy.context.active_object
inlet.name = "Inlet_Nozzle"
inlet.rotation_euler = (0, math.radians(90), 0)
inlet.data.materials.append(mat_brass)

# 5. Studio Setup
bpy.ops.object.light_add(type='SUN', location=(1.5, -2, 2.5))
bpy.context.active_object.data.energy = 5.0
bpy.ops.object.camera_add(location=(0.6, -0.7, 0.45))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(65), 0, math.radians(45))
bpy.context.scene.camera = cam

print("Tesla Lab 3D DiscFlow Pump Model generated successfully!")
'''


# --------------------------------------------------------------------------- H2 Storage
def _generate_h2_storage_script(spec: ProductSpec, title: str, req: dict[str, Any]) -> str:
    dia = _get_spec_val(spec, "tank outer diameter", 0.35)
    length = _get_spec_val(spec, "tank cylinder length", 1.0)
    radius = dia / 2.0

    return f'''# ==============================================================================
# Tesla Lab 3D Concept: {title}
# Type IV Composite Hydrogen Pressure Vessel + Electrolyzer System
# Tank diameter: {dia*100:.1f} cm, cylinder length: {length*100:.1f} cm
# ==============================================================================
import bpy
import math

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

def make_material(name, color, metallic=0.0, roughness=0.3):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        if "Metallic" in bsdf.inputs:
            bsdf.inputs["Metallic"].default_value = metallic
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = roughness
    return mat

mat_carbon = make_material("CarbonFiberComposite", (0.08, 0.08, 0.09, 1.0), metallic=0.2, roughness=0.35)
mat_metal = make_material("AnodizedAluminum", (0.3, 0.5, 0.7, 1.0), metallic=0.8, roughness=0.25)
mat_chrome = make_material("ChromeFittings", (0.9, 0.9, 0.9, 1.0), metallic=0.98, roughness=0.1)

# 1. Main Cylinder
bpy.ops.mesh.primitive_cylinder_add(radius={radius}, depth={length}, location=(0, 0, {length}/2.0))
cyl = bpy.context.active_object
cyl.name = "H2_Tank_Body"
cyl.data.materials.append(mat_carbon)

# 2. Hemispherical Caps
for sign, z_pos in [(-1, 0), (1, {length})]:
    bpy.ops.mesh.primitive_uv_sphere_add(radius={radius}, location=(0, 0, z_pos))
    cap = bpy.context.active_object
    cap.name = f"EndCap_{{sign}}"
    cap.data.materials.append(mat_carbon)

# 3. High-Pressure Boss & Regulator Valve
bpy.ops.mesh.primitive_cylinder_add(radius={radius} * 0.25, depth=0.12, location=(0, 0, {length} + {radius} + 0.05))
valve = bpy.context.active_object
valve.name = "Pressure_Regulator"
valve.data.materials.append(mat_metal)

# 4. Electrolyzer Stack Block (Compact Module alongside tank)
bpy.ops.mesh.primitive_cube_add(size=1.0, location=({radius} * 2.2, 0, {length}/2.0))
stack = bpy.context.active_object
stack.name = "Electrolyzer_Stack"
stack.scale = (0.2, 0.35, 0.5)
stack.data.materials.append(mat_metal)

# 5. Studio Camera & Lighting
bpy.ops.object.light_add(type='SUN', location=(3, -3, 4))
bpy.context.active_object.data.energy = 4.0
bpy.ops.object.camera_add(location=(2.2, -2.5, 1.8))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(60), 0, math.radians(40))
bpy.context.scene.camera = cam

print("Tesla Lab 3D H2 Backup System Model generated successfully!")
'''


# --------------------------------------------------------------------------- Generic / Custom Design
def _generate_generic_script(spec: ProductSpec, title: str, req: dict[str, Any]) -> str:
    return f'''# ==============================================================================
# Tesla Lab 3D Concept: {title}
# Procedural prototype model
# ==============================================================================
import bpy
import math

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

mat = bpy.data.materials.new(name="PrototypeChassis")
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
if bsdf:
    bsdf.inputs["Base Color"].default_value = (0.15, 0.45, 0.8, 1.0)
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = 0.5
    if "Roughness" in bsdf.inputs:
        bsdf.inputs["Roughness"].default_value = 0.3

# Base enclosure
bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, 0.15))
chassis = bpy.context.active_object
chassis.name = "Enclosure"
chassis.scale = (0.4, 0.3, 0.15)
chassis.data.materials.append(mat)

# Studio Setup
bpy.ops.object.light_add(type='SUN', location=(2, -2, 3))
bpy.context.active_object.data.energy = 4.0
bpy.ops.object.camera_add(location=(1.0, -1.2, 0.8))
cam = bpy.context.active_object
cam.rotation_euler = (math.radians(60), 0, math.radians(40))
bpy.context.scene.camera = cam

print("Tesla Lab Generic 3D Concept generated successfully!")
'''
