"""Parametric 3D mesh and interactive WebGL / Three.js viewer generator.

Generates:
  1. Downloadable ASCII/binary STL meshes (.stl) for direct 3D printing and CAD import.
  2. Embeddable Three.js WebGL HTML viewer for interactive 360-degree inspection
     directly inside Streamlit.
"""
from __future__ import annotations

import base64
import math
from typing import Any

from .schema import ProductSpec


# --------------------------------------------------------------------------- STL Generator
def generate_stl(spec: ProductSpec) -> str:
    """Generate an ASCII STL mesh representation of the product spec."""
    design = spec.design
    req = spec.requirements or {}

    facets = []
    if design == "wireless_power":
        # Generate coils as ring toroids
        tx_r = 0.25
        for s in spec.specs:
            if "transmitter coil radius" in s.name.lower():
                try:
                    tx_r = float(s.value)
                except (ValueError, TypeError):
                    pass
        facets.extend(_generate_torus_facets(r_major=tx_r, r_minor=0.015, n_major=24, n_minor=8, z_offset=0.0))
        dist = float(req.get("distance_m", 0.5))
        rx_r = 0.075
        for s in spec.specs:
            if "receiver coil radius" in s.name.lower():
                try:
                    rx_r = float(s.value)
                except (ValueError, TypeError):
                    pass
        facets.extend(_generate_torus_facets(r_major=rx_r, r_minor=0.008, n_major=20, n_minor=8, z_offset=dist))
    elif design == "disc_pump":
        r_out = 0.1
        for s in spec.specs:
            if "disc outer" in s.name.lower():
                try:
                    r_out = float(s.value)
                except (ValueError, TypeError):
                    pass
        # Generate shaft + stacked discs
        facets.extend(_generate_cylinder_facets(radius=0.015, height=0.15, z_offset=-0.075, segments=16))
        for z in [-0.03, -0.01, 0.01, 0.03]:
            facets.extend(_generate_cylinder_facets(radius=r_out, height=0.003, z_offset=z, segments=24))
    elif design == "h2_storage":
        dia = 0.35
        for s in spec.specs:
            if "tank outer diameter" in s.name.lower():
                try:
                    dia = float(s.value)
                except (ValueError, TypeError):
                    pass
        r = dia / 2.0
        facets.extend(_generate_cylinder_facets(radius=r, height=0.8, z_offset=-0.4, segments=20))
    else:
        # Generic box/cylinder
        facets.extend(_generate_cylinder_facets(radius=0.15, height=0.2, z_offset=-0.1, segments=16))

    lines = [f"solid {spec.design or 'tesla_lab_model'}"]
    for (n1, n2, n3), (v1, v2, v3) in facets:
        lines.append(f"  facet normal {n1:.4e} {n2:.4e} {n3:.4e}")
        lines.append("    outer loop")
        lines.append(f"      vertex {v1[0]:.4e} {v1[1]:.4e} {v1[2]:.4e}")
        lines.append(f"      vertex {v2[0]:.4e} {v2[1]:.4e} {v2[2]:.4e}")
        lines.append(f"      vertex {v3[0]:.4e} {v3[1]:.4e} {v3[2]:.4e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {spec.design or 'tesla_lab_model'}\n")
    return "\n".join(lines)


def _generate_cylinder_facets(radius: float, height: float, z_offset: float, segments: int = 16):
    """Parametric cylinder facets."""
    facets = []
    z0 = z_offset
    z1 = z_offset + height
    theta_step = 2.0 * math.pi / segments

    for i in range(segments):
        th1 = i * theta_step
        th2 = (i + 1) * theta_step
        x1, y1 = radius * math.cos(th1), radius * math.sin(th1)
        x2, y2 = radius * math.cos(th2), radius * math.sin(th2)

        # Side quad as two triangles
        nx = math.cos((th1 + th2) / 2.0)
        ny = math.sin((th1 + th2) / 2.0)
        facets.append(((nx, ny, 0.0), ((x1, y1, z0), (x2, y2, z0), (x2, y2, z1))))
        facets.append(((nx, ny, 0.0), ((x1, y1, z0), (x2, y2, z1), (x1, y1, z1))))

        # Top and bottom caps
        facets.append(((0.0, 0.0, 1.0), ((0.0, 0.0, z1), (x1, y1, z1), (x2, y2, z1))))
        facets.append(((0.0, 0.0, -1.0), ((0.0, 0.0, z0), (x2, y2, z0), (x1, y1, z0))))
    return facets


def _generate_torus_facets(r_major: float, r_minor: float, n_major: int = 20, n_minor: int = 8, z_offset: float = 0.0):
    """Parametric torus facets."""
    facets = []
    u_step = 2.0 * math.pi / n_major
    v_step = 2.0 * math.pi / n_minor

    for i in range(n_major):
        u1 = i * u_step
        u2 = (i + 1) * u_step
        for j in range(n_minor):
            v1 = j * v_step
            v2 = (j + 1) * v_step

            def pt(u, v):
                x = (r_major + r_minor * math.cos(v)) * math.cos(u)
                y = (r_major + r_minor * math.cos(v)) * math.sin(u)
                z = z_offset + r_minor * math.sin(v)
                return (x, y, z)

            p1 = pt(u1, v1)
            p2 = pt(u2, v1)
            p3 = pt(u2, v2)
            p4 = pt(u1, v2)

            facets.append(((0.0, 0.0, 1.0), (p1, p2, p3)))
            facets.append(((0.0, 0.0, 1.0), (p1, p3, p4)))
    return facets


# --------------------------------------------------------------------------- Interactive Colorfigures-style Viewer
def generate_threejs_viewer_html(spec: ProductSpec) -> str:
    """Generate self-contained Colorfigures-inspired hand-inked isometric HTML viewer for Streamlit."""
    design = spec.design
    title = spec.title or "Tesla Lab 3D Concept"
    req = spec.requirements or {}

    # Extract geometry variables
    if design == "wireless_power":
        tx_r = 0.25
        rx_r = 0.075
        for s in spec.specs:
            if "transmitter coil radius" in s.name.lower():
                try:
                    tx_r = float(s.value)
                except (ValueError, TypeError):
                    pass
            if "receiver coil radius" in s.name.lower():
                try:
                    rx_r = float(s.value)
                except (ValueError, TypeError):
                    pass
        dist = float(req.get("distance_m", 0.5))
        js_scene_builder = f'''
        // Bookshelf casing
        const shelfGeo = new THREE.BoxGeometry(1.2, 0.8, 0.28);
        const shelfMat = new THREE.MeshToonMaterial({{ color: 0xdfcbb5, transparent: true, opacity: 1.0 }});
        const shelf = new THREE.Mesh(shelfGeo, shelfMat);
        shelf.position.set(0, 0.4, -0.08);
        casingGroup.add(shelf);
        addEdges(shelf, 0x231e16);

        // Transmitter coil (Copper)
        const txGeo = new THREE.TorusGeometry({tx_r}, 0.02, 16, 48);
        const copperMat = new THREE.MeshToonMaterial({{ color: 0xef6a4c }});
        const txCoil = new THREE.Mesh(txGeo, copperMat);
        txCoil.position.set(0, 0.4, 0.04);
        partsGroup.add(txCoil);
        addEdges(txCoil, 0x4a2219);
        addLabelTag("TRANSMITTER COIL (2 TURNS)", 0, 0.4 + {tx_r} + 0.08, 0.04);

        // Tuning capacitors at base
        const capGeo = new THREE.CylinderGeometry(0.025, 0.025, 0.08, 16);
        const capMat = new THREE.MeshToonMaterial({{ color: 0x3d3935 }});
        const cap1 = new THREE.Mesh(capGeo, capMat);
        cap1.position.set(-0.08, 0.4 - {tx_r} - 0.02, 0.04);
        partsGroup.add(cap1);
        addEdges(cap1, 0x231e16);
        const cap2 = new THREE.Mesh(capGeo, capMat);
        cap2.position.set(0.08, 0.4 - {tx_r} - 0.02, 0.04);
        partsGroup.add(cap2);
        addEdges(cap2, 0x231e16);
        addLabelTag("TUNING CAPACITORS", 0, 0.4 - {tx_r} - 0.08, 0.04);

        // Receiver lamp base + coil
        const rxGeo = new THREE.TorusGeometry({rx_r}, 0.012, 16, 36);
        const rxCoil = new THREE.Mesh(rxGeo, copperMat);
        rxCoil.position.set(0, 0.4, 0.04 + {dist});
        partsGroup.add(rxCoil);
        addEdges(rxCoil, 0x4a2219);

        // Desk Lamp receiver body
        const lampBaseGeo = new THREE.CylinderGeometry({rx_r} * 1.25, {rx_r} * 1.3, 0.04, 24);
        const lampMat = new THREE.MeshToonMaterial({{ color: 0x93d3b0, transparent: true, opacity: 1.0 }});
        const lampBase = new THREE.Mesh(lampBaseGeo, lampMat);
        lampBase.position.set(0, 0.4, 0.04 + {dist});
        lampBase.rotation.x = Math.PI / 2;
        partsGroup.add(lampBase);
        addEdges(lampBase, 0x1f4034);
        addLabelTag("RECEIVER LAMP ({dist:.2f} m)", 0, 0.4 + {rx_r} + 0.08, 0.04 + {dist});

        // Faint resonant field loop
        const fieldGeo = new THREE.TorusGeometry(({tx_r} + {rx_r}) / 2, {dist} / 2, 8, 24);
        const fieldMat = new THREE.MeshBasicMaterial({{ color: 0x76a0e2, wireframe: true, transparent: true, opacity: 0.35 }});
        const field = new THREE.Mesh(fieldGeo, fieldMat);
        field.position.set(0, 0.4, 0.04 + {dist} / 2);
        field.rotation.y = Math.PI / 2;
        partsGroup.add(field);
        '''
        camera_pos = "1.2, 0.9, 1.4"
    elif design == "disc_pump":
        r_out = 0.1
        for s in spec.specs:
            if "disc outer" in s.name.lower():
                try:
                    r_out = float(s.value)
                except (ValueError, TypeError):
                    pass
        js_scene_builder = f'''
        const steelMat = new THREE.MeshToonMaterial({{ color: 0xc6ccce }});
        const brassMat = new THREE.MeshToonMaterial({{ color: 0xf4b829 }});
        const glassMat = new THREE.MeshToonMaterial({{ color: 0xe4f2f4, transparent: true, opacity: 0.85 }});
        
        // Rotor Shaft
        const shaftGeo = new THREE.CylinderGeometry(0.015, 0.015, 0.28, 24);
        const shaft = new THREE.Mesh(shaftGeo, steelMat);
        shaft.rotation.z = Math.PI / 2;
        partsGroup.add(shaft);
        addEdges(shaft, 0x231e16);
        addLabelTag("DRIVE SHAFT", 0, 0.08, 0);

        // Disc Stack
        for (let i = -5; i <= 5; i++) {{
            const discGeo = new THREE.CylinderGeometry({r_out}, {r_out}, 0.003, 32);
            const disc = new THREE.Mesh(discGeo, steelMat);
            disc.position.x = i * 0.018;
            disc.rotation.z = Math.PI / 2;
            partsGroup.add(disc);
            addEdges(disc, 0x8f989b);
        }}
        addLabelTag("DISC STACK ({r_out*200:.0f} mm)", 0, {r_out} + 0.04, 0);

        // Volute Casing
        const casingGeo = new THREE.CylinderGeometry({r_out} * 1.18, {r_out} * 1.18, 0.22, 32);
        const casing = new THREE.Mesh(casingGeo, glassMat);
        casing.rotation.z = Math.PI / 2;
        casingGroup.add(casing);
        addEdges(casing, 0x231e16);

        // Tangential Nozzle
        const nozGeo = new THREE.CylinderGeometry(0.018, 0.018, 0.1, 16);
        const noz = new THREE.Mesh(nozGeo, brassMat);
        noz.position.set({r_out} * 1.1, 0.06, 0);
        partsGroup.add(noz);
        addEdges(noz, 0x231e16);
        addLabelTag("TANGENTIAL INLET", {r_out} * 1.1, 0.12, 0);
        '''
        camera_pos = "0.5, 0.4, 0.5"
    elif design == "h2_storage":
        dia = 0.35
        for s in spec.specs:
            if "tank outer diameter" in s.name.lower():
                try:
                    dia = float(s.value)
                except (ValueError, TypeError):
                    pass
        r = dia / 2.0
        js_scene_builder = f'''
        const carbonMat = new THREE.MeshToonMaterial({{ color: 0x3d3935, transparent: true, opacity: 1.0 }});
        const metalMat = new THREE.MeshToonMaterial({{ color: 0x76a0e2 }});
        const brassMat = new THREE.MeshToonMaterial({{ color: 0xf4b829 }});

        // Main cylinder
        const tankGeo = new THREE.CylinderGeometry({r}, {r}, 0.65, 32);
        const tank = new THREE.Mesh(tankGeo, carbonMat);
        casingGroup.add(tank);
        addEdges(tank, 0x231e16);

        // Hemispherical caps
        const capGeo = new THREE.SphereGeometry({r}, 32, 16);
        const cap1 = new THREE.Mesh(capGeo, carbonMat);
        cap1.position.y = 0.325;
        casingGroup.add(cap1);
        addEdges(cap1, 0x231e16);
        const cap2 = new THREE.Mesh(capGeo, carbonMat);
        cap2.position.y = -0.325;
        casingGroup.add(cap2);
        addEdges(cap2, 0x231e16);
        addLabelTag("CARBON-COMPOSITE TANK", 0, 0, {r} + 0.08);

        // Pressure Regulator Valve
        const valveGeo = new THREE.CylinderGeometry(0.035, 0.035, 0.12, 16);
        const valve = new THREE.Mesh(valveGeo, brassMat);
        valve.position.y = 0.325 + {r} + 0.06;
        partsGroup.add(valve);
        addEdges(valve, 0x231e16);
        addLabelTag("700-BAR REGULATOR", 0, 0.325 + {r} + 0.14, 0);

        // Electrolyzer Module Block
        const stackGeo = new THREE.BoxGeometry(0.18, 0.4, 0.25);
        const stack = new THREE.Mesh(stackGeo, metalMat);
        stack.position.set({r} * 1.8, 0, 0);
        partsGroup.add(stack);
        addEdges(stack, 0x1c2948);
        addLabelTag("PEM ELECTROLYZER", {r} * 1.8, 0.25, 0);
        '''
        camera_pos = "0.9, 0.6, 1.1"
    else:
        js_scene_builder = '''
        const mat = new THREE.MeshToonMaterial({ color: 0x76a0e2 });
        const boxGeo = new THREE.BoxGeometry(0.4, 0.25, 0.2);
        const box = new THREE.Mesh(boxGeo, mat);
        partsGroup.add(box);
        addEdges(box, 0x231e16);
        addLabelTag("PROTOTYPE CHASSIS", 0, 0.18, 0);
        '''
        camera_pos = "0.6, 0.5, 0.7"

    return f'''<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; padding: 0; overflow: hidden;
      background: #f5f2eb; color: #231e16;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      user-select: none;
    }}
    #canvas-container {{ width: 100vw; height: 100vh; position: relative; }}
    
    /* Top Header Bar */
    .header-bar {{
      position: absolute; top: 12px; left: 16px; right: 16px;
      display: flex; justify-content: space-between; align-items: center;
      pointer-events: none; z-index: 10;
    }}
    .fig-title {{
      background: #ffffff; color: #231e16; font-size: 13px; font-weight: 700;
      padding: 6px 14px; border-radius: 9999px; border: 1.5px solid #231e16;
      box-shadow: 2px 2px 0 #231e16; pointer-events: auto; letter-spacing: 0.03em;
    }}
    .btn-group {{
      display: flex; gap: 8px; pointer-events: auto;
    }}
    .pill-btn {{
      background: #ffffff; color: #231e16; font-size: 12px; font-weight: 600;
      padding: 6px 14px; border-radius: 9999px; border: 1.5px solid #231e16;
      box-shadow: 2px 2px 0 #231e16; cursor: pointer; transition: all 0.15s ease;
      display: flex; align-items: center; gap: 5px;
    }}
    .pill-btn:hover {{
      transform: translate(-1px, -1px); box-shadow: 3px 3px 0 #231e16;
    }}
    .pill-btn.active {{
      background: #231e16; color: #f5f2eb;
    }}

    /* Bottom Control Bar */
    .bottom-bar {{
      position: absolute; bottom: 14px; left: 16px; right: 16px;
      display: flex; justify-content: space-between; align-items: center;
      pointer-events: none; z-index: 10;
    }}
    .swatches {{
      display: flex; gap: 8px; align-items: center; pointer-events: auto;
      background: #ffffff; padding: 5px 12px; border-radius: 9999px;
      border: 1.5px solid #231e16; box-shadow: 2px 2px 0 #231e16;
    }}
    .swatch-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; margin-right: 4px; }}
    .swatch {{
      width: 18px; height: 18px; border-radius: 50%; border: 1.5px solid #231e16;
      cursor: pointer; transition: transform 0.15s;
    }}
    .swatch:hover {{ transform: scale(1.2); }}
    
    .zoom-group {{
      display: flex; align-items: center; gap: 4px; pointer-events: auto;
      background: #ffffff; padding: 4px 10px; border-radius: 9999px;
      border: 1.5px solid #231e16; box-shadow: 2px 2px 0 #231e16; font-size: 11px;
    }}
    .zoom-btn {{
      background: none; border: none; font-size: 14px; font-weight: bold;
      cursor: pointer; padding: 0 4px; color: #231e16;
    }}

    /* 3D Label Callouts */
    .callout-tag {{
      position: absolute; background: #ffffff; color: #231e16;
      font-size: 10px; font-weight: 700; padding: 3px 8px; border-radius: 4px;
      border: 1.5px solid #231e16; box-shadow: 2px 2px 0 #231e16;
      pointer-events: none; transform: translate(-50%, -50%);
      transition: opacity 0.2s ease; opacity: 1; white-space: nowrap;
    }}
    .callout-tag.hidden {{ opacity: 0; }}
  </style>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
  <div id="canvas-container">
    <div class="header-bar">
      <div class="fig-title">Fig. 01 — {title}</div>
      <div class="btn-group">
        <button id="btn-see-through" class="pill-btn">See-through</button>
        <button id="btn-labels" class="pill-btn active">Labels</button>
        <button id="btn-turntable" class="pill-btn active">Turntable</button>
      </div>
    </div>

    <div class="bottom-bar">
      <div class="swatches">
        <span class="swatch-label">Accent:</span>
        <div class="swatch" style="background:#f4b829" onclick="setAccent(0xf4b829)"></div>
        <div class="swatch" style="background:#ef6a4c" onclick="setAccent(0xef6a4c)"></div>
        <div class="swatch" style="background:#93d3b0" onclick="setAccent(0x93d3b0)"></div>
        <div class="swatch" style="background:#76a0e2" onclick="setAccent(0x76a0e2)"></div>
      </div>
      <div class="zoom-group">
        <button class="zoom-btn" onclick="zoomCam(1.15)">-</button>
        <span id="zoom-val">100%</span>
        <button class="zoom-btn" onclick="zoomCam(0.85)">+</button>
      </div>
    </div>
  </div>

  <script>
    const container = document.getElementById('canvas-container');
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf5f2eb);

    // Warm paper shadow plane (Colorfigures flat blue cast shadow #41538f)
    const shadowGeo = new THREE.PlaneGeometry(3.5, 3.5);
    const shadowMat = new THREE.MeshBasicMaterial({{ color: 0x41538f, transparent: true, opacity: 0.15 }});
    const shadowPlane = new THREE.Mesh(shadowGeo, shadowMat);
    shadowPlane.rotation.x = -Math.PI / 2;
    shadowPlane.position.y = -0.01;
    scene.add(shadowPlane);

    // Camera (Isometric angle)
    const camera = new THREE.PerspectiveCamera(40, container.clientWidth / container.clientHeight, 0.05, 100);
    camera.position.set({camera_pos});

    const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.0;

    // Lighting (Colorfigures toon lighting: overhead + fill)
    const ambient = new THREE.AmbientLight(0xfff8ee, 0.85);
    scene.add(ambient);
    const sun = new THREE.DirectionalLight(0xffffff, 0.95);
    sun.position.set(3, 5, 4);
    scene.add(sun);

    // Groups for See-through toggling
    const casingGroup = new THREE.Group();
    const partsGroup = new THREE.Group();
    scene.add(casingGroup);
    scene.add(partsGroup);

    // Helper: Add inked edges to meshes
    function addEdges(mesh, inkColor = 0x231e16) {{
      const edges = new THREE.EdgesGeometry(mesh.geometry, 28);
      const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({{ color: inkColor, linewidth: 2 }}));
      mesh.add(line);
    }}

    // 3D Tag Tracking
    const tags = [];
    function addLabelTag(text, x, y, z) {{
      const el = document.createElement('div');
      el.className = 'callout-tag';
      el.innerText = text;
      container.appendChild(el);
      tags.push({{ el: el, pos: new THREE.Vector3(x, y, z) }});
    }}

    // Construct Scene Geometry
    {js_scene_builder}

    // Interactive State
    let seeThrough = false;
    let showLabels = true;

    const btnSeeThrough = document.getElementById('btn-see-through');
    btnSeeThrough.addEventListener('click', () => {{
      seeThrough = !seeThrough;
      btnSeeThrough.classList.toggle('active', seeThrough);
      casingGroup.traverse(child => {{
        if (child.isMesh && child.material) {{
          child.material.opacity = seeThrough ? 0.22 : 1.0;
          child.material.transparent = true;
        }}
      }});
    }});

    const btnLabels = document.getElementById('btn-labels');
    btnLabels.addEventListener('click', () => {{
      showLabels = !showLabels;
      btnLabels.classList.toggle('active', showLabels);
      tags.forEach(t => t.el.classList.toggle('hidden', !showLabels));
    }});

    const btnTurntable = document.getElementById('btn-turntable');
    btnTurntable.addEventListener('click', () => {{
      controls.autoRotate = !controls.autoRotate;
      btnTurntable.classList.toggle('active', controls.autoRotate);
    }});

    function zoomCam(factor) {{
      camera.position.multiplyScalar(factor);
    }}

    function setAccent(hex) {{
      partsGroup.traverse(child => {{
        if (child.isMesh && child.material && child.material.color) {{
          child.material.color.setHex(hex);
        }}
      }});
    }}

    // Window Resize
    window.addEventListener('resize', () => {{
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    }});

    // Update screen positions of tags
    function updateTags() {{
      if (!showLabels) return;
      const w = container.clientWidth, h = container.clientHeight;
      tags.forEach(t => {{
        const v = t.pos.clone().project(camera);
        if (v.z < 1) {{
          const sx = (v.x * 0.5 + 0.5) * w;
          const sy = (-v.y * 0.5 + 0.5) * h;
          t.el.style.left = `${{sx}}px`;
          t.el.style.top = `${{sy}}px`;
        }}
      }});
    }}

    // Animation Loop
    function animate() {{
      requestAnimationFrame(animate);
      controls.update();
      updateTags();
      renderer.render(scene, camera);
    }}
    animate();
  </script>
</body>
</html>'''

