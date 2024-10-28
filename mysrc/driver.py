
import MatterSim
import time
import math
import cv2
import os
import json
import numpy as np


def state_to_dict(mat_state):
    """
    {
        "scanId" : "2t7WUuJeko7"  // Which building the agent is in
        "step" : 5,               // Number of frames since the last newEpisode() call
        "rgb" : <image>,          // 8 bit image (in BGR channel order), access with np.array(rgb, copy=False)
        "depth" : <image>,        // 16 bit single-channel image containing the pixel's distance in the z-direction from the camera center 
                                // (not the euclidean distance from the camera center), 0.25 mm per value (divide by 4000 to get meters). 
                                // A zero value denotes 'no reading'. Access with np.array(depth, copy=False)
        "location" : {            // The agent's current 3D location
            "viewpointId" : "1e6b606b44df4a6086c0f97e826d4d15",  // Viewpoint identifier
            "ix" : 5,                                            // Viewpoint index, used by simulator
            "x" : 3.59775996208,                                 // 3D position in world coordinates
            "y" : -0.837355971336,
            "z" : 1.68884003162,
            "rel_heading" : 0,                                   // Robot relative coords to this location
            "rel_elevation" : 0,
            "rel_distance" : 0
        }
        "heading" : 3.141592,     // Agent's current camera heading in radians
        "elevation" : 0,          // Agent's current camera elevation in radians
        "viewIndex" : 0,          // Index of the agent's current viewing angle [0-35] (only valid with discretized viewing angles)
                                // [0-11] is looking down, [12-23] is looking at horizon, is [24-35] looking up
        "navigableLocations": [   // List of viewpoints you can move to. Index 0 is always the current viewpoint, i.e. don't move.
            {                     // The remaining valid viewpoints are sorted by their angular distance from the image centre.
                "viewpointId" : "1e6b606b44df4a6086c0f97e826d4d15",  // Viewpoint identifier
                "ix" : 5,                                            // Viewpoint index, used by simulator
                "x" : 3.59775996208,                                 // 3D position in world coordinates
                "y" : -0.837355971336,
                "z" : 1.68884003162,
                "rel_heading" : 0,                                   // Robot relative coords to this location
                "rel_elevation" : 0,
                "rel_distance" : 0
            },
            {
                "viewpointId" : "1e3a672fa1d24d668866455162e5b58a",  // Viewpoint identifier
                "ix" : 14,                                           // Viewpoint index, used by simulator
                "x" : 4.03619003296,                                 // 3D position in world coordinates
                "y" : 1.11550998688,
                "z" : 1.65892004967,
                "rel_heading" : 0.220844170027,                      // Robot relative coords to this location
                "rel_elevation" : -0.0149478448723,
                "rel_distance" : 2.00169944763
            },
            {...}
        ]
    }
    """
    dd = {}
    dd['scanId'] = mat_state.scanId
    dd['step'] = mat_state.step
    dd['location'] = {
        'viewpointId': mat_state.location.viewpointId,
        'ix': mat_state.location.ix,
        'x': mat_state.location.x,
        'y': mat_state.location.y,
        'z': mat_state.location.z,
        'rel_heading': mat_state.location.rel_heading,
        'rel_elevation': mat_state.location.rel_elevation,
        'rel_distance': mat_state.location.rel_distance
    }
    dd['heading'] = mat_state.heading
    dd['elevation'] = mat_state.elevation
    dd['viewIndex'] = mat_state.viewIndex
    dd['navigableLocations'] = []
    for loc in mat_state.navigableLocations:
        dd['navigableLocations'].append({
            'viewpointId': loc.viewpointId,
            'ix': loc.ix,
            'x': loc.x,
            'y': loc.y,
            'z': loc.z,
            'rel_heading': loc.rel_heading,
            'rel_elevation': loc.rel_elevation,
            'rel_distance': loc.rel_distance
        })
    return dd


def get_sim_instance(WIDTH=800, HEIGHT=600, vfovdeg=60, DEPTH_ENABLED=False, scan_id='TbHJrupSAjP', viewpoint_id='ce5a75d3715b49c5b6fe193235e52c27', heading_rad=0, elevation_rad=0):
    VFOV = math.radians(vfovdeg)

    sim = MatterSim.Simulator()
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(VFOV)
    sim.setDepthEnabled(DEPTH_ENABLED)  # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
    sim.initialize()
    if viewpoint_id is None:
        sim.newRandomEpisode([scan_id])
    else:
        sim.newEpisode([scan_id], [viewpoint_id], [heading_rad], [elevation_rad])
    return sim


def get_manual_control_view(sim, WIDTH=800, HEIGHT=600, vfovdeg=60, DEPTH_ENABLED=False, RECORD_VIDEO=False, output_path='.'):
    VFOV = math.radians(vfovdeg)
    HFOV = 2 * math.atan(math.tan(VFOV / 2) * WIDTH / HEIGHT)
    TEXT_COLOR = [40, 40, 230]
    ANGLEDELTA = 5 * math.pi / 180

    cv2.namedWindow('Python RGB')
    if DEPTH_ENABLED:
        cv2.namedWindow('Python Depth')
    if RECORD_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(output_path, 'video.mp4'), fourcc, 1.0, (WIDTH, HEIGHT))

    while True:
        state = sim.getState()[0]
        locations = state.navigableLocations
        rgb = np.array(state.rgb, copy=False)
        if RECORD_VIDEO:
            out.write(rgb)
        for idx, loc in enumerate(locations[1:]):
            # Draw actions on the screen
            fontScale = 0.9 / loc.rel_distance
            x = int(WIDTH / 2 + loc.rel_heading / HFOV * WIDTH)
            y = int(HEIGHT / 2 - loc.rel_elevation / VFOV * HEIGHT)
            cv2.putText(rgb, f"{idx+1}:{loc.viewpointId[:4]}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, TEXT_COLOR, thickness=2)
        cv2.imshow('Python RGB', rgb)
        if DEPTH_ENABLED:
            depth = np.array(state.depth, copy=False)
            cv2.imshow('Python Depth', depth)

        print("Enter action key")
        k = cv2.waitKey(0)
        k = (k & 255)
        location = 0
        heading = 0
        elevation = 0
        if k == ord('q'):
            break
        elif k == ord('p'):
            # save state to files
            cv2.imwrite(os.path.join(output_path, 'rgb.png'), rgb)
            if DEPTH_ENABLED:
                cv2.imwrite(os.path.join(output_path, 'depth.png'), depth)
            state = state_to_dict(state)
            with open(os.path.join(output_path, 'state.json'), 'w') as f:
                json.dump(state, f, indent=4)
        elif ord('1') <= k <= ord('9'):
            location = k - ord('0')
            if location >= len(locations):
                location = 0
        elif k == 81 or k == ord('a'):
            heading = -ANGLEDELTA
        elif k == 82 or k == ord('w'):
            elevation = ANGLEDELTA
        elif k == 83 or k == ord('d'):
            heading = ANGLEDELTA
        elif k == 84 or k == ord('s'):
            elevation = -ANGLEDELTA
        print(f"Making action: {location}, {heading}, {elevation}")
        sim.makeAction([location], [heading], [elevation])

    if RECORD_VIDEO:
        out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sim = get_sim_instance()
    get_manual_control_view(sim=sim)
