# Kuavo 5W head camera — fact-check (2026-11-11)

Checked against the live Kuavo MuJoCo wheel-arm sim (container `kuavo_m1_sim`,
ROBOT_VERSION=62) and the vendor sources in `kuavo_ws`. The ROS-master topics were listed
live (243 topics, zero image topics); the camera code was read in `kuavo_ws`.

## Real robot (per `kuavo_ws/src/kuavo_sdk/sdk/03_use_camera/readme.md`)

RealSense-style head camera, factory-calibrated:

- RGB: `/camera/color/image_raw` (`sensor_msgs/Image`), intrinsics `/camera/color/camera_info`
- Depth: `/camera/depth/image_rect_raw`, intrinsics `/camera/depth/camera_info`
- Aligned depth-to-RGB: `/camera/aligned_depth_to_color/image_raw` + `camera_info`

To be re-confirmed on-site during M2 before wiring `snapshot()`.

## Kuavo MuJoCo wheel-arm sim (the container sim this release runs)

- **No image topic of any kind is published.** `biped_s62.xml` carries only a
  `head_camera` STL *visual* geom, not a MuJoCo `<camera>` element.
- The sim's built-in depth camera (`kuavo_ws/src/mujoco/src/mujoco_node.cc`) activates
  only when the scene XML defines a camera named `waist_camera` AND `/run_mujoco_camera`
  is true; it then publishes `/waist_camera/depth/image_raw` — 64x36, `32FC1`, normalized
  by max range (2.5 m), **depth only, no RGB**.
- The gazebo stack publishes `/camera/color/image_raw`, but this release runs the mujoco
  sim, not gazebo.

## Consequence (per the M3 spec)

On the Kuavo sim `snapshot()` returns a valid blank frame (black rgb, no metric depth,
no labels) — the session loop runs, navigate plans over the fallback raster, and any
click-driven skill is refused at its own argument check. Vision on the sim would need a
`waist_camera` element in a scene XML plus an RGB publisher in the kuavo-side nodelet —
a `kuavo_ws` change, out of this repo's scope.
