# Ariel: Autonomous Robots through Integrated Evolution and Learning

<!--
[ ] Friction increase (rev2)
[ ] CMA only
[ ] full rotation of hinges (45 increments)
[ ] add top/bot
[ ] Trimesh collision detect
[ ] Spawn point + bounding box
 -->

## TODO: Installation

## Notes

### This project is managed using `uv`

### Python Code Style Guide

This repository uses the `numpydoc` documentation standard.
For more information checkout: [numpydoc-style guide](https://numpydoc.readthedocs.io/en/latest/format.html#)

<!-- ### Units

To ensure that Ariel uses a consistent set of units for all simulations, we use [SI units](https://www.wikiwand.com/en/articles/International_System_of_Units), and (astropy)[https://docs.astropy.org/en/stable/index.html] to enforce it (we automatically convert where we can).

For more information, see: [astropy: units and quantities](https://docs.astropy.org/en/stable/units/index.html) and [astropy: standard units](https://docs.astropy.org/en/stable/units/standard_units.html#standard-units). -->

### MuJoCo

#### Attachments

Robot parts should be attached using the `site` functionality (from body to body), while robots should be added to a world using the `frame` functionality (from spec to spec).

- [Python → Attachment](https://mujoco.readthedocs.io/en/stable/python.html#attachment)
- [mjsFrame](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html#mjsframe)

NOTE: when attaching a body, only the contents of `worldbody` get passed, meaning that, for example, `compiler` options are not!
