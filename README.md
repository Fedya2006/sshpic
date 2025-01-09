# sshpic: SSH Key-Based Art Generation

`sshpic` is a Python library that generates artistic collages based on SSH key fingerprints using a distilled version of Stable Diffusion 1.4 and Tiny VAE (`taesd-diffusers`). This tool provides a unique way to visualize cryptographic keys as beautiful art.

---

## Features

- **SSH Key Art**: Generate images and textures derived from SSH key fingerprints.
- **Distilled Model**: Utilizes a lightweight Stable Diffusion 1.4 for efficient generation.
- **Tiny VAE**: Integrates `taesd-diffusers` Tiny VAE for fast and efficient image processing.
- **Collage Generation**: Automatically arranges generated images into a cohesive collage with textured backgrounds.
- **Simple Interface**: Easy-to-use API for seamless integration into projects.

---
## Theoretical evaluation of generation variability:

A 4-byte seed and a 1-byte guidance_scale are used to generate each image.

Then in theory it is possible to generate in the case of a 32-byte fingerprint:

**2^((4×8+1×8)×6 + 2×8) = 2^256** different collage variants, which with is enough for **2^256** different ssh fingerprints.
There are 6 images on the collage + texture on seed which takes 2 bytes in this case.

---

## Installation

Install `sshpic` from the wheel file:

```bash
pip install sshpic-0.1.0-py3-none-any.whl --use-deprecated=legacy-resolver
```
## Usage Example:
```
from sshpic import SshPicGen

z = SshPicGen.SSHPicture("input.txt")
z.run()
```
