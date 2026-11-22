# Sources and licenses

- PPO implementation and portions of the ball detector/tracker originate from the Panthera adaptation of `zecice/so101-pingpong-balance`. The supplied MIT notice is retained in `licenses/pingpong-MIT.txt`.
- The reference Panthera follower arm (including enabled camera_link), meshes and source package metadata came from `wEch1ng/Panthera-HT_pingpong-balance`, whose source is the HighTorque Robotics SDK. Its MIT notice and the source package's BSD declaration are retained at `licenses/HighTorque-SDK-MIT.txt` and `licenses/arm-package.xml`. No attempt is made to resolve or overwrite this upstream license distinction.
- Chassis CAD and earlier combined/arm exports were already supplied in this repository. The original files are preserved under `archive/source_exports/`; this reorganization does not grant a new license to assets whose license was not explicitly supplied.
- Paddle meshes retain `assets/tools/paddle/PADDLE_LICENSE.txt` (MIT, Samuel Galla and Tim Huber). The source-scene notice is `assets/tools/paddle/SCENE_LICENSE` (Apache-2.0). Additional supplied provenance is retained in that directory.
- `checkpoints/pingpong/arm_recovery.pt` is a byte-identical copy of the user's reference single-arm recovery checkpoint. The experimental residual checkpoint was produced in this project. SHA256, iteration and intended usage are in `checkpoints/manifest.json`.
- MuJoCo, NumPy, OpenCV, PyTorch and other dependencies retain their own licenses; they are installed from their package repositories rather than vendored here.
