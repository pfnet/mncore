# `mncore-packages`: APT repository for MN-Core packages

`mncore-packages` is an APT repository providing the packages for MN-Core.
You can install the MN-Core driver and SDK via `apt` after adding this repository to your Ubuntu system.

## Packages in `mncore-packages`

The repository provides packages such as the following:

* Kernel modules: To enable your Ubuntu system to recognize MN-Core devices as PCIe devices
  * `gpfn3-dkms`
* User-space drivers: To enable your applications to use MN-Core
  * `libgpfn3-0`
  * `libgpfn3-dev`
* Utilities: To verify MN-Core's operational status and perform system management tasks
  * `gpfn3-smi`
  * `gpfn3-loader`
* SDK: To develop programs that run on MN-Core
  * `mncore-sdk`

These packages are provided for select Ubuntu LTS releases. Non-LTS Ubuntu releases are not supported.

## Typical use cases

### On a bare-metal machine (MN-Core 2 Devkit and MN-Server 2)

Users who own an MN-Core bare-metal machine, such as MN-Core 2 Devkit and MN-Server 2, need to add `mncore-packages` to the system so that they can install related packages.

See [MN-Core 2 Devkit・MN-Server 2 インストール・運用マニュアル](https://projects.preferred.jp/mn-core/assets/MN-Core2-Devkit-MN-Server-2-installation-operation-manual.pdf) for operational use.

See [sdk/0.5/README.md](../sdk/0.5/README.md) for development use.

### In the cloud, Preferred Computing Platform (PFCP)

The standard environments in PFCP are preconfigured with `mncore-packages`. Typical users do not need to add `mncore-packages` or install packages manually.

See the [Preferred Computing Platform（PFCP）ユーザガイド](https://docs.pfcomputing.com/) for details.

## Adding `mncore-packages` in your Ubuntu system

To add the `mncore-packages` repository in your Ubuntu system, use `add_mncore_packages.sh` in this directory. Just run the script as `root` as follows:

```bash
sudo bash add_mncore_packages.sh
```

If the script finishes successfully, `mncore-packages` has been registered on the system. You can then install packages by running commands such as `sudo apt install gpfn3-smi`.
