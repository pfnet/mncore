# `mncore-packages`: APT repository for MN-Core packages

`mncore-packages` is an APT repository providing the packages for MN-Core.
You can install the MN-Core driver and SDK via `apt` after adding this repository to your Ubuntu system.

## Supported platforms

`mncore-packages` supports select Ubuntu LTS releases. Non-LTS Ubuntu releases are not supported.

## Adding `mncore-packages` in your Ubuntu system

Run the script as `root` as follows.

```bash
sudo bash apt/add_mncore_packages.sh
```

## Installing packages from `mncore-packages`

Install MN-Core packages via `apt install` after adding `mncore-packages` in your system.

```bash
sudo apt install gpfn3-smi
```

## Tips

Google Artifact Registry, which hosts `mncore-packages`, does not provide a web interface for everyone to search packages. To find installable packages and versions, see the APT index file instead.

```bash
zgrep '^Package:' /var/lib/apt/lists/*mncore-packages*  # will show available packages
```

## See also

Dockerfiles under `sdk/<version>` directories use the same `mncore-packages` repository to install MN-Core packages inside the container.
