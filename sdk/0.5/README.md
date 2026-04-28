MN-Core SDK 0.5
================

Documents
----------

### MLSDK

* English: https://dev.mn-core.com/sdk/0.5/MLSDK/docs/en/
* 日本語: https://dev.mn-core.com/sdk/0.5/MLSDK/docs/ja/

Dockerfiles
------------

This directory contains Dockerfiles to build your own MN-Core SDK 0.5 Docker images.

```
# Build mncore-sdk-minimal:0.5.
$ docker build -t mncore-sdk-minimal:0.5 -f mncore-sdk-minimal.Dockerfile .

# Build mncore-sdk-full:0.5 based on mncore-sdk-minimal:0.5.
$ docker build -t mncore-sdk-full:0.5 -f mncore-sdk-full.Dockerfile --build-arg minimal_image_ref=mncore-sdk-minimal:0.5 .
```

### For Podman users

The Dockerfiles are designed to be compliant with Podman. Replace `docker` with `podman` in the examples above if you prefer to use Podman.

The resulting images will be OCI-compliant. They will function correctly regardless of whether you use Docker or Podman.

#### For your information: build warnings

You may observe warning messages like below during the build process on some versions of Podman (v4.9 - v5.2).

```
time="YYYY-MM-DDThh:mm:ss+ZZ:ZZ" level=warning msg="can't raise ambient capability CAP_*****: operation not permitted"
```

They are a known diagnostic noise in rootless environments. They do not affect the integrity of the resulting image. It has been addressed in Podman v5.3 and later. (See: [containers/podman#27967](https://github.com/containers/podman/discussions/27967))

Utility Script
---------------

This directory also contains a helper script to start a development container with
MN-Core devices attached.

### `create_dev_ctr.sh`

Use `create_dev_ctr.sh` when developing, testing, and evaluating programs for
MN-Core with `mncore-sdk-full:0.5` in a container on a bare metal machine.

#### Usage

On a machine used by a single user, `-A` is a simple way to attach all available
devices.

```bash
$ bash create_dev_ctr.sh -A
```

If you want to use specific MN-Core devices, specify them explicitly.

```bash
$ bash create_dev_ctr.sh /dev/mnc2p1s0
```

After the container starts, enter the container and use the MN-Core SDK there
for sample execution, application development, debugging, or performance
evaluation.

#### Options

* `-A`: Attach all MN-Core devices found on the host
* `-i IMAGE[:TAG]`: Use a specific Docker image instead of `mncore-sdk-full:0.5`
* `-s MOUNT`: Specify the mount used for exclusive control of devices
* `-O DOCKER_OPTS`: Pass additional options to `docker run`
