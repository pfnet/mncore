# syntax=docker/dockerfile:1

# Dockerfile to build the MN-Core SDK 0.5 full Docker image (mncore-sdk-full:0.5) based on mncore-sdk-minimal:0.5.

# The ref of your own mncore-sdk-minimal:0.5 image, which is built from mncore-sdk-minimal.Dockerfile.
ARG minimal_image_ref

# Example command line to build this Docker image to run on this directory:
#
# docker build -t mncore-sdk-full:0.5 -f mncore-sdk-full.Dockerfile --build-arg minimal_image_ref=mncore-sdk-minimal:0.5 .
#
# NOTE: The "--build-arg" option specifies the base minimal image. You have to replace this "mncore-sdk-minimal:0.5"
# if you've given your own minimal image (built from mncore-sdk-minimal.Dockerfile) a different image name or tag.


################################################################################
# stage: download-vscode-cli
#
# * Downloads the Visual Studio Code CLI package.
################################################################################

FROM curlimages/curl:8.15.0 AS download-vscode-cli

# Install Visual Studio Code CLI.
ARG VSCODE_CLI_VERSION=1.103.1

# https://code.visualstudio.com/docs/supporting/faq#_previous-release-versions
RUN curl -kL "https://update.code.visualstudio.com/${VSCODE_CLI_VERSION}/cli-linux-x64/stable" -o vscode_cli_linux_x64_cli.tar.gz \
    && mkdir /tmp/vscode-cli \
    && tar xvf vscode_cli_linux_x64_cli.tar.gz -C /tmp/vscode-cli


################################################################################
# stage: (final)
#
# * Builds the final "mncore-sdk-full" Docker image from "mncore-sdk-minimal".
################################################################################

FROM ${minimal_image_ref}

RUN <<-EOF
    rm -f /etc/apt/apt.conf.d/docker-clean
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
    ls -R /var/cache/apt/archives

    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        automake \
        make \
        cmake \
        git \
        libpng++-dev \
        pkgconf \
        zlib1g-dev \
        `# nodejs/npm for Jupyter extensions` \
        nodejs \
        npm \
        `# nbconvert dependencies for PDF export support in Jupyter` \
        texlive-xetex \
        texlive-fonts-recommended \
        texlive-plain-generic \
        vim \
        emacs \
        nano \
        file \
        gawk \
        less \
        curl \
        wget

    ls -R /var/cache/apt/archives
EOF

# Install JupyterLab
RUN npm install -g n && n stable && node -v

# This `requirements-jupyter.txt` is embedded in the Dockerfile so that the Dockerfile builds standalone.
COPY <<-EOF /var/tmp/requirements-jupyter.txt
    nbconvert
    notebook
    widgetsnbextension
    black # used by jupyterlab-code-formatter
    python-language-server[all]
    jupyter==1.0.0
    jupyter-console==6.6.3
    jupyter-events==0.6.3
    jupyter-lsp==2.2.0
    jupyter-ydoc==0.2.4
    jupyter_client==8.2.0
    jupyter_core==5.3.1
    jupyter_server==2.6.0
    jupyter_server_fileid==0.9.0
    jupyter_server_terminals==0.4.4
    jupyter_server_ydoc==0.8.0
    jupyterlab==4.0.2
    jupyterlab-code-formatter
    jupyterlab-pygments==0.2.2
    jupyterlab-widgets==3.0.7
    jupyterlab_server==2.23.0
EOF

RUN <<-EOF
    python3 -m pip install ipykernel
    python3 -m ipykernel install
    python3 -m pip install -r /var/tmp/requirements-jupyter.txt
    python3 -m pip install -U jupyterlab
    python3 -m jupyter lab build
    python3 -m jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

    # This symlink /app/jupyter/bin/jupyter is here for compatibility with older versions of MN-Core SDK.
    # It should be removed in some future release.
    mkdir -p /app/jupyter/bin
    ln -s /usr/local/bin/jupyter /app/jupyter/bin/jupyter
EOF

# Set pre-installed fonts on macOS, Windows and Ubuntu in various versions for Jupyter.
COPY <<-EOF /root/.jupyter/lab/user-settings/@jupyterlab/terminal-extension/plugin.jupyterlab-settings
    {
        "fontFamily": "'SF Mono', Menlo, 'Courier New', Consolas, 'Ubuntu Mono', 'DejaVu Sans Mono', monospace",
    }
EOF

# Copy Visual Studio Code CLI.
COPY --from=download-vscode-cli /tmp/vscode-cli/code /usr/local/bin/code

LABEL "org.opencontainers.image.description"="This is a container image of the MN-Core SDK's rich development environment. Development environments such as JupyterLab and Visual Studio Code CLI are pre-installed. Before using this MN-Core SDK image, please review the MN-Core_SDK_End-User_License_Agreement.pdf located in either /opt/pfn/licenses/ or /opt/pfn/pfcomp/licenses/ within the image. By using the SDK, you acknowledge and agree to the terms and conditions set forth therein."
