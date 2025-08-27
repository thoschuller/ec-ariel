<!-- TODO:
1. check docker section
2. add images to docker section -->
# Installation guide

## Prerequisites

- Python 3.12 or newer
- [uv](https://github.com/astral-sh/uv) package manager
- [pip](https://pip.pypa.io/en/stable/)
- [Docker](https://www.docker.com/)

## Installation with uv

1. Install `uv` (if not already installed):

    ```bash
    pip install uv
    ```

3. Install ariel:
    ```bash
    uv pip install ariel
    ```

## Installation with pip

1. Ensure you have `pip` installed.
2. Install ariel:

    ```bash
    pip install ariel
    ```

## Installation with Docker

### I am not very proficient with docker. So double check!!!!
1. Build the Docker image:

    ```bash
    docker build -t ariel .
    ```

2. Run the Docker container:

    ```bash
    docker run -d -p 8000:8000 ariel
    ```

3. Edit docker container:

![image1](../resources/docker_img_1.jpeg)

![image2](../resources/docker_img_2.jpeg)

![image3](../resources/docker_img_2.jpeg)


## Verifying Installation

- For uv/pip: Run the application as described in the project README.
- For Docker: Access the platform at `http://localhost:8000`.

## Troubleshooting

- Ensure all prerequisites are installed and up to date.
- Check logs for errors and consult documentation for further help.
- For Docker issues, verify Docker is running and ports are available.