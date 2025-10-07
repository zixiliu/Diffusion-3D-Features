from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="diff3f",
    version="0.1.0",
    author="Niladri Shekhar Dutt, Sanjeev Muralikrishnan, Niloy J. Mitra",
    author_email="",
    description="Diffusion 3D Features (Diff3F): Decorating Untextured Shapes with Distilled Semantic Features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://diff3f.github.io/",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=[
        "matplotlib>=3.7.1",
        "numpy>=1.25.0",
        "scikit-learn>=1.2.2",
        "scipy>=1.10.1",
        "torch>=2.1.0",
        "torchvision>=0.17.1",
        "torchaudio>=2.1.0",
        "diffusers==0.21.4",
        "einops==0.7.0",
        "huggingface-hub==0.17.3",
        "meshio==5.3.4",
        "opencv-python==4.8.1.78",
        "plyfile==1.0.1",
        "transformers==4.34.1",
        "trimesh==4.0.0",
        "potpourri3d==1.0.0",
        "robust_laplacian==0.2.7",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "jupyter",
        ],
        "gpu": [
            "xformers==0.0.21",
        ]
    },
    entry_points={
        "console_scripts": [
            "diff3f-extract-shrec=extract_shrec:main",
            "diff3f-extract-tosca=extract_tosca:main",
            "diff3f-eval-shrec=evaluate_pipeline_shrec:main",
            "diff3f-eval-tosca=evaluate_pipeline_tosca:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["meshes/*.obj", "meshes/*.off", "assets/*.jpg", "*.yaml", "*.md"],
    },
    zip_safe=False,
)
