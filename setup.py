import os
from setuptools import find_packages, setup


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

def _parse_requirements(path):
  with open(os.path.join(ROOT_PATH, path)) as f:
    return [
        line.rstrip()
        for line in f
        if not (line.isspace() or line.startswith('#'))
    ]

def _parse_long_description():
  with open(os.path.join(ROOT_PATH, 'README.md')) as f:
    return f.read()


if __name__ == "__main__":
    setup(
        name="autoxgb",
        description="autoxgb: tuning xgboost with optuna",
        long_description=_parse_long_description(),
        long_description_content_type="text/markdown",
        author="Abhishek Thakur",
        author_email="abhishek4@gmail.com",
        url="https://github.com/abhishekkrthakur/autoxgb",
        license="Apache 2.0",
        package_dir={"": "src"},
        packages=find_packages("src"),
        entry_points={"console_scripts": ["autoxgb=autoxgb.cli.autoxgb:main"]},
        include_package_data=True,
        install_requires=_parse_requirements('requirements.txt'),
        platforms=["linux", "unix"],
        python_requires=">=3.6",
    )
