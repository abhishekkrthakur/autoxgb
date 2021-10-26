from setuptools import find_packages, setup


with open("README.md") as f:
    long_description = f.read()

INSTALL_REQUIRES = [
    "optuna>=2.10.0",
    "xgboost>=1.5.0",
]

if __name__ == "__main__":
    setup(
        name="autoxgb",
        description="autoxgb: tuning xgboost with optuna",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Abhishek Thakur",
        author_email="abhishek4@gmail.com",
        url="https://github.com/abhishekkrthakur/autoxgb",
        license="Apache 2.0",
        package_dir={"": "src"},
        packages=find_packages("src"),
        entry_points={"console_scripts": ["autoxgb=autoxgb.cli.autoxgb:main"]},
        include_package_data=True,
        install_requires=INSTALL_REQUIRES,
        platforms=["linux", "unix"],
        python_requires=">3.5.2",
    )
