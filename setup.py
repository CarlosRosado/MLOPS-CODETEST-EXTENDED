from setuptools import setup

with open("VERSION") as f:
    version = f.read().strip()

setup(
    name="seedtag_text_classifier",
    version=version,
    packages=["seedtag_text_classifier"]
)
