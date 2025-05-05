from setuptools import setup, find_packages

setup(
    name="deep-research-v",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "pydantic-ai>=0.1.9",
        "python-dotenv>=1.0.0",
        "anthropic>=0.3.0",
        "openai>=1.1.0",
    ],
) 