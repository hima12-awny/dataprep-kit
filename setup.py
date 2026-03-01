from setuptools import setup, find_packages

setup(
    name="dataprep-kit",
    version="0.1.1",
    description="AI-powered data preparation toolkit built with Streamlit",
    author="Ibrahim Awny",
    author_email="hima12awny@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "streamlit>=1.32.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
        "openpyxl>=3.1.0",
        "xlsxwriter>=3.1.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "plotly>=5.18.0",
        "jsonschema>=4.20.0",
        "pyarrow>=14.0.0",
        "uuid6>=2024.1.12",
        "streamlit-sortables>=0.3.0",
        "pydantic-ai>=0.0.14",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    project_urls={
        "LinkedIn": "https://www.linkedin.com/in/ibrahim-awny/",
        "Source": "https://github.com/hima12awny/dataprep-kit",
    },
)
