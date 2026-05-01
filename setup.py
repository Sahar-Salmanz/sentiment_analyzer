from setuptools import setup, find_packages


setup (
    name = "sentiment_analysis",
    version = "0.1.0", 
    description = "Multi-model NLP sentiment analysis with explainability", 
    author = "SaharSalmanz",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "app": ["gradio>=4.0.0"],
        "explain": ["shap>=0.43.0"],
        "dev": ["pytest>=7.4.0", "flake8"],
    },
)