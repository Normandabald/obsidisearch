# Obsidisearch

This is a Python app that provides an interface for a personal Obsidian notebook. The app allows a user to ask a question, and the app will answer it using the information in the notebook.

## Installation

To install the app, follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Create a `config.yaml` file with the required options (see `config.yaml` for an example).
4. Run the app by running `python obsidisearch.py`.

## Usage

To use the app, follow these steps:

1. Start the app by running `python obsidisearch.py`.
2. Enter a question when prompted.
3. The app will provide an answer based on the information in the notebook.
4. The app will create a data file in the `persistence_directory` directory. This file will be used to store the information that the app has learned from the notebook. If you want to reset the app or have added new notes, simply delete this file and restart the app.

## Configuration

The `config.yaml` file contains the following options:

- `openai_api_key`: Your OpenAI API key.
- `obisidian_directory`: The directory path where your Obsidian notes are stored.
- `persistence_directory`: The directory path where the app will store its data.
- `openai_model`: The OpenAI model to use for generating answers.
- `temperature`: The temperature to use when generating answers.
- `prompt`: The prompt to use when generating answers.