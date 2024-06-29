# Alexander Texts Analysis

This project processes and analyzes ancient texts related to Alexander the Great using embedding techniques. It supports both BERT and OpenAI's GPT models for generating embeddings.

## Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package installer)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/nac-codes/alexander_db.git
   cd alexander-texts-analysis
   ```

2. Create a virtual environment:
   ```
   python -m venv env
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     .\env\Scripts\activate
     ```
   - On macOS and Linux:
     ```
     source env/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. If you're using the OpenAI GPT model, you need to set up your API key:
   - Create a `.env` file in the project root directory
   - Add your OpenAI API key to the file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

2. By default, the script will look for a folder named `all_primaries` in the current directory.

## Usage

1. To process texts and generate embeddings (optional if you've already processed the texts):
   ```
   python process_texts.py --model [model]
   ```
   - Replace `[model]` with either `bert` or `openai-gpt`.

   This script will process all `.txt` files in the all_primaries folder, generate embeddings, and save the results as NumPy arrays.

   Note: You only need to run this step if you're processing the texts for the first time or if you've made changes to the text files. Otherwise, you can skip to the analysis step.

2. To analyze the processed texts:
   ```
   python analyze_texts.py [model]
   ```
   - `[model]`: Either `bert` or `openai-gpt` (must match the model used in `process_texts.py`)

   This script will load the previously generated embeddings, find similar texts to your query, and display the results.

## Example

Process texts (if needed):
```
python process_texts.py --model openai-gpt 
```

Analyze texts:
```
python analyze_texts.py openai-gpt 
```

## Troubleshooting

- For BERT-related issues, ensure you have enough RAM available, as BERT models can be memory-intensive.
- If you get a "File not found" error, make sure you've either run `process_texts.py` first or that you're pointing to the correct folder containing the processed data.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
