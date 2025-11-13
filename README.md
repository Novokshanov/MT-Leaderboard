# Machine Translation Leaderboard

A Gradio-based web application for visualizing and analyzing machine translation benchmark results across multiple models and datasets.

## Features

- **Interactive Visualization**: View translation scores across multiple models and translation directions
- **Flexible Filtering**: Select specific benchmark datasets, metrics, translation directions, and models
- **Table Transposition**: Switch between viewing directions as rows vs. columns
- **Dynamic Sorting**: Sort results by any column in ascending or descending order
- **Responsive Design**: Fixed first column for easy reference while scrolling horizontally
- **Colorization**: Colorize scores according to customizable zones of quality

## Supported Metrics

The application supports common machine translation evaluation metrics:

- **BLEU**: Bilingual Evaluation Understudy - measures n-gram overlap between translation and reference
- **chrF**: Character F-score - measures character-level n-gram precision and recall
- **chrF++**: Enhanced version of chrF with additional parameters
- **Meteor**: Metric for Evaluation of Translation with Explicit ORdering - considers synonyms and stemming
- **Comet-wmt22**: COMET (Crosslingual Optimized Metric for Evaluation of Translation) from WMT 2022
- **XComet-XXL**: Extended COMET model with larger architecture

\* All metrics except chrF and chrF++ are multiplied by 100 for better readability and unified format.

## Covered Languages

We have determined automatic metrics for translation direction in conjunction with Russian for the following languages: Azerbaijani, Armenian, Bashkir, Belarussian, Chuvash, Dari, English, Erzya, French, German, Hebrew, Hindi, Indonesian, Italian, Japanese, Kazakh, Korean, Kyrgyz, Mongolian, Modern Standard Arabic, Persian, Portuguese, Pashto, Simplified Chinese, Spanish, Tajik, Tatar, Thai, Turkmen, Turkish, Ukrainian, Uzbek, Vietnamese and Yiddish.

## Benchmarked Models

* Traditional Translation Models:
 - NLLB-200 Family
 - Madlad-400 Family
 - Mitre 913m
* LLM-based Translation Models:
 - Seed-X-Pro 7b
 - Hunyuan-MT 7b
 - X-ALMA
* LLMs:
 - Gemma 3 27b
 - Aya Expanse 32b
 - Qwen3 Family
 - Deepseek R1 distill llama 70b
 - Deepseek R1 distill qwen
 - LLaMA 3.3 70b
 - gpt-oss 120b
 - gpt-oss 20b
 - GigaChat A3B Instruct 20b
 - Yandex GPT 5 Lite 8b Instruct
 - T-Pro-it-2.0
 - Vistral 24b Instruct
* Proprietary Systems
 - m-translate
 - ...

## Getting Started

### Directory Structure

The application expects benchmark datasets in subdirectories with names containing `-scores`:

```
mt-benchmarking-gradio/
├── benchmark1-scores/
│   ├── model1.csv
│   ├── model2.csv
│   └── ...
├── benchmark2-scores/
│   ├── model1.csv
│   ├── model2.csv
│   └── ...
└── app.py
```

### CSV Format

Each model's results should be stored in a CSV file with the following structure:
- First column: `Direction` (e.g., `eng_Latn_2_rus_Cyr`, `fra_Latn_2_eng_Latn`)
- Other columns: metric scores (e.g., `BLEU`, `chrF`, `Meteor`, etc.)

Example:
```csv
Direction,BLEU,chrF,chrF++,Meteor
eng_Latn_2_rus_Cyr,28.5,0.45,0.47,0.62
fra_Latn_2_eng_Latn,32.1,0.48,0.50,0.65
```

### Launching the Application

1. Clone or download the repository
2. Install required dependencies:
   ```bash
   pip install gradio pandas
   ```
3. Place your benchmark results in appropriately named subdirectories (with `-scores` in the name)
4. Run the application:
   ```bash
   python app.py
   ```
5. The application will launch in your default browser at `http://localhost:7860`

## Interface Guide

### Main Controls

1. **Benchmark Dataset**: Select which benchmark dataset to analyze
2. **Automatic Metric**: Choose which evaluation metric to display
3. **Translation Directions**: Select specific directions to include (leave empty for all)
4. **Models**: Select specific models to include (leave empty for all)
5. **Transpose Table**: Toggle between directions as rows vs. columns
6. **Colorize Scores**: Colorize scores according to customizable zones of quality

### Sorting Controls

1. **Sort By Column**: Select which column to sort the results by
2. **Sort Order**: Choose ascending or descending order

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request to add new features, improve documentation, or fix bugs.

## Contact

For questions or support, please open an issue in the repository.