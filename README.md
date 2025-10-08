# Detecting ADR with Ontology Infused LLM

## Overview

This project focuses on detecting Adverse Drug Reactions (ADR) using a finely tuned Large Language Model (LLM) integrated with ontology. The combination of prompt engineering and a robust ontology aims to significantly enhance the performance of the model, achieving a recall metric of above 99%.

## Features

- **Prompt Engineering**: The project includes advanced prompt engineering techniques to optimize the LLM's performance for the specific task of detecting ADR.
- **Fine-tuned LLM**: We have fine-tuned an open-source 7B LLM to accurately detect ADR, ensuring high precision and recall.
- **Ontology Integration**: An ontology is used to ground the LLM, providing contextual understanding and enhancing its ability to detect ADR accurately.
- **High Recall**: The model achieves a recall metric of above 99%, ensuring that almost all relevant ADR cases are detected.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ryuzakace/ADE_ontology_LMM.git
    cd detecting-adr-ontology-llm
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset and ensure it is in the correct format.
2. Run the fine-tuning script:
    ```bash
    python finetune_llm.py --data_path path_to_your_data
    ```
3. Use the trained model to detect ADR:
    ```bash
    python detect_adr.py --model_path path_to_finetuned_model --input_path path_to_input_data
    ```

## Results

Our model achieves a recall metric of over 99%, making it highly reliable for detecting ADR in various contexts.

## Authors

Shirish Bajpai, Suman Roy, VSS Anirudh Sharma


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or feedback, please open an issue or contact us.


