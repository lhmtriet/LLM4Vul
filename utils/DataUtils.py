import os
import pandas as pd
from tree_sitter import Language, Parser
from utils.TSParse import TSParse
import shutil
from sklearn.model_selection import train_test_split
import json

# Remove comments functions
def get_current_folder():
    return os.getcwd()

def init_parser(working_folder, list_languages, tree_sitter_folder="utils/build"):
    
    tree_sitter_path = os.path.join(working_folder, tree_sitter_folder)

    parser = TSParse(treesitter_folder=tree_sitter_path, list_languages=list_languages)
    
    return parser

def create_library(lang, working_folder):
    
    Language.build_library(
        # Store the library in the `build` directory
        f'build/{lang}.so',

        # Include one or more languages
        [
            f'vendor/tree-sitter-{lang}'
        ]
    )

def remove_comments(parser, code, language):
    if language.lower() == 'swift':
        language = 'rust'
        
    return parser.noisy_lines(code, language)[1]


# Read data functions
def check_target_distribution(data, target_column='target', tab='\t'):
    # Print result
    print(f"{tab}Total data: {len(data)}")
    
    # Get the value counts
    distribution = data[target_column].value_counts().to_dict()
    
    for key, value in distribution.items():
        print(f"{tab*2}Target {key}: {value}")
    
    return distribution

def get_dataset_path(language: str, working_folder: str) -> str:
    # Capitalize the language name to ensure it starts with an uppercase letter
    formatted_language = language.capitalize()
    
    # Return the dataset path
    full_path = os.path.join(working_folder, f"data/{formatted_language}_data.parquet")
        
    return full_path
    
def extract_flaw_lines(code, list_flaw_index):
    if list_flaw_index.strip() == '':
        return ""
    list_flaw_index = [int(item) for item in list_flaw_index.split(",")]
    lines = code.split('\n')
    
    flaw_lines = [lines[i] for i in list_flaw_index]  # Adjusting for 0-based index
    return '/~/'.join(flaw_lines)

def compute_flaw_line(row):
    if row['label'] == 1:
        return extract_flaw_lines(row['code'], row['mod_lines'])
    else:
        return ""

def pre_process_data(data, parser, language):
    """
    Process the provided DataFrame by filtering, computing, and transforming data.
    
    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - parser (object): Parser used for removing comments.
    - language (str): Language identifier for removing comments.

    Returns:
    - pd.DataFrame: The processed DataFrame.
    """

    # Check if necessary columns exist in the DataFrame
    required_columns = ['before_change', 'label', 'added_only', 'code', 'mod_lines']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in the provided DataFrame.")
    
    # 1. Filter before_change
    data['before_change'] = data['before_change'].astype(str).str.lower() == 'true'
    filtered_data = data[data['before_change']].copy()

    # 2. Remove label = 1 and added_only = True
    # Convert the 'label' column to integer type
    filtered_data['label'] = filtered_data['label'].astype(int)
    filtered_data['added_only'] = filtered_data['added_only'].astype(str).str.lower() == 'true'
    
    condition = (filtered_data['label'] == 1) & (filtered_data['added_only'] == True)
    filtered_data.drop(filtered_data[condition].index, inplace=True)

    # 3. Compute flaw_line for each row in filtered_data
    filtered_data['flaw_line'] = filtered_data.apply(compute_flaw_line, axis=1)

    # 4. Remove comments
    # filtered_data['code'] = filtered_data['code'].apply(lambda code: remove_comments(parser, code, language))

    # 5. TODO: Remove multiple functions records (manual intervention required)

    # 6. Rename columns
    column_mapping = {
        'code': 'processed_func',
        'label': 'target',
        'mod_lines': 'flaw_line_index'
    }
    filtered_data.rename(columns=column_mapping, inplace=True)

    # 7. Choose columns
    return filtered_data[['processed_func', 'target', 'flaw_line_index', 'flaw_line']]

def read_language_dataset(language, working_folder, parser, data_path=None):
    
    # I. Read data
    formatted_language = language.lower()
    
    # Read from parquet or the specific path
    if data_path == None:
        data_path = get_dataset_path(language, working_folder)
    
        # Read dataset
        data = pd.read_parquet(data_path)
        
        csv_path = os.path.join(working_folder, f"data/{formatted_language}_data.csv")
        data.to_csv(csv_path, index=False)
    else:
        data = pd.read_csv(data_path)
        
    print("Data path:", data_path)
    
    # II. Pre-process
    data = pre_process_data(data, parser, language)

    # III. Save pre-processed data to csv
    csv_path = os.path.join(working_folder, f"data/{formatted_language}/data.csv")
    data.to_csv(csv_path, index=False)
    
    return data

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def build_paths(language, configuration_name, experiment_id):
    base_path = f'data/{language}/{configuration_name}/{experiment_id}'
    paths = {
        "train": f'{base_path}/train.csv',
        "val": f'{base_path}/val.csv',
        "test": f'{base_path}/test.csv',
        "train_undersampling": f'{base_path}/train_undersampling.csv',
        "val_undersampling": f'{base_path}/val_undersampling.csv',
        "train_oversampling": f'{base_path}/train_oversampling.csv',
        "val_oversampling": f'{base_path}/val_oversampling.csv'
    }
    return paths

def process_language_data_with_experiment_number(language, data, experiment_id, random_state, configuration_name, val_test_ratio=0.4, read_only=False):
    paths = build_paths(language, configuration_name, experiment_id)
    
    if read_only:
        datasets = {key: pd.read_csv(path) for key, path in paths.items()}
        return datasets
    
    # Split data
    train, remaining = train_test_split(data, test_size=val_test_ratio, random_state=random_state, stratify=data['target'])
    val, test = train_test_split(remaining, test_size=0.5, random_state=random_state, stratify=remaining['target'])
    
    # Undersampling
    train_undersampling = balance_dataframe(train)
    val_undersampling = balance_dataframe(val)
    
    # Oversampling
    train_oversampling = oversampling(train)
    val_oversampling = oversampling(val)

    ensure_dir(paths["train"])

    data_frames = {
        "train": train,
        "val": val,
        "test": test,
        "train_undersampling": train_undersampling,
        "val_undersampling": val_undersampling,
        "train_oversampling": train_oversampling,
        "val_oversampling": val_oversampling
    }

    for key, df in data_frames.items():
        df.to_csv(paths[key], index=False)

    return data_frames

# Balanced dataset
def balance_dataframe(df):
    # Determine the minimum number of records between the two targets
    min_count = min(df[df['target'] == 0].shape[0], df[df['target'] == 1].shape[0])

    # Randomly sample records from both groups
    df_0 = df[df['target'] == 0].sample(min_count, random_state=42)
    
    df_1 = df[df['target'] == 1].sample(min_count, random_state=42)
    
    # Concatenate the two dataframes and shuffle the rows
    balanced_df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df

def oversampling(df, target_column='target'):
    # Separate the two classes
    class_0 = df[df[target_column] == 0]
    class_1 = df[df[target_column] == 1]
    
    # Count the number of samples in both classes
    count_class_0 = len(class_0)
    count_class_1 = len(class_1)
    
    # Check which class has fewer samples and oversample it
    if count_class_1 < count_class_0:
        # Oversample the class_1 by randomly sampling with replacement
        oversampled_class_1 = class_1.sample(count_class_0, replace=True, random_state=42)
        df_oversampled = pd.concat([class_0, oversampled_class_1], axis=0)
    else:
        # If the '1' class has more or equal samples, then oversample the '0' class
        oversampled_class_0 = class_0.sample(count_class_1, replace=True, random_state=42)
        df_oversampled = pd.concat([class_1, oversampled_class_0], axis=0)
    
    # Shuffle the data to mix the rows
    df_oversampled = df_oversampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df_oversampled

# Fine-tuning
def create_fine_tuning_dataset(language, dataset, prompt_template):
    conversations = []
    
    # Convert all 'target' column values to integers
    dataset['target'] = dataset['target'].astype(int)

    for _, example in dataset.iterrows():
        
         # Populate user_content
        user_content = prompt_template['fine_tuning_template']['user_content_template'].format(
            new_source_code=example['processed_func'], language=language)

        # Determine analysis status
        if example['target'] == 1:
            analysis_status = "Vulnerable"
            
            # flaw_lines = example['flaw_line'].split('/~/')
            # flaw_indices = example['flaw_line_index'].split(',')
            
            # analysis_explanation = "Vulnerability detected.\n"
            
            # # Combining flaw lines and their respective indices into the desired format
            # for index, line in zip(flaw_indices, flaw_lines):
            #     analysis_explanation += f"Vulnerable Line {index}: {line}\n"
            analysis_explanation = f"[{example['flaw_line_index']}]"
        else:
            analysis_status = "Non-Vulnerable"
            analysis_explanation = "[]"

        
        # Populate assistant_content
        assistant_content = prompt_template['fine_tuning_template']['assistant_content_template'].format(
            analysis=analysis_status)
            #explanation=analysis_explanation)
        
        # print(assistant_content)
        # print("-------")
        
        system_content = prompt_template['system_content_template'].format(language=language)
        
        conversation = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        
        conversations.append(conversation)

    print(f"Number of records in fine-tuning json: {len(conversations)}")
    return conversations

def write_fine_tuning_json(fine_tuning_file, conversations):
    with open(fine_tuning_file, 'w') as file:
            for conversation in conversations:
                json.dump(conversation, file)
                file.write('\n')

    print(f"Conversations written to {fine_tuning_file}")
### OLD functions

# Split dataset
# Write train/test/val to files
# Copy train/test/val to linevul folder
def process_language_data(language, data, linevul_data_folder_path, val_test_ratio=0.4):
    # folder path
    folder_path = linevul_data_folder_path + f"/{language}_dataset"
    
    # Split data
    train, remaining = train_test_split(data, test_size=val_test_ratio, random_state=0)
    val, test = train_test_split(remaining, test_size=0.5, random_state=0)
    
    # Bigtrain
    big_train = pd.concat([train, val], ignore_index=True)
    
    # Undersampling
    train_balanced = balance_dataframe(train)
    val_balanced = balance_dataframe(val)
    
    # Oversampling
    train_oversampling = oversampling(train)
    val_oversampling = oversampling(val)
    
    # Save to CSV files
    train_path = f'data/{language}/train.csv'
    val_path = f'data/{language}/val.csv'
    test_path = f'data/{language}/test.csv'
    train_balanced_path = f'data/{language}/train_balanced.csv'
    val_balanced_path = f'data/{language}/val_balanced.csv'
    train_oversampling_path = f'data/{language}/train_oversampling.csv'
    val_oversampling_path = f'data/{language}/val_oversampling.csv'
    
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    train_balanced.to_csv(train_balanced_path, index=False)
    val_balanced.to_csv(val_balanced_path, index=False)
    train_oversampling.to_csv(train_oversampling_path, index=False)
    val_oversampling.to_csv(val_oversampling_path, index=False)

    # Define source and destination paths
    sources_dests = [
        (train_path, os.path.join(folder_path, 'train.csv')),
        (val_path, os.path.join(folder_path, 'val.csv')),
        (test_path, os.path.join(folder_path, 'test.csv')),
        (train_balanced_path, os.path.join(folder_path, 'train_balanced.csv')),
        (val_balanced_path, os.path.join(folder_path, 'val_balanced.csv')),
        (train_oversampling_path, os.path.join(folder_path, 'train_oversampling.csv')),
        (val_oversampling_path, os.path.join(folder_path, 'val_oversampling.csv'))
    ]

    # Copy files
    for src, dest in sources_dests:
        shutil.copy(src, dest)
        print(f"Copied '{src}' to '{dest}'")
        
    # Print result
    # print()
    # print(f"Train: {len(train)}")
    # display(train.value_counts('target'))
    # print(f"Val: {len(val)}")
    # display(val.value_counts('target'))
    # print(f"Test: {len(test)}")
    # display(test.value_counts('target'))
    
    # print(f"BigTrain: {len(big_train)}")
    # display(big_train.value_counts('target'))
    # print(f"Balanced Train: {len(train_balanced)}")
    # display(train_balanced.value_counts('target'))
    # print(f"Balanced Val: {len(val_balanced)}")
    # display(val_balanced.value_counts('target'))
    
    return train, val, test, big_train, train_balanced, val_balanced, train_oversampling, val_oversampling_path



